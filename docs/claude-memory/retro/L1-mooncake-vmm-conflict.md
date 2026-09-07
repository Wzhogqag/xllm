---
name: l1-mooncake-vmm-conflict
description: "第一层横切难点——Mooncake(RDMA)要求注册的内存有稳定 pinned 物理后备,但 GlobalXTensor 的机制是大虚拟地址+物理页动态流转,两者核心假设冲突。项目用独立 weight region + 先map后register + 按模型切片来局部化解。跨 xtensor 与 kv_cache 两层。"
metadata:
  node_type: memory
  type: project
  tags:
    - retro
    - multimodel
    - xtensor
    - mooncake
    - d2d
    - layer1
---

# 难点(横切):Mooncake D2D 与 VMM 流转的核心假设冲突

跨 [[xtensor_memory]](VMM 底盘)和 kv_cache 层。代码:`mooncake_weight_transfer.cpp`、`xtensor_allocator.cpp`。

## 冲突本质:RDMA 注册 vs 物理页流转,天生互斥

- **Mooncake 底层是 RDMA**:注册给它的内存必须有**稳定的 pinned 物理后备**,网卡才能拿到可 DMA 的物理地址、建立 memory region(MR)。
- **GlobalXTensor 的机制是"大虚拟地址 + 物理页动态流转"**(见 [[L1-global-xtensor-ring]]):同一虚拟地址此刻有物理页、下一刻被 unmap 还回池子。
- **矛盾**:把流转的虚拟地址注册给 Mooncake,网卡记下的物理地址随时失效 → 传输时读到**空洞或错页**。这就是"没绑定物理页的地址无法传输"。

> 项目最初设想:权重一开始就绑定好物理页 + 把地址注册给 Mooncake 就能传。但权重区**不可能全部预绑定物理页**(整个机制就是靠大虚拟空间 + 物理页流转省显存),矛盾由此暴露。

## 解法:不让权重走 infer arena 的流转,而是划独立 region + 收紧时序

**关键前提:权重区和激活/KV 是两套独立 region。**
- infer arena(环形流转)= `GlobalXTensor`,见 [[L1-global-xtensor-ring]]。
- 权重区 = `XTensorAllocator::weight_xtensor_`,**独立的 220GB 虚拟段**(`xtensor_allocator.cpp:1168`),按 128GB 段对齐(`:1194-1198`)。Mooncake 只注册**后者**,不碰 infer arena。

**四步化解:**
1. **先 map 物理页,再注册(after map)—— 时序是关键。**
   `alloc_weight_pages_local` 先给该模型权重逐页 `map_external_page`(`:1224-1248`)绑定物理页,`weight_base_ptr` 指向已绑定的连续虚拟段(`:1200`);**之后**才回调 `mooncake_weight_register_fn_`。这样 Mooncake 注册的地址底下**确实有 pinned 物理页**。`mooncake_weight_transfer.cpp:51` 注释 "per-model register_model_weight_slice **after map**" 就是强调这个顺序。
2. **按模型切片注册,不注册整个大虚拟空间。**
   `register_model_weight_slice`(`mooncake_weight_transfer.cpp:56`)每次只注册**一个模型的权重段** `[weight_base_ptr, +num_pages*pgsz)`(`:71-75`),只有真正绑好物理页的小段进 MR。
3. **幂等 + 记 buffer index**:`mooncake_weight_buffer_index >= 0` 就跳过(`:68`);注册后存 buffer ordinal 进 `ModelTensors`(`:80`),传输按 model_id 反查(`pull_weights:141`)。
4. **对称部署简化寻址**:`pull/push_weights` 里 `remote_i = local_i`(`:150`、`:177`),假设两端同一模型用相同 buffer 序号,省掉跨节点 buffer 映射协商。

## 残留代价(矛盾只是被限制,没消除)

1. **权重区被迫牺牲流转灵活性**:权重页一旦 map+register 就得保持 pinned,不能像 infer arena 自由流转。是对"大虚拟地址+流转"初衷的**局部妥协**(infer arena 流转,weight region 相对固定)。
2. **与 layer offload 直接打架**:`unmap_weight_region`(`:1338`)是分层卸载要 unmap 权重页的,但这些页**已注册进 MR**,unmap = 注册失效 → offload 与"已注册"之间必然要 re-register / 时序协调。这是 [[L1-online-migration-race]] 那类"异步搬迁 vs 稳定引用"矛盾在**权重区的翻版**。
3. **对称假设很脆**:`remote_i = local_i` 要求两端加载/buffer 分配顺序完全一致;fork/异构并行下顺序错位会 pull 到错误远端 buffer —— **静默传错权重、不崩溃**(呼应 [[00-multimodel-retro-moc]] 元经验3:静默正确性 bug 最贵)。

## ⚠️ 当前状态:注册调用被整段注释掉

`xtensor_allocator.cpp:1258-1268` 里调用 `mooncake_weight_register_fn_` 的那段被 `/* ... */` **整段注释**(`#if defined(USE_NPU)` 块)。也就是说:**当前代码路径下权重不会真正注册给 Mooncake**,D2D 权重传输实际处于关闭/实验态。这与第一层其他"占位/注释"疤痕(admission gate 恒 false、restore 分支注释)是**同一类现象** —— 底层还在震荡时,上层能力先关掉保评估。相关 commit:`fa16c68f`、`94edfca9`(support weight D2D via Mooncake)。

## 内化经验
> **当两个子系统的核心假设冲突(动态流转 vs 稳定注册),别试图让一方"适配"另一方,而是划出一块专用区、收紧时序,把冲突局部化。** 代价是该区放弃另一方的灵活性,且交界处(此处是 offload 要 unmap 已注册页)会长期是张力点。

关联:底盘分区 [[L1-global-xtensor-ring]];同源的搬迁-vs-稳定引用矛盾 [[L1-online-migration-race]];静默正确性风险见 [[00-multimodel-retro-moc]]。
