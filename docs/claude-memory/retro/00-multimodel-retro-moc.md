---
name: 00-multimodel-retro-moc
description: "多模型 colocation 实现难点复盘的总索引(MOC)。自底向上分三层:显存地基→协调→SLO策略。想回顾\"这条分支难在哪、踩过什么坑、内化成什么经验\"时从这里进。"
metadata:
  node_type: memory
  tags:
    - retro
    - multimodel
    - moc
  type: project
---

# 多模型 Colocation 难点复盘 · MOC

> [!abstract] 一句话总纲
> **多模型 colocation 的本质困难,是把单模型时代所有"隐式独占假设"逐一打破。**
> 单模型独占整卡、权重常驻、激活交给 PyTorch allocator、无人抢页、无优先级 —— 这五个假设被推翻后,每个"想当然"都退化成竞态 / 时序 / 再平衡问题。

复盘按**依赖顺序自底向上**组织:上层的每个难点都站在下层的肩膀上。

## 面试讲解版(合成产出)
- [[interview-multimodel-colocation]] —— 完整故事 + 各模块设计取舍 + 高频追问 Q&A + 对比 vLLM/main;侧重负责模块(prefix cache / 权重 offload / D2D)
- [[interview-5min-and-qa]] —— **背诵版**:5 分钟口播稿 + 20 个高频问答(精炼到能张口就来)

## 第一层 · 显存地基(VMM)—— 最深、救火最多
基于 [[xtensor_memory]] 的 VMM 底盘。核心是 `GlobalXTensor` 这个"环形 + 在线迁移"分配器。

- [[L1-process-thread-topology]] —— 背景板:进程/线程/RPC 拓扑(谁和谁真并发)。**竞态笔记的公共前提,先读**
- [[L1-global-xtensor-ring]] —— 心智模型:双向 offset 追逐 + 段尾环绕搬迁
- [[L1-free-offset-dual-meaning]] —— 难点1:`free_offset_` 一变量两义(作者 TODO 亲自求救)
- [[L1-map-page-check-then-act]] —— 难点2:`map_page` 放锁靠"未写进代码的契约"兜底
- [[L1-online-migration-race]] —— 难点3:在线迁移的半原子握手(混用 atomic 和裸 bool)
- [[L1-alloc-hidden-rpc-block]] —— 难点4:`wait_enough_pages` 热路径藏同步 RPC(**串起第三层的因果桥**)
- [[L1-lazy-unmap-tradeoff]] —— lazy unmap 动机(unmap 比 map 慢好几倍)与代价(阻塞换形式回到热路径);异步化不消灭慢操作只搬地方
- [[L1-mooncake-vmm-conflict]] —— 横切难点:Mooncake D2D(RDMA 稳定注册)与 VMM 物理页流转的核心假设冲突;注册调用当前被注释

## 第二层 · 协调层(offload/forward 时序)—— 待深潜
对应 commit `8eb9ee99→c893f615→bb8cf4fb`(负载驱动加载分 3 次才做完)。
核心问题:**权重正被逐层搬走时,forward 怎么保证不读到半张 unmap 的权重。**
- 待写:`layer-offload-forward-handshake`
- 待写:`async-copy-silent-correctness`(missing sync 反复出现,静默错数据)

## 第三层 · 策略层(SLO / 优先级)—— 待深潜
对应 commit `6a4fe80e`、`f7655e76`、`e5d15d43`。
核心反思:**底层还在震荡时,自动策略层被迫退化成占位/被注释。**
- 待写:`admission-gate-always-false`(`enable_prism && !enable_prism` 恒假)
- 待写:`policy-needs-stable-substrate`(为什么策略层落不了地 ← [[L1-alloc-hidden-rpc-block]])

## 横切放大器
- 待写:`tp-is-a-bug-amplifier`(几乎每个特性都跟一个 "under TP" 补丁)
- 待写:`long-branch-git-hygiene`(重复 commit = rebase 失控信号)

## 元经验速查(一页纸带走)
1. 先问"**哪个独占假设被打破了**" —— 最快的定位框架
2. 异步化 = 主动引入竞态,必答"释放前谁会用""用前是否同步好"
3. **静默的正确性 bug 比崩溃更贵** —— 异步拷贝加显式 sync + 断言,让错误尽早 crash
4. **底层没稳,别上自动策略层**
5. 难点指纹:"分多个 commit 才做完" + "反复 under-TP 补丁"
6. **先对后快**:粗锁保正确,跑通再换细粒度协调
7. 正交开关别合并、语义别反向(否则埋 `enable_prism && !enable_prism` 恒假坑)
