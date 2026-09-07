---
name: L1-alloc-hidden-rpc-block
description: "第一层难点4——GlobalXTensor::wait_enough_pages 在推理热路径上分配页时,页不够会同步阻塞在跨节点 emergency_eviction RPC + 对端逐层 offload 上。这是把第一层和第三层(SLO策略落不了地)串起来的因果桥。"
metadata:
  node_type: memory
  tags:
    - retro
    - multimodel
    - xtensor
    - slo
    - layer1
    - bridge
  type: project
---

# 难点4:`wait_enough_pages` 热路径藏同步 RPC(因果桥)

场景见 [[L1-global-xtensor-ring]]。代码 `global_xtensor.cpp:362-407`。

## 现象
`allocate_from_left`(推理热路径)→ 页不够 → `wait_enough_pages` 里:
- **单机**(`nnodes==1`):直接调 `PageAllocator::emergency_eviction`(`:384`)—— 选模型逐层 offload。
- **多机**:发**同步 RPC** 等对端驱逐(`:388-395`,`.get()` 阻塞)。

也就是说:**一次激活分配,可能同步阻塞在"跨节点 RPC + 对端逐层 offload"上。**

且熔断缺失:`emergency_eviction_count_ >= 100` 只打个 ERROR log 继续转(`:392`),不真正停。

## 为什么这是"桥"
这条直接解释了**第三层 SLO 策略为什么落不了地**:
- forward admission 要按 deadline 预测每个 batch 的 cost。
- 但**分配本身能同步阻塞在它管不到的下层救火上** → 最大延迟来自 allocate 内部的隐藏阻塞 → **上层的时延预测模型必然失效**。
- 于是策略层只能被关掉(见待写的 `admission-gate-always-false` / `policy-needs-stable-substrate`)。

## 内化经验
> **热路径上的隐藏阻塞是 SLO 杀手。** 把"救火"藏在 `allocate` 里,上层再精确的 deadline 计算都不准。
>
> **通用规则**:热路径的资源获取,要么保证 O(1) 非阻塞,要么把慢路径**显式异步化并让上层可见**(返回"需等待"让调度器决策,而不是就地 block)。隐藏的同步阻塞 = 上层 SLO 全废。

关联:上承 [[L1-global-xtensor-ring]];下接第三层(策略退化),见 [[00-multimodel-retro-moc]] 第三层。异步化引入竞态的另一面见 [[L1-free-offset-dual-meaning]]。
