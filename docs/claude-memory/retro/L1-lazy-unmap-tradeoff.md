---
name: l1-lazy-unmap-tradeoff
description: "第一层——lazy unmap 的真实动机(实测 unmap 比 map 慢好几倍,同步 unmap 直接阻塞推理热路径,故甩到后台异步)与代价(页堆在 unmap 队列'在途',池子被掏空时阻塞换形式回到热路径)。核心洞察:异步化不消灭慢操作,只搬地方。"
metadata:
  node_type: memory
  type: project
  tags:
    - retro
    - multimodel
    - xtensor
    - concurrency
    - performance
    - layer1
---

# lazy unmap:动机与代价(为什么快 + 为什么仍会阻塞)

场景见 [[L1-global-xtensor-ring]]。这条补上前几条竞态笔记只讲"后果"、没讲"为什么要 lazy"的动机。

## 真实动机(一手实测):unmap 比 map 慢好几倍

> 实测:**unmap 的耗时是 map 的好几倍**。若推理要页时**同步** unmap 旧页腾地方,推理线程直接卡在 unmap 上 → TTFT/TPOT 被大幅拖高。

所以把 unmap 改成**异步**:热路径只把页"扔进 unmap 队列"就立刻返回(轻),后台 `unmap_worker` 线程慢慢真正 unmap。这是经典的**关键路径卸载**——热路径只做最轻的事(入队),重活甩后台。

### 为什么 unmap 比 map 慢(底层原因,讲这个加分)
- **TLB shootdown**:map 只是往页表加条目;unmap 要**让所有核缓存的旧地址翻译失效**,需发核间中断(IPI)通知每个核、等确认——跨核同步,天然慢。
- **页表拆解 + 物理页归还**:map 是"填表",unmap 是"拆表 + 归还物理页给驱动/池子",路径更长。
- **驱动/NPU 侧同步**:`vmm::unmap` 底层常要确保设备端无正在进行的访问,可能带隐式同步。

一句话:**map 是"登记一下"(轻),unmap 是"注销 + 通知全世界地址作废 + 归还资源"(重)。**

## 代价:阻塞没被消灭,只是换了形式

lazy unmap 本为**不阻塞**推理,但物理页有限:

```
页扔进 unmap 队列 → 还没真正还回池子(处于"在途")
  → 池子可用页变少
  → 推理要新页时池子空了(页都卡在队列里"在途")
  → 推理只能等后台消化队列 → 又被阻塞 💥
```

**你们的"map 优先"策略雪上加霜**:`unmap_worker` 里 `while (pending_free_to_right_tasks_ == 0)` 才干活——只要有 map 任务,unmap 就让路。map 多时更快,但 unmap 被无限推迟、队列越堆越高,最后 migration 环绕时撞上"起点页还没 unmap 完"→ CHECK 崩溃(见 [[L1-online-migration-race]])。

## 核心洞察(面试金句)

> **异步化从来不会"消灭"慢操作,只会把它"搬个地方"。** unmap 还是那么慢,lazy 没让它变快——只是从"推理热路径"搬到"后台线程"。代价是:**若后台消化速度 < 产生速度,阻塞会以另一种形式(拿不到页)回到热路径。** 用"在途显存"换"低时延",这个交换有上限——在途页堆到掏空池子,时延就回来了。

## 成立条件(三量平衡)
```
平均: unmap 速度 ≥ 还页速度  → 队列不堆积,系统稳
瞬时: 还页速度 > unmap 速度  → 队列堆积 → 迟早掏空池子 → 阻塞回来
缓冲: 空闲页 + 在途容忍度    → 决定能扛多久突发
```
所以它是**优化,非根治**:用"更多在途显存"+"平均 unmap 追得上"两个假设换低时延;没消除 unmap 慢这个物理事实,极端情况(突发还页 / map 优先饿死 unmap)会破功。这与延迟回收(SMR)类方案的内在 blocking 属性是同一回事。

关联:根子是意图/事实错位 [[L1-free-offset-dual-meaning]];环绕撞车后果 [[L1-online-migration-race]];底盘 [[L1-global-xtensor-ring]]。
