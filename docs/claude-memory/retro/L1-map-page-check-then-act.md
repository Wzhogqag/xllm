---
name: L1-map-page-check-then-act
description: "第一层难点2——GlobalXTensor::map_page 在 check(offset未映射) 和 act(vmm::map+登记) 之间释放锁,正确性依赖\"调用方保证 offset 唯一\"这个未写进代码的契约。release build 里 CHECK 被跳过会直接 double-map。"
metadata:
  node_type: memory
  tags:
    - retro
    - multimodel
    - xtensor
    - concurrency
    - layer1
  type: project
---

# 难点2:`map_page` 的 check-then-act 竞态窗口

场景见 [[L1-global-xtensor-ring]]。代码 `global_xtensor.cpp:137-150`。

## 结构
```cpp
{ std::shared_lock lock(page_map_mtx_);       // ① 读锁:确认 offset 没被映射
  CHECK(page_map_.find(offset) == end); }
vmm::map(vaddr, phy_handle);                   // ② 无锁:真正映射(最慢!)
{ std::unique_lock lock(page_map_mtx_);        // ③ 写锁:登记 page_map_[offset]=page
  page_map_[offset] = page; }
```

①和③之间**放锁了**,中间夹着最慢的 `vmm::map`。经典 **check-then-act**:两个线程可能同时通过①的 CHECK,然后都去 map 同一个 offset → double-map。

## 为什么当前没爆
offset 由原子 `allocate_offset_` / `free_offset_` **串行发号**,上层保证不同线程拿到的 offset 不同 —— **靠上层不冲突兜底,而不是这层自己保证**。这个契约**没写进代码**。

## 危险点
- release build 里 `CHECK` 被编译掉 → 保护消失。
- 一旦有新路径不遵守发号契约(比如 migration 和正常分配并发写同一 offset,见 [[L1-online-migration-race]]),直接 double-map,难查。

## 内化经验
> **"暂时安全"≠"设计安全"。** 正确性依赖"未写进代码的契约"是定时炸弹。
>
> **通用规则**:check 和 act 之间释放锁,必须能**证明**"这个 key 在此期间不可能被别人碰";证不出来,就别放锁(改成 map 全程持锁),或用 per-key 锁。用慢操作(`vmm::map`)当理由放锁,是在拿正确性换性能 —— 至少要把契约写成断言/文档。

关联:[[L1-free-offset-dual-meaning]](同 map_page 的锁语义)、[[L1-online-migration-race]](可能打破 offset 唯一契约的路径)。
