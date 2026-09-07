---
name: L1-online-migration-race
description: "第一层难点3——GlobalXTensor 段尾环绕时的在线迁移(边搬家边住人),分配线程和搬迁线程通过3个atomic+1个裸bool(allocate_offset_migrated_)握手,那个裸bool被两线程无锁读写是真实数据竞争。并发难度天花板。"
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

# 难点3:在线 migration 的半原子握手

场景见 [[L1-global-xtensor-ring]]。代码 `global_xtensor.cpp:197-246`。

## 触发与搬运区间(易错点!)
`free_to_right_async` 里 `free_offset_` 撞到 `total_size_`(整个空间总末尾)→ 触发迁移。

**触发瞬间的排布**(此刻 `free_offset_ == total_size_`):
```
infer_start        allocate_offset_                    total_size_(=free_offset_)
   │                     │                                    │
   ├────[已分配/在用]─────┼──────[可用区:已map待分配]──────────┤
   │  (左侧,不搬)         └────── 搬的就是这段 ──────────────┘
```

**搬运区间 = `(allocate_offset_, total_size_)`,即 allocate 右侧到段尾**(代码:`migration_src_end = allocate_offset_` :216,`migration_src_next` 从 `total_size_-page` 往下,循环条件 `next > end` :217)。

⚠️ **两个关键订正(别再搞错)**:
1. 搬的**不是"正在用的页"**,而是**"已 map 好、还没分配出去的可用页"**(allocate 与 free 之间那段 = 可用区库存)。allocate **左侧**真正在用的页 **完全不碰**。
2. 触发时刻,"allocate 和 free 之间"就是可用区本身,不存在"另一批已用页"。搬它是因为 free 环绕回段头后,要把这批未消费库存挪到环头让可用区重新连续。

搬法是 `move_one_page` 的 **zero-copy 重映射**(unmap 旧 offset + map 同一物理页到段头新 offset),**数据一字节不动**;但逐页 `vmm::map` 仍阻塞、有开销。**同时 `allocate_from_left` 还在继续分配** —— 边搬家边住人。

## 握手协议(赛跑)
两个线程靠 4 个共享状态协调:
| 状态 | 类型 | 谁写 |
|---|---|---|
| `migration_in_flight_` | atomic | 搬迁线程 |
| `migration_src_next_` | atomic | 搬迁线程 fetch_sub |
| `migration_src_end_` | atomic | 两边都写(追踪住人边界) |
| `allocate_offset_migrated_` | **裸 bool** | **两边都写** ⚠️ |

- `maybe_switch_to_migration_dst`(`:237`,**不持 `mtx_`**):分配方发现要越过迁移前沿,主动把 `allocate_offset_` 跳到 dst 头,置 `allocate_offset_migrated_=true`。
- 搬迁线程(`:217-224`,**持 `mtx_`**):一边搬一边 `migration_src_end_.store(allocate_offset_.load())`,并读/写 `allocate_offset_migrated_`。

## 真实 bug
`allocate_offset_migrated_`(`.h:144`)是**非原子 bool**,被搬迁线程(持 `mtx_`)和分配线程(`maybe_switch_to_migration_dst` **不持 `mtx_`**)**同时读写** —— 这是一个 data race,UB。3 个 atomic 混 1 个裸 bool,协议就破功了。`7661f522`(extreme circumstances 下 migration 失败)几乎肯定是这类赛跑的产物。

## 内化经验
> **"在线数据迁移"是并发难度的天花板**,因为搬迁目标同时是活跃工作集。
>
> **通用规则**:握手协议里**所有共享状态必须同一种同步纪律** —— 不能一部分 atomic、一部分裸变量。凡"在线搬迁 + 并发分配":要么全程一把锁(慢但对),要么用严格验证过的无锁协议(难),**绝不能半原子半裸**。

关联:可能打破 [[L1-map-page-check-then-act]] 的 offset 唯一契约;设计背景见 [[L1-global-xtensor-ring]]。
