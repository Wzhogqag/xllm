---
name: L1-global-xtensor-ring
description: "GlobalXTensor 的心智模型——一个\"环形缓冲 + 在线迁移\"的虚拟显存分配器。理解第一层所有难点的前提。文件:xllm/core/framework/xtensor/global_xtensor.cpp。"
metadata:
  node_type: memory
  tags:
    - retro
    - multimodel
    - xtensor
    - layer1
  type: project
---

# GlobalXTensor:环形 + 在线迁移分配器(心智模型)

第一层所有难点的**共同舞台**。读 [[L1-free-offset-dual-meaning]]、[[L1-map-page-check-then-act]]、[[L1-online-migration-race]]、[[L1-alloc-hidden-rpc-block]] 之前先建立这个模型。底盘概念见 [[xtensor_memory]]。

## 地址布局

```
vaddr_ ┌────────────────────┬──────────────────────────────────────┐
       │  init arena (384GB) │  infer arena                          │
       │  启动期权重          │  ← allocate_offset_ 从左长出(激活/KV)  │
       │                     │  free_offset_ 异步回收的页往这填 →      │
       └────────────────────┴──────────────────────────────────────┘
                              ↑ infer_arena_start_
```

- **左端 `allocate_offset_`**(atomic):分配点,`fetch_add` 往右走。
- **右端 `free_offset_`**:回收点,异步 unmap 线程把回收的页往右填映射。
- 两者**在同一段内同向追逐**:
  - `allocate` 追上 `free` → 页不够 → `wait_enough_pages` 等待(见 [[L1-alloc-hidden-rpc-block]])。
  - `free_offset_` 走到**整个空间总末尾 `total_size_`** → 触发 **migration**:环绕回段头,把 allocate 右侧到段尾那段**已 map 待分配的可用页**(不是在用页!)zero-copy 搬到段头(见 [[L1-online-migration-race]])。**注意:不是每个 128GB 段尾都 migration,见下节。**

## ⚠️ 两套"段尾"机制,边界不同(极易混淆)

代码里有两套不同的段尾处理,**触发边界不一样**,别混为一谈:

| | 触发边界 | 做什么 | 代码 |
|---|---|---|---|
| **allocate 跳段** | 每个 **128GB** 段尾 | `allocate_offset_` 直接跳到下一段开头,并 unmap+回收 `[old_offset, seg_end)` 的残留页。**不是 migration** | `allocate_from_left` CAS 循环 `:257-301` |
| **free 环绕 migration** | 整个 **`total_size_`**(所有段拼起来的总末尾) | `free_offset_` 环绕回 `infer_arena_start_`,把 `(allocate_offset_, total_size_)` 的**可用区待分配页** zero-copy 搬到段头(**非在用页**) | `free_to_right_async` `:207-229` |

**为什么边界不同(关键)**:
- 每个 128GB 段是**独立 `vmm::create_vir_ptr` 预留**的,虚拟地址未必真连续。
- `allocate` 一次分配是**多页连续块**(`count * page_size_`),不能跨两个独立预留的段 → 放不下就必须在 128GB 边界**跳段**。
- `free_offset_` 是**逐页** map(每次恰好 `page_size_`、页对齐),单页永不跨段 → 能安全 march 穿过所有 128GB 边界,直到 `total_size_` 才环绕。

一句话:**allocate"多页块"怕跨段,每 128GB 就跳;free"逐页"不怕跨段,一路走到总末尾才 migration。** 而 128GB 跳段释放的残留页,又会喂给 `free_to_right_async` 去 remap,成为推动 `free_offset_` 逼近 `total_size_` 的燃料之一。

## 本质:环形缓冲 + 在线搬迁

这不是普通 bump allocator。它是**环形 + 搬迁前沿同时移动**的结构 —— 难度天花板就在于:**搬迁的目标区间,同时是活跃工作集**。

## 为什么选这个设计
- 追求**虚拟地址连续**(权重要连续段,免逐页 RPC map)。
- 异步 unmap(`free_to_right_async` / `unmap_worker`)是为了**治碎片 + 不阻塞热路径**(commit `61067ca2` lazy unmap)。
- 但异步 = 主动引入竞态,后续几乎所有 bugfix commit(`b9cc4103`/`e3ec6e5e`/`7661f522`)都在还这笔债。

## 关键状态一览
| 变量 | 类型 | 语义 | 坑 |
|---|---|---|---|
| `allocate_offset_` | atomic | 左端分配点 | migration 时会被强制跳到 dst |
| `free_offset_` | 裸 size_t | 右端回收点 | **双语义**,见 [[L1-free-offset-dual-meaning]] |
| `migration_in_flight_` | atomic bool | 是否在搬迁 | — |
| `allocate_offset_migrated_` | **裸 bool** | 分配点是否已跳到 dst | **数据竞争**,见 [[L1-online-migration-race]] |
| `page_map_` | map+shared_mutex | offset→PhyPage | check-then-act,见 [[L1-map-page-check-then-act]] |
