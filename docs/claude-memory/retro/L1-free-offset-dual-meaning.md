---
name: L1-free-offset-dual-meaning
description: "第一层难点1——GlobalXTensor 的 free_offset_ 一个变量承担\"调度意图\"和\"实际水位\"两个语义,是 lazy unmap 系列竞态的根子。作者在源码顶部 TODO 亲自求救。"
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

# 难点1:`free_offset_` 一变量两义

场景见 [[L1-global-xtensor-ring]]。

## 铁证:作者自己的 TODO(`global_xtensor.cpp:27-30`)
作者在文件顶部留了求救注释(原文摘要):
> "将 `free_offset_` 拆为两个变量:**操作逻辑上的**(控制后台该 map 到哪)和**实际事实上的**(后台线程 map 到了哪)…我理解上只有 `wait_enough_pages` 的 `free_offset_` 是事实上的,其余都是逻辑上的?请核实是否可行。"

这是作者留给未来自己的信号 —— **他知道这里有病,但没时间治。**

## 病在哪
`free_offset_` 同时是:
1. **调度意图**:`free_to_right_async`(`:204`)里 "我打算把下一页映射到这" —— `free_offset_ += page_size_`。
2. **实际水位**:`wait_enough_pages`(`:367`)用 `allocated <= free_offset_` 判断"页够不够",**假设它是事实**。

但 `map_page` 真正让页可用的 `notify_all`(`:151`)在锁外,且 `free_offset_ += ` 在 `map_page` **之后**才加 —— **意图和事实在多线程下会短暂不一致**:等待方可能看到"水位够了"但页其实还没 map 完。

## 为什么这是根子
lazy unmap 的历史 bug 全挂在这:
- `b9cc4103` fix unmap race condition
- `e3ec6e5e` fix lazy unmap bug
- `7661f522` activation migration failure under extreme circumstances

它们本质都是"意图/事实错位"在不同边界上的显形。

## 内化经验
> **一个变量承担两个语义,是并发 bug 的头号来源。** 当你要在注释里解释"这变量有时指 A 有时指 B",就是该拆变量的信号。
>
> **通用规则**:任何"生产者-消费者 + 水位判断"结构,水位线必须区分 **committed(意图)** 和 **completed(事实)** 两个值;消费者等待时**只能看 completed**。

关联:[[L1-map-page-check-then-act]](同样是 map_page 的锁问题)、lazy unmap 的动机与代价见 [[L1-lazy-unmap-tradeoff]]、元经验见 [[00-multimodel-retro-moc]] #2。
