---
name: project-overview
description: "xLLM repository overview — what the project is, what the active branch implements, where the canonical entry points live."
metadata:
  node_type: memory
  type: project
---

xLLM 是京东开源的面向国产芯片（NPU 主线，CUDA/MLU/MUSA 兼容）的多后端推理引擎，支持 LLM/VLM/DiT/Rec 四类模型后端。本仓库位于 `${REPO_ROOT}`，当前分支 `feat/final_multi_model` 实现 **同卡多模型部署**：

- **Activation + KV + Weight 统一池化**（基于 VMM API 的 xtensor 虚拟连续显存）
- **权重分层加卸载**（按 transformer layer 粒度，由水位线 / SLO 触发；支持 Mooncake D2D）
- **基于优先级 + SLO 的请求/模型调度**（多副本派发、Engine forward admission、模型自动 degrade/restore）

**Why:** 用户在该分支上开发；后续提问大概率围绕这套体系，需要快速定位多模型相关代码路径。

**How to apply:** 任何关于"为什么显存不够""模型为什么不响应""副本怎么选""权重怎么换"等问题，先想到这条主线。详见 [[xllm-arch-map]] 和 [[multimodel-mechanism]]。

入口：
- 启动：`xllm.cpp::main → run` + `start.sh`（实际启动命令样例）
- 路由：`xllm/api_service/api_service.cpp`
- 多模型核心：`xllm/core/framework/xtensor/page_allocator.{h,cpp}` + `layer_offload_manager.{h,cpp}`
- 调度：`xllm/core/scheduler/scheduler_factory.cpp`
- 引擎单步：`xllm/core/distributed_runtime/llm_engine.cpp`

文档：`docs/zh/{features,dev_guide}/*.md`，重点 `xtensor_memory.md`, `overview.md`, `xllm_service_overview.md`。

项目 skills 位于 `.claude/skills/`，至少包含：`xllm-arch`, `xllm-multimodel`, `xllm-memory-mgmt`, `xllm-scheduler`, `xllm-conventions`, `xllm-debug`, `xllm-feature-recipes`。
