---
name: retro-notes-in-chinese
description: retro/ 复盘笔记（给用户看的）正文必须用中文，技术术语保留英文；纯给 AI 看的记忆可用英文。
metadata:
  node_type: memory
  tags:
    - preference
    - retro
    - language
  type: feedback
---

# retro/ 复盘笔记用中文

`memory/retro/` 下的复盘笔记正文一律用**中文**书写,技术术语(如 `free_offset_`、check-then-act、migration、atomic、offload、forward、SLO 等)保留**英文**不翻译。

**Why:** 这些复盘是给用户本人长期阅读/内化的(用 Obsidian 打开),不是纯给 AI 消费的记忆。用户明确要求给他看的内容用中文。

**How to apply:**
- 写/改 `retro/` 下任何 `.md` 时,标题、说明、正文、表格都用中文。
- 术语、变量名、函数名、commit hash、文件路径保持英文/原样。
- 纯给 AI 看的常规 memory(user/feedback/project/reference 类)不受此限,可用英文。
- 这条偏好可推广到未来其他"给用户看"的产出:默认中文 + 英文术语。

**文件命名约定(用户确认):**
- retro 笔记**文件名主体保留英文 kebab-case**(与 frontmatter `name:` slug 一致,保证 memory 双链不断),但正文用中文。
- 文件名加**层级前缀**表示属于复盘哪一层:`L1-xxx.md`(显存地基)、`L2-xxx.md`(协调层)、`L3-xxx.md`(策略层);MOC 索引用 `00-` 前缀置顶。
- **Why:** 用户要在 Obsidian 侧边栏用肉眼按文件名判断层级 + 排序聚集。frontmatter `name:`、文件名、所有 `[[双链]]` 三处必须同步带前缀,否则断链。

**何时可以不问、直接改 retro 笔记:**
- 对已有 retro 笔记的**事实性修正/补充**(纠正错误、补精确细节、加 worked example 等),**直接执行,不用 AskUserQuestion 确认**。
- **Why:** 用户明确说过"这种不用问,肯定要修正"。复盘笔记的准确性是刚需,事实层面的订正没有决策分歧,反复确认反而拖慢。
- **How to apply:** 需要用户拍板的仍要问(如组织结构、命名方案、放哪层这类有多种合理选择的决策);但"内容对不对、够不够精确"这类事实问题,发现了就直接修。
