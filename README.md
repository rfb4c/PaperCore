# PaperCore

Convert academic PDFs to structured, compressed Markdown for LLM consumption.

## Three-Zone Strategy

| Zone | Sections | Treatment |
|------|----------|-----------|
| **A** (Metadata) | Title, Authors, Year | Extracted to front-matter |
| **B** (Full Retention) | Abstract, Introduction, Discussion, Conclusion | All text preserved |
| **C** (Compression) | Methods, Materials, Results, Experiments | Subheaders + captions + first sentence only |

Unknown sections default to Zone B (preserve rather than lose content).

## Install

```bash
pip install -r requirements.txt
```

Note: `docling` will download ML models (~1-2 GB) on first run.

## Usage

```bash
# Convert all PDFs in a folder
python papercore.py ./papers/

# Convert to a different output directory
python papercore.py ./papers/ -o ./markdown/

# Convert a single PDF
python papercore.py ./papers/paper.pdf

# Verbose mode (shows zone classification per section)
python papercore.py ./papers/ -v

# Disable compression (treat everything as Zone B)
python papercore.py ./papers/ --no-compress
```

## Output Format

```markdown
# Paper Title

**Authors:** Author Names
**Year:** 2024

---

## Abstract
[Full text...]

## Introduction
[Full text...]

## Methods
### Subsection Header
First sentence of each paragraph. [...]
*Figure 1: Caption text*

## Discussion
[Full text...]

## Conclusion
[Full text...]
```

## GUI

```bash
python papercore.py --gui
# or
python papercore_gui.py
# or double-click
PaperCore.cmd
```

Create a desktop shortcut (one-time):

```bash
python create_shortcut.py
```

No extra dependencies needed (uses built-in `tkinter`).

## Fallback Behavior

If section structure cannot be detected (e.g., unstructured PDFs), the tool falls back to full-text extraction with a warning banner at the top.

---

# 中文说明

## 简介

PaperCore 将学术 PDF 转换为结构化压缩的 Markdown，供 LLM 消费。避免将全文喂给大模型时产生的 token 浪费和噪音。

## 三区策略（Three-Zone Strategy）

| 区域 | 章节 | 处理方式 |
|------|------|----------|
| **A** (元数据) | 标题、作者、年份 | 提取至文件头 |
| **B** (全量保留) | 摘要、引言、讨论、结论 | 保留全部文本 |
| **C** (智能压缩) | 方法、材料、结果、实验 | 仅保留子标题 + 图表标题 + 每段首句 |

未识别的章节默认归入 Zone B（宁可多保留，不丢失内容）。

## 安装

```bash
pip install -r requirements.txt
```

注意：`docling` 首次运行会从 HuggingFace 下载模型（约 1-2 GB）。

## 命令行用法

```bash
# 转换文件夹中的所有 PDF
python papercore.py ./papers/

# 指定输出目录
python papercore.py ./papers/ -o ./markdown/

# 转换单个 PDF
python papercore.py ./papers/paper.pdf

# 详细模式（显示每个章节的区域分类）
python papercore.py ./papers/ -v

# 禁用压缩（所有章节均全量保留）
python papercore.py ./papers/ --no-compress
```

## 图形界面

```bash
python papercore.py --gui
# 或直接运行
python papercore_gui.py
# 或双击
PaperCore.cmd
```

创建桌面快捷方式（运行一次即可）：

```bash
python create_shortcut.py
```

无需额外依赖（使用 Python 内置 `tkinter`）。

## 输出格式

```markdown
# 论文标题

**Authors:** 作者名
**Year:** 2024

---

## 摘要
[完整文本...]

## 引言
[完整文本...]

## 方法
### 子章节标题
每段首句。[...]
*Figure 1: 图表标题*

## 讨论
[完整文本...]

## 结论
[完整文本...]
```

## 回退机制

当无法识别章节结构时（如纯文本 PDF），工具会回退到全文提取模式，并在文件顶部添加警告标注。
