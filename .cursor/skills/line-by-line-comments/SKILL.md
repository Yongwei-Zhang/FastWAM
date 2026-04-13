---
name: line-by-line-comments
description: 在指定文件或选区为代码段添加「段首整体说明 + 每行行尾注释（代码与注释之间空两格）」；仅当用户明确要求添加/给出注释或「注释」时触发；就地插入注释，禁止删改原代码逻辑。Adds segment header plus trailing comments (two spaces before comment token); only on explicit comment requests; in-place insertion only.
---

# 段首总述 + 行尾注释

## 触发与范围

- **触发**：仅当提示词含「给出注释」、「添加注释」或「注释」时执行。
- **范围**：仅限指定文件或高亮选区，绝不越界。

## 布局（位置）

1. **段首 / 文件首**：在每一段逻辑代码块的**正上方**（或整个文件最上方，若注释整文件）用当前语言注释符写**一段整体说明**，概括本段/本文件在做什么。
2. **逐行**：对每一段内每一行**有效代码**，在该行**末尾**追加注释：先**两个空格**，再写行尾注释符与内容（如 Python：`  # …`）。多行语句（括号续行）在**语义完整**的一行末写注释，**不拆改原换行**。
3. **语言**：保持文件/片段所用语言的注释符（`#`、`//`、`/* */` 等）。

## 内容

- 中文，极简单句；直击意图；禁复述字面量；禁长篇原理解释。

## 底线

- **禁止**修改、删减、重排或重构原代码的任何逻辑、标识符与无关格式（缩进、空行、换行策略保持不变）。
- **交付方式**：在仓库内**直接插入注释**（编辑工具打补丁 / apply），**不要**先删除整段再贴「带注释版」全文；用户需要的是对现有代码的就地增补。

## 示例（虚构，Python）

```python
        # 按用户 id 拉取档案并做缺省填充：查询库表 → 转字典 → 补默认邮箱。
        row = db.fetch_one(
            "SELECT id, name, email FROM users WHERE id = ?",
            (user_id,),
        )  # 返回一行或 None
        profile = dict(row) if row else {}  # 有结果则转 dict，否则空 dict
        profile.setdefault("email", "noreply@example.com")  # 无邮箱时写入占位
```
