# Git：分支、提交、暂存与本地仓库（概念梳理）

本文档归纳 **branch、commit、暂存区、`.gitignore`、`.git` 目录** 的关系与常规工作流，与网络推送问题无关的内容见 [git-push-github-troubleshooting.md](./git-push-github-troubleshooting.md)。

---

## 1. 核心概念

| 概念 | 含义 |
|------|------|
| **提交（commit）** | 一次**不可变快照**：记录某一时刻整棵目录树的内容、父提交指针、作者、时间与说明。每个 commit 有唯一 **SHA** 标识。 |
| **分支（branch）** | 指向**某个 commit** 的可移动**指针**（名字），如 `main`、`fastwam_dev`。新建提交时，当前分支指针会**前移**到新 commit。 |
| **HEAD** | 指向**当前检出位置**（通常是当前分支名，再解析到具体 commit）。`git checkout` / `git switch` 会移动 HEAD。 |
| **工作区（working tree）** | 磁盘上可见的项目文件；你编辑的是这里。 |
| **暂存区（index / staging area）** | 介于工作区与下一次 commit 之间的**中间层**：`git add` 把变更登记到这里；`git commit` 把暂存区内容打成新快照。 |

**要点**：不同分支 = 可能指向**不同 commit** = 同一路径下**文件内容可以不同**；切换分支不是「隐藏/显示注释」，而是**检出另一份快照**。

---

## 2. `.git` 目录与「commit 存在哪里」

- 仓库根目录下的 **`.git`**（点开头，多数界面默认**隐藏**，终端用 `ls -a` 可见）是 **Git 本地对象库与元数据**所在位置。
- **commit、blob（文件内容）、tree（目录结构）** 等以对象形式存放在 **`.git/objects/`**；分支指针在 **`.git/refs/heads/`** 等文件中（记录分支名 → commit SHA）。
- 执行 **`git commit`** 时：新 commit **写入本地** `.git`；**尚未 push 时，远端还没有这些对象**。
- **`git push`**：把本地已有、远端缺失的 commit（及相关对象）**同步到远端**；远端仓库同样以对象形式存储。  
→ **commit 首先且始终存在于本地仓库**；push 是复制/同步，不是「只在云端才有提交」。

---

## 3. `.gitignore` 与 `git add`

- **`.gitignore`**：规定哪些路径 **不被跟踪**（untracked 时被忽略；已跟踪文件需先 `git rm --cached` 等才能停止跟踪）。
- **`git add -A`**：把**已跟踪文件**的修改/删除，以及**未跟踪且未被忽略**的新文件，加入暂存区。
- **被忽略的路径**：正常 **`git add` 不会加入**（除非 `git add -f` 强制）。  
→ 逻辑上是：**忽略规则下的路径不参与版本控制**；不是「先全部 add 再删掉忽略的」。

---

## 4. 标准工作流（本地 → 远端）

```
工作区编辑 → git add（写入暂存区）→ git commit（生成 commit，分支前移）→ git push（同步到 origin 对应分支）
```

1. **修改文件**：只影响工作区；与上一次 commit 比较产生 diff。  
2. **`git add`**：选择**哪些变更**进入下一次提交（暂存区）。  
3. **`git commit`**：用暂存区生成**新 commit**，当前分支指向该 commit。  
4. **`git push origin <分支名>`**：把该分支上远端没有的 commit 推到远程；成功后 **远端该分支的「最新」即这次推送的 tip**（与本地一致，除非他人又推送）。

**`git push -u origin <分支>`**：同时设置上游跟踪关系，之后可用 `git push` / `git pull` 省略远程与分支名。

---

## 5. 切换分支时看到什么

- **`git switch <分支>`**（或 `git checkout <分支>`）：把 **HEAD** 移到该分支指向的 commit，并把**工作区文件**更新成该 commit 的树。  
- 因此：在 **A 分支**做的提交，在 **B 分支**若未合并，则 B 的工作区是 B 的**快照**，不会出现 A 上才有的修改。  
- **不是**「注释丢失」，而是**两个分支当前指向的提交不同**。

若希望另一分支也包含相同改动，需 **merge / rebase / cherry-pick** 或在另一分支上再提交。

---

## 6. 常用命令速查

```bash
git status              # 工作区与暂存区相对当前分支的差异
git diff                # 工作区相对暂存区/最后提交
git diff --cached       # 暂存区相对最后提交（即将提交的内容）
git log --oneline -n 5  # 最近提交简表
git branch -vv          # 本地分支及与 origin 跟踪关系
```

---

## 7. 与本文档相关的另一篇

- 推送失败（HTTPS / SSH / 代理）：见 [git-push-github-troubleshooting.md](./git-push-github-troubleshooting.md)。
