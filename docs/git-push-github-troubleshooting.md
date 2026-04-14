# Git 推送到 GitHub：前期失败与后期成功的原因说明

本文档归纳在本仓库开发过程中，使用 `git push -u origin fastwam_dev` 时**先失败、后成功**的常见原因与对应处理，便于在受限网络或代理环境下排查。

---

## 1. 前期为什么不成功

### 1.1 使用 HTTPS 远程时：TLS 握手失败 / 连接被异常终止

**现象**：`fatal: unable to access 'https://github.com/...'`；`gnutls_handshake() failed: The TLS connection was non-properly terminated`；或 `curl` 访问 `https://github.com` 出现 SSL 读错误、长时间无响应。

**原因**：到 GitHub 的 **HTTPS（TLS）链路**在本地网络、代理或中间设备上**未完整建立或被中途掐断**，与具体 commit 内容无关。

**处理方向**：

- 检查系统/代理是否能稳定访问 `https://github.com`（换网络、修正代理、关闭冲突的全局代理后再试）。
- 若 HTTPS 长期不可用，可改用 **SSH 远程**（见下文）。

---

### 1.2 使用 HTTPS 时：`git push` 报 HTTP 408 / 写入中断

**现象**：`Writing objects` 已完成后出现 `RPC failed; HTTP 408`、`send-pack: unexpected disconnect`、`curl 22` 等；而 `git fetch` / `git ls-remote` 往往仍可用。

**原因**：`fetch`/`ls-remote` 多为**短请求、以下行为主**；`push` 需要在同一条 HTTPS 连接上**长时间上传 pack 文件**（持续上行）。不稳定代理、防火墙或 NAT 对**长连接大上行**更容易超时或丢连接，中间层返回 **408 Request Timeout** 或断开，表现为「读还行、写超时」。

**处理方向**：

- 换更稳定的网络或代理节点，或减少单次推送体积（拆分 commit、浅层相关优化为辅助手段，根因仍是链路）。
- **改用 SSH 推送**通常可绕开同一条 HTTPS 上传路径（见第 2 节）。

---

### 1.3（可选）Clash 订阅添加报 403

**现象**：`clashsub add "<订阅 URL>"` 时出现 `curl: (22) ... 403 Forbidden`、`订阅无效`。

**常见原因**：

- 订阅 URL **被错误转义**（例如在 `?`、`=` 前多加 `\`），实际请求地址错误，服务端返回 403。
- 订阅本身**过期、作废、或服务商限制 IP/客户端**，也会 403。

**处理方向**：

- 用**单引号**包裹原始订阅链接，**不要**在 `?` 与 `=` 前随意加反斜杠。
- 向订阅提供商确认链接有效；无效则更换订阅。

---

### 1.4 改用 SSH 后：`Permission denied (publickey)`

**现象**：`git remote` 已为 `git@github.com:...`，执行 `git push` 或 `ssh -T git@github.com` 报 `Permission denied (publickey)`。

**原因**：本机 **SSH 私钥**未加载，或 **公钥未添加到 GitHub 账户**（或用了另一台机器生成的密钥），GitHub 无法识别身份。

**处理方向**：

1. 在本机生成密钥（推荐 ed25519），例如：  
   `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -C "github"`
2. 将 **`~/.ssh/id_ed25519_github.pub` 全文**添加到：  
   [GitHub → Settings → SSH and GPG keys](https://github.com/settings/keys)
3. 使用 `~/.ssh/config` 为 `Host github.com` 指定 `IdentityFile` 与 `IdentitiesOnly yes`（若有多把密钥时尤其建议）。
4. `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519_github`，再执行 `ssh -T git@github.com`，应出现 `Hi <用户名>! You've successfully authenticated...`。

---

## 2. 后期为什么可以成功

满足以下条件时，`git push -u origin fastwam_dev` 通常即可成功：

1. **远程地址与协议正确**：`origin` 指向你的仓库，且使用 **SSH**（`git@github.com:用户名/仓库.git`）时，推送走 **SSH 协议与 22 端口（或你配置的端口）**，不再依赖此前不稳定的 **HTTPS 大文件上传**路径。
2. **SSH 认证已通过**：`ssh -T git@github.com` 能显示 GitHub 欢迎语，说明公钥已绑定账户，`git push` 具备写权限（对私有库还需账户有权限）。
3. **网络能维持 SSH 会话**：一般比「代理环境下长时间 HTTPS POST」更稳定，故在相同网络下常表现为「前期 HTTPS push 失败、后期 SSH push 成功」。

**注意**：若仅换代理节点而未解决 HTTPS 链路本质问题，HTTPS push 仍可能失败；**SSH 成功**是因为认证与传输路径已切换。密钥必须放在**执行 push 的那台机器**上；换电脑需在新机器重新生成并添加公钥。

---

## 3. 推荐排障顺序（简表）

| 步骤 | 操作 |
|------|------|
| 1 | `git remote -v` 确认使用 HTTPS 还是 SSH |
| 2 | HTTPS 异常时：测 `git ls-remote` / 浏览器访问 github；仍失败则考虑 SSH |
| 3 | 改用 SSH：`git remote set-url origin git@github.com:<用户>/<仓库>.git` |
| 4 | `ssh -T git@github.com` 直至认证成功，再 `git push -u origin <分支>` |

---

## 4. 相关命令备忘

```bash
# 查看远程
git remote -v

# 改为 SSH（示例仓库请改成自己的）
git remote set-url origin git@github.com:Yongwei-Zhang/FastWAM.git

# 测试 SSH 与 GitHub
ssh -T git@github.com

# 推送并设置上游分支
git push -u origin fastwam_dev
```
