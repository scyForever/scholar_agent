# GitHub 敏感信息历史清理方案

## 1. 适用背景

本方案针对当前仓库 `scholar_agent` 的 Git 历史敏感信息清理，目标是删除历史中曾被提交过的 `api_keys.py`，并给出强推后的协作同步方式。

当前检查日期：`2026-04-03`

## 2. 已确认的暴露点

基于本地 Git 历史检查，确认如下事实：

- `api_keys.py` 曾被 Git 跟踪，并在 `2026-03-25` 到 `2026-04-01` 之间出现在 `9` 个历史提交中。
- 相关提交起止如下：
  - 首次进入历史：`fd07cf7a19dc147d1dd6ee63774555e8af8c7317`
  - 停止跟踪：`cdc72dc8988009d5884f0e8305c4e05ddfff19cb`
- 已确认在历史中出现过非空默认值的变量名包括：
  - `SILICONFLOW_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `ZHIPU_API_KEY`
  - `SCNET_API_KEY`
- 当前工作区中的 `api_keys.py` 虽然已被 `.gitignore` 忽略，但本地文件仍包含明文默认值，后续不要再将该文件强制加入版本库。

结论：

- 需要先轮换上面已经暴露过的 key，再重写 Git 历史。
- 对这个仓库，按文件路径彻底移除 `api_keys.py` 比按字符串替换更稳妥。

## 3. 推荐执行顺序

1. 先到对应平台撤销并重建已暴露的 key。
2. 在一个新的镜像克隆里执行历史重写，避免污染当前工作区。
3. 验证历史中已不存在 `api_keys.py`。
4. 强制推送到 GitHub。
5. 通知协作者重新同步本地仓库。

## 4. 历史清理命令

建议不要直接在当前工作目录执行，改用镜像克隆：

```bash
cd /tmp
git clone --mirror git@github.com:scyForever/scholar_agent.git scholar_agent-history-cleanup.git
cd scholar_agent-history-cleanup.git
python -m pip install git-filter-repo
git filter-repo --path api_keys.py --invert-paths --force
```

清理后的本地验证：

```bash
git log --all -- api_keys.py
git rev-list --objects --all | rg '(^|/)api_keys\\.py$'
```

上面两条命令都应无输出。

确认 GitHub 分支保护允许强推后，再执行：

```bash
git push origin --force --all
git push origin --force --tags
```

如果你更倾向镜像方式，也可以改为：

```bash
git push origin --force --mirror
```

## 5. 清理完成后的仓库验证

建议至少验证以下项目：

- GitHub 默认分支 `dev` 的最新代码中已不存在 `api_keys.py`
- GitHub Web 界面的提交历史中不再能通过仓库树访问到 `api_keys.py`
- 本地执行 `git log --all -- api_keys.py` 无输出
- 本地执行 `git rev-list --objects --all | rg '(^|/)api_keys\\.py$'` 无输出

如果历史清理后，旧 PR diff、缓存页面或搜索结果仍能看到敏感内容，需要联系 GitHub Support 处理缓存展示。

## 6. 协作者同步指令

### 6.1 推荐方式：重新克隆

这是最稳妥的方式：

```bash
mv scholar_agent scholar_agent.bak
git clone git@github.com:scyForever/scholar_agent.git
cd scholar_agent
git checkout dev
```

### 6.2 复用已有本地仓库

仅适用于协作者确认本地没有未备份改动的情况：

```bash
git fetch origin --prune
git checkout dev
git reset --hard origin/dev
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now
```

如果本地还有未跟踪文件需要保留，应先手工备份，再决定是否执行 `git clean -fd`。

## 7. 风险与回滚

### 7.1 主要风险

- 强推会改写远端历史，所有协作者都必须同步。
- 任何已经克隆过旧历史的副本仍可能保留旧 key，因此轮换 key 不能省略。
- 如果 GitHub 开启了分支保护，需要暂时允许 force push。

### 7.2 回滚思路

执行清理前，建议保留一个镜像备份：

```bash
cd /tmp
git clone --mirror git@github.com:scyForever/scholar_agent.git scholar_agent-history-backup.git
```

如果历史重写本身出现技术问题，可以先基于该备份恢复远端；但由于旧历史包含敏感信息，除非已经完成 key 失效处理，否则不建议恢复到旧历史。
