---
title: 记忆
summary: 在会话储存之上建立 FTS5 + 向量记忆、选择 embedding 提供者，以及常见检索模式。
tags:
 - guides
 - memory
 - embedding
---

# 记忆

给想要搜索过去会话事件的读者：无论是从 CLI、从 Python，还是让 agent 在执行时自己查。

会话的事件日志，同时也是一个小型的本地知识库。替它建立搜索索引后，你会得到 FTS 关键字搜索（免费、快速）、语意搜索（需要 embedder），以及融合关键字和语义排名的混合搜索。Agent 可以通过内置的 `search_memory` 工具查询当前会话的记忆。

概念先读：[记忆与压缩](../concepts/modules/memory-and-compaction.md)、[会话](sessions.md)。

## 哪些内容可搜索

Memory 将当前选中分支的用户输入、Trigger 内容、Assistant 文本、工具调用
及非空工具结果投影为可检索 Block。分支选择沿用 Resume 已保存的 Branch
View，未指定时使用主干默认选择。内容准入不依赖 Processing 生命周期，
因此迟到的后台结果和后续同步工具调用同样可以进入索引。非空 Error 与
Output 共同使用已有的索引长度预算。Round/Block 保留为兼容性分组标签，
Source Anchor 用于定位确切物理事件。

搜索会回传 `SearchResult` 记录，包含：

- `content`：命中的文字
- `agent`：由哪个Creature产生
- `block_type`：`text` / `tool` / `trigger` / `user`
- `round_num`, `block_num`：兼容性分组标签，不是精确事件定位符
- `score`：命中品质
- `ts`：时间戳记

## Embedding 提供者

共有三种提供者，选一个符合你环境的：

| Provider | 需要什么 | 说明 |
|---|---|---|
| `model2vec`（默认） | 不需要 torch、纯 NumPy | 极快，安装最精简。对接近关键字的检索品质不错，但长文本语意搜索较弱。 |
| `sentence-transformer` | `torch` | 较慢，但语意品质强很多。也适合 GPU。 |
| `api` | 网路 + API key | 远端 embedder（OpenAI、Jina、Gemini）。品质最好，但按次计费。 |
| `auto` | （无） | 若可用 API，优先用 `jina-v5-nano`，否则退回 `model2vec`。 |

默认模型名称（可跨 provider 使用）：

- `@tiny`：最小、最快
- `@base`：默认平衡
- `@retrieval`：为检索调校
- `@best`：最高品质
- `@multilingual`, `@multilingual-best`：非英文会话
- `@science`, `@nomic`, `@gemma`：特化用途

你也可以直接传入 Hugging Face 路径。

## 建立索引

```bash
kt embedding ~/.kohakuterrarium/sessions/swe.kohakutr
```

指定明确选项：

```bash
kt embedding swe.kohakutr \
  --provider sentence-transformer \
  --model @best \
  --dimensions 384
```

`--dimensions` 是 Matryoshka truncation：如果模型支持，可用它在执行时直接缩小向量维度。

增量建立：再次执行 `kt embedding` 时，只会索引新增事件。

## 为旧 Session 重建记忆索引

先关闭使用该 Session 的**全部读写进程**，包括外部进程。
对整个 Session 执行重建，不需要逐个指定 Agent：

```bash
kt memory rebuild SESSION
kt memory rebuild SESSION --dry-run
kt memory rebuild SESSION --vectors drop
kt memory rebuild SESSION --vectors rebuild --provider sentence-transformer --model @best
```

若存在旧向量，必须通过参数或交互提示明确选择：重新计算 Embedding
（可能下载模型或产生 API 费用），或者舍弃旧向量，直到重新建立向量索引
前禁用语义检索。没有向量时默认仅重建 FTS，并保留 Embedding 设置。

重建恢复物理事件定位信息，以及旧投影规则遗漏的内容。它沿用当前选中
分支，因此旧全分支索引的检索范围可能缩小。不会修改原始事件、ancestry
或外部 artifacts。缺失已索引来源、不可读事件及不支持的归档格式会在
替换索引前被拒绝。

命令先创建经过验证的 SQLite 逻辑备份，保留旧索引和已提交的 WAL 数据。
重建在原数据库内进行。若索引更新开始后维护失败，索引与 Embedding
配置可能处于不一致状态。**保持 Session 停止，直至重建成功或恢复备份。**
Resume 和普通搜索不保证检测或修复部分重建的索引。即使进程中断而未能
输出错误提示，也应遵守这一要求。保留备份，不要手动删除 WAL/SHM 文件。

普通搜索不会自动触发这种显式维护。分支切换仍可能通过正常索引生命周期
刷新受影响的索引。

## 从 CLI 搜索

```bash
kt search swe "auth bug"                # auto 模式（若已有向量则 hybrid，否则 fts）
kt search swe "auth bug" --mode fts     # 仅关键字
kt search swe "auth bug" --mode semantic
kt search swe "auth bug" --mode hybrid
kt search swe "auth bug" --agent swe -k 5
```

模式：

- **`fts`**：在 FTS5 上跑 BM25。不需要 embedding。最快，适合精确片语。
- **`semantic`**：纯向量相似度。需要索引。适合同义改写。
- **`hybrid`**：融合关键字和向量检索的排名（Reciprocal Rank Fusion）。
- **`auto`**：自动选择该会话支持的最完整模式。

`-k` 用来限制结果数量。`--agent` 可把搜索范围限制在Terrarium会话中的单一Creature。

## 从 agent 搜索

内置的 `search_memory` 工具，把同一套搜索引擎暴露给 controller：

```yaml
# creatures/my-agent/config.yaml
tools:
  - read
  - write
  - search_memory
memory:
  embedding:
    provider: model2vec
    model: "@base"
```

当 LLM 调用 `search_memory` 时，工具会对 *目前* 会话的索引执行搜索。这是 seamless-memory 的基本原语：agent 不需要额外搭 RAG 架构，就能查出自己（或队友）在前几轮说过什么。

工具参数（形状；实际语法取决于你的 `tool_format`，下面示范默认 bracket 格式）：

```
[/search_memory]
@@query=auth bug
@@mode=hybrid
@@k=5
@@agent=swe
[search_memory/]
```

### 从检索结果展开上下文

将返回的 Anchor 传给 `search_memory(anchor=..., before=3, after=3)`。
使用相邻事件返回的 Anchor 继续导航；长事件则使用同一 Anchor、scope
和返回的 `next_offset` 分页读取。

- `scope="selected"`（默认）：沿 Anchor 保存的分支视图读取。跳过归属
  未知的相邻事件；中心事件归属未知时需使用 raw。
- `scope="raw"`：读取同一 Agent 的物理相邻事件，包括其他分支，并显示
  已记录的 Turn/Branch 坐标。普通检索仍只使用选中分支索引。

工具调用及回执中的搜索自回声不会进入排名。预算耗尽时返回可用结果及
不完整提示，不表示没有更多证据。缺少物理来源的旧索引仍可检索，但需
显式重建才能获得 Anchor。

如果你要对 *外部* 数据源做 RAG，请自己做一个 custom tool，或做一个会调用向量数据库的 [插件指南](plugins.md)。

## 在Creature里设置记忆

```yaml
memory:
  embedding:
    provider: model2vec       # 或 sentence-transformer、api、auto
    model: "@retrieval"      # preset 或 HF 路径
```

实时搜索会按需刷新当前 Agent 的索引；FTS 不需要 Embedding Provider。

## 用程式检查

```python
from kohakuterrarium.session.store import SessionStore
from kohakuterrarium.session.memory.index import SessionMemory
from kohakuterrarium.session.memory.source_events import sourced_events

store = SessionStore("~/.kohakuterrarium/sessions/swe.kohakutr")
memory = SessionMemory(store.path, store=store)

memory.index_events("swe", sourced_events(store, "swe"))
results = memory.search("refactor", mode="fts", k=5)
for r in results:
    print(r.agent, r.source_ref, r.content[:120])

memory.close()
store.close()
```

## 疑难排解

- **`No vectors in index`**。 你用了 `--mode semantic`，但还没先执行 `kt embedding`。请先建立索引，或改用 `--mode fts`。
- **`kt embedding` 很慢**。 `sentence-transformer` 默认是 CPU-bound。请安装支持 CUDA 的 torch，或改用 `model2vec`。
- **Provider 安装失败**。 `kt embedding --provider model2vec` 没有 native 依赖，在哪里都能跑。`sentence-transformer` 需要 `torch`；`api` 需要对应 provider 的 SDK（`openai`、`google-generativeai` 等）。
- **Hybrid 模式结果噪声很多**。 把 `-k` 调低；如果查询很多改写语句，偏向用 `semantic` 而不是 `hybrid`；如果查的是精确片语，偏向用 `fts`。
- **旧索引缺少命中或 Anchor**。先尝试 FTS 并检查选中分支。使用 `kt memory rebuild SESSION` 恢复历史覆盖和来源信息；不完整提示及文本长度上限也可能限制结果。

## 延伸阅读

- [会话](sessions.md)：记忆建立在 `.kohakutr` 格式之上。
- [插件指南](plugins.md)：seamless-memory 插件模式（`pre_llm_call` 检索）。
- [参考 / CLI](../reference/cli.md)：`kt embedding`、`kt search`、`kt memory rebuild` 的旗标。
- [概念 / 记忆与压缩](../concepts/modules/memory-and-compaction.md)：背后的设计理由。
