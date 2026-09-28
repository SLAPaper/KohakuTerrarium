# M-lite P3 交付与验收记录

## 状态与范围

**P3完成提案已获Drive接受：`goal-6ze7ne`状态completed，revision 152。实现、限定独立审查和当前源码完整验证已收敛；这不是M-lite整体完成或进入P4/P5的声明。** H2/H3最后两处校验顺序问题已RED→修复→复审：已知反馈来源distinct校验在新injected constituent写入前；compact replay在ACK/feedback投影前做canonical冲突检查。修后完整后端三层测试均通过。

- Goal：`goal-6ze7ne`。
- 唯一实现目录：`C:/Users/slapa/workspace/KohakuTerrarium/.worktrees/m-lite-affinity-probe`。
- 分支：`wip/m-lite-affinity-probe`。
- 验收基线WIP：`993c99a18172b0a06a0e5fdf3a7410a3a657e469`；其父为任务指定检查点`5a412b1af1b80606a16e47d5f48ab00df2b77eb0`。
- 本记录对应上述HEAD之后的完整P3增量，不是只针对WIP。验收时增量未提交；随后用户明确授权“整体提交 P3”，本报告随P3源码和测试纳入整体提交。不包含push、amend或新分支授权。
- 下表按既有工作范围归为六组用于逐项验收，不扩展P4/P5实现范围。历史过程、真实RED和失败批次保留在本地`m-lite-affinity-progress.md`及来源快照中；该过程日志不纳入源码提交。
- 提交前核对2543个冻结文件全部SHA256相同，额外140项关键unit/integration通过（60.65s），Black 1807文件与Ruff通过；没有为提交更改已验收源码。

## 六组范围与实证

| 工作组 | 当前实现 | 主要证据入口与边界 |
|---|---|---|
| 1. 辅助writer、ACK、拒绝及反馈 | 原始native声明先确认持久化再派发；veto保留原调用身份；native后台ACK独立于终态；text promotion/ACK与direct、USER、trigger、命名输出反馈按实际occurrence及组成来源确认。owner只提交短内存变更，插件与外部输出在外执行。 | `core/native_declaration.py`、`native_ack.py`、`feedback_admission.py`、`feedback_dispatch.py`、`output_feedback.py`；`tests/unit/core/test_native_declaration.py`、`test_native_ack.py`、`test_feedback_composition.py`、`test_output_feedback.py`。未证明的组合不伪造exact覆盖。 |
| 2. Compact与恢复提交 | 捕获来源、生成摘要、owner应用、canonical事件与计数确认分离。owner应用同时登记未确认token，持久化及owner确认后解除；manager重建不绕过失败封锁。取消或owner回执丢失按是否已应用、已确认分别处理，不把已应用失败称为skipped。 | `core/compact_commit.py`、`compact_state.py`、`compact.py`、`model_switch_guard.py`；`tests/unit/core/test_compact_commit_owner.py`、`test_compact_snapshot_owner.py`。splice可用信号不等待后续需要turn锁的快照。 |
| 3. H1运行期绑定与交接 | 源准入封锁、有限已接受写入排空、独立目标准备、owner绑定与selector安装、发布、源退役。覆盖connect、split、remove、unwire、输出、子任务和Drive。Drive事务锁前后检查实际协调task；外部admission/完整turn不进入交接join，晚结算按精确delivery后继路由。 | `terrarium/topology_admission.py`、`session_merge_async.py`、`session_split_async.py`；`session/output_handoff.py`；`terrarium/drive/handoff.py`、`delivery_handoff.py`。`test_drive_handoff.py`、`test_topology_affinity.py`、`test_live_job_affinity_handoff.py`等真实屏障用例。 |
| 4. H2来源、快照与精确覆盖 | canonical origin为store UUID/namespace/original event ID；复制回执保留跨hop来源并仅重映射本地坐标。运行期快照显式绑定exclusive prefix；compact对实际被替换来源编译membership，来源校验在遮盖与native规范化之前。frontend同步消费覆盖字段。 | `session/event_origin.py`、`snapshot_coverage.py`、`compact_compile.py`、`compact_coverage.py`、`history_replay.py`、`resume_replay.py`；`tests/unit/session/test_snapshot_coverage.py`、`test_compact_coverage.py`；frontend `stores/compactCoverage.js`、`chat.test.js`。`user_pair_id`是opaque token。 |
| 5. H3运行期状态与工件 | runtime keys按源replace-or-pop，scratchpad、插件及子任务状态通过受管绑定保存；工件由源、目标各自worker读写，验证路径与冲突，拒绝逃逸/symlink，不以全目标覆盖掩盖冲突。 | `terrarium/session_runtime_state.py`、`session_resources.py`、`session_subagent_state.py`、`session_plugin_state.py`；`tests/unit/terrarium/test_session_runtime_state.py`、`test_session_resources.py`、`test_threaded_plugin_handoff.py`。整树工件内存占用仍是明确资源边界。 |
| 6. 契约、负例、恢复与收口 | 当前源码完整unit/integration、六个选定E2E文件及全部前端测试已获得通过证据；旧fat旅程扩展而非以零散integration新增函数代替。保留此前失败与未解释环境风险。 | `tests/integration/test_core.py`、`test_terrarium.py`；Terrarium旅程真实LLM等待跨merge→split，晚结果在正确目标唯一ACK，关闭再重开仍同delivery ID。独立审查结论见下方；源码冻结清单随交付保存。 |

## 最终当前源码验证

以下任务均已实际返回，不是根据running状态推测。

| 检查 | 结果 | 任务证据 |
|---|---:|---|
| Black全Python源码/测试 | 1807文件无需修改 | `bash_61b0a49e` |
| Ruff全Python源码/测试 | 通过 | `bash_61b0a49e` |
| 完整unit | **17128 passed、19 skipped、45 warnings，1478.35s** | `bash_61b0a49e` |
| 完整integration | **175 passed、3 warnings，129.16s** | `bash_61b0a49e`第二独立进程 |
| 选定E2E | **15 passed，29.07s** | 同任务第三独立进程；prog/api × creature/terrarium/studio六文件 |
| 前端Prettier | 全`src/`通过 | `bash_d051fe3c`格式阶段 |
| 前端完整Vitest | **173文件、1803测试通过，164.78s** | `bash_1f8f82e3` |
| 前端production build及surrogate后处理 | 通过；3622 modules，build 11.37s | `bash_d1db6794` |
| git diff whitespace | 通过，保留既有CRLF提示 | `bash_61b0a49e`检查阶段；只读命令级设置不改变全局Git配置 |

unit中的19项skip不是本轮为换绿添加；必须以测试自身既有平台/可选依赖条件解释。45条warning是Pydantic字段弃用42条及Path URI弃用3条；integration的3条为同Pydantic弃用。

前端使用现有主仓库安装的依赖。候选工作树没有node_modules，Vitest通过外部配置将root指向候选源码；Node26参数通过Vitest `--execArgv=--no-experimental-webstorage`传入worker，未改测试或替换localStorage断言。构建在新临时目录中对604个候选前端文件逐一SHA256核对后使用原`vite.config.js`、`uno.config.js`及原后处理脚本执行，依赖junction只在该副本中创建。未覆盖用户web_dist或修改安装。

构建输出保留在：
`C:/Users/slapa/workspace/m-lite-checkpoint-backups/incident-recovery-20260926-104354/p3-final-build-RhCoYa/kohakuterrarium/web_dist`。

## 失败证据与普通失败/崩溃保证

- 失败后封锁不是事务回滚。只有备份恢复已确认且绑定/拓扑没有变化时才解除封锁；写入或绑定结果不明确时要求冷恢复。
- 多KV、跨store、工件目录rename没有整体崩溃原子性。部分事件、组成行、工件或归档残留可能存在；不能以错误返回断言没有副作用。
- Drive失败交付的身份、结果与错误保留于运行期failed receipt；无法安全重试的拓扑项移出普通pending队列并保留失败记录，不饿死后续独立图。两者不是新durable crash journal，也没有自动修复保证。
- 单个repository事务的准入封锁保证已接受的成功写入被排空，后来的旧绑定写入被拒；不承诺包含多个await的整个manager调用具有事务原子性。
- 外部命名输出先发生的副作用不可回滚；后续来源或持久化确认失败会封锁续聊，不伪造成功回执。
- 不承诺subscriber、FTS、mirror或外部工具effect exactly-once。持久化事实与通知是不同确认层级。
- 历史Windows `WinError10055`及merge 503根因未被本轮证明解决。
- 集中把unit/integration/E2E放在同一进程时发生过E2E原60s限制超时，快照显示读取会话元数据后的临时目录unlink；进程中有大量既有worker/daemon。单独该旅程3.75s通过，独立E2E进程15项33.03s通过；这证明当前指定运行方式可通过，不证明长进程资源积累的根因已解决。没有加大case timeout。
- 该集中运行此前两项unit失败已明确修复：native声明测试替身转发新增guard keyword；Drive topology超600行通过移除陈旧解释注释降回限制内，没有放宽门禁。
- 前端初次81项失败全部在worker localStorage初始化；读取Vitest实现确认父execArgv被过滤，显式传给worker后原81项及全部1803项通过。初次失败记录仍保留。

## Bootstrap、standalone与legacy边界

- 运行中的受管Agent使用owner/worker协议；bootstrap中尚未进入并发运行的初始化，以及显式standalone组件的同步兼容行为，不能混称同一种执行域。
- 受管child host链：Agent先初始化executor并设置`executor._agent`，再创建SubAgentManager并设置`_parent_executor`，spawn先复制host再申请run receipt；`bind_children`使用同一host来源。单独构造无host的SubAgentManager不自动获得Terrarium热交接保证。该不变量已由绝对路径复审`agent_critic_a0406b6c`确认。
- 公共`save_conversation()`继续接受原list/JSON语义；运行时需要来源证明的写入点使用显式bound snapshot，不偷偷改变公共契约。
- 旧native缺失原始参数或来源证明时不合成新的可信声明，不声称可恢复任意旧损坏历史。可证明的兼容路径保留；不能证明的exact覆盖或native配对须明确拒绝/保留legacy边界。
- 借用Drive provider不能根据`durability`或`_path`猜成SQLite；原生session companion才可按其确定身份重开。未提供受支持事务交接能力的custom provider明确拒绝交接，不悄然丢行。
- 工件全量物化、大payload deepcopy、worker长时间卡住、retired handle和失败证据保留，仍有内存/等待时间风险，没有引入watchdog或资源限额来掩盖。

## P4验收输入（不是授权开始P4）

1. 本候选的源码哈希清单、基础HEAD与未提交文件清单；排除无关旧前端配置改动、临时日志和诊断脚本。
2. 六组证据表、最终独立审查结果及本节明确保留的边界。
3. 原始失败批次与成功批次均保留，尤其资源耗尽、长进程超时及普通失败/崩溃区别。
4. 以当前owner短提交、affinity worker、canonical来源、有限receipt和后继绑定协议作为下一阶段约束；不得把fallback或运行期失败证据当成durable事务。
5. 如下一阶段要改变上述资源/崩溃/兼容保证，先独立明确需求和验收标准，不在P3完成提案中默认承诺。

## 来源冻结

- 已保存2543个源码／测试／配置文件，每个文件的SHA256列入ZIP内`SOURCE-MANIFEST.json`；打包后逐项核对ZIP内容与当前工作树一致，ZIP完整性检查通过。
- 路径：`C:/Users/slapa/workspace/m-lite-checkpoint-backups/incident-recovery-20260926-104354/p3-final-source-_cy8z24j/candidate-source.zip`。
- ZIP SHA256：`6ecfc08699fbd44008c4719480aa0fc39b49cced03d86e70b29a5f29164e8908`。
- 明确排除：`src/_affinity_baseline.py`、`tests/unit/session/_probe_foreign.py`两个诊断文件，以及原有无关`MacroShell.test.js`、`vitest.config.js`改动。排除项记录于manifest，未删除其工作树文件。临时日志和根目录诊断脚本未列入源码包。
- ZIP另含打包时的交付报告和过程记录；本文补充的快照路径/Drive提案状态属于打包后的文档更新，不改变已验证源码。

## 独立审查与交付

- `agent_critic_d883e28e`：H1运行期交接、owner/worker、有限join、失败封锁检查；child host不变量交由后续绝对路径审查确认，其余范围内无新的确定阻塞项。
- `agent_critic_c8c34a0e`：H2/H3来源、快照、exact覆盖、状态及工件迁移检查，提出最后两条校验顺序问题。新增负例复现后修复，未把静态意见直接标为已解决。
- `agent_critic_a0406b6c`：确认上述两条RED已修、投影前canonical冲突检查正确、受管child host链成立；standalone边界保留。
- `agent_critic_a1c475a3`及此前Drive限定复审：失败队列、准入、晚结算、恢复与扫描修正已核查；父追加无live receipt恢复反例并验证，未把临时pause误当永久失败。
- `agent_critic_3966f2c5`引用了与绝对候选不符的源码状态，**该报告作废，不纳入验收证据**。
- 上述测试、格式与已确认审查结果均针对验收时工作树，提交前通过冻结哈希再次确认源码未变。来源ZIP/manifest为证据快照；Drive完成提案已接受，证据报告ID为`4cd9913b93f44a36adf5ec0f2adc15d5`。本次commit来自用户后续明确授权，不包含push。
