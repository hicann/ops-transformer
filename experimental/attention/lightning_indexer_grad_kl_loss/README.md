# LightningIndexerGradKLLoss

计算 LightningIndexer 的梯度与 KL loss，提供普通和 skip-padding 两种 Torch 接口。

## 接口

```python
d_query_index, d_key_index, d_weight, loss = torch.ops.cann_ops_transformer.lightning_indexer_grad_kl_loss(
    query, key, query_index, key_index, weights, sparse_indices,
    softmax_max, softmax_sum, query_rope, key_rope,
    cur_seq_lengths_query=None, cur_seq_lengths_key=None,
    scale_value=1.0, layout="BSND", sparse_mode=3,
    pre_tokens=2147483647, next_tokens=2147483647,
    block_size=1, deterministic=False,
)
```

同时提供 `lightning_indexer_grad_kl_loss_skip_padding` 接口：在 `key_rope` 后增加 `mask` 参数，末尾增加 `validTokenNum=-1`，其他参数与普通接口相同。

- `query/key/query_index/key_index`：FP16 或 BF16，ND 格式。
- `weights/softmax_max/softmax_sum`：FP32；`sparse_indices`：int32。
- `query_rope/key_rope`：可选 Tensor；mask 为可选 int8 Tensor。
- 序列长度使用可选整数列表，由 ACLNN 的 `aclIntArray` 接收。
- 输出梯度 shape 分别与 query_index、key_index、weights 一致；loss 为 shape `[1]` 的 FP32 Tensor。
- 支持构建目标 `ascend910b` 和 `ascend910_93`；已验证设备为 910B2C。

## 实现说明

- `op_kernel/` 包含 Ascend C 内核；`op_host/` 包含算子定义、Tiling、shape/dtype 推导。
- CMake 通过 `add_modules_sources` 接入仓库构建，生成 `aclnnLightningIndexerGradKLLoss`，并提供独立 CANN 构建入口。
- Torch 层提供普通版和 skip-padding 版接口，序列长度列表使用持有数据的 vector 接入 pybind。
- ND 格式转换使用公开的 `torch_npu.npu_format_cast` 接口。

## 相比主库 A2（arch22）的 event ID 与同步优化

本节对照主库
[`sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22`](../../../attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22)。
它与本目录具有对应的 V0/C1/V1/C2/V2 流水阶段和同名 Service 类。

**实验版围绕减少跨核通知、明确编号职责、加强阶段隔离和完善边界分支进行了同步优化。**
核心优点包括：

- **减少通知指令。** 合并 P/SY 的输入就绪与结果就绪通知，使这两段正常任务中的 Set/Wait 调用数分别减半，同时保留完整的数据依赖。
- **降低编号复用的维护成本。** 为 AIV 间交换、deterministic 汇合和结果传递分配明确的 flag，便于检查发送方、接收方及信号生命周期。
- **加强初始化与计算阶段的隔离。** 使用 C/V 联合同步确认工作区准备完成，并将其编号与主循环 flag 14 分开。
- **完善 skip-padding 和空任务的同步协议。** 数据处理被跳过时仍维护必要的同步配对，降低漏发、漏等和缓冲区代际错配的风险。

优化集中在跨核同步协议；核内 `HardEvent` 的编号分配继续保持一致。

### 1. 保持核内依赖，集中优化跨核同步

| 同步方式 | 本算子中的用途 | 本次对比结果 |
| --- | --- | --- |
| `SetFlag/WaitFlag<HardEvent::…>(EVENT_IDx)` | 同一个核内 MTE2、MTE3、Vector、Matrix、Fixpipe 之间的数据依赖和缓冲区复用 | 三个 Service 的 `AllocEventID` / `FreeEventID` 函数在忽略空白、注释后逻辑一致 |
| `CrossCoreSetFlag/CrossCoreWaitFlag(flagId)` | AIC/AIV 之间、两个 AIV 之间以及 deterministic 阶段的跨核同步 | P/SY 通知合并，多个 flag ID 调整 |
| `SyncAll<…>()` | 工作区初始化结束、主计算结束后的全核同步 | 初始化处由默认 `SyncAll()` 改为 `SyncAll<false>()` |

核内的 `EVENT_ID3` 与跨核的 flag 3 不是同一类同步资源。
这里的 `AllocEventID` 主要是给变量赋固定 `EVENT_IDx`，并预置可复用缓冲区的初始信号；
`FreeEventID` 主要通过 Wait 消费末尾信号，不应理解为跨核 flag 的动态申请与释放。

对应代码：[Vector Alloc/Free](op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h)（L386）、
[Cube Alloc/Free](op_kernel/sparse_lightning_indexer_grad_kl_loss_service_cube.h)（L246）、
[后处理 Vector2 Alloc/Free](op_kernel/sparse_lightning_indexer_grad_kl_loss_vector2.h)（L152）。

### 2. 重新规划 flag ID，明确各阶段职责

主库定义见 [common.h](../../../attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_common.h)（L27），
实验版定义见 [common.h](op_kernel/sparse_lightning_indexer_grad_kl_loss_common.h)（L28）。
下表的 mode 指发送端使用的模式；有些接收端使用默认模板参数。

| 同步关系 | 主库 A2 | 实验版 | 调整后的作用 |
| --- | --- | --- | --- |
| V0 → C1，gather 的 P/SY 输入就绪，mode 2 | P=`{0,1}`，SY=`{2,3}` | P+SY=`{0,1}` | 依据 S2 分块序号 `i & 1`，两份输入共用一次通知 |
| C1 → V1，MM1/MM2 输出就绪，mode 2 | P=`{4,5}`，SY=`{6,7}` | P+SY=`{4,5}` | 依据 `taskIdMod2`，整个任务只通知一次 |
| V1 两个 AIV 交换 P/SY，mode 1 | 硬编码 `0x8` | 命名常量 `SYNC_V1_TO_V1_PSY_FLAG=6` | 为 AIV 间交换分配独立的命名编号，明确其与 `{8,9}` 的不同职责 |
| V1 → C2，reluGrad 就绪，mode 2 | `{8,9}` | `{8,9}` | 仍按任务奇偶区分 |
| C2 → V2，MM5 的 dK 部分结果就绪，mode 2 | `{10,11}` | `{10,3}` | 复用空出的 3，将业务通知与 SDK 全核同步编号分开 |
| deterministic V2 收齐结果后的汇合，mode 0 | `{0,1}[taskIdMod2]` | 单个 7 | 用专用汇合编号保持同步，并简化奇偶编号管理 |
| deterministic scatter 顺序控制，mode 0 | 2 | 2 | 保持原编号，增加 mask 跳过路径的配对等待 |
| 主循环轮次推进，mode 2 | 14 | 14 | 保留 AIC → AIV 的轮次推进约束 |
| `SYNC_V2_TO_C2_DETER_SA_FLAG` | 声明为 12 | 删除 | 清理未使用的常量，减少阅读和维护歧义 |

P/SY 合并释放了 `{2,3}`、`{6,7}`，为其他同步角色腾出编号。
实验版由此形成清晰分工：0～10 用于业务阶段，14 用于轮次推进，11～13 留给下文的全核同步实现。
这种分配减少了不同角色共用编号时需要考虑的上下文，便于检查依赖关系和维护后续流水改动。
通知调用数的减半收益具体发生在 V0 → C1、C1 → V1 的 P/SY 数据链。

### 3. V0 → C1：成对处理分块，将通知次数减半

主库 [ProcessVector0](../../../attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h)（L437）
先完成所有 P 分块的 gather 和通知，再完成所有 SY 分块。
Cube 侧也先执行整个 `ComputeMm1`，再执行整个 `ComputeMm2`，分别等待各自的 flag。

实验版 [ProcessVector0](op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h)（L445）
按 S2 分块成对生产，Cube 侧通过 [ComputeMm12](op_kernel/sparse_lightning_indexer_grad_kl_loss_service_cube.h)（L488）
成对消费：

```text
主库 V0：for i: gather P_i，Set P[i%2]
         for i: gather SY_i，Set SY[i%2]
主库 C1：for i: Wait P[i%2]，MM1_i
         for i: Wait SY[i%2]，MM2_i

实验 V0：for i: gather P_i，gather SY_i，Set P_SY[i%2]
实验 C1：for i: Wait P_SY[i%2]，MM1_i，MM2_i
```

实验版 Set 位于两次 `MergeKv` 之后，绑定 `PIPE_MTE3`；因此它覆盖同一发送核此前提交的
P、SY 两份 GM 搬出。Cube 在 `ComputeMm1(info,i)` 中等一次，紧接着的 `ComputeMm2(info,i)`
沿用这个已满足的数据依赖，不再另等 SY flag。

若每个任务有 m 个 S2 分块，这一段每个 AIV 的 Set 调用从 2m 次降为 m 次，
对应 Cube 的 Wait 调用也从 2m 次降为 m 次，直接减少这条数据链的跨核通知指令。
成对生产、成对消费还让每个通知明确对应同一分块的完整输入，简化 P/SY 两条进度链的管理。

这项优化通过配套调整生产、计算与通知位置实现。MM1 的启动条件变为当前分块的 P、SY 均已准备好，
实际时延收益取决于新的流水重叠情况。

### 4. C1 → V1：以一个完整就绪通知代替两次等待

主库在整个 MM1 完成后发 P，在整个 MM2 完成后发 SY；
[ProcessVector1](../../../attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_vector.h)（L691）
连续等待这两个信号后才开始 Vector 计算。

实验版 [ComputeMm2](op_kernel/sparse_lightning_indexer_grad_kl_loss_service_cube.h)（L593）
只在最后一个 S2 分块结束时发送 `P_SY[taskIdMod2]`，绑定 `PIPE_FIX`。
此前 MM1、MM2 的 Fixpipe 输出均已排在该通知之前，因而
[ProcessVector1](op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h)（L692）
等一次就能取得同一任务的两类结果。

实验版以一个通知完整表达“P 和 SY 都已完成”，在保持两路结果依赖的同时压缩同步操作。
正常非空任务中，C1 的 Set 从两次降为一次，每个 AIV 的 Wait 也从两次降为一次。
V1 因而只需维护一个任务完成条件，减少两路通知分别发送、消费时的状态管理。

V1 后续仍由两个 AIV 分别处理 P 和 SY，把中间结果写到 `psySyncGm`，
通过 mode 1 的 flag 6 汇合后，再读取另一侧结果并计算梯度。
flag 6 专门表示交换阶段完成，`{8,9}` 专门表示 reluGrad 就绪，两种职责可以直接从编号辨认。

### 5. 分离同步角色，降低同号复用的耦合

在当前验证使用的 CANN 8.5 `dav_c220` SDK 实现中，跨核接口可概括为：

```text
Set(mode, pipe, flagId) → ffts_cross_core_sync(pipe, 编码后的 mode/flagId)
Wait(mode, pipe, flagId) → wait_flag_dev(flagId)
```

`GetffstMsg` 取 flagId 的低 4 位进行编码；`WaitEventImpl` 最终只使用 flagId，
不把 mode、pipe 组成独立的等待编号。因此，不能仅凭 mode 不同就认定同号信号隔离。
同样，发送端 `Set<2,...>`、接收端省略模板参数的 Wait，在这个 SDK 下也不意味着模式不匹配。

主库的 V1 mode 1 使用 8，V1 → C2 也使用 8。实验版将前者改为专用编号 6，
使两种同步角色在编号上直接分开，减少对接收方向和阶段上下文的隐式依赖，便于审查同步关系。
实际冲突的判断仍需结合信号接收核、发送/消费次数和生命周期；同号本身并不等同于已发生死锁。

这部分结论按当前 A2 SDK 实现成立；其他架构、SDK 或特殊编译模式的保留编号需要另行核对。

### 6. C/V 联合同步，隔离初始化与主循环阶段

工作区初始化后，主库调用
[`SyncAll()`](../../../attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/arch22/sparse_lightning_indexer_grad_kl_loss_base.h)（L345），
实验版改为 [`SyncAll<false>()`](op_kernel/sparse_lightning_indexer_grad_kl_loss_base.h)（L323）。

这里的布尔参数是 `isAIVOnly`，默认值为 true。**false 表示同步 AIC 和 AIV，
不是关闭同步。** 当前 `dav_c220` 的普通 SDK 路径使用：

| 调用 | SDK 使用的跨核编号 | 同步范围 |
| --- | --- | --- |
| `SyncAll()` / `SyncAll<true>()` | 14 | AIV-only 同步路径 |
| `SyncAll<false>()` | 11、12、13 | AIC/AIV 联合同步路径 |

两版主循环都使用硬编码 flag 14：AIC 在 `PIPE_MTE2` 发出，AIV 在进入当前轮 V0 前等待。
初始化默认同步与主循环因而出现同号复用；实验版同时做了两项调整：

- 初始化显式等待 C/V 全部到达，建立清零工作区与后续计算之间的同步边界。
- 初始化使用 SDK 的 11～13，主循环继续使用 14；业务 C2 → V2 的 pong 从 11 移到 3，
  使全核同步和业务通知的编号分工更清晰。

这项调整同时加强了同步范围和编号隔离：初始化阶段等待 C/V 全体核到达，主循环 flag 14 只承担轮次推进职责，
降低阶段交接时通知被误配的风险。后处理前继续保留两版已有的 `SyncAll<false>()`，
保证全局梯度写入完成后再进入最终读取与转换。

### 7. 完善 deterministic 与跳过分支的同步配对

[ProcessDeterVector2](op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h)（L931）
先等待本组 Cube 的 MM5 结果，再发 mode 0 通知并等待汇合。
主库按任务奇偶使用 0/1，实验版使用单个 7：

```text
Wait C2_result[taskId%2]
Set mode0, flag7
Wait flag7
```

每次进入该阶段都会先 Set 再完成 Wait，因此可以用专用编号 7 表达一个完整的跨核汇合，
省去额外的 ping/pong 编号选择。该设计在保留汇合作用的同时简化同步状态；
实现时仍须保持所有参与核的执行次数一致。

接着的 mode 0 flag 2 控制 deterministic scatter 的阶段顺序。
正常路径在 `Vector2ScatterAdd` 内对偶数 `idx` 等待；新增的
[`maskValid == false` 分支](op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h)（L1011）
虽然跳过数据处理，也对偶数 `idx` 补相同 Wait，外层再按奇数 `idx` 或末任务条件 Set。
这样 skip-padding 路径在跳过无效数据写入时，仍保持与正常路径一致的同步消费，完善了新增分支的同步保障。

空任务有两层处理，需要结合调用条件理解：

- [GetRunInfo](op_kernel/sparse_lightning_indexer_grad_kl_loss_base.h)（L673）
  对 `kRealSize==0` 标记 `isValid=false`，正常主循环会跳过对应 V0/C1/V1/C2/V2 调用；
  主循环的 flag 14 仍按轮次发送、等待。
- `ComputeMm12` 保留 `s2LoopTimes==0` 时发送完成通知、推进相关缓冲区状态的防御分支；
  `ProcessVector1` 也保留 `kLoopTimes==0` 时向 C2 发送 `{8,9}` 的防御分支。
  在前述正常跳过路径中，这些阶段函数不会被调用，不能把它们算成每个零任务必然发生的额外通知。

deterministic 模式还保留了主循环尾部的参与核补齐逻辑，使数据量不足一整组核时仍能维持必要的汇合参与。
这些处理让正常计算、跳过数据和流水收尾都具有明确的同步职责；零长度、mask 和非整核尾任务的组合验证
仍是确认其覆盖范围的必要步骤。

### 8. 优化收益与适用范围

| 数据阶段 | 实验版同步依据 | 需要保持的约束 |
| --- | --- | --- |
| V0 gather → C1 读 GM | P/SY 合并通知 `{0,1}` | 两路搬出都要先于 MTE3 通知；生产与消费使用同一个 S2 分块奇偶 |
| C1 输出 → V1 读 GM | 合并结果通知 `{4,5}` | 只能在最后一块 MM1/MM2 的 Fixpipe 输出之后发送 |
| 两个 AIV 交换 P/SY | mode 1，flag 6 | 两侧写出各自中间结果后汇合，不能提前读另一侧 |
| V1 reluGrad → C2 | `{8,9}` | 梯度搬出完成后通知，任务奇偶须与 GM 缓冲区匹配 |
| C2 MM5 → V2 scatter | `{10,3}` | 此通知保证 MM5 的 dK 部分结果就绪，不表示后续 MM6 也已结束 |
| 主计算 → 最终 Vector2 后处理 | `SyncAll<false>()` | 全体核的输出必须先于后处理读取 |

实验版的优势体现在三个层面：**P/SY 两段链路的通知指令减少、业务与全核屏障的编号职责更清晰、
边界分支的同步配对更完整**。这些变化为降低同步开销、减少跨阶段误配风险和维护流水逻辑提供了明确的代码基础。
保留生产者完成、消费者等待及缓冲区代际约束，是这些优化成立的前提。

以上通知数量变化已由源码确认；实际性能增益与运行稳定性的覆盖范围还取决于 MM1/MM2 的流水重叠、
输入配置和参与核的一致性。本节未进行仅切换 event ID 的性能对照，也未新增多流或长期循环压力测试。
两版还包含 TopK 容量、mask 语义和向量计算拆分等差异，既有精度结果及未解决问题保留在
“验证结果与已知限制”中，不以 event ID 调整替代这些问题的独立验证。

## 构建

仓库标准入口（需要准备齐整仓构建依赖）：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --ops=lightning_indexer_grad_kl_loss --soc=ascend910b -j8
```

也可使用独立构建入口，在仓库根目录执行：

```bash
bash experimental/attention/lightning_indexer_grad_kl_loss/tools/build_standalone.sh \
  /workspace/lightning_grad_build ascend910b
```

它编译 Host、FP16/BF16 内核，安装到 `<BUILD_DIR>/stage/packages/vendors/experimental_lightning_indexer_grad_kl_loss`，并生成 `<BUILD_DIR>/package/custom_opp_*.run`。

构建同时包含前向和梯度接口的 Torch wheel：

```bash
bash experimental/attention/lightning_indexer_grad_kl_loss/tools/build_torch.sh \
  /workspace/lightning_grad_python codex_lig /workspace/lightning_grad_wheels
```

公共 `builder.py` 保持仓库原样。该入口调用 `experimental/tools/build_torch_vendor.py`，
复制框架及前向/梯度接口到临时目录，仅在副本中修复 vendor 相对导入后构建 wheel。
wheel 保存在第三个参数指定的目录；省略该参数时为安装目录父目录下的 `wheels/`。

## 执行 golden 测试

`tests/lightning_indexer_klloss_golden.py` 生成输入、前向参考结果和 autograd 梯度参考，并按以下方式验证：

1. 前向 indexer 和梯度均通过 `torch.ops.cann_ops_transformer` 接口调用。
2. 前置 SFA 使用 `torch_npu.npu_sparse_flash_attention`，参数名为 `actual_seq_lengths_*`，序列长度为 int32；MLA-absorb 模式使用 `attention_mode=2`，窗口使用 INT64_MAX。
3. 梯度使用 `rtol=atol=5e-3`；超过阈值的元素比例大于 0.1% 时判为失败。
4. loss 检查有限性及 `rtol=atol=5e-3`。测试结束时汇总失败信息，以非零退出码报告。
5. 默认使用小规模 Query：`T1=16, TopK=2048, T2=4096/8192`。golden 测试范围限制为 `seq_k<=8192`；更大值会明确打印并跳过，不改变算子的 API 或内核支持范围。可通过 `NPU_DEVICE`、`LIG_T1`、`LIG_TOPK`、`LIG_T2_LIST` 调整配置。

为降低参考计算显存占用，teacher Attention 在 `torch.no_grad()` 下计算，仅对 KL loss 求反向；TopK 索引已经 detach，Attention 输出不会贡献 indexer 参数梯度，因此不改变三路 indexer 梯度的参考定义。

可通过 `LIG_STATS_SOURCE=reference` 使用 golden 自身的 max/sum 做对照；默认仍为 `sfa`。

golden 运行于 PyTorch/NPU，其反向参考由 autograd 计算，并非 CPU 参考。测试不启用性能循环。

```bash
# 参数分别是：梯度构建目录、前向 indexer 构建目录、wheel 安装目录、Python 包名。
bash experimental/attention/lightning_indexer_grad_kl_loss/tools/run_golden.sh \
  /workspace/lightning_grad_build /workspace/lightning_indexer_build \
  /workspace/lightning_grad_python cann_ops_transformer_codex_lig

# 小规模验证，使用 TopK=2048。
LIG_T1=16 LIG_TOPK=2048 LIG_T2_LIST=4096 \
bash experimental/attention/lightning_indexer_grad_kl_loss/tools/run_golden.sh \
  /workspace/lightning_grad_build /workspace/lightning_indexer_build \
  /workspace/lightning_grad_python cann_ops_transformer_codex_lig

# 验证 skip_padding 接口的全有效 mask 路径。
LIG_USE_SKIP_PADDING=1 LIG_T1=16 LIG_TOPK=2048 LIG_T2_LIST=4096 \
bash experimental/attention/lightning_indexer_grad_kl_loss/tools/run_golden.sh \
  /workspace/lightning_grad_build /workspace/lightning_indexer_build \
  /workspace/lightning_grad_python cann_ops_transformer_codex_lig
```

已知模板限制：`TopK=128` 虽能通过参数范围检查，但 Tiling 会向下对齐产生 `TOPK_RANGE=0`，不在已声明模板集合中，因此无法执行。验证采用 TopK=2048。

## 验证结果与已知限制

以下配置已在 Ascend 910B2C、CANN 8.5.0、Python 3.11、PyTorch 2.9.0 / torch_npu 2.9.0 环境验证。Host、FP16/BF16 内核、安装包和 Torch 桥接编译、加载成功，精度结果使用上述阈值：

| 测试 | 结果 |
| --- | --- |
| 普通接口，T1=16、T2=4096、TopK=2048 | dQ_index、dK_index、dW 全部通过阈值；loss 差约 1.53e-5 |
| skip_padding，全有效 mask，同上形状 | 三路梯度和 loss 均通过 |
| T1=1024、T2=4096、TopK=2048 | dQ_index/dW/loss 通过；dK_index 超阈值元素为 881/524288，即 0.168037%，超过允许的 0.1% |
| T1=1024、T2=8192、TopK=2048 | 三路梯度和 loss 通过；loss 差约 0.001 |
| T1=1024、T2=12288、TopK=2048 | golden 存在问题，排除出精度验证范围；该配置的差异不作为算子精度结论 |
| T1=1024、T2=16384、TopK=2048 | 超出当前验证范围；已有尝试出现显存分配失败及同步阶段 MTE 异常 |

对 T2=4096 的对照中，将 SFA max/sum 换成 golden max/sum 后，dK_index 仍有相同的 0.168037% 超阈值比例。此对照只排除了该用例中单纯由 SFA 统计量来源造成差异的解释；T2=4096 的 dK_index 阈值问题仍单独保留，尚未定位根因。

以上结果不代表完整精度验收通过。测试默认使用小规模 Query，并将 seq_k 限定在 8192 以内；未提供大规模压力测试或性能结论。

如需验证范围内的较大 Query 配置，在仓库根目录执行：

```bash
LIG_T1=1024 LIG_TOPK=2048 LIG_T2_LIST=4096,8192 \
bash experimental/attention/lightning_indexer_grad_kl_loss/tools/run_golden.sh \
  /workspace/lightning_grad_build /workspace/lightning_indexer_build \
  /workspace/lightning_grad_python cann_ops_transformer_codex_lig
```
