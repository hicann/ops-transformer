# mega_recurrent_kda PyTorch API

[English](torchapi_mega_recurrent_kda_en.md)

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

对 BSH 或 TH 输入执行融合的 KDA 递推，读取分页 checkpoint 并更新 Conv 和 replay 池。公式与阶段说明见[算子概览](../README.md#功能说明)。

## 接口说明

调用 `torch.ops.cann_ops_transformer.mega_recurrent_kda`。

### PyTorch 接口原型

```python
def mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

## 参数说明

本文按从 1 开始的方式描述张量的第几维。代码索引如 `shape[2]` 和 `stride(0)` 从 0 开始。

### 维度符号说明

- `B` 本次调用的逻辑请求数。BSH 取输入第 1 维，TH 取 `cu_seqlens.numel() - 1`。
- `b` 优化路径条件中的模型批量，满足 `B = b * TP`。
- `seq_len` BSH 中每个请求的 token 数。
- `total_tokens` TH 中各请求 token 数之和。
- `H` 隐藏维度，固定为 7168。
- `N` 当前 TP 实例的 head 数。TP4、TP8、TP16 分别对应 24、12、6。
- `D` 每个 head 的维度，固定为 128。
- `P` 投影通道数，`P = N * D`。
- `block_num` 五个状态池共用的物理块数，对应各池第 1 维。
- `L` 每个 Conv 块的行数，对应 `conv_state` 第 2 维，至少为最长序列长度加 2。
- `S` 每个 replay 块中每个 head 的 token 容量，对应 replay 池第 3 维，至少覆盖最长序列。

### 参数表

| 参数名 | 输入/输出/属性 | 必选/可选 | 数据类型/格式 | 形状或取值 | 描述 |
| --- | --- | --- | --- | --- | --- |
| `hidden_states` | 输入 | 必选 | BF16 / ND | `[B,seq_len,H]` 或 `[total_tokens,H]` | BSH 或 TH 输入。 |
| `qkv_projection_weight` | 输入 | 必选 | BF16 / NZ | `[3*P,H]` | Q、K、V 投影权重。 |
| `decay_projection_a_weight` | 输入 | 必选 | BF16 / NZ | `[D,H]` | 第一级 decay 投影。 |
| `decay_projection_b_weight` | 输入 | 必选 | BF16 / NZ | `[P,D]` | 第二级 decay 投影。 |
| `beta_projection_weight` | 输入 | 必选 | BF16 / NZ | `[N,H]` | beta 投影。 |
| `output_gate_projection_weight` | 输入 | 必选 | BF16 / NZ | `[P,H]` | 输出门控投影。 |
| `output_norm_weight` | 输入 | 必选 | FP32 / ND | `[D]` | 输出 RMSNorm 权重。 |
| `output_projection_weight` | 输入 | 必选 | BF16 / NZ | `[H,P]` | 输出投影。 |
| `conv1d_weight` | 输入 | 必选 | BF16 / ND | `[4,3*P]`、`[3,N,4,D]` 或 `[3,4,D]` | Conv 权重，卷积核长度为 4。 |
| `conv_state` | 输入/输出 | 必选 | BF16 / ND | `[block_num,L,3*P]` | PA Conv 池，原地更新。 |
| `checkpoint_state` | 输入 | 必选 | FP32 / ND | `[block_num,N,D,D]` | PA checkpoint 池，只读。 |
| `replay_u` | 输入/输出 | 必选 | FP32 / ND | `[block_num,N,S,D]` | PA 更新向量 U 记录池，原地更新。 |
| `replay_k` | 输入/输出 | 必选 | FP16 / ND | `[block_num,N,S,D]` | PA 归一化 K 记录池，原地更新。 |
| `replay_decay` | 输入/输出 | 必选 | FP32 / ND | `[block_num,N,S,D]` | PA 指数化衰减系数记录池，原地更新。 |
| `state_indices` | 输入 | 必选 | int32 / ND | `[B]` | 每个逻辑 batch 在五个池中共用的块号。 |
| `conv_num_accepted_tokens` | 输入 | 必选 | int32 / ND | `[B]` | 选择 Conv 历史窗口的计数。 |
| `A_log` | 输入 | 必选 | FP32 / ND | `[N]` | log decay 参数。 |
| `dt_bias` | 输入 | 必选 | FP32 / ND | `[N,D]` | decay bias。 |
| `scale` | 属性 | 必选 | FP32 scalar | — | 有限 query scale，常用 `D**-0.5`。 |
| `lower_bound` | 属性 | 必选 | FP32 scalar | — | decay gate 下界，在 `[-5,0]` 内。 |
| `rms_norm_eps` | 属性 | 可选（默认 `1e-5`） | FP32 scalar | — | 输出 RMSNorm epsilon。 |
| `cu_seqlens` | 输入 | 可选（TH 必传） | int32/int64 / ND | `[B+1]` 或 None | TH 的序列前缀和；BSH 必须为 None。 |
| `out` | 输出 | — | BF16 / ND | `[B,seq_len,H]` / `[total_tokens,H]` | 主输出，形状与输入一致。 |

## 返回值

返回 `(out, conv_state, replay_u, replay_k, replay_decay)`。`out` 是与 `hidden_states` 同形状的 BF16 张量，后四项依次返回对应的输入池并保留其存储布局。`checkpoint_state` 为只读输入。

## 约束说明

### 约束类型说明

- **单参数约束** 说明单个张量的形状、类型、格式和数值范围，或单个属性的取值。
- **存在性约束** 说明特定场景需要提供或省略的参数。
- **一致性约束** 说明同组参数的维度、类型或存储关系。
- **特性交叉约束** 说明不同组之间的维度、容量或执行条件。

### 特性参数组

分组按计算阶段和职责划分。可选字段以外的输入均须提供。四个 `INPUT/OUTPUT` 张量原地更新并作为返回值返回。

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
| --- | --- | --- | --- |
| 公共 | `hidden_states` | INPUT | Tensor |
|  | `out` | OUTPUT | Tensor |
| QKV 投影 | `qkv_projection_weight` | INPUT | Tensor |
| 因果 Conv1D | `conv1d_weight` | INPUT | Tensor |
|  | `conv_state` | INPUT/OUTPUT | Tensor |
|  | `conv_num_accepted_tokens` | INPUT | Tensor |
| 衰减投影与门控 | `decay_projection_a_weight` | INPUT | Tensor |
|  | `decay_projection_b_weight` | INPUT | Tensor |
|  | `A_log` | INPUT | Tensor |
|  | `dt_bias` | INPUT | Tensor |
|  | `lower_bound` | ATTR | float |
| Beta 投影与查询缩放 | `beta_projection_weight` | INPUT | Tensor |
|  | `scale` | ATTR | float |
| 分页递推与回放 | `state_indices` | INPUT | Tensor |
|  | `checkpoint_state` | INPUT | Tensor |
|  | `replay_u` | INPUT/OUTPUT | Tensor |
|  | `replay_k` | INPUT/OUTPUT | Tensor |
|  | `replay_decay` | INPUT/OUTPUT | Tensor |
| 输出归一化、门控与投影 | `output_norm_weight` | INPUT | Tensor |
|  | `rms_norm_eps` | ATTR(OPTIONAL) | float |
|  | `output_gate_projection_weight` | INPUT | Tensor |
|  | `output_projection_weight` | INPUT | Tensor |
| 变长输入 | `cu_seqlens` | INPUT(OPTIONAL) | Tensor |

### 基准信息说明

| 命名 | 含义 |
| --- | --- |
| `max_seq_len` | BSH 取 `seq_len`，TH 取 `cu_seqlens` 的最大相邻差值。 |
| `slot` | 当前请求的物理块号，取 `state_indices[b]`。 |
| `pool.stride(0)` | 各状态池相邻物理块之间的元素间距。 |

### 参数组约束

各条约束按拦截行为标注，标注含义如下：

- 【已拦截】调用时同步校验，违反即抛出 `ValueError` 或 `TypeError`，且算子不改动任何状态池（原子拒绝）。校验覆盖形状、数据类型、格式、布局、标量取值范围和存储别名关系，不校验张量数值内容（如权重、`A_log`、`dt_bias` 和 checkpoint 初值的数值），数值正确性由调用方保证。
- 【已拦截（主机侧）】取值校验需要把设备张量读到主机执行；图捕获期间这类值校验自动跳过，重放前须由调用方写入合法值（见[图捕获](#图捕获)）。
- 【不拦截】算子不做校验或无需校验的情形，包括由调用方负责的内容与语义约定，以及由算子执行行为天然保证的结果。
- 同一条约束内的混合语句在句内单独注明拦截状态。

#### 公共参数组

- **单参数约束【已拦截】** `hidden_states` 为 BF16/ND，隐藏维度固定为 7168。BSH 的批量和序列长度为正，TH 的 token 数为正，批量满足 `1 <= B <= block_num`。两种布局的总 token 数均小于 `2**31`（主机侧值校验，图捕获期间跳过）。

- **存在性约束【已拦截】** 接口用于推理。除 `rms_norm_eps` 和 `cu_seqlens` 外，所有输入均须提供，缺失参数由接口 schema 直接拒绝。使用普通 Tensor，Parameter 先调用 `.detach()`，未 detach 的 Parameter 输入会被拒绝。

- **一致性约束【已拦截】** 所有张量位于同一 NPU，设备一致性逐张量校验并要求位于 NPU 设备。主输出与输入的形状、布局和数据类型一致（不拦截：输出由算子按输入形状和布局分配，天然成立）。

- **特性交叉约束【已拦截】** 六个投影权重须预先转换为 NZ，其余张量使用 ND，NZ/ND 格式逐张量校验。五个状态池以外的输入张量必须连续。

#### QKV 投影参数组

- **单参数约束【已拦截】** `qkv_projection_weight` 为 BF16/NZ，形状为 `[3*P,H]`，依次对应 Q、K、V。Q、K、V 通道的排列顺序属于数据内容约定（不拦截：算子按固定顺序切分，不检查权重内容排布）。

- **特性交叉约束【已拦截】** 输入宽度与 `hidden_states` 的 `H` 一致。输出的 `3*P` 通道与 Conv 权重及历史池匹配（Conv 权重形状枚举和 `conv_state` 第 3 维均按同一 `3*P` 校验）。

#### 因果 Conv1D参数组

- **单参数约束【已拦截】** `conv1d_weight` 支持 `[4,3*P]`、`[3,N,4,D]` 或 `[3,4,D]`，固定 4 个卷积 tap。`conv_state` 为 BF16/ND，形状为 `[block_num,L,3*P]`，`L >= 3` 且 `block_num` 为正。

- **一致性约束【已拦截（主机侧）】** `conv_num_accepted_tokens` 为 `[B]` 的 int32 张量，取值为 `[0,L-2]`。历史从 `max(count-1,0)` 开始读取三行、后两行写回块首、本轮未卷积的 QKV 投影紧随其后均为执行行为（不拦截）；计数值与真实历史长度是否语义一致由调用方保证（不拦截）。

- **特性交叉约束【已拦截】** 按 `state_indices` 选择历史块。若本次最长请求有 `max_seq_len` 个 token，须满足 `L >= max_seq_len + 2`（BSH 在调用入口校验；TH 依赖主机侧序列长度，图捕获期间跳过）。

#### 衰减投影与门控参数组

- **单参数约束【已拦截】** 两级权重均为 BF16/NZ，形状依次为 `[D,H]` 和 `[P,D]`。`A_log`、`dt_bias` 为 FP32/ND。`lower_bound` 为有限 FP32，取值为 `[-5,0]`。

- **一致性约束【已拦截】** 第一级的输出宽度等于第二级的输入宽度 `D`。第二级的输出按 `[N,D]` 分组，与 `A_log[N]` 和 `dt_bias[N,D]` 的 head 对齐。

#### Beta 投影与查询缩放参数组

- **单参数约束【已拦截】** `beta_projection_weight` 为 BF16/NZ，形状为 `[N,H]`。`scale` 须可表示为有限 FP32；其具体取值不做限制（不拦截：常用值为 `D**-0.5`，数值效果由调用方负责）。Q/K 归一化使用固定的 `1e-6`（编译期常量，无需拦截）。

- **特性交叉约束【已拦截】** `N` 取 24、12 或 6，分别对应 TP4、TP8、TP16。所有权重与状态池按同一 `N` 和 `D=128` 组织。

#### 分页递推与回放参数组

- **单参数约束【已拦截（主机侧）】** `state_indices` 为 `[B]` 的 int32 张量，块号在 `[0,block_num)` 内，同次调用互不重复（含块 0，块 0 是普通有效块）。块号取值与去重校验需读到主机，图捕获期间跳过。`checkpoint_state` 为只读输入属于行为承诺（不拦截：算子不写入该池，无需校验）。

- **一致性约束【已拦截】** 三个 replay 池的形状均为 `[block_num,N,S,D]`，分别存储 FP32 更新向量、FP16 归一化 key 和 FP32 指数化衰减系数，`S` 为正。只写所选块的有效 token 前缀，保留其他块、记录尾部及块间间隙均为执行行为（不拦截）。

- **特性交叉约束【已拦截】** Conv、checkpoint 和三个 replay 池共享 `block_num` 与每个请求的物理块号。各池的 `stride(0)` 可独立设置，但至少为一个块的元素数，其余轴必须连续。四个可写池须设置 `requires_grad=False`，且不得与其他输入的有效存储区重叠（别名重叠逐对校验）。回放容量须满足 `S >= max_seq_len`（BSH 在调用入口校验；TH 依赖主机侧序列长度，图捕获期间跳过）。

例如，`state_indices=[5,0]` 且两个请求分别有 1 和 3 个 token 时，算子分别写入物理块 5 的第 0 条 replay 记录和块 0 的前 3 条记录。两块剩余记录以及其他块保留原值。若第二个请求的 Conv 计数为 2，则从块 0 的第 1 行开始读取历史。

回放元素偏移按各池自身数据类型的元素数计算。

```text
slot * replay.stride(0) + (head * S + token) * D + channel
```

#### 输出归一化、门控与投影参数组

- **单参数约束【已拦截】** `output_norm_weight` 为 `[D]` 的 FP32/ND 张量，门控权重和输出投影权重分别为 `[P,H]` 和 `[H,P]` 的 BF16/NZ 张量。`rms_norm_eps` 必须为正的正规 FP32 数。

- **存在性约束【已拦截】** 省略 `rms_norm_eps` 时使用 `1e-5`（由接口默认值保证）。

- **特性交叉约束【已拦截】** RMSNorm 权重在各 head 间共享。门控投影与拼接后的 `P` 个通道对齐，最终投影恢复到隐藏宽度 `H`。

#### 变长输入参数组

- **单参数约束【已拦截】** `cu_seqlens` 为 `[B+1]` 的 int32 或 int64 连续 ND 张量。

- **存在性约束【已拦截】** TH 必须传入 `cu_seqlens`，BSH 使用 `None`。

- **一致性约束【已拦截（主机侧）】** 前缀和从 0 开始，严格递增，末项等于 `total_tokens`。每个相邻差值是对应请求的有效 token 数。值校验需读到主机，图捕获期间跳过。

- **特性交叉约束【已拦截（主机侧）】** 最长相邻差值决定 `max_seq_len`，同时用于 Conv 和 replay 容量检查。TH 需要将前缀和读到主机，因此不支持图捕获（属执行限制而非校验项，无法在算子内拦截，由调用方避免在 TH 输入上捕获）。

### 执行路径与核数配额

| 实现目录 | 路径 | 矩阵核配额 | 每个矩阵核对应的向量核数 |
|---|---|---|---|
| `arch35` | 等长 4 或 8 token 且 `B=b*TP`、`1 <= b <= 32` 时使用对应的优化路径，其余合法输入使用通用路径。 | 8、16、24、28、32 | 2 |
| `arch92` | BSH 和 TH 均使用通用路径。 | 8、16、24、32、64 | 1 |

配额来自当前执行流，调用方可通过执行流资源限制指定。

运行 pytest 或导出用例时可用 `KDA_ARCH=arch92` 选择该实现的用例。TTK 用例和运行命令见[用例说明](../tests/ttk/README.md)。

### 图捕获

本文的图捕获指 `torch_npu.npu.NPUGraph` / `torch_npu.npu.graph` 使用的 ACL Graph 流捕获机制。
捕获期间跳过设备张量的输入值校验，以及包装层的事件等待和事件记录。静态参数校验保留，块号可用重复的 0 占位，重放前须写入合法块号和 Conv 计数。
普通调用（包括捕获前的预热）继续执行值校验。共享调用锁和完成事件保护工作区及池的复用。编译和工作区预热须在捕获前完成，
调用方负责图与共享工作区的生命周期和串行使用。通用路径的 BSH 输入在设备上生成等距序列前缀。TH 输入仍读取 `cu_seqlens` 到主机以选择执行路径，
因此 TH 输入的捕获不在支持范围内。

Torch 扩展为 `torch.ops.cann_ops_transformer.mega_recurrent_kda` 注册 Meta 和 PrivateUse1 实现。通过 TorchDynamo 构图时可直接调用该算子。

## 调用示例

下面的 BSH 示例使用形状为 `[2,4,7168]` 的 BF16/ND `hidden_states`。六个 NZ 投影权重、`conv1d_weight`、`output_norm_weight`、`A_log` 和 `dt_bias` 按参数表准备。安装仓库构建的算子包和 PyTorch 扩展后，导入扩展以注册接口。示例使用默认扩展包名 `cann_ops_transformer`，实际包名随构建配置而定。

```python
import torch
import cann_ops_transformer

block_num, N, D = 6, 12, 128
S, L = 8, 10
P = N * D
device = hidden_states.device

conv_state = torch.zeros(2*block_num, L, 3*P, dtype=torch.bfloat16, device=device)[::2]
checkpoint_state = torch.zeros(3*block_num, N, D, D, dtype=torch.float32, device=device)[::3]
replay_u = torch.zeros(2*block_num, N, S, D, dtype=torch.float32, device=device)[::2]
replay_k = torch.zeros(3*block_num, N, S, D, dtype=torch.float16, device=device)[::3]
replay_decay = torch.zeros(4*block_num, N, S, D, dtype=torch.float32, device=device)[::4]
state_indices = torch.tensor([5, 0], dtype=torch.int32, device=device)
conv_num_accepted_tokens = torch.zeros(2, dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states=hidden_states,
    qkv_projection_weight=qkv_projection_weight,
    decay_projection_a_weight=decay_projection_a_weight,
    decay_projection_b_weight=decay_projection_b_weight,
    beta_projection_weight=beta_projection_weight,
    output_gate_projection_weight=output_gate_projection_weight,
    output_norm_weight=output_norm_weight,
    output_projection_weight=output_projection_weight,
    conv1d_weight=conv1d_weight,
    conv_state=conv_state,
    checkpoint_state=checkpoint_state,
    replay_u=replay_u,
    replay_k=replay_k,
    replay_decay=replay_decay,
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    A_log=A_log,
    dt_bias=dt_bias,
    scale=D**-0.5,
    lower_bound=-5.0,
)
# 第一个请求的记录写入块 5，第二个请求的记录写入块 0。
```

示例返回的后四项分别引用传入的四个可写池。

TH 调用使用相同接口。例如，两个请求各有 1 和 3 个 token 时，将输入按请求顺序拼成 `[4,7168]`，传入同设备的 `cu_seqlens=[0,1,4]`（int32 或 int64 张量），仍为每个请求提供一个块号和一个 Conv 计数。
