# FlashKDA

[English](FlashKDA_en.md)

FlashKDA 基于 CANNBot-DSL 实现 Kimi Delta Attention 的 prefill 计算，训练前向和推理共用同一套实现。

- `flash_kda` 是算子公开入口，固定返回 `(out, final_state, chunk_states)`。
- `flash_kda_fwd` 使用相同输入和模板，固定返回 16 项，供需要读取训练缓存的调用方使用。
- `flash_kda_metadata` 是公开的调度元数据生成接口。调用方应令 `enable_grad` 与主算子的运行模式一致：推理为 False，训练为 True。
- 运行模式由 `torch.is_inference_mode_enabled()` 决定，`torch.no_grad()` 不会切换到推理分支。当前支持训练前向，尚未实现自动反向。

## 产品支持情况

| 产品 | FlashKDA 支持情况 |
|---|---|
| Ascend 950PR 系列 | 支持。 |
| Ascend 950DT 系列 | 支持，构建目标为 `ascend950`，metadata 生成器使用 DT 专用调度参数。 |
| Atlas A2、A3 系列，Atlas 200I/500 A2 推理产品及其他 Atlas 推理、训练系列 | 未列入当前 FlashKDA 的适配范围。 |

## 目录与入口

| 目录 | 用途 |
|---|---|
| `op_dsl/` | 前向实现、调度元数据和参数校验 |
| `torch_extension/` | 三个公开接口的 Torch schema 与分发注册 |
| `tests/ut/` | pytest 前向与 metadata 测试 |
| `tests/ttk/` | TTK 适配器、参考实现、CSV 和运行文档 |

DSL 模块分工如下。

- [`flash_kda.py`](../op_dsl/flash_kda.py)：公开算子入口及内部前向实现。
- [`flash_kda_metadata.py`](../op_dsl/flash_kda_metadata.py)：Host 模板选择、AICPU metadata 生成与解码。
- [`flash_kda_staged.py`](../op_dsl/flash_kda_staged.py)：按 chunk group 分阶段调度，通过 workspace 交换阶段结果。
- [`flash_kda_persistent.py`](../op_dsl/flash_kda_persistent.py)：一个核连续处理序列头的 chunk，使递推状态驻留。
- [`flash_kda_validation.py`](../op_dsl/flash_kda_validation.py)：主入口、metadata 与 Meta/FakeTensor 共用的参数和调度校验。

## 功能说明

按需融合 Q/K 行级 L2 normalize、gate 激活和 beta sigmoid，将序列按 64 个 token 切分，在 chunk 内构造因果 QK 矩阵及下三角逆矩阵，再跨 chunk 递推 FP32 状态。支持非对齐尾块和 GQA。训练模式按开关生成反向所需的缓存。

以下 $Q,K$ 表示按开关完成归一化后的输入，$g,\beta$ 表示完成对应激活后的值。每个 chunk 独立累加对数衰减，令

$$
G_t=\sum_{i=0}^{t}g_i,\quad \gamma_t=\exp(G_t),\quad
\tilde K=\gamma\odot K,\quad \bar K=\gamma^{-1}\odot K,\quad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q).
$$

块内求解为

$$
T_c=\operatorname{stril}\!\left(\operatorname{diag}(\beta)\tilde K\bar K^\top\right),\quad
A_c=(I+T_c)^{-1},\quad
U=A_c\operatorname{diag}(\beta)V,\quad
W=A_c\operatorname{diag}(\beta)\tilde K,
$$

$$
M_{qk}=\operatorname{tril}(\tilde Q\bar K^\top),\quad
K_{\mathrm{restored},t}=\exp(G_{\mathrm{last}}-G_t)\odot K_t.
$$

$\operatorname{stril}$ 取严格下三角，$\operatorname{tril}$ 保留下三角及对角。令 chunk 入口状态 $H_c$ 的维度为 $D_v\times D_k$，与公开状态张量一致，$H_0=$ `initial_state`，则

$$
V_{\mathrm{new}}=U-WH_c^\top,\qquad
O_c=\tilde QH_c^\top+M_{qk}V_{\mathrm{new}},
$$

$$
H_{c+1}=H_c\operatorname{diag}(\gamma_{\mathrm{last}})
       +V_{\mathrm{new}}^\top K_{\mathrm{restored}}.
$$

内部训练缓存 `Aqk`、`Akk`、`w`、`u`、`v_new` 分别对应 $M_{qk}$、$A_c$、$W$、$U$、$V_{\mathrm{new}}$；`qg` 对应未乘 scale 的 $\gamma\odot Q$，`kg` 对应 $K_{\mathrm{restored}}$，`g_cumsum` 对应 $G$。公开的 `chunk_states` 保存各 chunk 的入口状态 $H_c$，仅在推理模式按开关返回。训练缓存通过 `flash_kda_fwd` 的后十三个槽位按开关返回。状态使用 FP32 递推。

metadata 的 Host 入口根据布局、有效长度、`enable_grad` 和调度配置，沿用原成本模型自动选择 staged/persistent 模板及分组上限；AICPU 根据选择结果生成调度表。两个模板共用公开接口和缓存布局。公开 metadata 接口不接受 `template`；定向模板测试使用内部构建函数，见同仓库测试用例。

## 快速开始

先安装下文的 Flash KDA Torch 扩展 wheel，并按下文设置 Native 包路径。下面使用 BNSD 布局、完整序列长度和默认缩放值：

```python
import torch
import torch_npu

from cann_ops_transformer import flash_kda, flash_kda_metadata

B, Nqk, Nv, S, D = 1, 1, 2, 65, 128
q = torch.randn(B, Nqk, S, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D, device="npu", dtype=torch.bfloat16)
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S, device="npu", dtype=torch.float32)
initial_state = torch.zeros(B, Nv, D, D, device="npu", dtype=torch.float32)
A_log = torch.zeros(Nv, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(Nv, D, device="npu", dtype=torch.float32)

# Inference: explicitly enter inference_mode and generate inference metadata.
with torch.inference_mode():
    metadata = flash_kda_metadata(
        batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S,
        enable_grad=False, layout="BNSD",
    )
    out, final_state, chunk_states = flash_kda(
        q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
        layout="BNSD", output_final_state=True,
        return_intermediate_states=True,
    )

# Training: use the same flash_kda entry outside inference_mode.
metadata = flash_kda_metadata(
    batch_size=B, nheads_qk=Nqk, nheads_v=Nv, seqlens=S,
    enable_grad=True, layout="BNSD",
)
out, final_state, chunk_states = flash_kda(
    q, k, v, g, beta, metadata, initial_state, A_log, dt_bias,
    layout="BNSD", output_final_state=True,
)
assert chunk_states is None
```

## PyTorch 接口原型

导入 `cann_ops_transformer` 后可调用以下三个接口。两个前向入口的第六个参数 `metadata` 均无默认值，执行时需传入有效调度张量。`flash_kda_fwd` 固定返回 16 个槽位，顺序及各项返回条件见[返回值说明](#返回值说明)。

### `flash_kda`

```text
torch.ops.cann_ops_transformer.flash_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    metadata: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    scale: Optional[float] = 0.08838834764831845,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
```

### `flash_kda_fwd`

```text
torch.ops.cann_ops_transformer.flash_kda_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    metadata: Optional[Tensor],
    initial_state: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    scale: Optional[float] = 0.08838834764831845,
    output_final_state: bool = False,
    return_intermediate_states: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[
    Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor],
    Optional[Tensor], Optional[Tensor], Optional[Tensor]
]
```

### `flash_kda_metadata`

```text
torch.ops.cann_ops_transformer.flash_kda_metadata(
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    *,
    batch_size: int,
    nheads_qk: int,
    nheads_v: int,
    seqlens: int,
    headdim_qk: int = 128,
    headdim_v: int = 128,
    enable_grad: bool = False,
    layout: str = "TND",
) -> Tensor
```

## 参数说明

`B` 为逻辑序列数，`Nqk`、`Nv` 为 Q/K、Value 头数，`S` 为存储序列长度，`T` 为 packed token 总数，`Dk=Dv=D=128`。

| 参数 | BNSD shape / 取值 | 类型 | 必选/可选 | 说明 |
|:---|:---|:---|:---|:---|
| `q`, `k` | `[B,Nqk,S,D]` | BF16 | 必选 | Query/Key；可在内核归一化，也可由调用方预处理。 |
| `v` | `[B,Nv,S,D]` | BF16 | 必选 | Value。 |
| `g` | 与 `v` 相同 | BF16 / FP32 | 必选 | 原始 gate 或预处理的对数衰减；FP32 输入会转换为 BF16。 |
| `beta` | `[B,Nv,S]` | BF16 / FP32 | 必选 | 原始 logits 或预处理 beta；核内激活使用 FP32 操作数。 |
| `metadata` | `[M]` | int32 Tensor | 必传 | DSL 类型注解为 `Optional[torch.Tensor]`。执行时须传入合法的调度 Tensor。 |
| `initial_state` | `[B,Nv,Dv,Dk]` | FP32 | 可选 | 可为 None，此时使用全零状态；已传入的状态必须符合规格。 |
| `A_log` | `[Nv]` | FP32 / None | 条件必选 | 内核激活 gate 时必传；外部激活时可为 None。 |
| `dt_bias` | `[Nv,Dk]` | FP32 / None | 可选 | gate 偏置；None 按零处理。 |
| `cu_seqlens` | `[B+1]` | int32 / None | 条件必选 | 有效长度前缀和；TND 必传，首项为 0、末项为 T。 |
| `seqused` | `[B]` | int32 / None | 可选 | 每条序列的有效长度；同时提供 cu 时必须与相邻差值一致。 |
| `scale` | 默认 `D ** -0.5` | float / None | 可选 | 传 None 时按头维度计算缩放。 |
| `output_final_state` | 默认 False | bool | 可选 | 控制是否返回最终状态。 |
| `return_intermediate_states` | 默认 False | bool | 可选 | 仅推理且 True 时返回各 chunk 入口状态。 |
| `use_qk_l2norm_in_kernel` | 默认 True | bool | 可选 | True 执行 L2Norm；训练时由内部实现保存 norm/rstd。 |
| `use_gate_in_kernel` | 默认 True | bool | 可选 | False 时 g 已激活，忽略 `A_log`、`dt_bias`、`lower_bound`。 |
| `use_beta_sigmoid_in_kernel` | 默认 True | bool | 可选 | True 执行 sigmoid；训练时由内部实现保存激活结果。 |
| `allow_neg_eigval` | 默认 False | bool | 可选 | True 时 sigmoid beta 乘 2；要求开启 beta sigmoid。 |
| `disable_recompute` | 默认 False | bool | 可选 | 控制训练前向的中间缓存保存；True 时保留用于避免重计算的内部缓存，六个对应槽位由 `flash_kda_fwd` 返回。 |
| `lower_bound` | 默认 None，数值范围 `[-5,0)` | int / float / None | 可选 | gate 开启时 None 使用 -exp(A_log) × softplus(g + dt_bias)，有限数值使用有界 sigmoid；关闭时忽略。 |
| `layout` | `"BNSD"` / `"BSND"` / `"TND"` | str | 可选 | 默认 `TND`。 |

BSND 的 `q/k` 为 `[B,S,Nqk,D]`，`v/g` 为 `[B,S,Nv,D]`，`beta` 为 `[B,S,Nv]`。TND 分别为 `[T,Nqk,D]`、`[T,Nv,D]`、`[T,Nv]`。

`flash_kda_metadata` 接收尺寸属性及可选的长度张量，用于生成传给 `flash_kda` 的调度元数据。

| 参数 | shape / 取值 | 类型 | 必选/可选 | 说明 |
|:---|:---|:---|:---|:---|
| `cu_seqlens` | `[B+1]` / None | int32 Tensor / None | 条件必选 | TND 布局必传；有效长度前缀和，首项为 0、末项为 T。BNSD/BSND 可省略。张量须连续且位于当前 NPU。 |
| `seqused` | `[B]` / None | int32 Tensor / None | 可选 | 每条序列的有效长度；同时提供 `cu_seqlens` 时必须与相邻差值一致。张量须连续且位于当前 NPU。 |
| `batch_size` | `B >= 1` | int | 必选（关键字） | 逻辑序列数。 |
| `nheads_qk` | `1 <= Nqk <= Nv` | int | 必选（关键字） | Query/Key 头数。 |
| `nheads_v` | `Nv <= 96`，`Nv % Nqk == 0` | int | 必选（关键字） | Value 头数。 |
| `seqlens` | 正整数 | int | 必选（关键字） | BNSD/BSND 的存储长度 S，或 TND 的总 token 数 T。 |
| `headdim_qk` | 默认 128，当前仅支持 128 | int | 可选 | Query/Key 头维度。 |
| `headdim_v` | 默认 128，当前仅支持 128 | int | 可选 | Value 头维度。 |
| `enable_grad` | 默认 False | bool | 可选 | 推理使用 False，训练前向使用 True；与主算子运行模式保持一致。 |
| `layout` | `"BNSD"` / `"BSND"` / `"TND"` | str | 可选 | 默认 `"TND"`，与主算子的布局保持一致。 |

调用时必须显式提供 `seqlens`：BNSD/BSND 为存储长度 S，TND 为总长度 T。TND 必须传连续 INT32 `[B+1]` 的 `cu_seqlens`；padded 布局可用 cu 或 `seqused` 指定有效长度，不传时均为 S。

给定的长度张量必须位于当前 NPU，生成的 metadata 也位于当前 NPU。metadata 和主算子必须使用一致的长度参数；主算子不会只凭 metadata 替代 `cu_seqlens` / `seqused` 入参。

`enable_grad=False` 用于推理，训练使用 True。主算子不会从 `enable_grad` 切换运行模式；切换模式、设备、设备有效核数、形状、布局或有效长度后重新生成 metadata。相同调度配置下可以复用 metadata。真实前向会阻塞回读 metadata 和给定的长度张量，校验本次调用的序列边界与调度；Meta/FakeTensor 只执行静态检查。尺寸、长度、连续性和 GQA 等非法参数抛出 `ValueError`。

直接调用 DSL 函数时，第六个输入 `metadata` 可按位置或关键字传入；省略时抛出 `TypeError`，显式传 None 时抛出 `ValueError: metadata is required`；从 `scale` 开始的属性按关键字传入；`scale` 的签名默认值为 `128 ** -0.5`，显式传 None 时仍使用该缩放值。

## 返回值说明

`flash_kda` 固定返回以下三项，未启用的可选输出为 None：

```python
(out, final_state, chunk_states)
```

令 BNSD/BSND 的 `J=ceil(S/64)`，TND 的容量 `J=ceil(T/64)+B`：

| 参数名 | 参数类型 | 必选/可选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 序列输出，布局与 `v` 相同；metadata 标记的有效序列范围之外，输出值未定义。 | bfloat16 | 与 `v` 相同 |
| `final_state` | Tensor 或 None | 可选 | `output_final_state=True` 时返回每条序列的最终递归状态，训练和推理均生效；不原地修改 `initial_state`。 | float32 | `[B,Nv,Dv,Dk]` |
| `chunk_states` | Tensor 或 None | 可选 | 各 chunk 入口状态；仅推理且 `return_intermediate_states=True` 时返回，训练时为 None。 | float32 | BNSD/BSND：`[B,J,Nv,Dv,Dk]`；TND：`[J,Nv,Dv,Dk]` |

非 None 的张量均位于输入所在设备。状态递推、`initial_state`、`final_state` 和保存的 `chunk_states` 均使用 FP32。

`flash_kda_fwd` 接受与 `flash_kda` 相同的参数，始终按以下顺序返回 16 项。前三项与上表一致。

```python
(out, final_state, chunk_states, Aqk, Akk, w, u, qg, kg, v_new,
 g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_activated)
```

推理时后十三项均为 None。训练时返回 `Aqk`、`Akk`；`disable_recompute=True` 时额外返回 `w/u/qg/kg/v_new/g_cumsum`。Q/K 归一化和 beta 激活的缓存分别由对应的融合开关控制。未启用的槽位保留为 None。

TND 的 chunk 状态按 `token_start//64 + logical_batch` 定位各序列首槽，容量中可能有未使用槽。只消费有效序列和有效 chunk 范围；out 的无效尾区未定义。内部 out 按 64 对齐，公开 out 在非对齐场景可能为不连续视图。

`flash_kda_metadata` 返回当前 NPU 上的一维 int32 Tensor `metadata`，形状为 `[M]`，其中 M 由调度容量决定。将其作为 `flash_kda` 的第六个输入传入。

## 约束说明

- `Dk=Dv=128`，`B>=1`，序列长度大于 0；支持 BNSD、BSND、TND 和非对齐尾块。
- 当前实现支持 GQA，要求 `1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`。
- 参与计算和校验的公开输入必须连续；不校验输入之间的设备一致性。状态、`A_log`、`dt_bias` 使用 FP32，Q/K/V 使用 BF16。
- TND 前缀和首项为 0、末项为 T，严格递增；BNSD/BSND 的有效长度必须在 `[1,S]` 内。
- metadata 必须由本目录的配套模块生成，并与当前布局、形状、长度、运行模式和设备有效核数匹配。应使用 `flash_kda_metadata` 生成；合并前的旧推理 metadata 或其他 ABI 版本的 metadata 不可混用。
- TND 的 chunk 缓存按容量 `ceil(T/64)+B` 分配，仅每条序列实际使用的 chunk 有效。

自定义参数拦截统一抛出 `ValueError`。检查参与计算的输入类型、shape、dtype、设备和连续性；七个开关严格为 bool，`allow_neg_eigval=True` 要求开启 beta sigmoid。`scale` 可为 None 或有限数值，不接受 bool。关闭输出不免除计算输入检查。

Host 检查长度张量的规格和内容，metadata 设备内核也会报告非法长度；主入口读取长度并核对 ABI4 metadata 的当前调用一致性，在启动前检查内部 buffer 的规格及可访问范围。底层设备、分配、编译和执行异常保留原诊断。

## 构建 Torch 扩展

在仓库根目录，激活装有配套 PyTorch 和 torch_npu 的 Python 环境后执行以下命令。需要已安装仓库的基础 Python 包 `cann_ops_transformer`。

```bash
(
    cd torch_extension
    TORCH_EXTENSION_EXPERIMENTAL=TRUE TORCH_EXTENSION_OPS=flash_kda TORCH_EXTENSION_VENDOR=flash_kda \
        python setup.py bdist_wheel
    python -m pip install --no-deps --force-reinstall dist/cann_ops_transformer_flash_kda-*.whl
)
```

该 wheel 包含 PyTorch 注册代码及 `op_dsl` Python 源码。`import cann_ops_transformer` 后可使用以下接口。

- `torch.ops.cann_ops_transformer.flash_kda` 调用三项前向。
- `torch.ops.cann_ops_transformer.flash_kda_fwd` 调用十六项前向。
- `torch.ops.cann_ops_transformer.flash_kda_metadata` 生成调度张量。

两个前向导出共用 staged 和 persistent 内核。设备执行还需要下文的 Native 包。

## 构建 Native 包并验证

加载 CANN 环境，将 shell 变量 `SOC_VERSION` 设为目标设备对应的 `build.sh --soc` 取值，然后在仓库根目录执行以下命令。示例将自定义算子包安装到独立的 OPP 目录。上卡验证需要与目标设备匹配的基础算子包和上文的 Torch 扩展。

```bash
bash build.sh --pkg --experimental --ops=flash_kda --soc="${SOC_VERSION:?}"
WORK_DIR="$(mktemp -d)"
bash build/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path="$WORK_DIR/opp"
source "$WORK_DIR/opp/vendors/custom_transformer/bin/set_env.bash"
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT="$WORK_DIR/hits.json"
```

在同一环境中启动 Python 并调用一次 `flash_kda` 后，查看 `$WORK_DIR/hits.json` 中的 `counts`。
`installed_hit` 大于 0 且 `installed_miss` 为 0（或未出现）表示本次调用命中了已安装的二进制。

## UT 与 TTK

pytest 直接调用 Python DSL；TTK 通过独立适配器在 NPU 上运行精度、异常、泛化和性能用例。红线用例、CPU 对照服务和结果判读见 [TTK 运行说明](../tests/ttk/README.md)。

在仓库根目录运行 `flash_kda` 和 `flash_kda_metadata` 的测试：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q \
    experimental/attention/flash_kda/tests/ut/
```

默认精度容差为 `atol=rtol=5e-3`。`experimental/attention/flash_kda/tests/ut/test_flash_kda.py` 直接调用 Python 入口，检查返回契约、Meta 输出和 NPU 前向精度。

Eager 前向读取并校验长度和 metadata。使用 NPU 图时，先在图外生成 metadata，用同一配置预热，再直接捕获公开入口。捕获使用生成时已校验的 Host 调度信息。

```python
metadata = flash_kda_metadata(**metadata_args)
flash_kda(**inputs, metadata=metadata)
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    outputs = flash_kda(**inputs, metadata=metadata)

inputs["v"].copy_(next_v)
graph.replay()
```

从生成 metadata 到最后一次重放，保留原始 metadata 和长度张量，保持其内容、形状和调度配置固定。改变序列边界时重新生成 metadata 并捕获。
