# FusedQuantQkvProjNormRopeCache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：输入调用方完成MXFP8量化的激活与权重，执行反量化QKV投影（MatMul）、K RmsNorm、Q/K部分RoPE以及K/V Cache更新的融合操作，输出q_out、k_out、v_out、mirror_k_out（可选）、mirror_v_out（可选），并原地更新k_cache、v_cache。面向推理场景，支持普通层和KV mirror源层。
- 本算子目前支持的场景如下表：

  |         场景类型           |            情况概要            |
  |:----------------------|:-----------------------------|
  |<ul><li>普通层（不提供mirror输出）</li><li>激活与权重已完成MXFP8量化</li></ul>|令Q和KV的宽度分别为QW = H<sub>q</sub> * 256、KVW = H<sub>kv</sub> * 256，投影输出顺序为[Q, K, V]。主要计算过程与输出对应关系：<br><ul><li>X、W经Dequant、MatMul->C</li><li>Q经RoPE->q_out</li><li>K经RmsNorm、RoPE->k_out、k_cache</li><li>V直通->v_out、v_cache</li></ul>|
  |<ul><li>mirror源层（提供mirror输出）</li><li>激活与权重已完成MXFP8量化</li></ul>|投影输出顺序为[Q, K, V, mirror K, mirror V]，N = (H<sub>q</sub>+4 * H<sub>kv</sub>) * 256。Q/K/V的计算过程与普通层一致，此外：<br><ul><li>mirror K、mirror V为投影后的原始值，不做RmsNorm或RoPE，直通->mirror_k_out、mirror_v_out</li></ul>|

- 计算公式：

  (1) 反量化与投影（MatMul）：

  下式中，X为量化激活[M, K]，W为量化权重[N, K]，S<sub>A</sub>、S<sub>B</sub>为对应的E8M0缩放因子。Dequant按MX规范每32个元素一组，使用对应E8M0缩放因子还原浮点数值；BF16_RN表示按RN模式舍入到BF16。普通层N = (H<sub>q</sub>+2 * H<sub>kv</sub>) * 256，mirror源层N = (H<sub>q</sub>+4 * H<sub>kv</sub>) * 256。

  $$
  C = \text{BF16\_RN}(\mathrm{Dequant}(X, S_A) @ \mathrm{Dequant}(W, S_B)^{\mathrm{T}})
  $$

  (2) RmsNorm:

  此处$x$和$y$分别表示RmsNorm的输入张量和输出张量，归一化沿最后一维（feature dimension）进行，该计算规则仅作用于K分量（每个head的256维独立计算），Q不做RmsNorm。

  $$
  \mathrm{squareX} = x \cdot x
  $$

  $$
  \mathrm{meanSquareX} = \mathrm{mean}(\mathrm{squareX}, \text{dim} = -1, \text{keepdim} = \text{True})
  $$

  $$
  \mathrm{rms} = \sqrt{\mathrm{meanSquareX} + \epsilon}
  $$

  $$
  y = \frac{x}{\mathrm{rms}} \cdot \gamma
  $$

  (3) RoPE (NeoX rotate-half):

  此处的$y$指代参与RoPE计算的输入：Q分量为投影结果$C$，K分量为RmsNorm输出经BF16_RN舍入的结果。RoPE仅作用于每个head的末尾64维（前32维为$y_1$、后32维为$y_2$），其余维直通；cos、sin由cos_sin表按positions索引的行给出（每行前32个为cos，后32个为sin）。

  $$
  y_1 = y[\ldots, -64:-32]
  $$

  $$
  y_2 = y[\ldots, -32:]
  $$

  $$
  y_{\mathrm{RoPE}} = \mathrm{concat}(-y_2, y_1)
  $$

  $$
  y_{\mathrm{embed}} = y \cdot \cos + y_{\mathrm{RoPE}} \cdot \sin
  $$

  (4) Cache写入（Scatter）与行映射：

  K/V计算结果同时写入dense输出与Cache，Cache写入行由slot_mapping指定，slot为-1时跳过Cache写入。使用chunk模式时（L = tokens_per_rank、C = chunk_tokens_per_rank、a = chunk_token_offset），局部输入行i映射到输出和元数据行：

  $$
  g(i) = \lfloor i/C \rfloor \times L + a + (i \bmod C)
  $$

## 参数说明

接口使用预分配输出和Cache，执行后原地修改，返回`None`。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| `hidden_quant` | 输入 | `[M, K]`。调用方已量化的激活payload，K为正的256倍数。 | FLOAT8_E4M3FN / FLOAT8_E5M2 | ND |
| `hidden_scale` | 输入 | `[M, K//64, 2]`。每32个激活元素对应一个正的E8M0缩放因子，最后一维按两组打包。 | FLOAT8_E8M0 / INT8 / UINT8 | ND |
| `weight_quant` | 输入 | `[N, K]`。调用方已量化的权重payload，可与hidden_quant使用不同的FP8格式。 | FLOAT8_E4M3FN / FLOAT8_E5M2 | ND |
| `weight_scale` | 输入 | `[N, K//64, 2]`。每32个权重元素对应一个E8M0缩放因子。 | FLOAT8_E8M0 / INT8 / UINT8 | ND |
| `k_gamma` | 输入 | `[256]`。K RmsNorm的一维gamma，所有K head共享。 | BFLOAT16 | ND |
| `positions` | 输入 | `[T]`。位置索引；调用方必须初始化所有有效行g(i)，包括跨tile、跨chunk时各tile的首行。 | INT64 | ND |
| `cos_sin` | 输入 | `[P, 64]`。RoPE表，每行前32个值为cos，后32个值为sin。 | FLOAT32 | ND |
| `slot_mapping` | 输入 | `[T]`。Cache槽位；有效行的值须为-1或[0,S)内的唯一非负整数，-1跳过Cache写入。 | INT64 | ND |
| `q_out` | 输入/输出 | `[T, QW]`。预分配Q缓冲区；chunk未选中行保持输入值，读取这些行前须预先初始化。 | BFLOAT16 | ND |
| `k_out` | 输入/输出 | `[T, KVW]`。预分配K缓冲区；chunk未选中行保持输入值，读取这些行前须预先初始化。 | BFLOAT16 | ND |
| `v_out` | 输入/输出 | `[T, KVW]`。预分配V缓冲区；chunk未选中行保持输入值，读取这些行前须预先初始化。 | BFLOAT16 | ND |
| `k_cache` | 输入/输出 | `[S, KVW]`。原地更新的K Cache，未写行保持输入值。 | BFLOAT16 | ND |
| `v_cache` | 输入/输出 | `[S, KVW]`。原地更新的V Cache，未写行保持输入值。 | BFLOAT16 | ND |
| `mirror_k_out` | 可选输入/输出 | `[T, KVW]`。原始mirror-K投影，不做RmsNorm/RoPE。须与mirror_v_out同时提供或同时省略；chunk未选中行保持输入值，读取前须初始化。 | BFLOAT16 | ND |
| `mirror_v_out` | 可选输入/输出 | `[T, KVW]`。原始mirror-V投影，不做RmsNorm/RoPE。须与mirror_k_out同时提供或同时省略；chunk未选中行保持输入值，读取前须初始化。 | BFLOAT16 | ND |
| `epsilon` | 可选属性 | RmsNorm epsilon，必须为有限且大于0的浮点数，默认1e-6。 | FLOAT32 | - |
| `positions_contiguous` | 可选属性 | 默认False。True时每个Q/K处理tile读取自身首个position，并在tile内递增推导；所有有效positions仍须初始化并满足连续性。 | BOOL | - |
| `tokens_per_rank` | 可选属性 | chunk输出步长L；三个chunk参数须同时提供或同时省略。 | INT64 | - |
| `chunk_tokens_per_rank` | 可选属性 | chunk局部行数C；三个chunk参数须同时提供或同时省略。 | INT64 | - |
| `chunk_token_offset` | 可选属性 | chunk偏移a；三个chunk参数须同时提供或同时省略。 | INT64 | - |

## 约束说明

- M >= 0，K为正的256倍数，head维度固定为256。令QW = Hq*256、KVW = Hkv*256，Hq/Hkv为正整数。普通层N = QW+2*KVW，mirror源层N = QW+4*KVW；S、P必须为正数。
- `hidden_scale` 与 `weight_scale` 的INT8/UINT8输入承载E8M0原始编码；使用`view`按位解释，不进行整数数值转换。
- `mirror_k_out` 与 `mirror_v_out` 必须同时提供或同时省略。
- 不使用chunk时T = M。使用chunk时L > 0、C > 0、M%C = 0、0 <= a、a+C <= L，T = (M//C)*L，g(i) = (i//C)*L+a+i%C。只有g(i)对应的dense/mirror行被写入；调用方必须初始化需要保留和读取的其余行。
- 每个有效position位于[0,P)，有效slot为-1或[0,S)内的整数，一次调用中的有效非负slot必须唯一。-1只跳过Cache写入，仍产生dense和mirror输出。未访问的元数据行不参与计算。
- `positions_contiguous=True` 表示局部处理顺序的positions[g(i)]连续递增。每个Q/K tile在g(local_row_start)重新读取首个position，其余行按tile内偏移推导。调用方应初始化全部有效positions，不能只初始化整个张量的第一个元素；跨tile、跨chunk均使用g(i)定位元数据。
- `epsilon` 必须有限且大于0。所有Tensor位于同一NPU、连续存储且 `requires_grad=False`；可写Tensor之间及与输入之间不得存储重叠。
- M = 0时校验元数据后返回，不修改缓冲区。索引数值不在运行时校验；调用方负责范围、唯一性、位置连续性以及跨stream的依赖和Cache写入冲突。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| torch接口 | [test_fused_quant_qkv_proj_norm_rope_cache](examples/test_fused_quant_qkv_proj_norm_rope_cache.py) | 使用 `torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache`。 |

Host按M选择tiling：当M <= 3072时使用baseM = 128、kL1 = 512，否则使用baseM = 256、kL1 = 256。两档均预编译到RUN；任务在4行token tile窗口内沿M遍历，相邻窗口反向遍历head。

### 构建与部署

在仓库根目录执行，安装与目标分支匹配的CANN和Torch/TorchNPU，替换 `/path/to` 为实际路径：

```bash
source /path/to/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=fused_quant_qkv_proj_norm_rope_cache --vendor_name=custom
bash build.sh --torch_extension --experimental \
  --ops=fused_quant_qkv_proj_norm_rope_cache --vendor_name=custom
```

安装当前主机架构对应的RUN和配套wheel，再加载环境：

```bash
bash build_out/cann-ops-transformer-custom_linux-*.run --quiet --install-path=/path/to/opp
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_transformer_custom-*.whl
source /path/to/opp/vendors/custom_transformer/bin/set_env.bash
```

wheel的vendor后缀由仓库打包规则生成，对外调用仍使用标准 `torch.ops.cann_ops_transformer` 命名空间。

### 运行示例

从仓库根目录执行（`PYTHONPATH` 供未安装wheel的源码环境导入基础包，已安装可省略）：

```bash
export PYTHONPATH="$PWD/torch_extension:${PYTHONPATH:-}"
OP=experimental/posembedding/fused_quant_qkv_proj_norm_rope_cache
python3 "$OP/examples/test_fused_quant_qkv_proj_norm_rope_cache.py"
python3 "$OP/examples/test_fused_quant_qkv_proj_norm_rope_cache.py" --mirror --chunk --contiguous
```

[示例源码](examples/test_fused_quant_qkv_proj_norm_rope_cache.py)构造完整的预分配输入并自检Q/K/V、mirror输出与Cache的原地更新，成功后打印PASS。

### 运行测试用例

从仓库根目录执行（测试自行装配导入路径，无需设置`PYTHONPATH`）：

```bash
OP=experimental/posembedding/fused_quant_qkv_proj_norm_rope_cache
CANNBOTDSL_NATIVE_BINARY_MODE=off QKV_TEST_BACKEND=source python3 -m pytest "$OP/tests/pytest" -q
CANNBOTDSL_NATIVE_BINARY_MODE=require QKV_TEST_BACKEND=opp python3 -m pytest "$OP/tests/pytest" -q
```

测试覆盖接口校验、Meta/Fake、普通与mirror层、动态维度、chunk、尾块、Cache更新与重复调用。
