# GroupedMatmulQuant

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2系列产品 | 是 |
| 其他产品 | 否 |

## 功能说明

GroupedMatmulQuant 是面向 MoE 场景的 W4A16 分组矩阵乘算子。激活 `x` 为 FP16 或 BF16，权重以有符号
INT4 保存，并通过 INT32 张量承载。算子先按 K 轴分组反量化权重，再对每个专家对应的 token 区间执行矩阵乘。

对专家 `e`、K 轴量化组 `s`，权重反量化公式为：

```text
weight[e, k, n] = (quantized_weight[e, k, n] + weight_offset[e, s, n])
                  * weight_scale[e, s, n]
s = k // scale_group_size
```

设 `group_list[e]` 为截至专家 `e` 的累计 token 数，`group_list[-1] = M`，则：

```text
start_e = 0                         (e == 0)
          group_list[e - 1]         (e > 0)
y[start_e:group_list[e], :] = x[start_e:group_list[e], :] @ weight[e, :, :]
```

当专家数 `G` 为 1 时可以不传 `group_list`，此时全部 M 个 token 使用同一组权重。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 输入 | 激活，shape 为 `[M, K]`。 | FLOAT16、BFLOAT16 | ND |
| `quantized_weight` | 输入 | INT4 权重的 INT32 载体，shape 为 `[G, K/16, N/16, 16, 2]`。 | INT32 | ND |
| `weight_scale` | 输入 | K 轴分组反量化 scale，shape 为 `[G, K/scale_group_size, N]`。类型与 `x` 一致。 | FLOAT16、BFLOAT16 | ND |
| `weight_offset` | 输入 | K 轴分组反量化 offset，shape 与 `weight_scale` 相同。 | FLOAT16 | ND |
| `group_list` | 可选输入 | 各专家累计 token 数，shape 为 `[G]`；仅 `G=1` 时可省略。 | INT64 | ND |
| `scale_group_size` | 可选属性 | K 轴量化分组大小。当前调用必须显式传入有效值。 | INT64 | - |
| `y` | 输出 | 分组矩阵乘结果，shape 为 `[M, N]`，类型与 `x` 一致。 | FLOAT16、BFLOAT16 | ND |

## INT4 打包布局

逻辑权重 shape 为 `[G, K, N]`，元素值域为 `[-8, 7]`。存储前按以下方式转换：

1. 将 K 和 N 分别切分为 `K1 * K0` 和 `N1 * N0`，其中 `K0=N0=16`。
2. 排列为 `[G, K1, N1, K0, N0]`。
3. N0 方向每 8 个有符号 INT4 按低位在前的顺序打包进一个 INT32。
4. 最终 INT32 载体 shape 为 `[G, K1, N1, 16, 2]`。

测试目录中的 `golden.py` 提供了 `pack_int4_to_int32` 和 `unpack_int32_to_int4` 参考实现。

## 约束说明

- 仅支持 Ascend 910B。
- M、N、K 均应为正数，专家数应满足 `1 <= G <= 256`。
- `x` 必须为二维张量，输出为二维张量。
- K 和 N 必须按 INT4 权重布局要求对齐到 16。
- 算子定义中 `scale_group_size` 是默认值为 0 的可选属性，但当前实现不能使用该默认值。调用方必须显式传入正数；
  该值必须按 32 对齐，并且 K 必须能够被 `scale_group_size` 整除。
- `weight_scale` 的数据类型必须与 `x` 一致；`weight_offset` 固定为 FLOAT16。
- `group_list` 使用累计计数语义，正常输入应单调非降、取值位于 `[0, M]`，最后一个值应为 M。
- 当前实现内部最多为 256 个专家预留状态。
- 当前 Host 侧不会验证 `group_list` 的数据内容，调用方必须保证其满足上述语义。

## 调用说明

Torch 适配函数定义在 `grouped_matmul_quant_torch_adpt.h`，接口如下：

```cpp
at::Tensor grouped_matmul_quant(
    const at::Tensor &x,
    const at::Tensor &quantized_weight,
    const at::Tensor &weight_scale,
    const at::Tensor &weight_offset,
    const c10::optional<at::Tensor> &group_list,
    int64_t scale_group_size);
```

适配层通过 `aclnnGroupedMatmulQuant` 调用算子并返回 `[M, N]` 输出。具体 PyTorch 命名空间由集成工程的注册方式决定。

## 测试说明

Host/Kernel UT：

```bash
bash build.sh -u --experimental --ops=grouped_matmul_quant --soc=ascend910b
```

Ascend 910B 真机精度测试：

```bash
bash experimental/gmm/grouped_matmul_quant/run_precision_test.sh
```

脚本会根据自身路径定位仓库，日志默认保存在 `build_out/grouped_matmul_quant_precision_logs`。可以通过以下环境变量适配服务器环境：

- `CANN_ENV_SCRIPT`：CANN 环境脚本路径。
- `PYTHON_BIN`：执行测试使用的 Python 解释器。
- `OPP_INSTALL_ROOT`：experimental 自定义算子包的安装目录，当前用户必须具有写权限。

例如：

```bash
CANN_ENV_SCRIPT=/opt/Ascend/ascend-toolkit/set_env.sh \
PYTHON_BIN=/path/to/python3 \
OPP_INSTALL_ROOT=/opt/Ascend/opp \
bash experimental/gmm/grouped_matmul_quant/run_precision_test.sh
```

未设置 `OPP_INSTALL_ROOT` 时，脚本优先使用 `ASCEND_OPP_PATH`，该变量为空时使用 `${ASCEND_HOME_PATH}/opp`。

上述脚本会从算子包构建开始执行完整流程。如果需要手动分步执行：

```bash
cd experimental/npu_ops_transformer_ext
NPU_OPS_TRANSFORMER_EXT_OPS=grouped_matmul_quant \
  python3 -m pip install --no-build-isolation -e .

cd experimental/gmm/grouped_matmul_quant/tests
python3 -m pytest -sv test_grouped_matmul_quant.py
```

运行测试前，需要先安装包含 GroupedMatmulQuant 的 experimental 自定义算子包；随后构建上述 PyTorch 扩展。
默认注册路径为 `torch.ops.npu_ops_transformer_ext.grouped_matmul_quant`。

如果集成工程注册的 PyTorch 路径不是测试中的默认候选路径，可通过环境变量指定。路径相对于 `torch`，例如
`ops.vllm.grouped_matmul_quant`：

```bash
export GROUPED_MATMUL_QUANT_MODULES=vllm_ascend
export GROUPED_MATMUL_QUANT_OP=ops.vllm.grouped_matmul_quant
pytest -sv test_grouped_matmul_quant.py
```

精度测试使用固定随机种子，并按照生态算子浮点输出混合容差规则校验：逐元素满足
`|actual-golden| <= atol + rtol * |golden|` 的比例不低于 0.99，同时最大绝对误差不能超过对应数据类型上限。

| 输出类型 | `atol` | `rtol` | 最大绝对误差上限 |
| :--- | :---: | :---: | :---: |
| FLOAT16 | `2^-9` | `2^-9` | `0.1` |
| BFLOAT16 | `2^-6` | `2^-6` | `1.0` |
