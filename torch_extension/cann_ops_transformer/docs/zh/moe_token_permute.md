# moe_token_permute

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

`moe_token_permute`是基于`torch_npu`的`cann_ops_transformer`扩展接口。该接口根据`indices`将输入`tokens`扩展，并按照专家索引排序，返回排序后的token及其与原始token的映射关系。

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：该接口还支持在permute过程中将输出量化为MXFP8或MXFP4，并返回对应的per-token分块scale。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持量化模式，按非量化路径执行。
<!-- end id8 -->

## 函数原型

```python
cann_ops_transformer.ops.moe_token_permute(
    tokens,
    indices,
    num_out_tokens=None,
    padded_mode=False,
    quant_mode=-1,
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `tokens` | Tensor | 必选 | 输入token特征，记shape为`[N, H]`。量化模式仅支持float16和bfloat16。 | 非量化：float16、bfloat16、float32；Ascend 950非量化额外支持int8；量化：float16、bfloat16 | `[N, H]` |
| `indices` | Tensor | 必选 | token对应的专家索引。可以是一维索引，也可以表示每个token对应K个专家。元素个数记为`F`。 | int32、int64 | `[N]`或`[N, K]` |
| `num_out_tokens` | int | 可选 | 控制`permuted_tokens`的有效输出行数。`None`或`0`表示保留全部；正数表示最多保留指定行数；负数表示从完整结果尾部删除对应行数。默认值为`None`。 | int64 | - |
| `padded_mode` | bool | 可选 | 是否使用padded模式。当前仅支持`False`。默认值为`False`。 | bool | - |
| `quant_mode` | int | 可选 | 量化模式：`-1`表示不量化；`2`表示MXFP8 E5M2；`3`表示MXFP8 E4M3FN；`9`表示MXFP4 E2M1，默认值为`-1`。 | int64 | - |

<!-- npu="A3,910b" id9 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`quant_mode`仅支持-1。
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR/Ascend 950DT</term>：`quant_mode`支持-1/2/3/9。取值为9时，隐藏维`H`必须为偶数。取值2/3/9时，不支持autograd。
<!-- end id10 -->

## 返回值说明

接口返回三个Tensor：

```text
permuted_tokens, sorted_indices, expanded_scale
```

定义：

```text
F = indices.numel()
H = tokens.shape[1]

num_out_tokens is None或num_out_tokens == 0：M = F
num_out_tokens > 0：                              M = min(num_out_tokens, F)
num_out_tokens < 0：                              M = max(F + num_out_tokens, 0)
```

各量化模式的输出如下：

| `quant_mode` | `permuted_tokens` | `sorted_indices` | `expanded_scale` |
| --- | --- | --- | --- |
| `-1` | shape为`[M, H]`，dtype与`tokens`相同 | shape为`[F]`，dtype为int32 | shape为`[0]`的空Tensor，dtype为float32 |
| `2` | shape为`[M, H]`，dtype为float8_e5m2 | shape为`[F]`，dtype为int32 | shape为`[M, AlignUp(CeilDiv(H, 32), 2)]`，dtype为float8_e8m0 |
| `3` | shape为`[M, H]`，dtype为float8_e4m3fn | shape为`[F]`，dtype为int32 | shape为`[M, AlignUp(CeilDiv(H, 32), 2)]`，dtype为float8_e8m0 |
| `9` | PyTorch物理shape为`[M, H / 2]`，dtype为uint8 | shape为`[F]`，dtype为int32 | PyTorch物理shape为`[M, CeilDiv(H, 64), 2]`，dtype为uint8 |

其中：

```text
CeilDiv(a, b)  = (a + b - 1) // b
AlignUp(a, b)  = CeilDiv(a, b) * b
```

- `permuted_tokens`：根据`indices`扩展并按专家索引排序后的token；仅该输出和`expanded_scale`的第一维受`num_out_tokens`影响。
- `sorted_indices`：`permuted_tokens`与原始`tokens`的行映射关系，长度始终为`indices.numel()`。
- `expanded_scale`：量化输出对应的per-token分块scale；非量化模式返回空Tensor。

> [!NOTE]
>
> MXFP4类型当前通过uint8 Tensor承载物理存储。调用aclnn接口时，`permuted_tokens`会被解释为`ACL_FLOAT4_E2M1`，`expanded_scale`会被解释为`ACL_FLOAT8_e8m0`。因此表中的uint8表示PyTorch侧的物理存储类型，不表示量化数据的逻辑类型。

## 约束说明

- `indices`元素个数必须小于`16777215`，元素值必须大于等于`0`且小于`16777215`。
- MXFP8的每32个量化值共享一个E8M0 scale，scale数量向2对齐。
- MXFP4的两个4 bit值打包在一个uint8中，因此`permuted_tokens`的PyTorch物理隐藏维为`H / 2`。

## 确定性计算

默认支持确定性计算。

## 调用示例

### 非量化调用

```python
import torch
import torch_npu
from cann_ops_transformer.ops import moe_token_permute

tokens = torch.randn(4, 128, dtype=torch.float16, device="npu")
indices = torch.tensor(
    [[1, 0], [0, 1], [1, 0], [0, 1]],
    dtype=torch.int32,
    device="npu",
)

permuted_tokens, sorted_indices, expanded_scale = moe_token_permute(
    tokens,
    indices,
    num_out_tokens=6,
)

print(permuted_tokens.shape)  # torch.Size([6, 128])
print(sorted_indices.shape)   # torch.Size([8])
print(expanded_scale.shape)   # torch.Size([0])
```

<!-- npu="950" id11 -->
### Ascend 950 MXFP8调用

```python
permuted_tokens, sorted_indices, expanded_scale = moe_token_permute(
    tokens,
    indices,
    quant_mode=2,
)

print(permuted_tokens.shape)  # torch.Size([8, 128])
print(expanded_scale.shape)   # torch.Size([8, 4])
```
<!-- end id11 -->
