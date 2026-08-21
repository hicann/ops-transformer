# moe_re_routing

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。相较于`torch_npu.npu_moe_re_routing`，新增可选输入`expert_topk_weight`和可选输出`permute_topk_weight`，支持对topkWeight按专家顺序进行重排，使得`permute_topk_weight`与`permute_tokens`一一对应。`expert_topk_weight`与`permute_topk_weight`必须同时传入或同时不传入。

- 计算公式：

  $$SrcOffset = \sum_{i=0}^{cur_rank} \left( \sum_{j=0}^{cur_expert} expert_token_num_per_rank(i,j) \right)$$

  $$DstOffset = \sum_{j=0}^{cur_expert} \left( \sum_{i=0}^{cur_rank} expert_token_num_per_rank(i,j) \right)$$

  $$permute_tokens[DstOffset + k] = tokens[SrcOffset + k]$$

  $$permute_per_token_scales[DstOffset + k] = per_token_scales[SrcOffset + k]$$

  $$permute_topk_weight[DstOffset + k] = expert_topk_weight[SrcOffset + k]$$

  - SrcOffset指当前需要移动的token源偏移，根据输入`expert_token_num_per_rank`的值进行计算。
  - DstOffset指当前需要移动的token目的偏移。
  - cur_rank是`expert_token_num_per_rank`的纵轴索引，表示该token原本在的卡。
  - cur_expert是`expert_token_num_per_rank`的横轴索引，表示该token由卡上专家cur_expert计算。
  - k表示当前expert下第k个token的偏移（0 ≤ k < currTokenNum）。
  - topkWeight与token一一对应，搬运偏移量与token完全一致，直接复用token的SrcOffset和DstOffset。

## 函数原型

```python
cann_ops_transformer.moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    *,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_topk_weight: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## 参数说明

> [!NOTE]
> Tensor中shape使用的变量说明：
>
> - A：表示token个数，取值要求Sum(expert_token_num_per_rank)=A。
> - H：表示token长度，取值要求0 < H < 16384。
> - N：表示卡数，取值无限制。
> - E：表示卡上的专家数，取值无限制。

|  参数名   | 参数类型 |  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  tokens  |  Tensor |  必选  |  表示待重新排布的token。要求为2维，shape为[A, H]。  |   float16、bfloat16、int8、float8_e5m2、float8_e4m3fn、hifloat8、float4_e2m1、float4_e1m2   |  2  |
|  expert_token_num_per_rank  |  Tensor |  必选  |  二维矩阵，矩阵中元素[i, j]表示当前卡上从卡i获取到的专家j处理的token数。要求为2维，shape为[N, E]，取值必须大于0。  |  int32、int64   | 2  |
|  per_token_scales  |  Tensor |  可选  |  表示每个token对应的scale，需要随token同样进行重新排布。默认值为None，不输入表示不使用scale，输出`permute_per_token_scales`中的值无意义。支持1维shape [A]（`float32`）、2维shape [A, S]（`float32`）、3维shape [A, K/64, 2]（`float8_e8m0`，用于FP8量化token）。  |  float32、float8_e8m0   | 1-3  |
|  expert_topk_weight  |  Tensor |  可选  |  表示每个token对应的topk权重值，需要随token同样进行重新排布。默认值为None，不输入表示不输出`permute_topk_weight`。输入要求为2维，shape为[A, 1]，与`permute_topk_weight`联动：必须同时传入或同时不传入。  |  float32   | 2  |
|  expert_token_num_type  |  int |  可选  |  表示输出`expert_token_num`的模式。默认值为1，0为cumsum模式，1为count模式。当前只支持为1。  |  -  | - |
|  idx_type  |  int |  可选  |  表示输出`permute_token_idx`的索引类型。默认值为0，0为gather索引，1为scatter索引。  |  -  | - |

## 返回值说明

|  参数名   | 参数类型 |  可选/必选 |    描述    |     数据类型    |   维度(shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  permute_tokens  |  Tensor |  必选  |  表示重新排布后的token。shape为[A, H]，数据类型同`tokens`。  |  与tokens一致  | 2  |
|  permute_per_token_scales  |  Tensor |  必选  |  表示重新排布后的`per_token_scales`。`per_token_scales`输入时，shape和数据类型与`per_token_scales`一致；`per_token_scales`未输入时，shape为[A]，数据类型为`float32`，该输出无意义。  |  float32、float8_e8m0  | 1-3  |
|  permute_token_idx  |  Tensor |  必选  |  表示每个token在原排布方式的索引。shape为[A]。  |  int32   | 1  |
|  expert_token_num  |  Tensor |  必选  |  表示每个专家处理的token数。shape为[E]，数据类型同`expert_token_num_per_rank`。  |  int32、int64   | 1  |
|  permute_topk_weight  |  Tensor |  可选  |  表示重新排布后的`expert_topk_weight`，与`permute_tokens`一一对应。`expert_topk_weight`输入时必须同时输出，shape为[A, 1]；`expert_topk_weight`未输入时输出为空tensor（shape为(0,)）。  |  float32   | 2  |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- `expert_token_num_type`当前只支持为1（count模式）。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  tokens_num = 16384
  tokens_length = 7168
  rank_num = 16
  expert_num = 16

  tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8).npu()
  expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32).npu()
  per_token_scales = torch.randn(tokens_num, dtype=torch.float32).npu()

  # 不传入expert_topk_weight，permute_topk_weight为空tensor
  permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
      cann_ops_transformer.moe_re_routing(tokens, expert_token_num_per_rank,
                                           per_token_scales=per_token_scales,
                                           expert_token_num_type=1, idx_type=0)

  # 传入expert_topk_weight，permute_topk_weight为有效数据
  expert_topk_weight = torch.randn(tokens_num, 1, dtype=torch.float32).npu()
  permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
      cann_ops_transformer.moe_re_routing(tokens, expert_token_num_per_rank,
                                           per_token_scales=per_token_scales,
                                           expert_topk_weight=expert_topk_weight,
                                           expert_token_num_type=1, idx_type=0)
  ```

- TorchAir图模式调用：

  ```python
  import torch
  import torch.nn as nn
  import torch_npu
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig
  import cann_ops_transformer

  config = CompilerConfig()
  config.experimental_config.keep_inference_input_mutations = True
  npu_backend = tng.get_npu_backend(compiler_config=config)

  class MoeReRoutingModel(nn.Module):
      def __init__(self):
          super().__init__()

      def forward(self, tokens, expert_token_num_per_rank, *,
                   per_token_scales=None, expert_topk_weight=None,
                   expert_token_num_type=1, idx_type=0):
          return cann_ops_transformer.moe_re_routing(tokens, expert_token_num_per_rank,
                                                      per_token_scales=per_token_scales,
                                                      expert_topk_weight=expert_topk_weight,
                                                      expert_token_num_type=expert_token_num_type,
                                                      idx_type=idx_type)

  def main():
      tokens_num = 16384
      tokens_length = 7168
      rank_num = 16
      expert_num = 16

      tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8).npu()
      expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32).npu()
      per_token_scales = torch.randn(tokens_num, dtype=torch.float32).npu()
      expert_topk_weight = torch.randn(tokens_num, 1, dtype=torch.float32).npu()

      model = MoeReRoutingModel().npu()
      model = torch.compile(model, backend=npu_backend, dynamic=False)
      permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
          model(tokens, expert_token_num_per_rank, per_token_scales=per_token_scales,
                expert_topk_weight=expert_topk_weight, expert_token_num_type=1, idx_type=0)

  if __name__ == '__main__':
      main()
  ```
