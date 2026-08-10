# moe\_re\_routing

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                                      |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  |    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|<term>Atlas 推理系列产品</term>    |     ×    |
|<term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- API功能：MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。相较于`torch\_npu.npu\_moe\_re\_routing`，新增可选输入`expert\_topk\_weight`和可选输出`permute\_topk\_weight`，支持对topkWeight按专家顺序进行重排，使得`permute\_topk\_weight`与`permute\_tokens`一一对应。`expert\_topk\_weight`与`permute\_topk\_weight`必须同时传入或同时不传入。

- 计算公式：

    $$SrcOffset = \sum_{i=0}^{cur\_rank} \left( \sum_{j=0}^{cur\_expert} expert\_token\_num\_per\_rank(i,j) \right)$$

    $$DstOffset = \sum_{j=0}^{cur\_expert} \left( \sum_{i=0}^{cur\_rank} expert\_token\_num\_per\_rank(i,j) \right)$$

    $$permute\_tokens[DstOffset + k] = tokens[SrcOffset + k]$$

    $$permute\_per\_token\_scales[DstOffset + k] = per\_token\_scales[SrcOffset + k]$$

    $$permute\_topk\_weight[DstOffset + k] = expert\_topk\_weight[SrcOffset + k]$$

    - SrcOffset指当前需要移动的token源偏移，根据输入`expert_token_num_per_rank`的值进行计算。
    - DstOffset指当前需要移动的token目的偏移。
    - cur\_rank是`expert_token_num_per_rank`的纵轴索引，表示该token原本在的卡。
    - cur\_expert是`expert_token_num_per_rank`的横轴索引，表示该token由卡上专家cur\_expert计算。
    - k表示当前expert下第k个token的偏移（0 ≤ k < currTokenNum）。
    - topkWeight与token一一对应，搬运偏移量与token完全一致，直接复用token的SrcOffset和DstOffset。

## 函数原型

```python
def moe_re_routing(
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
> - A：表示token个数，取值要求Sum\(expert\_token\_num\_per\_rank\)=A。
> - H：表示token长度，取值要求0<H<16384。
> - N：表示卡数，取值无限制。
> - E：表示卡上的专家数，取值无限制。

- **tokens** (`Tensor`)：必选参数，表示待重新排布的token。要求为2维，shape为\[A, H\]，数据格式要求为$ND$。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`、`int8`。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型额外支持`float8_e5m2`、`float8_e4m3fn`、`hifloat8`、`float4_e2m1`、`float4_e1m2`。

- **expert\_token\_num\_per\_rank** (`Tensor`)：必选参数，二维矩阵，矩阵中元素\[i, j\]表示当前卡上从卡i获取到的专家j处理的token数。要求为2维，shape为\[N, E\]，数据类型支持`int32`、`int64`，数据格式要求为$ND$。取值必须大于0。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **per\_token\_scales** (`Tensor`)：可选参数，默认为None，表示每个token对应的scale，需要随token同样进行重新排布。数据格式要求为$ND$。不输入表示不使用scale，输出`permute\_per\_token\_scales`中的值无意义。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求为1维，shape为\[A\]，数据类型支持`float32`。
    - <term>Ascend 950PR/Ascend 950DT</term>：支持1维shape \[A\]（数据类型`float32`）、2维shape \[A, S\]（数据类型`float32`）、3维shape \[A, K/64, 2\]（数据类型`float8_e8m0`，用于FP8量化token）。

- **expert\_topk\_weight** (`Tensor`)：可选参数，默认为None，表示每个token对应的topk权重值，需要随token同样进行重新排布。
    - 可选输入，不输入表示不输出`permute\_topk\_weight`。
    - 输入要求为2维，shape为\[A, 1\]，数据类型仅支持`float32`，数据格式要求为$ND$。
    - 与`permute\_topk\_weight`联动：必须同时传入或同时不传入。

- **expert\_token\_num\_type** (`int`)：可选参数，默认值为1，表示输出`expert_token_num`的模式。0为cumsum模式，1为count模式。当前只支持为1。

- **idx\_type** (`int`)：可选参数，默认值为0，表示输出`permute_token_idx`的索引类型。0为gather索引，1为scatter索引。scatter索引仅<term>Ascend 950PR/Ascend 950DT</term>支持。

## 返回值说明

- **permute\_tokens** (`Tensor`)：表示重新排布后的token。要求为2维，shape为\[A, H\]，数据类型同`tokens`，数据格式要求为$ND$。

- **permute\_per\_token\_scales** (`Tensor`)：表示重新排布后的`per_token_scales`，数据格式要求为$ND$。
    - `per_token_scales`输入时，shape和数据类型与`per_token_scales`一致。
    - `per_token_scales`未输入时，shape为\[A\]，数据类型为`float32`，该输出无意义。

- **permute\_token\_idx** (`Tensor`)：表示每个token在原排布方式的索引。要求为1维，shape为\[A\]，数据类型为`int32`，数据格式要求为$ND$。

- **expert\_token\_num** (`Tensor`)：表示每个专家处理的token数。要求为1维，shape为\[E\]，数据类型同`expert_token_num_per_rank`，数据格式要求为$ND$。

- **permute\_topk\_weight** (`Tensor`)：表示重新排布后的`expert_topk_weight`，与`permute\_tokens`一一对应，数据类型仅支持`float32`，数据格式要求为$ND$。
    - 可选输出，`expert_topk_weight`输入时必须同时输出；`expert_topk_weight`未输入时输出为空tensor（shape为(0,)）。
    - `expert_topk_weight`输入时，shape为\[A, 1\]。

## 约束说明

1. 该接口支持推理场景下使用。
2. 该接口支持图模式。
3. `tokens`和`expert_token_num_per_rank`必须是2D张量。
4. `expert_topk_weight`和`permute_topk_weight`必须同时传入或同时不传入（联动约束）。
5. `expert_topk_weight`仅支持`float32` dtype，shape必须为\[A, 1\]，其中A必须等于tokens的第0维。
6. `expert_topk_weight`/`permute_topk_weight`仅<term>Ascend 950PR/Ascend 950DT</term>支持，其他产品不支持该参数。
7. `expert_token_num_type`当前只支持为1（count模式）。
8. `idx_type`为1（scatter索引）仅<term>Ascend 950PR/Ascend 950DT</term>支持，其他产品仅支持0（gather索引）。
9. <term>Ascend 950PR/Ascend 950DT</term>上`per_token_scales`支持`float8_e8m0`数据类型和3维shape \[A, K/64, 2\]，其他产品仅支持`float32`数据类型和1维shape \[A\]。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import moe_re_routing

    tokens_num = 16384
    tokens_length = 7168
    rank_num = 16
    expert_num = 16

    tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8).npu()
    expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32).npu()
    per_token_scales = torch.randn(tokens_num, dtype=torch.float32).npu()

    # 不传入expert_topk_weight，permute_topk_weight为空tensor
    permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
        moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales=per_token_scales,
                       expert_token_num_type=1, idx_type=0)

    # 传入expert_topk_weight，permute_topk_weight为有效数据
    expert_topk_weight = torch.randn(tokens_num, 1, dtype=torch.float32).npu()
    permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
        moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales=per_token_scales,
                       expert_topk_weight=expert_topk_weight, expert_token_num_type=1, idx_type=0)
    ```

- 图模式调用

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    from cann_ops_transformer.ops import moe_re_routing

    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MoeReRoutingModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, tokens, expert_token_num_per_rank, *,
                     per_token_scales=None, expert_topk_weight=None,
                     expert_token_num_type=1, idx_type=0):
            return moe_re_routing(tokens, expert_token_num_per_rank,
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
