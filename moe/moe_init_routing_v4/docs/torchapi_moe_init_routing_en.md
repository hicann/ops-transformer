# moe_init_routing

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:28:57.372Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Routing computation of MoE. Performs routing processing based on the computation result of moe_gating_top_k_softmax, and supports the non-quantized, static quantization, and dynamic quantization modes. Supports the input `topk_weight` and the output `expanded_topk_weight`, and supports rearranging topkWeight according to the sorted indices so that `expanded_topk_weight` corresponds one-to-one with `expanded_x`. `topk_weight` and `expanded_topk_weight` must be passed in together or not passed in together.

- Formula:

  1.Sort the input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding index sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.Perform position mapping based on sortedRowIdx to obtain expandedRowIdxOut:
    - When rowIdxType is 1, output scatter indices

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - When rowIdxType is 0, output gather indices

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.Compute the histogram result for each expert in sortedExpertIdx to obtain expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.If quantMode is not -1, compute the quantization result:
     - Static quantization

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$

    - Dynamic quant:
        - If scale is not passed in:

            $$
            dynamicQuantScaleOutOptional = row_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - If scale is passed in:

            $$
            dynamicQuantScaleOutOptional = row_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - When quantMode is 13, dynamic quantization uses a symmetric quantization range of [-8, 7]. The denominator in the scale calculation is 7. Every two INT4 values in the quantization result are packed into 1 byte along the H dimension.

  5.If the active expert range is the full expert range, token index movement follows the Scatter index; otherwise, token index movement follows the Gather index. When dropPadMode is 1, the number of tokens to be processed by each expert is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and the insufficient ones are padded with 0. expandedXOut is obtained:
    - Non-quantized scenario
      - Index movement by Scatter index

        $$
        expandedXOut[i]=x[scatterRowIdx[i] // K]
        $$

      - Index movement by Gather index

        $$
        expandedXOut[gatherRowIdx[i]]=x[i // K]
        $$

    - Quantization scenario
      - Index movement by Scatter index

        $$
        expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
        $$

      - Index movement by Gather index

        $$
        expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
        $$

  6.If the input topkWeight is passed in, rearrange topkWeight according to the sorted indices, so that expandedTopkWeightOut and expandedXOut are in one-to-one correspondence:
    - Dropless scenario (dropPadMode=0)

      $$
      expandedTopkWeightOut[i] = topkWeight[sortedRowIdx[i]], \quad 0 \le i < effectiveNum
      $$

      where effectiveNum is:

      $$
      effectiveNum = availableIdxNum
      $$

    - DropPad scenario (dropPadMode=1)

      $$
      expandedTopkWeightOut[dstIdx] = topkWeight[globalSortIdx], \quad dstIdx \ne -1
      $$

      where globalSortIdx is the position of expertIdx in the flatten layout (that is, globalSortIdx = rowIdx * K + colIdx, corresponding to topkWeight[rowIdx, colIdx]), dstIdx is the destination index at the corresponding position in expandedRowIdxOut, and the unallocated expert capacity positions in expandedTopkWeightOut are filled with 0.

  7.expandedRowIdxOut has availableIdxNum valid elements, which is calculated as the number of elements in expertIdx within the range of activeExpertRangeOptional

    $$
    availableIdxNum = |\{x\in expertIdx| expert_start \le x<expert_end \ \}|
    $$

## Function Prototype

```python
cann_ops_transformer.moe_init_routing(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    topk_weight: Optional[torch.Tensor] = None,
    active_num: Union[int, torch.SymInt] = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
    x_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## Mandatory

|  Name   | Type |  Type |    Description    |     Data Type    |   Dimension (Shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  x  |  Tensor |  Required  |  Input of MoE, that is, the token feature input, corresponding to x in the formula. The shape is (NUM_ROWS, H). Supported data types are float16, bfloat16, float32, int8, hifloat8, float8_e5m2, float8_e4m3fn, and float4_e2m1 (types that cannot be expressed by native PyTorch dtypes are stored as native dtypes such as `torch.uint8` and need to be specified through the `x_dtype` parameter).  |   float16, bfloat16, float32, int8, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1   |  2  |
|  expert_idx  |  Tensor |  Required  |  K experts processed for each row of the features output by moe_gating_top_k_softmax, corresponding to expertIdx in the formula. The shape is (NUM_ROWS, K), and the expert ID cannot exceed the number of experts.  |  int32   | 2  |
|  scale  |  Tensor |  Optional  |  Parameter used to calculate the quantization result, corresponding to scaleOptional in the formula. The default value is None. If not passed in, scale is not used during computation, and the values in the output `expanded_scale` are meaningless. For input requirements in different scenarios, see the description below.  |  float32, float8_e8m0   | 1-3  |
|  offset  |  Tensor |  Optional  |  Offset value used to calculate the quantization result, corresponding to offsetOptional in the formula. The default value is None.  |  float32   | 1  |
|  topk_weight  |  Tensor |  Optional  |  Routing weight of the topk experts, used for reordering by the sorted index to have a one-to-one correspondence with `expanded_x`. The default value is None. If not passed in, `expanded_topk_weight` is not output. The shape is (NUM_ROWS, K). It is linked with `expanded_topk_weight`: they must be passed in together or not passed in together.  |  float32   | 2  |
|  active_num  |  Union[int, torch.SymInt] |  Optional  |  Maximum total number of rows to process, currently fixed at NUM_ROWS*K. The actual number of valid rows depends on routing rules such as expert range filtering. The default value is -1, which means processing with a row limit of NUM_ROWS*K. When explicitly specifying the row limit, only NUM_ROWS*K is supported. Active truncation is currently not supported. Both `int` and `torch.SymInt` are supported. Passing in `torch.SymInt` (such as `x.size(0) * k`) in graph mode prevents it from being fixed as a constant, enabling reuse of the dynamic shape compilation cache. In eager mode, passing in `int` is sufficient. The upper-layer calling script does not need to perceive the type difference, and the same code can run normally in both modes.  |  int  | - |
|  expert_capacity  |  int |  Optional  |  Number of tokens that each expert can process. The default value is -1. Input validation requires the value to be greater than 0 and less than NUM_ROWS. In the Dropless scenario, only its value is validated and this parameter is not used. In the DropPad scenario, the value range is (0, NUM_ROWS].  |  -  | - |
|  expert_num  |  int |  Optional  |  Number of experts. The default value is -1, which must be greater than 0. When `expert_tokens_num_type` is the key_value mode, the value range is [1, 5120]. In other modes, the value range is [1, 10240].  |  -  | - |
|  drop_pad_mode  |  int |  Optional  |  Whether it is the drop_pad scenario. The default value is 0. 0 indicates the Dropless scenario, in which `expert_capacity` is not validated. 1 indicates the Drop_Pad scenario.  |  -  | - |
|  expert_tokens_num_type  |  int |  Optional  |  Different modes of the histogram. The default value is 0, which takes the value 0, 1, or 2. 0 indicates the cumsum mode. 1 indicates the count mode. 2 indicates the key_value mode.  |  -  | - |
|  expert_tokens_num_flag  |  bool |  Optional  |  Whether to output `expert_token_cumsum_or_count`. The default value is False, and it only supports taking the value true.  |  -  | - |
|  quant_mode  |  int |  Optional  |  Quantization mode. The default value is -1. For supported values, see the description below.  |  -  | - |
|  active_expert_range  |  List[int] |  Optional  |  Range of active experts. The default value is None. The length is 2, and the values in the array are [expert_start, expert_end], which is left-closed and right-open. The values are required to be greater than or equal to 0, and expert_end cannot be greater than `expert_num`. In the Drop/Pad scenario, expert_start equals 0 and expert_end equals `expert_num`. If None is passed in, the range of active experts is regarded as 0 to `expert_num`.  |  -  | - |
|  row_idx_type  |  int |  Optional  |  Index type used by the output `expanded_row_idx`. The default value is 0, which takes the value 0 or 1. 0 indicates the gather-type index. 1 indicates the scatter-type index. In the DropPad scenario, this parameter only supports 0.  |  -  | - |
|  x_dtype  |  int |  Optional  |  Used to specify the non-native data type of `x` (types that cannot be expressed by native PyTorch dtypes and are stored as native dtypes such as `torch.uint8`). The value takes one of the dtype enumerations of `torch_npu`. The default value is None. All supported enumeration values are: `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.int4`. For value constraints under different quant_mode settings, see the description below.  |  -  | - |

**Input requirements for scale in different scenarios:**

- In the non-quantized scenario, if the input is passed in, it is required to be a 1-dimensional tensor with shape (NUM_ROWS,).
- In the non-quantized scenario where the data type of `x` is `float8_e5m2`, `float8_e4m3fn`, or `float4_e2m1` (specified by `x_dtype`), it is required to be a 3-dimensional tensor with shape (NUM_ROWS, CeilDiv(H, 64), 2) and data type `float8_e8m0`.
- In the static quantization scenario, it is a required input and must be a 1D tensor with shape (1,).
- In the dynamic quantization scenario, if the input is passed in, it is required to be a 2-dimensional tensor with shape (expert_end-expert_start, H) or (1, H).
- In the INT8 dynamic quantization scenario (quantMode is 1), it is an optional input. If the input is passed in, it is required to be a 2D tensor with shape (expert_end-expert_start, H). In the INT4 dynamic quantization scenario (quantMode is 13), it is an optional input. If the input is passed in, it is required to have shape (1, H), indicating the smooth scale broadcast along the H dimension.
- In the MXFP8 quantization scenario (quantMode is 2 or 3), it is not passed in.

**Supported values of quant_mode:**

-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, and 17. The meanings are as follows:

- -1: Indicates the non-quantized scenario.
- 0: indicates the static quantization scenario.
- 1: indicates the dynamic quantization scenario, where `expanded_x` is quantized to INT8.
- 2: indicates the MXFP8 quantization scenario, where `expanded_x` is quantized to float8_e5m2.
- 3: indicates the MXFP8 quantization scenario, where `expanded_x` is quantized to float8_e4m3fn.
- 4: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where `expanded_x` is quantized to float8_e5m2.
- 5: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where `expanded_x` is quantized to float8_e4m3fn.
- 6: indicates the HIF8 direct conversion quantization scenario, where `expanded_x` is quantized to hifloat8.
- 7: indicates the HIF8 PERTENSOR quantization scenario.
- 8: indicates the HIF8 PERTOKEN quantization scenario.
- 9: indicates the MXFP4 quantization scenario, where `expanded_x` is quantized to float4_e2m1.
- 11: Indicates the FP8 PerBlock quantization scenario (BlockSize=128), where `expanded_x` is quantized to float8_e5m2.
- 12: Indicates the FP8 PerBlock quantization scenario (BlockSize=128), where `expanded_x` is quantized to float8_e4m3fn.
- 13: Indicates the INT4 dynamic quantization scenario, where `expanded_x` is quantized to INT4.
- 14: Indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where `expanded_x` is quantized to float8_e5m2.
- 15: Indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where `expanded_x` is quantized to float8_e4m3fn.
- 16: Indicates the MXFP8 RoundScale+Amax quantization scenario, where `expanded_x` is quantized to float8_e5m2.
- 17: Indicates the MXFP8 RoundScale+Amax quantization scenario, where `expanded_x` is quantized to float8_e4m3fn.

**Value constraints of `x_dtype` in different scenarios:**

- When `quant_mode` is -1 (non-quantized pass-through), `x_dtype` supports `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`. `torch.float8_e5m2` and `torch.float8_e4m3fn` are native PyTorch dtypes and can be directly passed in as the data type of `x` without being specified through `x_dtype`.
- When `quant_mode` is 13 (INT4 dynamic quantization), `x_dtype` only supports `torch_npu.int4` or None.

## Return Values

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|---------|----------|----------|---------------|---------------|------------|
|  expanded_x  |  Tensor |  Required  |  Features expanded based on expert_idx. In the non-quantized scenario, the data type is the same as that of `x`; in the quantization scenario, the data type is determined by quant_mode. In the quantization scenario, if the data type of `x` is `int8`, the output value is meaningless. In the Dropless scenario, the shape is [NUM_ROWS *K, H]; in the Drop/Pad scenario, it is required to be a 3D tensor with shape [expertNum, expertCapacity, H].  |  float16, bfloat16, float32, int8, float8_e5m2, float8_e4m3fn, hifloat8, float4_e2m1, int4   | 2-3  |
|  expanded_row_idx  |  Tensor |  Required  |  Mapping between expanded_x and x, with shape [NUM_ROWS*K]. The first availableIdxNum elements are valid data. The remaining invalid data is determined by `row_idx_type`: if the value is 0, the invalid data is filled with -1; if the value is 1, it is uninitialized.  |  int32   | 1-2  |
|  expert_token_cumsum_or_count  |  Tensor |  Required  |  Statistical result or cumulative value of the number of tokens processed by each expert.  |  int64   | 1-2  |
|  expanded_scale  |  Tensor |  Required  |  Intermediate values of scale in different quantization processes. For the output shape and data type under different scenarios, see the description below.  |  float32, float8_e8m0   | 1-3  |
|  expanded_topk_weight  |  Tensor |  Optional  |  Routing weights rearranged by the sorting index, in one-to-one correspondence with `expanded_x`. This output is required when `topk_weight` is provided; when `topk_weight` is not provided, the output is an empty tensor (shape is (0,)). For the output shape under different scenarios, see the description below.  |  float32   | 2  |

**Shapes of expert_token_cumsum_or_count in different modes:**

- When `expert_tokens_num_type` is 0, the output indicates the prefix sum of the total number of tokens processed by experts within `active_expert_range` after sorting, and the shape is [expert_end-expert_start].
- When `expert_tokens_num_type` is 1, the shape is [expert_end-expert_start].
- When `expert_tokens_num_type` is 2, the shape is [expert_num, 2], indicating the experts whose total number of tokens is non-zero and the total number of tokens processed by each expert.

**Output shape and data type of expanded_scale under different scenarios:**

- In the non-quantized scenario, when `scale` is provided, the shape is [NUM_ROWS*K], the first availableIdxNum elements are valid data, and the output is of the `float32` type. When the data type of the input x is `float8_e5m2`, `float8_e4m3fn`, or `float4_e2m1`, if `scale` is provided, the shape is [NUM_ROWS*K, CeilDiv(H, 64), 2] and the output is of the `float8_e8m0` type.
- In the dynamic quantization scenario, when `scale` is provided, the first availableIdxNum elements are valid data.
- In the static quantization, HIF8 direct conversion quantization, and HIF8 PERTENSOR quantization scenarios, the output is an empty tensor.
- In the HIF8 PERTOKEN quantization scenario, the shape is [NUM_ROWS*K] and the output data type is `float32`.
- In the MXFP8 quantization scenario (quantMode is 2, 3, 16, or 17), the output data type is `float8_e8m0` and the Shape is [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H, 32), 2). The first availableIdxNum rows are valid data.
- In the MXFP4 quantization scenario, the output data type is `float8_e8m0` and the Shape is [NUM_ROWS*K, M, 2], where M=CeilDiv(H, 64). The first availableIdxNum rows are valid data.
- In the FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15), the output data type is `float32` and the Shape is [NUM_ROWS*K, CeilDiv(H, 128)]. The first availableIdxNum rows are valid data.
- In the FP8 PerBlock quantization scenario (quantMode is 11 or 12), the output data type is `float32` and the Shape is [NUM_ROWS*K, CeilDiv(H, 256), 2]. The first availableIdxNum rows are valid data.

**Output shape of expanded_topk_weight in different scenarios:**

- In the Dropless scenario, the shape is [NUM_ROWS*K, 1]. The first effectiveNum elements are valid data, and the rest are uninitialized. effectiveNum=availableIdxNum.
- In the DropPad scenario, the shape is [expertNum*expertCapacity, 1]. Positions with allocated capacity are valid data, and positions without allocated capacity are filled with 0.

## Constraints

- This API can be used in inference and training scenarios. In the training scenario, automatic reverse is supported when `x.requires_grad` is True. The backward operator is [moe\_init\_routing\_grad](../../moe_init_routing_v2_grad/docs/torchapi_moe_init_routing_grad_en.md) (which encapsulates `aclnnMoeInitRoutingV2Grad`).
- This API supports calls in single-operator mode and TorchAir graph mode.
- `topk_weight` is not affected by `quant_mode`. The data type of `expanded_topk_weight` is always `float32`.
- `expert_num` must be greater than 0.
- When `drop_pad_mode=1`, `row_idx_type` only supports taking the value 0 (gather index), `quant_mode` only supports -1 (non-quantized), and the data type of `x` only supports `float16`, `bfloat16`, `float32`, `int8`, and `hifloat8`.
- `expert_tokens_num_flag` only supports taking the value true.
- The default value of `active_num` is -1, which means processing with a row limit of NUM_ROWS*K. When explicitly specifying the row limit, only NUM_ROWS*K is supported. Active truncation is currently not supported.
- In the MXFP4/INT4 dynamic quantization scenarios where quantMode is 9 or 13, and in the non-quantized passthrough scenario where quantMode is -1 and the data type of `x` is `float4_e2m1` (specified by `x_dtype`), the last dimension H of `x` is required to be even. The INT4 dynamic quantization scenario where quantMode is 13 must also meet the following requirements: the data type of `x` is `float32` or `bfloat16`, `drop_pad_mode` is 0, and the input `offset` is not supported.
- Empty tensor handling: When NUM_ROWS=0 or K=0, there are no routing elements, and the empty Tensor handling path is used. The expert count is 0. The output shape must still meet the constraints of the corresponding mode. When NUM_ROWS*K>0 and H=0, the normal routing process is still used to generate `expanded_row_idx` and `expert_token_cumsum_or_count`.
- **Automatic reverse (autograd) constraints**: Automatic reverse is supported only when the forward pass degrades to the aclnnMoeInitRoutingV2 use case, that is, no specific feature of aclnnMoeInitRoutingV4 is used. Specific requirements: `scale` is not passed in, `offset` is not passed in, `topk_weight` is not passed in, `quant_mode=-1` (non-quantized), `row_idx_type=0` (gather index), `x_dtype` is None, and `drop_pad_mode` is 0 or 1. When a specific feature of aclnnMoeInitRoutingV4 is used (quantization, `scale`, `offset`, `topk_weight`, `x_dtype`, `row_idx_type` other than 0, etc.), calling automatic reverse raises `NotImplementedError`. `active_expert_range` does not affect the reverse and is not considered a specific feature of aclnnMoeInitRoutingV4.
- Automatic reverse computes gradients only for `x`. `expert_idx` is an integer index tensor and has no gradient. `expanded_row_idx` and other integer/statistics outputs have no gradient.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  n = 3
  h = 2
  k = 4
  expert_num = 256
  expert_capacity = -1
  drop_pad_mode = 0
  expert_tokens_num_type = 1
  expert_tokens_num_flag = True
  quant_mode = -1
  active_expert_range = [0, 4]
  row_idx_type = 1

  x = torch.randn((n, h), dtype=torch.float32).npu()
  expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
  scale = torch.randn((n,), dtype=torch.float32).npu()

  # topk_weight is not passed in, expanded_topk_weight is an empty tensor
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      cann_ops_transformer.moe_init_routing(x, expert_idx, scale=scale,
                                             active_num=-1, expert_capacity=expert_capacity,
                                             expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                                             expert_tokens_num_type=expert_tokens_num_type,
                                             expert_tokens_num_flag=expert_tokens_num_flag,
                                             quant_mode=quant_mode, active_expert_range=active_expert_range,
                                             row_idx_type=row_idx_type)

  # topk_weight is passed in, expanded_topk_weight is valid data
  topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      cann_ops_transformer.moe_init_routing(x, expert_idx, topk_weight=topk_weight, scale=scale,
                                             active_num=-1, expert_capacity=expert_capacity,
                                             expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                                             expert_tokens_num_type=expert_tokens_num_type,
                                             expert_tokens_num_flag=expert_tokens_num_flag,
                                             quant_mode=quant_mode, active_expert_range=active_expert_range,
                                             row_idx_type=row_idx_type)
  ```

- TorchAir graph mode call:

  ```python
  import torch
  import torch.nn as nn
  import torch_npu
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig
  import cann_ops_transformer

  config = CompilerConfig()
  npu_backend = tng.get_npu_backend(compiler_config=config)

  class MoeInitRoutingModel(nn.Module):
      def __init__(self):
          super().__init__()

      def forward(self, x, expert_idx, *, topk_weight=None, scale=None, offset=None,
                  active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0,
                  expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1,
                  active_expert_range=None, row_idx_type=0):
          return cann_ops_transformer.moe_init_routing(x, expert_idx, topk_weight=topk_weight,
                                                        scale=scale, offset=offset, active_num=active_num,
                                                        expert_capacity=expert_capacity, expert_num=expert_num,
                                                        drop_pad_mode=drop_pad_mode,
                                                        expert_tokens_num_type=expert_tokens_num_type,
                                                        expert_tokens_num_flag=expert_tokens_num_flag,
                                                        quant_mode=quant_mode,
                                                        active_expert_range=active_expert_range,
                                                        row_idx_type=row_idx_type)

  def main():
      n = 3
      h = 2
      k = 4
      expert_num = 256
      active_expert_range = [0, 4]

      x = torch.randn((n, h), dtype=torch.float32).npu()
      expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
      topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
      scale = torch.randn((1,), dtype=torch.float32).npu()
      offset = torch.randn((1,), dtype=torch.float32).npu()

      model = MoeInitRoutingModel().npu()
      model = torch.compile(model, backend=npu_backend, dynamic=False)
      expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
          model(x, expert_idx, topk_weight=topk_weight, scale=scale, offset=offset,
                active_num=-1, expert_capacity=-1, expert_num=expert_num,
                drop_pad_mode=0, expert_tokens_num_type=1,
                expert_tokens_num_flag=True, quant_mode=0,
                active_expert_range=active_expert_range, row_idx_type=1)

  if __name__ == '__main__':
      main()
  ```

- Automatic reverse call (training use case, non-quantized):

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_init_routing

  torch_npu.npu.set_device(0)

  n = 4
  h = 8
  k = 2
  expert_num = 8

  # autograd is automatically enabled when x.requires_grad=True
  x = torch.randn((n, h), dtype=torch.float32, device="npu", requires_grad=True)
  expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32, device="npu")

  # Forward: non-quantized scenario, scale/offset/topk_weight are not passed in, row_idx_type=0
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      moe_init_routing(x, expert_idx, active_num=-1, expert_capacity=-1,
                       expert_num=expert_num, drop_pad_mode=0,
                       expert_tokens_num_type=1, expert_tokens_num_flag=True,
                       quant_mode=-1, active_expert_range=[0, 4], row_idx_type=0)

  # moe_init_routing_grad is automatically triggered in the backward pass
  loss = expanded_x.sum()
  loss.backward()
  print(x.grad.shape)  # torch.Size([4, 8])
  ```
