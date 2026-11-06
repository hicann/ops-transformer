# MoeReRoutingV2

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> |    √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: In an MoE network, after the AlltoAll operation obtains the tokens to be computed from other devices, this operator rearranges the tokens in expert order. Compared with the MoeReRouting operator, this operator adds the optional input `expert_topk_weight` and the optional output `permute_topk_weight`, which support reordering topkWeight in expert order so that `permute_topk_weight` corresponds one-to-one with `permute_tokens`. `expert_topk_weight` and `permute_topk_weight` must be passed in or omitted together.

- Computation formula:
  Compute the offset of the current token at the source position through double summation:

  $$
  SrcOffset =
  \sum_{i=0}^{cur\_rank} \left( \sum_{j=0}^{cur\_expert} {expert\_token\_num\_per\_rank}(i,j) \right)
  $$

  Compute the offset of the current token at the target position through double summation:

  $$
  DstOffset =
  \sum_{j=0}^{cur\_expert} \left( \sum_{i=0}^{cur\_rank} {expert\_token\_num\_per\_rank}(i,j) \right)
  $$

  - SrcOffset is the source offset of the token to be moved, computed based on the value of the input expert_token_num_per_rank.
  - DstOffset is the destination offset of the token to be moved.
  - cur_rank is the vertical-axis index of expert_token_num_per_rank, indicating the device where the token originally resides.
  - cur_expert is the horizontal-axis index of expert_token_num_per_rank, indicating that the token is computed by the expert cur_expert on the device.

  The moving offsets of tokens, per_token_scales, and expert_topk_weight are exactly the same:

  $$
  permute\_tokens[DstOffset + k] = tokens[SrcOffset + k]
  $$

  $$
  permute\_per\_token\_scales[DstOffset + k] = per\_token\_scales[SrcOffset + k]
  $$

  $$
  permute\_topk\_weight[DstOffset + k] = expert\_topk\_weight[SrcOffset + k]
  $$

  - k indicates the offset of the k-th token under the current expert (0 ≤ k < currTokenNum).
  - topkWeight corresponds one-to-one with tokens, and its moving offsets are exactly the same as those of tokens, directly reusing the SrcOffset and DstOffset of tokens.

## Parameters

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Name</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>tokens</td>
        <td>Input</td>
        <td>Tokens to be rearranged.</td>
        <td>FLOAT16, BF16, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expert_token_num_per_rank</td>
        <td>Input</td>
        <td>Number of tokens processed by each expert on each device, corresponding to `expert_token_num_per_rank` in the formula.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>per_token_scales</td>
        <td>Optional input</td>
        <td>Scale of each token, which needs to be rearranged together with the tokens.</td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expert_topk_weight</td>
        <td>Optional input</td>
        <td>Topk weight value of each token, which needs to be rearranged together with the tokens. Linked with `permute_topk_weight`: they must be passed in or omitted together.</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permute_tokens</td>
        <td>Output</td>
        <td>Rearranged tokens.</td>
        <td>FLOAT16, BF16, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permute_per_token_scales</td>
        <td>Output</td>
        <td>Rearranged per_token_scales.</td>
        <td>FLOAT, FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permute_token_idx</td>
        <td>Output</td>
        <td>Index of each token in the original arrangement.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expert_token_num</td>
        <td>Output</td>
        <td>Number of tokens processed by each expert.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permute_topk_weight</td>
        <td>Optional output</td>
        <td>Rearranged topk weight, corresponding one-to-one with permute_tokens. Linked with `expert_topk_weight`: they must be passed in or omitted together.</td>
        <td>FLOAT</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expert_token_num_type</td>
        <td>Optional attribute</td>
        <td>Mode of the output expert_token_num. 0 indicates the cumsum mode, 1 indicates the count mode, and the default value is 1.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>idx_type</td>
        <td>Optional attribute</td>
        <td>Index type of the output permute_token_idx. 0 indicates the gather index, 1 indicates the scatter index, and the default value is 0.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

## Constraints

- Description of variables used in the tensor shape:
  - A: Number of tokens. The value must meet Sum(expert_token_num_per_rank)=A.
  - H: Token length. The value must meet 0 < H < 16384.
  - N: Number of devices. No value restriction.
  - E: Number of experts on a device. No value restriction.
- Input value range restrictions
  - expert_token_num_type supports only 1 (count mode) currently.
  - The input expert_topk_weight must be 2D with shape [A, 1], and only the FLOAT data type is supported.
- Output type restrictions
  - The type of expert_token_num must be the same as the type of the input expert_token_num_per_rank.

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode     | [test_geir_moe_re_routing_v2](examples/test_geir_moe_re_routing_v2.cpp) | Invoke the MoeReRoutingV2 operator by building a graph through the [operator IR](op_graph/moe_re_routing_v2_proto.h). |
| PyTorch API | - | Invoke the moe_re_routing operator through the [moe_re_routing](docs/torchapi_moe_re_routing_en.md) API. |
