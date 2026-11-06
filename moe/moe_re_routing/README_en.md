# MoeReRouting

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> |    √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function Description

- Operator function: In an MoE network, after the AlltoAll operation obtains the tokens to be computed from other devices, this operator rearranges the tokens in expert order.

- Computation formula:
  Assume that the shape of `expert_token_num_per_rank` is `[rank_num, expert_num]`, where `rank_num` is the number of devices and `expert_num` is the number of experts on each device. The following formulas compute the start offsets of the token segment corresponding to the current rank and expert in the source input and the target output, respectively:

  $$
  SrcOffset =
  \sum_{i=0}^{cur\_rank-1} \left( \sum_{j=0}^{expert\_num-1} {expert\_token\_num\_per\_rank}(i,j) \right) + \sum_{j=0}^{cur\_expert-1} {expert\_token\_num\_per\_rank}(cur\_rank,j)
  $$

  $$
  DstOffset =
  \sum_{j=0}^{cur\_expert-1} \left( \sum_{i=0}^{rank\_num-1} {expert\_token\_num\_per\_rank}(i,j) \right) + \sum_{i=0}^{cur\_rank-1} {expert\_token\_num\_per\_rank}(i,cur\_expert)
  $$

  - SrcOffset is the start offset of the token segment corresponding to the current rank and expert in the rank-major source input.
  - DstOffset is the start offset of the same token segment in the expert-major target output.
  - cur_rank is the vertical-axis index of expert_token_num_per_rank, indicating the device where the token originally resides.
  - cur_expert is the horizontal-axis index of expert_token_num_per_rank, indicating that the token is computed by the expert cur_expert on the device.

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
        <td>
        General: FLOAT16, BF16, INT8<br>
        <term>Ascend 950PR&950DT products</term>: FLOAT16, BF16, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2
        </td>
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
        <td>
        General: FLOAT<br>
        <term>Ascend 950PR&950DT products</term>: FLOAT, FLOAT8_E8M0
        </td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permute_tokens</td>
        <td>Output</td>
        <td>Rearranged tokens.</td>
        <td>
        General: FLOAT16, BF16, INT8<br>
        <term>Ascend 950PR&950DT products</term>: FLOAT16, BF16, INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2
        </td>
        <td>ND</td>
      </tr>
      <tr>
        <td>permute_per_token_scales</td>
        <td>Output</td>
        <td>Rearranged per_token_scales.</td>
        <td>
        General: FLOAT<br>
        <term>Ascend 950PR&950DT products</term>: FLOAT, FLOAT8_E8M0
        </td>
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

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- Description of variables used in the tensor shape:
  - A: Number of tokens. The value must meet Sum(expert_token_num_per_rank)=A.
  - H: Token length. The value must meet 0 < H < 16384.
  - N: Number of devices. No value restriction.
  - E: Number of experts on a device. No value restriction.
- Input value range restrictions
  - expert_token_num_type, that is, the mode of the output expert_token_num. 0 indicates the cumsum mode, 1 indicates the count mode, and the default value is 1. Currently, only 1 is supported.
  - idx_type, that is, the index type of the output permute_token_idx. 0 indicates the gather index, 1 indicates the scatter index, and the default value is 0. Ascend 950PR&950DT products support 0 or 1, and other products support only 0.
- Output type restrictions
  - The type of expert_token_num must be the same as the type of the input expert_token_num_per_rank.
