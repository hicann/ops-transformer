# MoeDistributeDispatchV3

English | [简体中文](./README.md)

## Product Support Status

| Product                                                      |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT products</term>                           |    √    |
| <term>Atlas A3 products</term>                               |    √    |
| <term>Atlas A2 products</term>                               |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×    |
| <term>Atlas inference products</term>                        |    ×    |
| <term>Atlas training products</term>                         |    ×    |

## Description

Operator function: Quantizes token data (optional) and performs AllToAllV communication in the EP (Expert Parallelism) domain. TP (Tensor Parallelism) domain communication is no longer supported in the current version; TP-related parameters are reserved parameters.

- Case 1: If quant_mode=0 (non-quantization scenario):

$$
allToAllXOut = AllToAllV(X)\\
expand\_x\_out = AllToAllV(X)
$$

- Case 2: If quant_mode=2 (pertoken dynamic quantization scenario):

$$
xFp32 = CastToFp32(X) \times scales \\
dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
quantOut = CastToInt8(xFp32 \times dynamicScales) \\
allToAllXOut = AllToAllV(quantOut) \\
allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
expand\_x\_out = AllToAllV(quantOut) \\
dynamic\_scales\_out = allToAllDynamicScalesOut
$$

Here, $emax$ indicates the value of the exponent part corresponding to the maximum normal number of the type.

- <term>Atlas A3 products</term>: This operator must be used together with `MoeDistributeCombineV3`.

Compared with the `MoeDistributeDispatchV3` operator, this operator has the following changes:

- Adds the `context` input, which stores the communicator-related information;
- Adds the `ccl_buffer_size` input, which specifies the size of the current communicator;
- Removes the `group_ep` and `group_tp` communicator name inputs;

For details, see the following parameter descriptions.

## Parameters

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter</th>
   <th>Input/Output/Attribute</th>
   <th>Description</th>
   <th>Data Type</th>
   <th>Data Format</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>context</td>
   <td>Input</td>
   <td>Communicator information data of the current card.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>Input</td>
   <td>Token data sent by the current card.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_ids</td>
   <td>Input</td>
   <td>TopK expert indexes of each token.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>scales_optional</td>
   <td>Optional input</td>
   <td>Quantization smoothing parameter of each expert. Pass a null pointer in non-quantization scenarios; valid data or a null pointer can be passed in dynamic quantization scenarios.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask_optional</td>
   <td>Optional input</td>
   <td>Indicates whether a token participates in communication. Valid data or a null pointer can be passed. When 1D, true values must be arranged before false values (for example, {true, false, true} is invalid). When 2D, a token does not participate in communication if all K values corresponding to the token are false. By default, all tokens participate in communication.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_scales_optional</td>
   <td>Optional input</td>
   <td>TopK expert weights of each token.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elastic_info_optional</td>
   <td>Optional input</td>
   <td>Dynamic scale-down information of the EP communicator.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>performance_info_optional</td>
   <td>Optional input</td>
   <td>Indicates the communication time the current card spends waiting for data from other cards, in us (microseconds). For a single operator call, the communication time of each card is accumulated into this tensor. The operator does not automatically clear the tensor. Therefore, you need to clear the tensor each time before using it to start recording the time consumption.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_world_size</td>
   <td>Attribute</td>
   <td>Size of the EP communicator.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_rank_id</td>
   <td>Attribute</td>
   <td>ID of the current card in the EP domain. Value range: [0, ep_world_size). The ep_rank_id values of the cards in the same EP communicator must be unique.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moe_expert_num</td>
   <td>Attribute</td>
   <td>Number of MoE experts. moe_expert_num % (ep_world_size - shared_expert_rank_num) = 0 must be satisfied.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ccl_buffer_size</td>
   <td>Required attribute</td>
   <td>Buffer size of the current communicator (INT64, in bytes), which must be obtained through the get_low_latency_ccl_buffer_size API.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_world_size</td>
   <td>Optional attribute</td>
   <td><li>Size of the TP communicator (reserved parameter; not supported in the current version; only 0 or 1 can be passed).</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_rank_id</td>
   <td>Optional attribute</td>
   <td><li>ID of the current card in the TP domain (reserved parameter; not supported in the current version; pass 0).</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_shard_type</td>
   <td>Optional attribute</td>
   <td><li>Indicates the distribution type of shared expert cards. Currently, only 0 can be passed, indicating that shared expert cards are arranged before MoE expert cards.</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Indicates the number of shared experts (one shared expert can be replicated and deployed on multiple cards).</li><li>The default value is 1.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_rank_num</td>
   <td>Optional attribute</td>
   <td><li>Indicates the number of shared expert cards. Value range: [0, ep_world_size). When it is 0, shared_expert_num must be 0 or 1. When it is not 0, shared_expert_rank_num % shared_expert_num = 0 must be satisfied.</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>quant_mode</td>
   <td>Optional attribute</td>
   <td><li>Indicates the quantization mode. Supported values: 0 (non-quantization) and 2 (dynamic quantization).</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>global_bs</td>
   <td>Optional attribute</td>
   <td><li>Global batch size in the EP domain. When the BS values of all ranks are the same, global_bs = BS * ep_world_size or 0. When the BS values of the ranks are inconsistent, global_bs = max_bs * ep_world_size (max_bs is the maximum BS of a single card).</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_token_nums_type</td>
   <td>Optional attribute</td>
   <td><li>Semantic type of the values in the output expert_token_nums. Supported values: 0 (the output in expert_token_nums is the prefix sum of the numbers of tokens processed by each expert) and 1 (the output in expert_token_nums is the number of tokens processed by each expert).</li><li>The default value is 1.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>Optional attribute</td>
   <td><li>Indicates the communication-affinity memory layout algorithm.</li><li>The default value is "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zero_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Number of zero experts.</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copy_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Number of copy experts.</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Number of constant experts.</li><li>The default value is 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_x_out</td>
   <td>Output</td>
   <td>Token features expanded based on expert_ids.</td>
   <td>FLOAT16, BFLOAT16, INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>dynamic_scales_out</td>
   <td>Output</td>
   <td>In quantization scenarios, indicates the quantization coefficient of the tokens output by the current card. This output exists only when quant_mode=2.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assist_info_for_combine_out</td>
   <td>Output</td>
   <td>Indicates the number of tokens sent to the same expert.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_token_nums_out</td>
   <td>Output</td>
   <td>Indicates the number of tokens received by each expert.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_recv_count_out</td>
   <td>Output</td>
   <td>Number of tokens received from each card in the EP communicator.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_recv_count_out</td>
   <td>Output</td>
   <td>Number of tokens received from each card in the TP communicator (reserved output; not supported in the current version; pass a null pointer). This output does not exist if there is no TP domain communication.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_scales_out</td>
   <td>Output</td>
   <td>Indicates the weights of the tokens output by the current card.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

* <term>Atlas A3 products</term>:
    * `expand_scales_out` is not supported.

## Constraints

- `MoeDistributeDispatchV3` must be used together with the `CombineV3` series operators. For details, see the calling instructions.

- With different product models, different communication algorithms, or different versions, the element values in the tensor outputs `assist_info_for_combine_out`, `ep_recv_count_out`, `tp_recv_count_out`, and `expand_scales_out` of `MoeDistributeDispatchV3` may differ. When using them, directly pass the preceding tensors to the corresponding parameters of the `CombineV3` series operators. Other service logic of the model must not depend on them.

- The values of the `ep_world_size`, `moe_expert_num`, `ccl_buffer_size`, `tp_world_size`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, `global_bs`, and `comm_alg` parameters used during operator calling must be the same on all cards, the same in different layers of the network, and consistent with the corresponding parameters of the `CombineV3` series operators.

- Shape format descriptions in the parameter descriptions:
    - `A`: Indicates the maximum number of tokens that the current card may receive. The value ranges are as follows:
        - For shared experts, `A` = `BS` *`ep_world_size`* `shared_expert_num` / `shared_expert_rank_num` must be satisfied.
        - For MoE experts, when `global_bs` is 0, `A` >= `BS` *`ep_world_size`* min(`local_expert_num`, `K`) must be satisfied; when `global_bs` is not 0, `A` >= `global_bs` * min(`local_expert_num`, `K`) must be satisfied.
    - `K`: Indicates the selection of topK experts. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num` must also be satisfied.
    - `local_expert_num`: Indicates the number of experts on the current card.
        - For shared expert cards, `local_expert_num` = 1
        - For MoE expert cards, `local_expert_num` = `moe_expert_num` / (`ep_world_size` - `shared_expert_rank_num`). When `local_expert_num` > 1, TP domain communication is not supported.

- Attribute constraints:
    - `zero_expert_num`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid ID values of zero experts are [`moe_expert_num`, `moe_expert_num` + `zero_expert_num`).
    - `copy_expert_num`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid ID values of copy experts are [`moe_expert_num` + `zero_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num`).
    - `const_expert_num`: Value range: [0, MAX_INT32), where MAX_INT32 = 2^31 - 1. The valid ID values of constant experts are [`moe_expert_num` + `zero_expert_num` + `copy_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num`).

- In the formulas in this document, "/" indicates integer division.
- Communicator usage constraints:
    - In a model, the `CombineV3` series operators and `MoeDistributeDispatchV3` support only the same EP communicator, and no other operators are allowed in this communicator.
    - TP domain communication is currently not supported.

- Communication mode constraints:
    - <term>Ascend 950DT products</term>: Only UB Memory communication is supported.

- <term>Atlas A3 products</term>:
    - In this scenario, a single card contains dual dies (referred to as "grains" or "bare dies" for short). Therefore, "the current card" in the parameter descriptions refers to a single die.
    - Parameter constraints:
        - `elastic_info_optional`: Not supported in the current version; pass a null pointer.
        - `ep_world_size`: Value range: [2, 768].
        - `moe_expert_num`: Value range: (0, 1024].
        - `shared_expert_num`: Supported value range: [0, 4].
        - `comm_alg`: The current version supports only the following input values: "", "fullmesh_v1", and "fullmesh_v2".
            - "": Default value; the fullmesh_v1 template is enabled.
            - "fullmesh_v1": The fullmesh_v1 template is enabled.
            - "fullmesh_v2": The fullmesh_v2 template is enabled. `comm_alg` takes effect only when `tp_world_size` is 1, and enabling it is not supported in the scenarios where the `BS` values of the cards are inconsistent, xActiveMask is input, or special experts are used.
        - `ep_recv_count_out`: The shape must be (`ep_world_size` * `local_expert_num`,).
        - `performance_Info_optional`: Reserved parameter; not supported in the current version; pass a null pointer.
        - `ccl_buffer_size`: Call the get_low_latency_ccl_buffer_size API (../common/torch_extension/moe_distribute_buffer.py).
    - Shape format descriptions in the parameter descriptions:
        - `H`: Indicates the hidden size of the hidden layer. Value range: [1024, 8192].
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. Value range: [1, 512].

## Calling Instructions

| Calling Method           | Sample Code                                                      | Description                                                                                             |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| low_latency_dispatch API | [moe_distribute_buffer.py](../common/torch_extension/moe_distribute_buffer.py) | Calls the moe_distribute_dispatch_v3 operator in the form of the [low_latency_dispatch](../common/docs/torchapi_low_latency_dispatch_en.md) API. |
