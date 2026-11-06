# MoeDistributeCombineV3

English | [简体中文](./README.md)

## Product Support

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950DT products</term>                             |    ×    |
| <term>Atlas A3 products</term>       |    √    |
| <term>Atlas A2 products</term> |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                               |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function Description

- Operator function: Performs AllToAllV communication and then integrates the received data (multiplies the data by weights and then sums the results). TP (Tensor Parallelism) domain communication is no longer supported in the current version, and TP-related parameters are reserved.

    Compared with the MoeDistributeCombineV2 operator, this operator has the following changes:
    - Adds the `context` input, which stores communicator-related information;
    - Adds the `ccl_buffer_size` input, which specifies the size of the current communicator;
    - Removes the `group_ep` and `group_tp` communicator name inputs;
    For details, see the following parameter descriptions.
- Calculation formula:

    $$
    ataOut = AllToAllV(expand_x)\\
    x_out = Sum(expert_scales * ataOut) + sharedExpertX
    $$

    Note that this operator must be used together with MoeDistributeDispatchV3. It is equivalent to returning the data collected by the MoeDistributeDispatchV3 operator along the original path.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
 <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>Parameter Name</th>
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
   <td>Communicator information of the current card.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_x</td>
   <td>Input</td>
   <td>Token features expanded based on expertIds.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_ids</td>
   <td>Input</td>
   <td>TopK expert indices of each token.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>assist_info_for_combine</td>
   <td>Input</td>
   <td>Indicates the number of tokens received by the same expert.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ep_send_counts</td>
   <td>Input</td>
   <td>Number of tokens received from each card in the EP communicator.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_scales</td>
   <td>Input</td>
   <td>Weights of the topK experts of each token.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_send_counts_optional</td>
   <td>Optional input</td>
   <td>Reserved parameter. TP domain communication is no longer supported. Pass a null pointer.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask_optional</td>
   <td>Optional input</td>
   <td>Indicates whether tokens participate in communication. Valid data or a null pointer can be passed. In the 1D case, true must be placed before false (for example, {true, false, true} is invalid). In the 2D case, a token does not participate in communication if all K values corresponding to the token are false. By default, all tokens participate in communication.</td>
   <td>BOOL</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activation_scale_optional</td>
   <td>Optional input</td>
   <td>Reserved parameter. This parameter is not supported in the current version. Pass a null pointer.</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scale_optional</td>
   <td>Optional input</td>
   <td>Reserved parameter. This parameter is not supported in the current version. Pass a null pointer.</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group_list_optional</td>
   <td>Optional input</td>
   <td>Reserved parameter. This parameter is not supported in the current version. Pass a null pointer.</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expand_scales_optional</td>
   <td>Optional input</td>
   <td>Indicates the weights of tokens on the current card, corresponding to the expand_scales output of MoeDistributeDispatchV3.</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_x_optional</td>
   <td>Optional input</td>
   <td>Indicates tokens computed by the shared expert. The data type must be the same as that of expand_x.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>elastic_info_optional</td>
   <td>Optional input</td>
   <td>Dynamic scale-in information of the EP communicator.</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ori_x_optional</td>
   <td>Optional input</td>
   <td>Indicates token data that has not passed through the FFN (Feed-Forward Neural network). This input is required in scenarios where copyExpert or constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_alpha_1_optional</td>
   <td>Optional input</td>
   <td>Calculation coefficient required as input in scenarios where constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_alpha_2_optional</td>
   <td>Optional input</td>
   <td>Calculation coefficient required as input in scenarios where constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_v_optional</td>
   <td>Optional input</td>
   <td>Calculation coefficient required as input in scenarios where constExpert is enabled.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>performance_info_optional</td>
   <td>Optional input</td>
   <td>Indicates the communication time spent by the current card waiting for data from other cards, in us (microseconds). The communication time consumed by each card in a single operator invocation is accumulated on this Tensor, and the operator does not automatically clear the Tensor. Therefore, you need to clear the Tensor before enabling it to record the time consumption each time.</td>
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
   <td>ID of the current card in the EP domain. Value range: [0, ep_world_size). The ep_rank_id values of cards in the same EP communicator must be unique.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moe_expert_num</td>
   <td>Attribute</td>
   <td>Number of MoE experts. The value must satisfy moe_expert_num % (ep_world_size - shared_expert_rank_num) = 0.</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>ccl_buffer_size</td>
   <td>Attribute</td>
   <td><li>Size of the Buffer of the current communicator.</li><li>Default value: "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_world_size</td>
   <td>Optional attribute</td>
   <td><li>Size of the TP communicator (reserved parameter, not supported in the current version, only 0 or 1 is allowed).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tp_rank_id</td>
   <td>Optional attribute</td>
   <td><li>ID of the current card in the TP domain (reserved parameter, not supported in the current version, pass 0).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_shard_type</td>
   <td>Optional attribute</td>
   <td><li>Indicates the distribution type of shared expert cards. Currently, only 0 is supported, indicating that shared expert cards are placed before MoE expert cards.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Indicates the number of shared experts (one shared expert can be replicated and deployed on multiple cards).</li><li>Default value: 1.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>shared_expert_rank_num</td>
   <td>Optional attribute</td>
   <td><li>Indicates the number of shared expert cards. Value range: [0, ep_world_size). If the value is 0, shared_expert_num must be 0 or 1. If the value is not 0, shared_expert_rank_num % shared_expert_num = 0 must be satisfied.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>global_bs</td>
   <td>Optional attribute</td>
   <td><li>Global batch size in the EP domain. If the BS values of all ranks are the same, global_bs = BS * ep_world_size or 0. If the BS values of ranks are different, global_bs = max_bs * ep_world_size (max_bs is the maximum BS of a single card).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>out_dtype</td>
   <td>Optional attribute</td>
   <td><li>Specifies the data type of the output x (reserved parameter, not supported in the current version, pass 0).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_quant_mode</td>
   <td>Optional attribute</td>
   <td><li>Communication quantization type. Value range: 0 or 2. 0 indicates that communication is not quantized, and 2 indicates int8 communication quantization.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>group_list_type</td>
   <td>Optional attribute</td>
   <td><li>Format of group_list (reserved parameter, not supported in the current version, pass 0).</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>Optional attribute</td>
   <td><li>Indicates the communication affinity memory layout algorithm.</li><li>Default value: "".</li></td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>zero_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Number of zero experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>copy_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Number of copy experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>const_expert_num</td>
   <td>Optional attribute</td>
   <td><li>Number of constant experts.</li><li>Default value: 0.</li></td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_out</td>
   <td>Output</td>
   <td>Indicates the processed tokens. The data type and data format are the same as those of expand_x.</td>
   <td>FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

- <term>Atlas A3 products</term>:
    * `expand_scales_optional` is not supported.
    * `comm_alg` is not supported.

## Constraints

- `MoeDistributeDispatchV3` and the `CombineV3` series operators must be used together. For details, see the invocation example.

- In different product models, communication algorithms, or versions, the element values in the Tensor outputs `assist_info_for_combine_out`, `ep_recv_counts_out`, and `expand_scales_out` of `MoeDistributeDispatchV3` may differ. When using these Tensors, pass them directly to the corresponding parameters of the `CombineV3` series operators. Other business logic of the model should not depend on these values. `tp_recv_counts_out` is a reserved output, which is not supported in the current version. Pass a null pointer.

- The values of the `ep_world_size`, `moe_expert_num`, `ccl_buffer_size`, `expert_shard_type`, `shared_expert_num`, `global_bs`, and `comm_alg` parameters used during operator invocation must be the same on all cards, the same in different layers of the network, and the same as the corresponding parameters of the `MoeDistributeDispatchV3` operator. `tp_world_size` is a reserved parameter, which is not supported in the current version.

- Shape format description in the parameter description:
    - `A`: Indicates the maximum number of tokens that the current card may receive. The value range is as follows:
        - For shared experts, `A` = `BS` *`ep_world_size`* `shared_expert_num` / `shared_expert_num` must be satisfied.
        - For MoE experts, if `global_bs` is 0, `A` >= `BS` *`ep_world_size`* min(`local_expert_num`, `K`) must be satisfied. If `global_bs` is not 0, `A` >= `global_bs` * min(`local_expert_num`, `K`) must be satisfied.
    - `K`: Indicates that topK experts are selected. The value range is 0 < `K` ≤ 16, and 0 < `K` ≤ `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num` must also be satisfied.
    - `local_expert_num`: Indicates the number of experts on the current card.
        - For a shared expert card, `local_expert_num` = 1
        - For a MoE expert card, `local_expert_num` = `moe_expert_num` / (`ep_world_size` - `shared_expert_rank_num`). TP domain communication is not supported in the current version.

- Parameter constraints:
    - `zero_expert_num`: Value range: [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Valid zero expert IDs are in the range [`moe_expert_num`, `moe_expert_num` + `zero_expert_num`).
    - `copy_expert_num`: Value range: [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Valid copy expert IDs are in the range [`moe_expert_num` + `zero_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num`).
    - `const_expert_num`: Value range: [0, MAX_INT32). MAX_INT32 = 2^31 - 1. Valid constant expert IDs are in the range [`moe_expert_num` + `zero_expert_num` + `copy_expert_num`, `moe_expert_num` + `zero_expert_num` + `copy_expert_num` + `const_expert_num`).
    - `ori_x_optional`: Valid data or a null pointer can be passed. If `copy_expert_num` is not 0 or `const_expert_num` is not 0, valid input must be passed. When valid data is passed, the shape must be (`BS`, `H`), and the data type must be the same as that of `expand_x`.
    - `const_expert_alpha_1_optional`: Valid data or a null pointer can be passed. If `const_expert_num` is not 0, valid input must be passed. When valid data is passed, the shape must be (`const_expert_num`,), and the data type must be the same as that of `expand_x`.
    - `const_expert_alpha_2_optional`: Valid data or a null pointer can be passed. If `const_expert_num` is not 0, valid input must be passed. When valid data is passed, the shape must be (`const_expert_num`,), and the data type must be the same as that of `expand_x`.
    - `const_expert_v_optional`: Valid data or a null pointer can be passed. If `const_expert_num` is not 0, valid input must be passed. When valid data is passed, the shape must be (`const_expert_num`, `H`), and the data type must be the same as that of `expand_x`.

- "/" in the formulas in this document indicates integer division.
- Communicator usage constraints:
    - In a model, `MoeDistributeCombineV3` and `MoeDistributeDispatchV3` support only the same EP communicator, and no other operators are allowed in the communicator.
    - TP domain communication is currently not supported.

- <term>Atlas A3 products</term>:
    - In this scenario, a single card contains dual DIEs (also referred to as dies). Therefore, the "current card" in the parameter description refers to a single DIE.
    - Shape format description in the parameter description:
        - `H`: Indicates the hidden size. Value range: [1024, 8192].
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. Value range: [1, 512].
    - Parameter constraints:
        - `ep_world_size`: Supported values: 8, 16, 32, 64, 128, 144, 256, 288.
        - `moe_expert_num`: Value range: (0, 1024].
        - `tp_world_size`: Reserved parameter, not supported in the current version. Only 0 or 1 is allowed.
        - `tp_rank_id`: Reserved parameter, not supported in the current version. Pass 0.
        - `shared_expert_num`: Current value range: [0, 4].
        - `comm_quant_mode`: int8 quantization is supported.
        - `performance_info_optional`: Reserved parameter, not supported in the current version. Pass a null pointer.
        - `ccl_buffer_size`: Call the get_low_latency_ccl_buffer_size API (../common/torch_extension/moe_distribute_buffer.py).

## Invocation Description

| Invocation Method | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| low_latency_combine API | [moe_distribute_buffer.py](../common/torch_extension/moe_distribute_buffer.py) | The moe_distribute_combine_v3 operator is invoked through the [low_latency_combine](../common/docs/torchapi_low_latency_combine_en.md) API. |
