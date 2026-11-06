# FfnWorkerBatching

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---------------------------- | :-----------: |
|<term>Ascend 950PR&950DT products</term>| √ |
|<term>Atlas A3 products</term>| √ |
|<term>Atlas A2 products</term>| √ |
|<term>Atlas 200I/500 A2 inference products</term>| × |
|<term>Atlas inference products</term>| × |
|<term>Atlas training products</term>| × |

## Function Description

- Operator function: `FFNWorkerBatching` performs token reordering on the FFN worker in scenarios where Attention and FFN are deployed separately. Attention sends tokens to the pre-allocated data area of the corresponding FFN worker based on expert routing. FFNWorkerBatching scans the scheduling information from this data area, aggregates and reorders tokens by expert dimension, and produces the contiguous token data blocks corresponding to each expert.

- Computation steps:

    1. Sort the expert IDs of all tokens in `expert_ids_in` (tokens that are masked are initialized to a large value) and generate the gather index.
    2. Multiple cores extract the hidden states and dynamic scale of tokens from `token_data` in parallel based on the gather index, and look up the corresponding `session_id`, `micro_batch_id`, and `token_id`.
    3. Count the number of tokens of each expert based on the sorted expert ID sequence and generate `group_list` (each row contains the expert ID and the number of tokens of that expert).

    Here $Y = A \times BS \times (K+1)$, where $A$ is the number of Attention workers, $BS$ is the micro batch size, and $K+1$ is topK plus the number of shared experts.

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
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
      <td>schedule_context</td>
      <td>Input</td>
      <td>Scheduling context data structure, which contains CommonArea, ControlArea, AttentionArea, and FfnArea. The operator reads token_info_buf and token_data_buf from the FfnArea to obtain the token data to be reordered and the description information, and obtains routing information such as layer_id, session_id, micro_batch_id, and expert_ids. It is a contiguous one-dimensional INT8 tensor with shape [1024]. It does not support an empty tensor.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_num</td>
      <td>Attribute</td>
      <td>Total number of local experts for this invocation; interpreted as the number of local experts per layer × layer_num only for asynchronous mixed-layer RECV. Used to derive the output size of group_list. Value range: (0, 8192].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_out_shape</td>
      <td>Attribute</td>
      <td>Upper limit of the output shape, in the format {A, BS, topK+1, H}. Used to derive the shape upper limit of output y, Y = A × BS × (topK+1), and the value of H. The length must be 4. A (number of Attention workers) ≤ 1024; the third dimension topK+1 (topK plus the number of shared experts) ≤ 64; BS (micro batch size) > 0 and has no hard upper limit (limited by memory); H (hidden size) > 0 and supports generalization.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>token_dtype</td>
      <td>Attribute</td>
      <td>Data type of the input token. The default value is 0. The value range is [0, 5] for <term>Ascend 950PR&950DT products</term> and [0, 2] for other supported products. Determines the storage width of each hidden state element and the data type of output y:<br>0: FP16, 2 bytes per element;<br>1: BF16, 2 bytes per element;<br>2: INT8 dynamic quantization, 1 byte per element, one FP32 scale per row;<br>3: FLOAT8_E5M2, 1 byte per element;<br>4: FLOAT8_E4M3FN (float8_e4m3), 1 byte per element;<br>5: FLOAT4_E2M1, two elements packed into 1 byte.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>need_schedule</td>
      <td>Attribute</td>
      <td>Scheduling mode. 0 means performing only batching without scanning data; 1 means scanning data first and then performing batching. The default value is 0. Value range: [0, 1].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layer_num</td>
      <td>Attribute</td>
      <td>Number of layers. Experts of each layer are indexed independently. The default value is 0. Value range: [0, expert_num]; for asynchronous receiving, the value must be greater than 0 and expert_num must be divisible by this value.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sync_flag</td>
      <td>Attribute</td>
      <td>The default is false, with values false/true. This attribute is interpreted only when need_schedule=1 on <term>Ascend 950PR&950DT products</term>: false waits for all sessions of the current micro batch to be ready; true scans micro batches cyclically and processes a micro batch as soon as a ready session is found for the first time.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Reordered token hidden states, stored contiguously after being sorted by expert ID. The logical shape is [Y, H]. The data type is determined by token_dtype: 0 is FP16, 1 is BF16, and 2 is INT8; <term>Ascend 950PR&950DT products</term> additionally support 3 as FLOAT8_E5M2, 4 as FLOAT8_E4M3FN, and 5 as FLOAT4_E2M1.</td>
      <td>FP16, BF16, INT8; <term>Ascend 950PR&950DT products</term> additionally support FLOAT8_E5M2, FLOAT8_E4M3FN, and FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>Output</td>
      <td>Number of tokens of each non-empty expert, with shape [expert_num, 2]. Stored compactly in ascending order of expert ID, with each row being [expert_id, expert_token_num] and remaining rows filled with [0, 0].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>session_ids</td>
      <td>Output</td>
      <td>Attention session ID corresponding to each output token, with shape [Y].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>micro_batch_ids</td>
      <td>Output</td>
      <td>Micro batch ID corresponding to each output token, with shape [Y].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>token_ids</td>
      <td>Output</td>
      <td>Position index of each output token in the original input, with shape [Y].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expert_offsets</td>
      <td>Output</td>
      <td>Offset of each output token in its expert group, with shape [Y].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dynamic_scale</td>
      <td>Output</td>
      <td>Scale values reordered together with tokens. When token_dtype=2, the value is FP32 with shape [Y]; on <term>Ascend 950PR&950DT products</term>, when token_dtype is 3, 4, or 5, the value is FLOAT8_E8M0 with shape [Y, ceil(H/32)]. When token_dtype is 0 or 1, this output contains no valid data.</td>
      <td>FP32; <term>Ascend 950PR&950DT products</term> additionally support FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_token_num</td>
      <td>Output</td>
      <td>Sum of the number of valid tokens of all experts, a single-element output with shape [1].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

> For the value range, shape, and format constraints of each parameter (attribute/input/output), see the preceding "Parameter Description" table. This section lists only global constraints across parameters.

- This API supports graph mode (GEIR) and single-operator mode (aclnn).
- The value of parameter M (number of micro batches, taken from schedule_context.common.micro_batch_num) must be ≤ 64.
- The upper limit of the number of output tokens is Y = A × BS × (topK+1), where A, BS, and topK+1 are the first, second, and third dimensions of max_out_shape respectively. Y supports generalization and has no hard upper limit (limited by memory).
- The precision standard is binary consistency (a non-computation operator: integer index sorting plus byte-for-byte data movement, with no floating-point operations).
- Only the first `actual_token_num` rows of `y`, the index outputs, and the valid `dynamic_scale` are guaranteed to be valid. The remaining capacity does not participate in result comparison.
- On <term>Ascend 950PR&950DT products</term>, `token_dtype=3, 4, 5` all support `need_schedule=0, 1`. Every 32 logical hidden state elements correspond to one E8M0 scale, and a tail with fewer than 32 elements still occupies one scale. All scales immediately follow the entire row of token data.
- On <term>Ascend 950PR&950DT products</term>, let S=ceil(H/32). The valid data of an FP8 row is the `H`-byte token plus the `S`-byte scale, and that of an FP4 row is the `H/2`-byte token plus the `S`-byte scale. FP4 requires H to be an even number. The low nibble of each byte stores the preceding element, and the high nibble stores the following element. `attn_to_ffn_token_size` records the byte stride of the source row, which may include row-end padding.
- Asynchronous receiving on <term>Ascend 950PR&950DT products</term> (`need_schedule=1, sync_flag=true`) requires `layer_num>0` and `layer_num` to divide `expert_num`. Cross-layer expert IDs are generated as `layer_id * (expert_num / layer_num) + expert_id` for sorting, token reordering, and `group_list` statistics. Synchronous receiving and NORM still use the original expert IDs.

## Invocation Instructions

| Invocation Method | Sample Code | Description |
| ---- | ---- | ---- |
| Graph mode invocation | [test_geir_ffn_worker_batching.cpp](examples/test_geir_ffn_worker_batching.cpp) | Calls the FfnWorkerBatching operator by constructing the graph through [operator IR](op_graph/ffn_worker_batching_proto.h). |
| aclnn API | [test_aclnn_ffn_worker_batching.cpp](examples/test_aclnn_ffn_worker_batching.cpp) | Calls the FfnWorkerBatching operator through the [aclnnFfnWorkerBatching](docs/aclnnFfnWorkerBatching_en.md) API. |

## Schedule Context Data Structure

```cpp
struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;           // Number of Attention nodes
    uint32_t micro_batch_num;       // Number of micro batch splits
    uint32_t micro_batch_size;      // batch_size / micro_batch_num
    uint32_t selected_expert_num;   // topK + 1 (including shared experts)
    uint32_t expert_num;            // Number of experts per layer
    uint32_t attn_to_ffn_token_size;// Size occupied by each token in the FFN window data area, aligned to 512
    uint32_t ffn_to_attn_token_size;// Size occupied by each token in the Attention window data area, aligned to 512
    int32_t  schedule_mode;         // 0: schedule FFN only, 1: schedule Attention only, 2: schedule both
    int8_t   reserve0[96];          // padding to 128 bytes
  };
  struct ControlArea {
    int32_t run_flag;               // Controls loop exit
    int8_t  reserve1[124];
  };
  struct AttentionArea {
    uint64_t token_info_buf;        // [M, DataDesc], where DataDesc contains flags[batch_size][topK+1]
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // [M, BS, K+1, HS]
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id;        // Used for polling, initial value micro_batch_num-1
    int8_t   reserve5[92];
  };
  struct FfnArea {
    // FFN input area
    uint64_t token_info_buf;        // [A, M, F], where DataDesc contains flag/layer_id/expert_ids
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;        // [A, M, BS, K+1, HS]
    uint64_t token_data_buf_size;
    uint64_t polling_index;
    int8_t   reserve3[88];
    // FFN output area
    uint64_t layer_ids_buf;         // [session_num]
    uint64_t layer_ids_buf_size;
    uint64_t session_ids_buf;       // [session_num]
    uint64_t session_ids_buf_size;
    uint64_t micro_batch_ids_buf;   // [session_num]
    uint64_t micro_batch_ids_buf_size;
    uint64_t expert_ids_buf;        // [session_num, BS, K+1]
    uint64_t expert_ids_buf_size;
    uint32_t out_num;               // Number of sessions actually fully received
    int8_t   reserve4[60];
  };
  CommonArea    common;
  ControlArea   control;
  AttentionArea attention;
  FfnArea       ffn;
  int8_t        reserve6[384];      // padding to 1024 bytes
}; // Total size: 1024 bytes
```
