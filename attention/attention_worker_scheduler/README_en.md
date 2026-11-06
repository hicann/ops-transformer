# AttentionWorkerScheduler

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR&950DT products</term>|      √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: A data scanning operator on the Attention side in the scenario where Attention and FFN are deployed separately. This operator receives the output data from the FFNToAttention operator and scans the data step by step to ensure that the data is ready.

  **This operator is not recommended for standalone use. It is recommended to use it together with the FFNToAttention and AttentionWorkerCombine operators to form a complete workflow.**

    1. Receives the data sent by the FFNToAttention operator. The data is stored in the ScheduleContext structure. The structure contains the CommonArea, ControlArea, AttentionArea, and FfnArea fields. This interface involves CommonArea (used to store configuration information such as session_num, micro_batch_num, micro_batch_size, and selected_expert_num), ControlArea (used by upper-layer processes to control whether to exit), and the AttentionArea field (responsible for managing the core data buffers and status information required during operator computation, where token_info_buf stores input-related data information).

    2. Reads the flag information stored in token_info_buf in the ScheduleContext.AttentionArea field to check whether the communication data is ready.

    3. After all data is ready, the data can be used by the AttentionWorkerCombine operator.

- Computation formula:

$$
\text{Initialize:} \quad \text{ready\_count} = 0, \quad \text{flag\_num} = \text{micro\_batch\_size} \times \text{selected\_expert\_num}
$$

$$
\text{Check if run\_flag is 0:}
\quad \text{if run\_flag} = 0, \quad \text{exit and log}
$$

$$
\text{Loop:} \quad \text{while run\_flag} \neq 0:
\quad \text{ready\_count} = \sum\_{i=1}^{\text{flag\_num}} \mathbf{1}\_{\{ \text{flag}[i] = 1 \}}; \quad \text{if ready\_count} = \text{flag\_num}, \quad \text{break}
$$

$$
\text{Reset flags:}
\quad \text{flag}[i] = 0 \quad \text{for} \quad i = 1, 2, \dots, \text{flag\_num}
$$

$$
\text{Set micro\_batch\_id:} \quad \text{micro\_batch\_id} = (\text{micro\_batch\_id} + 1) \% \text{micro\_batch\_num}
$$

Note: micro_batch_size, selected_expert_num, run_flag, and micro_batch_id are parameters of the input ScheduleContext structure. The structure information is demonstrated in the invocation example.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>schedule_context</td>
      <td>Input</td>
      <td>Data to be processed received on the Attention side, indicating the input ScheduleContext information. Empty tensors are not supported, and the shape is fixed to (1024).</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>schedule_context</td>
      <td>Output</td>
      <td>Processed ScheduleContext information, with the same shape, dtype, and format as the input.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraint Description

- The input `schedule_context` is a 1D INT8 tensor whose data contains at least `sizeof(ScheduleContext)` bytes (the current structure is 1024 bytes).
- `schedule_mode` must be `1` (Attention); `micro_batch_num`, `micro_batch_size`, and `selected_expert_num` must all be greater than 0.
- `token_info_buf` and `token_data_buf` must both be non-empty addresses; `token_info_buf_size` must be at least `micro_batch_num * (sizeof(AttentionDataDesc) + sizeof(int32_t) * micro_batch_size * selected_expert_num)` bytes.
- `micro_batch_id` must be in the range `[0, micro_batch_num)`.
- When `run_flag` is `0`, the operator exits; when it is a non-zero value, the operator waits until all flags of the current batch are ready and then continues processing.
- If an integer multiplication overflows when computing the required buffer length, the operator returns an internal error.

## Invocation Description

| Invocation method  | Sample code                                                                | Description                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn interface | [test_aclnn_inplace_attention_worker_scheduler](./examples/test_aclnn_inplace_attention_worker_scheduler.cpp) | Invokes the AttentionWorkScheduler operator through the [aclnnInplaceAttentionWorkerScheduler](docs/aclnnInplaceAttentionWorkerScheduler_en.md) interface. |
