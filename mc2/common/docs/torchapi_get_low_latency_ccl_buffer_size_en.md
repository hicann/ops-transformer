# get_low_latency_ccl_buffer_size

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:12:03.516Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
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

- **Description**:

    This API must be used together with [low_latency_dispatch](torchapi_low_latency_dispatch_en.md) and [low_latency_combine](torchapi_low_latency_combine_en.md) to compute the size (unit: MB) of the HCCL communication buffer required by the dispatch_v3 and combine_v3 operators. This API is a static method and can be called before `MoeDistributeBuffer` is initialized.

<!-- npu="950" id7 -->
- For <term>Ascend 950PR&950DT products</term>, passing `comm_alg="hierarchy"` to `get_low_latency_ccl_buffer_size` raises `ValueError`.
<!-- end id7 -->

- **Formula**:

  - Compute the actual token length:

        $$token\_actual\_len = Align32(hidden * max\_out\_dtype\_size) + scale\_expand\_index\_buffer$$

        where max_out_dtype_size is the size of the token data type in bytes, and scale_expand_index_buffer is scale (32B) + expand_idx (3 × 4B)

  - Compute the size required for each token in the dispatch phase based on `comm_alg`:

        If `comm_alg` is "fullmesh_v2":

        $$token\_need\_size\_dispatch = Align480(token\_actual\_len) / full\_mesh\_data\_align * win\_addr\_align$$

        If `comm_alg` is "fullmesh_v1" or "":

        $$token\_need\_size\_dispatch = Align512(token\_actual\_len)$$

  - Compute the size required for each token in the combine phase:

        $$token\_need\_size\_combine = Align512(hidden * max\_out\_dtype\_size)$$

        where full_mesh_data_align is 480B alignment and win_addr_align is 512B alignment

  - Compute the ccl_buffer_size:

        $$dispatch\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_dispatch * world\_size * local\_moe\_expert\_num$$
        $$combine\_buffer\_size = 2 * num\_max\_dispatch\_tokens\_per\_rank * token\_need\_size\_combine * (topk + num\_shared\_expert)$$
        $$ccl\_buffer\_size = Align2(Align1MB(dispatch\_buffer\_size + combine\_buffer\_size + 1MB) / 1MB) / 2$$

## Function Prototype

```python
MoeDistributeBuffer.get_low_latency_ccl_buffer_size(world_size, num_max_dispatch_tokens_per_rank, hidden, num_moe_expert, topk, num_shared_expert=0, num_shared_expert_ranks=0, comm_alg="") -> int
```

## Mandatory

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>Required</td>
        <td>Represents the size of the communicator. Value range: [2, 768].</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_dispatch_tokens_per_rank</td>
        <td>int</td>
        <td>required</td>
        <td>Number of tokens on each device. If BS differs across ranks, this parameter indicates the maximum BS size.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hidden</td>
        <td>int</td>
        <td>required</td>
        <td>Hidden size of the hidden layer. Value range: [1024, 8192].</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_moe_expert</td>
        <td>int</td>
        <td>Required</td>
        <td>Number of MoE experts. Value range: [1, 1024]. The following condition must be met: num_experts % (ep_world_size - shared_expert_rank_num) = 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>topk</td>
        <td>int</td>
        <td>Required</td>
        <td>Indicates the number of topK experts to select. Value range: 0 < K ≤ 16 and 0 < K ≤ num_experts + zero_expert_num + copy_expert_num + const_expert_num.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_shared_expert</td>
        <td>int</td>
        <td>Optional</td>
        <td>Indicates the number of shared experts. A shared expert can be replicated and deployed on multiple devices. Value range: [0, 4]. 0 indicates no shared expert. Default value: 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_shared_expert_ranks</td>
        <td>int</td>
        <td>optional</td>
        <td>Number of shared expert devices. The value must satisfy world_size - num_shared_expert_ranks > 0. Default value: 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>optional</td>
        <td>Communication algorithm. Value range: "fullmesh_v1", "fullmesh_v2", or "" (default).</td>
        <td>str</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Returns

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>ccl_buffer_size</td>
        <td>int</td>
        <td>Required</td>
        <td>Represents the size of the computed ccl_buffer_size, in MB.</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Constraints

- This API is supported in training and inference use cases.
- This API can be called in single-operator mode and torchair graph mode.
- get_low_latency_ccl_buffer_size, low_latency_dispatch, and low_latency_combine must be used together.
- The "/" in the formulas in this document indicates integer division.

Variables:

- BS: batch sequence size, that is, the number of tokens finally output by the local device. Value range: 0<BS≤512. If comm_alg is "fullmesh_v2", 0<BS≤256 must be met.

list[Tensor] length / transpose or not / non-contiguous tensor support:

This API has no tensor inputs.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode:

  ```python
  import os
  import torch
  import torch_npu
  from cann_ops_transformer.ops import MoeDistributeBuffer

  # 1. Prepare the communicator and operator specification parameters.
  server_num = 1
  dev_num = 16
  world_size = server_num * dev_num
  shared_expert_num = 1
  shared_expert_rank_num = 1  # Number of shared expert devices.
  num_experts = 32  # Number of MoE experts.
  bs = 8  # Number of tokens.
  h = 7168  # Length of each token.
  k = 8
  comm_alg = "fullmesh_v1"

  # 2. Call the static method to compute the CCL communication buffer size required by dispatch/combine (in MB).
  buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(world_size, bs, h, num_experts, k,\
      shared_expert_num, shared_expert_rank_num, comm_alg)

  # 3. Write the computation result to the HCCL_BUFFSIZE environment variable for use in subsequent low_latency_dispatch/low_latency_combine calls.
  os.environ['HCCL_BUFFSIZE'] = f'buffer_size'
  print(f"buffer_size is {buffer_size}")
  print("run ccl_buffer_size success.")
  ```
