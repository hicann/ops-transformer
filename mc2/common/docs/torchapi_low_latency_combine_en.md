# low_latency_combine

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:20:13.617Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

    Must be used together with [low_latency_dispatch](torchapi_low_latency_dispatch_en.md). It is equivalent to returning the data along the same path through which the low_latency_dispatch operator collected the data.
  - Supports data combine, performs alltoallv communication, and finally combines the received data (multiplied by weights and then added);
  - Supports special expert scenarios.

- **Formula**:

  - Data combination function:

      $$ata\_out = AlltoAllv(expand\_x)$$

      $$x = Sum(topk\_weights * ata\_out + topk\_weights * shared\_expert\_x)$$

  - Special expert scenarios:

      Zero expert scenario, i.e., `zero_expert_num` is not 0:

      $$Moe(ori\_x)=0$$

      Copy expert scenario, i.e., `copy_expert_num` is not 0:

      $$Moe(ori\_x)=ori\_x$$

      Constant expert scenario, i.e., `const_expert_num` is not 0:

      $$Moe(ori\_x)=const\_expert\_alpha\_1*ori\_x+const\_expert\_alpha\_2*const\_expert\_v$$

## Function Prototype

```python
MoeDistributeBuffer.low_latency_combine(x, topk_idx, topk_weights, assist_info_for_combine, ep_send_counts, *, num_experts=0, comm_alg="", comm_quant_mode=0, x_active_mask=None, expand_scales=None, shared_expert_x=None, elastic_info=None, ori_x=None, const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None, zero_expert_num=0, copy_expert_num=0, const_expert_num=0, expert_shared_type=0, shared_expert_num=1, shared_expert_rank_num=0, num_max_dispatch_tokens_per_rank=0) -> Tensor
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
    <th>Data Type</th>
    <th>Optional/Mandatory</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Mandatory</td>
        <td>Token features expanded according to `topk_idx`. The data format is ND.</td>
        <td>bfloat16, float16</td>
        <td>(A, H)</td>
    </tr>
    <tr>
        <td>topk_idx</td>
        <td>Tensor</td>
        <td>mandatory</td>
        <td>Indices of the topK experts for each token, with the ND data format. This corresponds to the `topk_idx` input of [low_latency_dispatch](torchapi_low_latency_dispatch_en.md). The value range of the tensor is [0, num_experts), and the K values in the same row cannot be duplicated.</td>
        <td>int32</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>topk_weights</td>
        <td>Tensor</td>
        <td>mandatory</td>
        <td>Indicates the weights of the topK experts for each token. Shared experts do not need to be multiplied by the weight coefficients and can be added directly. None is supported. The data format is ND.</td>
        <td>float</td>
        <td>(BS, K)</td>
    </tr>
    <tr>
        <td>assist_info_for_combine</td>
        <td>Tensor</td>
        <td>mandatory</td>
        <td>Indicates the number of tokens sent to the same expert. The data format is ND. Corresponds to the `assist_info_for_combine` output of [low_latency_dispatch](torchapi_low_latency_dispatch_en.md).</td>
        <td>int32</td>
        <td>(A*128, )</td>
    </tr>
    <tr>
        <td>ep_send_counts</td>
        <td>Tensor</td>
        <td>mandatory</td>
        <td>Indicates the number of tokens that each expert on this device sends to each device in the EP (Expert Parallelism) domain (the token count is expressed as a prefix sum). The data format is ND. Corresponds to the `ep_recv_counts` output of [low_latency_dispatch](torchapi_low_latency_dispatch_en.md).</td>
        <td>int32</td>
        <td>(ep_world_size*local_expert_num, )</td>
    </tr>
    <tr>
        <td>num_experts</td>
        <td>int</td>
        <td>optional</td>
        <td>Number of MoE experts, value range [1, 1024], and must satisfy the following condition: num_experts%(ep_world_size - shared_expert_rank_num)=0. Default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_alg</td>
        <td>str</td>
        <td>optional</td>
        <td>Indicates the communication affinity memory layout algorithm. The current version supports only "". The default value is "".</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_quant_mode</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the communication quantization type. Value range: 0 and 2. 0 indicates no quantization during communication, and 2 indicates int8 quantization during communication. The default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_active_mask</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Indicates whether a token participates in communication. Required to be a 1D or 2D tensor, and the data format is required to be ND. When the input is 1D, true indicates that the corresponding token participates in communication, and true must be placed before false; for example, {true, false, true} is invalid input. When the input is 2D, true indicates that the `topk_idx` values corresponding to the current token participate in communication. If all K bool values corresponding to the current token are false, the current token does not participate in communication. By default, all tokens participate in communication. When the BS quantity is not consistent across devices, all tokens must be valid. The default value is None.</td>
        <td>bool</td>
        <td>            <ul>
                <li>(BS, )</li>
                <li>(BS, K)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>expand_scales</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Corresponds to the `expand_scales` output of [low_latency_dispatch](torchapi_low_latency_dispatch_en.md). This parameter is not supported for now; the default value can be used. The default value is None.</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_x</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Data type must remain consistent with `expand_x`. Used only in the scenario where the number of shared expert cards `shared_expert_rank_num` is 0. It indicates the shared expert token, that is, the value to be added after combine_v2. If the tensor is 2D, the shape is (BS, H); if the tensor is 3D, the product of the first two dimensions must equal BS, and the third dimension must equal H. The default value is None.</td>
        <td>Consistent with x</td>
        <td>            <ul>
                <li>(BS, H)</li>
                <li>(*, *, H)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>elastic_info</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Reserved parameter, not supported in the current version. The default value None can be passed. The default value is None.</td>
        <td>int32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ori_x</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Indicates the token data that has not passed through the FFN. This input data is required in the scenario where `copy_expert_num` is not 0 or `const_expert_num` is not 0. The data type needs to remain consistent with `expand_x`, and the data format is required to be ND. Valid data can be optionally passed or filled with None. Valid data must be passed when `copy_expert_num` is not 0 or `const_expert_num` is not 0. The default value is None.</td>
        <td>consistent with x</td>
        <td>(BS, H)</td>
    </tr>
    <tr>
        <td>const_expert_alpha_1</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Computation coefficient that requires input in the scenario of `const_expert_num` not being 0. The data type needs to remain consistent with `expand_x`, and the data format is required to be ND. Valid data can be optionally passed or filled with None. Valid input must be passed when `const_expert_num` is not 0. The default value is None.</td>
        <td>Consistent with x</td>
        <td>(const_expert_num, H)</td>
    </tr>
    <tr>
        <td>const_expert_alpha_2</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Computation coefficient that requires input in the scenario of `const_expert_num` not being 0. The data type needs to remain consistent with `expand_x`, and the data format is required to be ND. Valid data can be optionally passed or filled with None. Valid input must be passed when `const_expert_num` is not 0. The default value is None.</td>
        <td>The same as x.</td>
        <td>(const_expert_num, H)</td>
    </tr>
    <tr>
        <td>const_expert_v</td>
        <td>Optional[Tensor]</td>
        <td>Optional</td>
        <td>Computation coefficients that require input in the scenario where `const_expert_num` is not 0. The data type must remain consistent with `expand_x`, and the data format is required to be ND. Valid data or None can be optionally passed. Valid input must be passed when `const_expert_num` is not 0. The default value is None.</td>
        <td>The same as x.</td>
        <td>(const_expert_num, H)</td>
    </tr>
    <tr>
        <td>zero_expert_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Indicates the number of zero experts. Value range: [0, MAX_int32), where MAX_int32 = 2^31 - 1. The valid ID value of zero experts is [num_experts, num_experts+zero_expert_num). Default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>copy_expert_num</td>
        <td>int</td>
        <td>Optional</td>
        <td>Indicates the number of copy experts. Value range: [0, MAX_int32), where MAX_int32 = 2^31 - 1. The valid ID value of copy experts is [num_experts+zero_expert_num, num_experts+zero_expert_num+copy_expert_num). Default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>const_expert_num</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the number of constant experts. Value range: [0, MAX_int32), where MAX_int32 = 2^31 - 1. Valid IDs of constant experts are [num_experts+zero_expert_num+copy_expert_num, num_experts+zero_expert_num+copy_expert_num+const_expert_num). Default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>expert_shared_type</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the layout type of shared expert devices. Currently, only 0 is supported, which indicates that shared expert devices are placed before MoE expert devices. Default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_num</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the number of shared experts. One shared expert can be replicated and deployed on multiple devices. Value range: [0, 4]. 0 indicates no shared expert. The default value is 1.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>shared_expert_rank_num</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the quantity of shared expert devices. Value range: [0, ep_world_size). A value of 0 indicates no shared expert. If the value is not 0, shared_expert_rank_num%shared_expert_num=0 must be satisfied. Default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_max_dispatch_tokens_per_rank</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the quantity of tokens on each device. When BS differs across ranks, this parameter is the maximum BS size. When BS is the same across ranks, the default value is 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Return Values

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
    <th>Optional/Mandatory</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Mandatory</td>
        <td>Indicates the processed tokens. The data type remains consistent with input `expand_x`, and the data format is ND. Non-contiguous tensors are not supported.</td>
        <td>bfloat16, float16</td>
        <td>(BS, H)</td>
    </tr>
</tbody>
</table>

## Constraints

- This API supports use in inference scenarios.
- This API can be called in single-operator mode and TorchAir graph mode.
- The constraints on the list[Tensor] length, transposition, and non-contiguous tensor support of each tensor parameter are as follows:

    <table style="undefined;table-layout: fixed; width:900px"><colgroup>
    <col style="width: 240px">
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 200px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>list[Tensor] Length</th>
        <th>Whether to Transpose</th>
        <th>Whether Non-contiguous Tensors Are Supported</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>x</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>topk_idx</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>topk_weights</td>
            <td>-</td>
            <td>No</td>
            <td>support</td>
        </tr>
        <tr>
            <td>assist_info_for_combine</td>
            <td>-</td>
            <td>No</td>
            <td>support</td>
        </tr>
        <tr>
            <td>ep_send_counts</td>
            <td>-</td>
            <td>No</td>
            <td>support</td>
        </tr>
        <tr>
            <td>x_active_mask</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>expand_scales</td>
            <td>-</td>
            <td>No</td>
            <td>-</td>
        </tr>
        <tr>
            <td>shared_expert_x</td>
            <td>-</td>
            <td>No</td>
            <td>-</td>
        </tr>
        <tr>
            <td>elastic_info</td>
            <td>-</td>
            <td>No</td>
            <td>-</td>
        </tr>
        <tr>
            <td>ori_x</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>const_expert_alpha_1</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>const_expert_alpha_2</td>
            <td>-</td>
            <td>No</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>const_expert_v</td>
            <td>-</td>
            <td>No</td>
            <td>Yes</td>
        </tr>
    </tbody>
    </table>

- Variables used in the parameter Shape are as follows:
  - A: indicates the maximum number of tokens received by this card. The value range is as follows:
    - For shared experts, A=BS\*ep_world_size\*shared_expert_num/shared_expert_rank_num must be satisfied.
    - For MoE experts, when num_max_dispatch_tokens_per_rank is 0, A >= BS\*ep_world_size \* min(local_expert_num, K) must be satisfied; when num_max_dispatch_tokens_per_rank is not 0, A >= num_max_dispatch_tokens_per_rank \* ep_world_size \* min(local_expert_num, K) must be satisfied.
  - H: indicates the hidden size. The value range is [1024, 8192].
  - BS: indicates the number of tokens to be sent. The value range is 0<BS≤512.
  - K: indicates the number of topK experts to be selected. The value range is 0<K≤16, and 0 < K ≤ num_experts + zero_expert_num + copy_expert_num + const_expert_num must also be satisfied.
  - local_expert_num: indicates the number of experts on this card.
    - For shared expert cards, local_expert_num=1.
    - For MoE expert devices, local_expert_num=num_experts/(ep_world_size-shared_expert_rank_num).
    - Ensure that 0 < local_expert_num \* ep_world_size ≤ 2048.

- `low_latency_dispatch` and `low_latency_combine` must be used together.
- The element values in the Tensor outputs `assist_info_for_combine`, `ep_recv_counts`, and `expand_scales` of `low_latency_dispatch` may differ across different product models, communication algorithms, or versions. Directly pass the above Tensors to the corresponding parameters of `low_latency_combine` when using them. Other model service logic should not depend on these values.
- During the API call, the values of parameters `group_ep`, `ep_world_size`, `num_experts`, `expert_shard_type`, `shared_expert_num`, `shared_expert_rank_num`, and `num_max_dispatch_tokens_per_rank` must remain consistent across all devices. The values of `group_ep`, `ep_world_size`, `expert_shard_type`, and `num_max_dispatch_tokens_per_rank` must also remain consistent across different layers in the network and with the corresponding parameters of [low_latency_dispatch](torchapi_low_latency_dispatch_en.md).
- In this scenario, a single card contains two DIE (short for "die" or "bare die"), so "this card" in the parameter description refers to a single DIE.
- num_experts + zero_expert_num + copy_expert_num + const_expert_num < MAX_int32.
- Related constraints:
  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>:
    - `topk_weights` only supports passing a valid Tensor for weighted aggregation.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - `topk_weights` can be passed a valid tensor, None, or an empty Tesenor. Passing a valid tensor enables the topK expert weight function. Passing None or an empty Tesenor disables the function and directly sums the expert outputs
  <!-- end id8 -->
- HCCL communicator buffer size:

    Before calling this API, check whether the configured HCCL communicator buffer size is appropriate. The unit is MB.
  - When the HCCL\_BUFFSIZE environment variable is used, the buffer size is managed at the communicator granularity and defaults to 200 MB. Each communicator exclusively occupies a memory region of size "2*HCCL\_BUFFSIZE".
  - In the EP communicator: the configured size must be >= 2 and satisfy >= 2 \* (local_expert_num \* max_bs \* ep_world_size \* Align512(Align32(2 \* h) + 64) + (k + shared_expert_num) \* max_bs\* Align512(2 \* h)).
  - where 480Align512(x) = ((x+480-1)/480)\*512,Align512(x) = ((x+512-1)/512)\*512,Align32(x) = ((x+32-1)/32)\*32.
  - The size allocated in the communicator can be calculated by calling the MoeDistributeBuffer.get_low_latency_ccl_buffer_size API.
  <!-- npu="950" id12 -->
  - For <term>Ascend 950PR&950DT products</term>, you can specify the size of the internally allocated communicator buffer using `ccl_buffer_size` when initializing `MoeDistributeBuffer`. This parameter is in MB and can use the value returned by `MoeDistributeBuffer.get_low_latency_ccl_buffer_size`.
  - For <term>Ascend 950PR&950DT products</term>, when both the HCCL\_BUFFSIZE environment variable and a positive `ccl_buffer_size` are configured, the buffer allocated through `ccl_buffer_size` takes precedence. When this parameter is 0, the default communicator buffer is used.
  <!-- end id12 -->

- The "/" in the formulas in this document indicates integer division.

- Communication domain usage constraints:

  - The `low_latency_dispatch` and `low_latency_combine` operators in a model support only the same EP communicator, and no other operators are allowed in this communicator.
  - Nodes in a communicator must be in the same SuperPoD, and cross-SuperPoD is not supported.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call:

  ```python
  import os
  import torch
  import random
  import torch_npu
  import numpy as np
  from torch.multiprocessing import Process
  import torch.distributed as dist
  from torch.distributed import ReduceOp
  from cann_ops_transformer.ops import MoeDistributeBuffer

  # Control mode
  quant_mode = 2  # 2 indicates dynamic quantization
  is_dispatch_scales = True  # scales can be optionally passed in dynamic quantization
  input_dtype = torch.bfloat16  # Output dtype
  server_num = 1
  server_index = 0
  port = 50001
  master_ip = '127.0.0.1'
  dev_num = 16
  world_size = server_num * dev_num
  rank_per_dev = int(world_size / server_num)  # Number of dies per host
  shared_expert_rank_num = 0  # Number of shared experts
  num_experts = 32  # Number of MoE experts
  bs = 8  # Number of tokens
  h = 7168  # Length of each token
  k = 8
  random_seed = 0
  ep_world_size = world_size
  moe_rank_num = ep_world_size - shared_expert_rank_num
  local_moe_expert_num = num_experts // moe_rank_num
  globalBS = bs * ep_world_size
  is_shared = (shared_expert_rank_num > 0)
  is_quant = (quant_mode > 0)
  zero_expert_num = 1
  copy_expert_num = 1
  const_expert_num = 0


  def get_new_group(rank):
      ep_ranks = list(range(ep_world_size))
      ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
      if rank in ep_ranks:
          print(f"rank:{rank} ep_ranks:{ep_ranks}")
      return ep_group


  def get_hcomm_info(rank, comm_group):
      if torch.__version__ > '2.0.1':
          hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
      else:
          hcomm_info = comm_group.get_hccl_comm_name(rank)
      return hcomm_info

  def run_npu_process(rank):
      torch_npu.npu.set_device(rank)
      rank = rank + 16 * server_index
      # Initialize the distributed process group.
      dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
      ep_group = get_new_group(rank)
      ep_hcomm_info = get_hcomm_info(rank, ep_group)

      # Create the input tensor.
      x = torch.randn(bs, h, dtype=input_dtype).npu()
      topk_idx = torch.tensor([[5, 7, 17, 4, 2, 6, 11, 16],
                              [10, 12, 13, 15, 19, 4, 18, 1],
                              [19, 33, 1, 17, 9, 5, 0, 32],
                              [19, 11, 17, 0, 10, 5, 7, 9],
                              [10, 16, 11, 17, 33, 8, 9, 3],
                              [12, 19, 5, 7, 1, 3, 18, 16],
                              [11, 9, 13, 16, 12, 33, 17, 14],
                              [16, 4, 9, 5, 0, 10, 11, 17]], dtype=torch.int32).npu()
      topk_weights = torch.randn(bs, k, dtype=torch.float32).npu()

      const_expert_alpha_1 = None
      const_expert_alpha_2 = None
      const_expert_v = None

      # Construct the communicator buffer.
      distribute_buffer = MoeDistributeBuffer(ep_group)

      # Call dispatch first to perform token quantization and alltoallv dispatch in the EP domain. The assist_info_for_combine and ep_recv_counts required by combine come from the output of this step.
      expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_recv_counts, _ = distribute_buffer.low_latency_dispatch(
          x=x,
          topk_idx=topk_idx,
          shared_expert_num=0,
          shared_expert_rank_num=shared_expert_rank_num,
          num_experts=num_experts,
          quant_mode=quant_mode,
          num_max_dispatch_tokens_per_rank=globalBS,
          zero_expert_num=zero_expert_num,
          copy_expert_num=copy_expert_num,
          const_expert_num=const_expert_num)

      if is_quant:
          expand_x = expand_x.to(input_dtype)

      # Then call combine to return the output of dispatch along the original path for collection: pass in assist_info_for_combine and ep_recv_counts directly as a pair, multiply the results by the weights in topk_weights, and then sum them.
      x = distribute_buffer.low_latency_combine(x=expand_x,
                                              topk_idx=topk_idx,
                                              topk_weights=topk_weights,
                                              assist_info_for_combine=assist_info_for_combine,
                                              ep_send_counts=ep_recv_counts,
                                              shared_expert_num=0,
                                              shared_expert_rank_num=shared_expert_rank_num,
                                              num_experts=num_experts,
                                              num_max_dispatch_tokens_per_rank=globalBS,
                                              ori_x=x,
                                              const_expert_alpha_1=const_expert_alpha_1,
                                              const_expert_alpha_2=const_expert_alpha_2,
                                              const_expert_v=const_expert_v,
                                              zero_expert_num=zero_expert_num,
                                              copy_expert_num=copy_expert_num,
                                              const_expert_num=const_expert_num)
      print(f'rank {rank} npu finished! \n')


  if __name__ == "__main__":
      print(f"bs={bs}")
      print(f"num_max_dispatch_tokens_per_rank={globalBS}")
      print(f"shared_expert_rank_num={shared_expert_rank_num}")
      print(f"num_experts={num_experts}")
      print(f"k={k}")
      print(f"quant_mode={quant_mode}", flush=True)
      print(f"local_moe_expert_num={local_moe_expert_num}", flush=True)
      print(f"ep_world_size={ep_world_size}", flush=True)

      if shared_expert_rank_num > ep_world_size:
          print("shared_expert_rank_num cannot be greater than ep_world_size")
          exit(0)
      if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
          print("ep_world_size must be an integer multiple of shared_expert_rank_num")
          exit(0)
      if num_experts % moe_rank_num != 0:
          print("num_experts must be an integer multiple of moe_rank_num")
          exit(0)
      p_list = []
      for rank in range(rank_per_dev):
          p = Process(target=run_npu_process, args=(rank,))
          p_list.append(p)

      for p in p_list:
          p.start()
      for p in p_list:
          p.join()

      print("run npu success.")
  ```

- Graph mode call:

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import time
    from cann_ops_transformer.ops import MoeDistributeBuffer

    # Control mode
    quant_mode = 2  # 2 indicates dynamic quantization.
    is_dispatch_scales = True  # For dynamic quantization, scales can be optionally passed.
    input_dtype = torch.bfloat16  # Output dtype
    server_num = 1
    server_index = 0
    port = 50001
    master_ip = '127.0.0.1'
    dev_num = 16
    world_size = server_num * dev_num
    rank_per_dev = int(world_size / server_num)  # Number of dies on each host
    shared_expert_rank_num = 0  # Number of shared experts
    num_experts = 32  # Number of moe experts
    bs = 8  # Number of tokens
    h = 7168  # Length of each token
    k = 8
    random_seed = 0
    ep_world_size = world_size
    moe_rank_num = ep_world_size - shared_expert_rank_num
    local_num_experts = num_experts // moe_rank_num
    globalBS = bs * ep_world_size
    is_shared = (shared_expert_rank_num > 0)
    is_quant = (quant_mode > 0)

    zero_expert_num = 1
    copy_expert_num = 1
    const_expert_num = 0

    class MOE_DISTRIBUTE_GRAPH_Model(torch.nn.Module):
        def __init__(self, group_ep):
            super().__init__()
            self.group = group_ep
            self.distribute_buffer = MoeDistributeBuffer(group_ep)
        def forward(self, x, topk_idx, group_ep,
                    ep_world_size, ep_rank_id, expert_shard_type, shared_expert_rank_num, num_experts,
                    scales, quant_mode, num_max_dispatch_tokens_per_rank, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v, zero_expert_num, copy_expert_num, const_expert_num):
            output_dispatch_npu = self.distribute_buffer.low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            expand_x_npu, _, assist_info_for_combine_npu, _, ep_recv_counts_npu, expand_scales = output_dispatch_npu
            if expand_x_npu.dtype == torch.int8:
                expand_x_npu = expand_x_npu.to(input_dtype)
            output_combine_npu = self.distribute_buffer.low_latency_combine(
                x=expand_x_npu,
                topk_idx=topk_idx,
                topk_weights=expert_scales,
                assist_info_for_combine=assist_info_for_combine_npu,
                ep_send_counts=ep_recv_counts_npu,
                shared_expert_num=0,
                shared_expert_rank_num=shared_expert_rank_num,
                num_experts=num_experts,
                num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
                elastic_info=elastic_info,
                ori_x=x,
                const_expert_alpha_1=const_expert_alpha_1,
                const_expert_alpha_2=const_expert_alpha_2,
                const_expert_v=const_expert_v,
                zero_expert_num=zero_expert_num,
                copy_expert_num=copy_expert_num,
                const_expert_num=const_expert_num
            )
            x = output_combine_npu
            x_combine_res = output_combine_npu
            return [x_combine_res, output_combine_npu]


    def gen_const_expert_alpha_1():
        const_expert_alpha_1 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_1


    def gen_const_expert_alpha_2():
        const_expert_alpha_2 = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_alpha_2


    def gen_const_expert_v():
        const_expert_v = torch.empty(size=[const_expert_num, h], dtype=input_dtype).uniform_(-1, 1)
        return const_expert_v


    def get_new_group(rank):
        ep_ranks = list(range(ep_world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            print(f"rank:{rank} ep_ranks:{ep_ranks}")
        return ep_group


    def get_hcomm_info(rank, comm_group):
        if torch.__version__ > '2.0.1':
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(rank)
        return hcomm_info

    def get_dispatch_kwargs_warmup(
        x_warm_up, topk_idx_warm_up,
    ):
        x_warm_up = x_warm_up.to(input_dtype).npu()
        topk_idx_warm_up = topk_idx_warm_up.to(torch.int32).npu()
        return {
            'x': x_warm_up,
            'topk_idx': topk_idx_warm_up,
            'x_active_mask': None,
            'expert_shard_type': 0,
            'shared_expert_num': 0,
            'shared_expert_rank_num': shared_expert_rank_num,
            'num_experts': num_experts,
            'scales': None,
            'quant_mode': 2,
            'num_max_dispatch_tokens_per_rank': 1,
        }


    def run_npu_process(rank):
        torch_npu.npu.set_device(rank)
        rank = rank + 16 * server_index
        dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=world_size,
            init_method=f'tcp://{master_ip}:{port}'
        )
        ep_group = get_new_group(rank)
        ep_hcomm_info = get_hcomm_info(rank, ep_group)
        # Create input tensors.
        x = torch.randn(bs, h, dtype=input_dtype).npu()
        topk_idx = torch.tensor([
            [0, 8, 4, 1, 6, 12, 14, 17],
            [14, 10, 7, 3, 0, 12, 11, 17],
            [12, 0, 5, 11, 19, 4, 6, 18],
            [17, 3, 4, 10, 18, 0, 1, 2],
            [13, 16, 9, 10, 15, 6, 7, 14],
            [17, 15, 14, 8, 16, 18, 3, 12],
            [4, 12, 2, 17, 15, 3, 9, 10],
            [16, 7, 12, 9, 18, 3, 19, 17]
        ], dtype=torch.int32).npu()
        expert_scales = torch.randn(bs, k, dtype=torch.float32).npu()
        scales_shape = (1 + num_experts, h) if shared_expert_rank_num else (num_experts, h)
        if is_dispatch_scales:
            scales = torch.randn(scales_shape, dtype=torch.float32).npu()
        else:
            scales = None
        elastic_info = None
        const_expert_alpha_1 = None
        const_expert_alpha_2 = None
        const_expert_v = None
        model = MOE_DISTRIBUTE_GRAPH_Model(ep_group)
        model = model.npu()
        npu_backend = torchair.get_npu_backend()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        output = model.forward(
            x, topk_idx, ep_hcomm_info, ep_world_size,
            rank, 0, shared_expert_rank_num, num_experts, scales,
            quant_mode, globalBS, expert_scales, elastic_info, const_expert_alpha_1, const_expert_alpha_2, const_expert_v,
            zero_expert_num, copy_expert_num, const_expert_num
        )
        torch.npu.synchronize()
        print(f'rank {rank} npu finished! \n')

        time.sleep(10)


    if __name__ == "__main__":
        print(f"bs={bs}")
        print(f"num_max_dispatch_tokens_per_rank={globalBS}")
        print(f"shared_expert_rank_num={shared_expert_rank_num}")
        print(f"num_experts={num_experts}")
        print(f"k={k}")
        print(f"quant_mode={quant_mode}", flush=True)
        print(f"local_num_experts={local_num_experts}", flush=True)
        print(f"ep_world_size={ep_world_size}", flush=True)

        if shared_expert_rank_num > ep_world_size:
            print("shared_expert_rank_num cannot be greater than ep_world_size")
            exit(0)

        if shared_expert_rank_num > 0 and ep_world_size % shared_expert_rank_num != 0:
            print("ep_world_size must be an integer multiple of shared_expert_rank_num")
            exit(0)

        if num_experts % moe_rank_num != 0:
            print("num_experts must be an integer multiple of moe_rank_num")
            exit(0)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_npu_process, args=(rank,))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```
