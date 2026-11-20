# ElasticBuffer

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:23:35.143Z -->

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

The ElasticBuffer class provides unified distributed communication buffer management:

- The Engram storage API is used for distributed Engram storage management. It supports registering the table of the local rank as communication-reachable storage (in inference mode, data is written to host pinned shared segments; in training mode, the upper-layer pinned table memory is directly mapped, with zero-copy), and fetching Engram data from remote ranks via RDMA. Must be used in conjunction with [get_engram_storage_size_hint](#get_engram_storage_size_hint-static-method).
- The Engram training API is built on the inference API and enables training mode via `with_grad=True`. The forward [engram_fetch](#engram_fetch) saves the communication metadata required for backward propagation (encapsulated as [EngramFetchCtx](#engramfetchctx)) while fetching data. The backward [engram_fetch_grad](#engram_fetch_grad) exchanges gradients in reverse along the forward path based on this metadata, performs sparse accumulation by local entry, and produces sparse gradients for optimizer updates.
- The Dispatch/Combine APIs are used for MoE Expert Parallelism (EP) deployment. Token data is dispatched to the corresponding expert devices via [dispatch](#dispatch), and the expert outputs are then aggregated back into the original sequence along the original routing via [combine](#combine). The required CCL communication buffers are automatically calculated and allocated by ElasticBuffer internally based on operator parameters, with no manual configuration required.

## Function Prototype

```python
class ElasticBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        *,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: Optional[int] = None,
        hidden: Optional[int] = None,
        num_topk: Optional[int] = None,
        with_grad: bool = False,
        explicitly_destroy: bool = False,
    )

    def engram_write(self, storage: torch.Tensor) -> None

    def engram_fetch(self, indices: torch.Tensor) -> Callable

    def engram_fetch_grad(
        self,
        grad_fetched: torch.Tensor,
        fetch_ctx: EngramFetchCtx,
    ) -> Tuple[torch.Tensor, torch.Tensor]

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None

    @staticmethod
    def get_engram_storage_size_hint(
        num_entries: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
               Optional[torch.Tensor], Optional[torch.Tensor], EPHandle]

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        *,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]

    @staticmethod
    def get_moe_ep_ccl_buffer_size(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        topk: int,
    ) -> int

    def destroy(self) -> None
```

## EngramFetchCtx

The save-for-backward context returned by the forward [engram_fetch](#engram_fetch) call when `with_grad=True`. It must be passed as-is to the backward [engram_fetch_grad](#engram_fetch_grad). This object is automatically constructed internally; no manual intervention is required.

| Attribute | Type | shape | Description |
|------|------|-------|------|
| `perm` | `Tensor` | `(num_tokens,)` int32 | Bucket permutation index used to rearrange gradients in backward propagation. |
| `send_counts` | `Tensor` | `(world_size*8,)` int32 | Number of indices sent to each rank (padded to 32-byte alignment, with the first `world_size` elements valid). Send counts for the backward a2a. |
| `recv_counts` | `Tensor` | `(world_size,)` int32 | Number of indices received from each rank. Recv counts for the backward a2a. |
| `recv_local_entry` | `Tensor` | `(R_max,)` int32 | Global indices received through the forward a2a exchange. The first `num_recv` elements are valid, where `R_max = num_max_tokens_per_rank × world_size`. |
| `num_recv` | `Tensor` | `(1,)` int32 | Actual number of received global indices `num_recv`, where `num_recv ≤ R_max`. |

## Member Function Description

### **init**

**Function**: Constructs an ElasticBuffer instance and records the configuration required by Engram and Dispatch/Combine. Engram runtime resources are initialized upon the first call to [engram_write](#engram_write); the Dispatch/Combine communication context is initialized upon the first call to [dispatch](#dispatch) or [combine](#combine).

**Parameters**:

- **group** (`torch.distributed.ProcessGroup`): required parameter. Distributed process group, used for cross-rank communication and synchronization.
- <strong>*</strong>: Variables before it are positional; variables after it are optional parameters that must be assigned using key-value pairs. If not assigned, default values are used.
- **num_cpu_bytes** (`int`): optional parameter, size of the CPU buffer in bytes, used for host pinned storage allocation (takes effect only in inference mode; in training mode, no internal storage is allocated and this parameter takes no effect). The default value is 0, and it must be 2MB-aligned.
- **num_max_tokens_per_rank** (`int`): optional parameter, indicates the maximum quantity of tokens on each device. When using [dispatch](#dispatch) and [combine](#combine), it must be specified together with `hidden` and `num_topk`; in training mode with `with_grad=True`, it must be specified and must be greater than 0.
- **hidden** (`int`): optional parameter, hidden layer size.
- **num_topk** (`int`): optional parameter, indicates the number of topK experts to select.
- **with_grad** (`bool`): optional parameter, indicates whether training is enabled. The default value is `False` (inference). If set to `True`, the forward [engram_fetch](#engram_fetch) additionally saves the communication metadata required for backward propagation (encapsulated as [EngramFetchCtx](#engramfetchctx)), and backward can be performed via [engram_fetch_grad](#engram_fetch_grad).
- **explicitly_destroy** (`bool`): optional parameter, indicates whether to explicitly call [destroy](#destroy) to release resources. The default value is `False`, and `destroy` is automatically called to release resources when the instance is garbage collected. If set to `True`, the caller must explicitly call `destroy` to release resources. It is recommended to set this parameter to `True` and manually call `destroy` to release resources after use, ensuring that resources are released at a deterministic time.

**Output**: No return value. Constructs an ElasticBuffer instance.

### engram_write

**Function**: Registers the Engram table of the local rank as a communication-accessible Engram storage so that other ranks can read the data. Two behaviors apply depending on `with_grad` specified at construction:

- **Inference** (`with_grad=False`): Copies the `storage` data to the internal host pinned shared memory segment.
- **Training** (`with_grad=True`): Directly maps the pinned memory of `storage` as the Engram storage (zero-copy, no data movement). In-place updates to `storage` are visible to subsequent [engram_fetch](#engram_fetch) calls.

After the write is complete in both modes, the communication context tensor, local storage address, communication buffer size, number of ranks, and other information are automatically obtained internally for use by subsequent [engram_fetch](#engram_fetch) calls.

> **Host pinned shared memory segment** (inference mode): Refers to the pinned host memory allocated through `aclrtMallocHost` and mapped through `aclrtHostRegisterV2` to obtain a device-accessible address. This memory segment can be directly accessed by the NPU of this device and read by the NPUs of remote ranks through RDMA, enabling cross-rank zero-copy data sharing. Its size is specified by the `num_cpu_bytes` parameter of the constructor.
> **Training mode memory mapping**: The pinned host memory of `storage` is registered through `aclrtHostRegisterV2` and mapped to a device address, and then registered to the HCCL communicator so that remote ranks can access it. Requirements: `storage` must be pinned memory (e.g., allocated by `pin_memory()` or `torch.empty(..., pin_memory=True)`); plain (non-pinned) CPU memory is not supported under the host registration path of the current driver, and `aclrtHostGetDevicePointer` fails on the first invocation; `storage` must remain alive with an unchanged address throughout the lifetime of this ElasticBuffer. Repeated calls to this API must pass the same `storage` (it can serve as a cross-rank synchronization point after in-place table updates); passing any other address raises an error.

**Formula**:

In inference mode, the data of `storage` is copied byte by byte to the starting position of the host pinned shared memory segment through `memcpy_s`:

$$HostPinnedBuf[0 : storage.nbytes()] \leftarrow storage.data()$$

In training mode, no copy is performed; `storage` is directly used as the Engram table of the local rank:

$$EngramTable[rank\_id] \equiv storage$$

In both modes, two `Barrier` operations are performed before and after the write, ensuring that writes from all ranks are complete and visible to each other:

$$Barrier \rightarrow write(storage) \rightarrow Barrier$$

Here, `storage.nbytes() = num_entries × hidden × dtype_size`, and `storage.nbytes() ≤ num_cpu_bytes` must be satisfied.

**Function Prototype**:

```python
ElasticBuffer.engram_write(storage) -> None
```

**Input parameters**:

- **storage** (`Tensor`): Required parameter. CPU tensor to be written, with shape `(num_entries, hidden)`, indicating `num_entries` entries, each of dimension `hidden`. Must be 2D and contiguous, and `hidden` must be greater than 0. Supported data types are `bfloat16`, `float16`, and `float32`.

**Output description**: No return value. In inference mode, data is written to host pinned memory; in training mode, the `storage` memory is registered as Engram storage.

### engram_fetch

**Function**: Fetches Engram data from the corresponding rank based on the input global indices. The behavior in inference and training modes is determined by `with_grad` specified at construction:

- **Inference** (`with_grad=False`): Fetches data from the host pinned shared memory of remote ranks through RDMA one-sided read. The API adopts an asynchronous design: it returns a callable immediately after the call. When the callable is executed, it blocks until the RDMA transfer is complete and returns the result tensor.
- **Training** (`with_grad=True`): Fetches data from remote ranks through all-to-all communication (bucket sort → a2a exchange of indices → local gather → a2a exchange of data → restore). It returns a callable. When the callable is executed, it waits for the transfer to complete and returns the `(fetched, fetch_ctx)` tuple, where `fetch_ctx` is the save-for-backward context [EngramFetchCtx](#engramfetchctx).

**Formula**:

Each global index is mapped to a target rank and a local entry index as follows:

$$rank\_id = \lfloor global\_idx / num\_entries \rfloor$$

$$local\_idx = global\_idx \bmod num\_entries$$

$$fetched[i] = EngramTable[rank\_id][local\_idx]$$

The variables are described as follows:

- $global\_idx$: Element value of the input index tensor `indices`, with the value range $[0, world\_size \times num\_entries)$.
- $num\_entries$: Number of entries written by each rank during [engram_write](#engram_write), that is, `storage.size(0)`.
- $world\_size$: Total number of ranks in the communicator.
- $rank\_id$: target rank ID that $global\_idx$ is mapped to, with the value range $[0, world\_size)$. When $rank\_id$ is this rank, data is read directly from the local host pinned memory; when $rank\_id$ is a remote rank, data is read across devices via RDMA.
- $local\_idx$: local entry index within the target rank, with the value range $[0, num\_entries)$.
- $EngramTable[rank\_id]$: Engram table data provided by the target rank via [engram_write](#engram_write) (host pinned shared memory in inference mode, directly mapped upper-layer table memory in training mode), with the shape `(num_entries, hidden)`,
where $EngramTable[rank\_id][local\_idx]$ indicates the $local\_idx$-th entry in it.
- $fetched[i]$: Engram data corresponding to the $i$-th token in the output tensor, where $i \in [0, num\_tokens)$ and $num\_tokens$ is the length of `indices`.

The index mapping in **training** (`with_grad=True`) is the same as that in inference (see formulas $rank\_id$ and $local\_idx$ above). The difference is that cross-rank data fetching is performed through all-to-all communication, and the backward context [EngramFetchCtx](#engramfetchctx) is saved. The forward process is:

1. Bucket sorting: bucket `indices` by $rank\_id_i = \lfloor indices[i] / num\_entries \rfloor$, and record the bucket permutation order $perm$ and the per-rank send volume $send\_counts$:

$$send\_counts[r] = |\{\,i \mid rank\_id_i = r\,\}|, \quad r \in [0, world\_size)$$

2. a2a exchange of indices: each rank sends the sorted indices (global indices) to the target rank and receives the global indices requested by other ranks, obtaining $recv\_local\_entry$ (all of which are global indices mapped to this rank, i.e., `global_idx / num_entries == rank_id`), $recv\_counts$, and the total received volume $num\_recv = \sum_{r=0}^{world\_size-1} recv\_counts[r]$.

3. Local gather: convert the global indices in $recv\_local\_entry$ according to $local\_idx = recv\_local\_entry[j] \bmod num\_entries$, and then read from the local $EngramTable[rank\_id]$:

$$local\_data[j] = EngramTable[rank\_id][recv\_local\_entry[j] \bmod num\_entries], \quad j \in [0, num\_recv)$$

4. a2a exchange of data: send $local\_data$ back along the original request path to obtain $recv\_data[i]$, where $i \in [0, num\_tokens)$.

5. Restore: reorder the tokens back to the original order according to $perm$:

$$fetched[perm[i]] = recv\_data[i], \quad i \in [0, num\_tokens)$$

6. Save the backward context $fetch\_ctx$ (i.e., [EngramFetchCtx](#engramfetchctx)), which contains five fields: $perm$, $send\_counts$, $recv\_counts$, $recv\_local\_entry$, and $num\_recv$, for use by the backward [engram_fetch_grad](#engram_fetch_grad).

**Function Prototype**:

```python
ElasticBuffer.engram_fetch(indices) -> Callable
```

**Input Parameters**:

- **indices** (`Tensor`): Required parameter. NPU tensor of query indices with shape `(num_tokens,)`, indicating the global indices of the entries to fetch (`len(indices)` must be the same on all devices in training use cases and can differ in inference use cases). The supported data type is `int32`, and the data format is $ND$. The value range of elements must be within `[0, world_size × num_entries)`. If the value of an element exceeds this range, the data at the corresponding position in the return value is 0.

**Output Description**:

- **wait_callable** (`Callable`): Returns a callable that returns different results depending on the mode when called.

**Inference mode**: calling `wait_callable()` returns:

- **fetched** (`Tensor`): NPU tensor with shape `(num_tokens, hidden)`. Its data type is the same as `storage.dtype` of [engram_write](#engram_write), and its data format is $ND$.

**Training** call to `wait_callable()` returns:

- **fetched** (`Tensor`): NPU tensor with shape `(num_tokens, hidden)`. The data type is the same as `storage.dtype`. The data format is $ND$.
- **fetch_ctx** (`EngramFetchCtx`): save-for-backward context that contains 5 communication metadata tensors required for backward propagation (see [EngramFetchCtx](#engramfetchctx)). Must be passed as is to [engram_fetch_grad](#engram_fetch_grad).

### engram_fetch_grad

**Function**: Training backward API. Must be used in conjunction with [engram_fetch](#engram_fetch) in training mode. Based on the [EngramFetchCtx](#engramfetchctx) saved in the forward pass, it reversely exchanges `grad_fetched` along the forward path and performs sparse accumulation by local entry, producing sparse gradients for optimizer updates.

**Formula**:

The backward pass is executed in reverse order along the forward path, $fetch\_ctx$ where all fields ($perm$, $send\_counts$, $recv\_counts$, $recv\_local\_entry$, $num\_recv$) come from the [EngramFetchCtx](#engramfetchctx) returned by the forward [engram_fetch](#engram_fetch) in training mode. The steps are as follows:

1. Unsort: rearrange $grad\_fetched$ back to the bucket sort order according to $perm$:

$$grad\_sorted[i] = grad\_fetched[perm[i]], \quad i \in [0, num\_tokens)$$

2. Reverse a2a exchange: exchange $grad\_sorted$ via reverse a2a ($send\_counts$ is the send size, $recv\_counts$ is the recv size), and each rank receives the gradients of the corresponding $recv\_local\_entry$:

$$grad\_recv[j] \leftrightarrow recv\_local\_entry[j], \quad j \in [0, num\_recv)$$

3. Unique: convert $recv\_local\_entry$ according to $local\_idx = recv\_local\_entry[j] \bmod num\_entries$ and deduplicate it to obtain the local entry index set of this rank:

$$unique\_local\_entry = \text{unique}(\{\,recv\_local\_entry[j] \bmod num\_entries \mid j \in [0, num\_recv)\,\})$$

4. ScatterAdd: for each unique entry, accumulate all gradients mapped to the entry in FP32, and then convert back to the output dtype:

$$grad\_unique[k] = \text{Cast}_{dtype}\!\left(\sum_{\substack{j \in [0,num\_recv) \\ recv\_local\_entry[j] \bmod num\_entries = unique\_local\_entry[k]}} \text{Cast}_{fp32}(grad\_recv[j])\right)$$

Here, $K$ is the number of local entries after deduplication (determined at runtime; only the first $K$ rows of the output are valid), $num\_tokens$ is the length of `indices`, and $num\_entries$ is the number of entries written by each rank during [engram_write](#engram_write). `grad_unique` and `unique_local_entry` can be directly used for sparse optimizer updates.

**Function Prototype**:

```python
ElasticBuffer.engram_fetch_grad(grad_fetched, fetch_ctx) -> (Tensor, Tensor)
```

**Input parameters**:

- **grad_fetched** (`Tensor`): required parameter, the gradient of the forward `fetched`. NPU tensor with shape `(num_tokens, hidden)`. Supported data types: `bfloat16`, `float16`, `float32`. Data format: $ND$.
- **fetch_ctx** (`EngramFetchCtx`): required parameter, the context object returned by the forward [engram_fetch](#engram_fetch) in training mode, containing five fields: `perm`, `send_counts`, `recv_counts`, `recv_local_entry`, and `num_recv`.

**Output description**:

- **grad_unique** (`Tensor`): NPU tensor with shape `(K, hidden)`. The data type is the same as `grad_fetched` (accumulated internally in FP32 and then converted back to the output dtype). Data format: $ND$. Here `K` is the number of local entries after deduplication (determined at runtime), and the first `K` rows are valid.
- **unique_local_entry** (`Tensor`): NPU tensor with shape `(K,)`. Data type: `int32`. Data format: $ND$. Indicates the local entry index corresponding to each row of `grad_unique`.

### barrier

**Function**: Cross-device synchronization.

**Function Prototype**:

```python
ElasticBuffer.barrier(use_comm_stream=True, with_cpu_sync=False) -> None
```

**Input parameters**:

- **use_comm_stream** (`bool`): optional parameter. Indicates whether to use a dedicated communication stream to execute the barrier. The default value is `True`, which uses the dedicated communication stream and synchronizes the computation stream and the communication stream through events before and after the barrier; if set to `False`, the current computation stream is used.
- **with_cpu_sync** (`bool`): optional parameter. Indicates whether to synchronize the device before and after the barrier. The default value is `False`. If set to `True`, `aclrtSynchronizeDevice` is called once before and once after the barrier to ensure that device-side operations are completed.

**Output description**: No return value.

### get_engram_storage_size_hint (static method)

**Function**: Calculates the CPU buffer size required for Engram storage.

**Formula**:

```text
dtype_size = elementSize(dtype)              # e.g., bfloat16=2, float16=2, float32=4
hidden_size_bytes = hidden × dtype_size
num_bytes_per_entry = Align32(hidden_size_bytes)
num_cpu_bytes = Align2MB(num_bytes_per_entry × num_entries)
```

Where `AlignX(value) = ((value + X - 1) / X) × X`, and `/` indicates integer division. `num_bytes_per_entry` is aligned to 32 bytes, and the final result is aligned to 2MB.

**Function Prototype**:

```python
ElasticBuffer.get_engram_storage_size_hint(
    num_entries, hidden, dtype=torch.bfloat16
) -> int
```

**Input Parameters**:

- **num_entries** (`int`): Required parameter, number of entries in the Engram storage, which must be non-negative.
- **hidden** (`int`): Required parameter, hidden layer size of each entry, which must be greater than 0.
- **dtype** (`torch.dtype`): Optional parameter, data type, default value is `torch.bfloat16`. Used only here to calculate the number of bytes by dtype.

**Output Description**:

- **num_cpu_bytes** (`int`): CPU buffer size (bytes), used as the local storage area for engram_write (inference mode only), aligned to 2MB.

### dispatch

**Function**: Must be used in conjunction with [combine](#combine) to implement token dispatch in the Expert Parallelism (EP) deployment of MoE. Based on the topK expert index of each token, this API dispatches token data to the corresponding expert device through alltoallv communication in the EP domain.

- Supports cached mode, that is, the handle of the first dispatch can be reused in the second dispatch, skipping the slot allocation phase to achieve lower latency.
- Supports specifying `num_max_tokens_per_rank` to control the upper limit of the receive buffer size.
- Graph mode is not supported in the current version.

**Formula**:

$$alltoall\_x\_out = alltoallv(x)$$

$$dst\_buffer\_slot\_idx = SlotAssignment(topk\_idx)$$

$$recv\_src\_metadata = MetadataAssignment(alltoall\_x\_out, topk\_idx)$$

**Function Prototype**:

```python
ElasticBuffer.dispatch(
    x,
    *,
    topk_idx=None,
    topk_weights=None,
    handle=None,
    num_experts=None,
    num_max_tokens_per_rank=None,
    expert_alignment=None,
    do_cpu_sync=None,
) -> (Tensor | Tuple[Tensor, Tensor], Tensor | None, Tensor | None, EPHandle)
```

**Input Parameters**:

- **x** (`Tensor` or `Tuple[Tensor, Tensor]`): Required parameter, indicating the token data used for computation, which needs to be sent to other devices based on `topk_idx`. If a tuple is passed in, the first tensor is the token data and the second tensor is the scales. The token must be a 2D tensor with shape `(BS, H)`, data type `bfloat16`, `float16`, `float8_e5m2`, or `float8_e4m3fn`, and data format $ND$. The scales must be a 2D tensor with shape `(BS, H1)`, data type `float32` or `float8_e8m0`, and data format $ND$, and are passed in only when the token data type is `float8_e5m2` or `float8_e4m3fn`.
- <strong>*</strong>: Variables before it are positional; variables after it are optional parameters that must be assigned as key-value pairs. If not assigned, default values are used.
- **topk_idx** (`Tensor`): Optional parameter, indicating the topK expert indexes of each token, which determine the experts to which each token is sent. It must be a 2D tensor with shape `(BS, K)`, data type `int32`, and data format $ND$. The value range of the tensor values is `[0, num_experts)`. This parameter is required in non-cached mode and must be `None` in cached mode.
- **topk_weights** (`Tensor`): optional parameter, indicates the topK expert weights corresponding to each token. Must be a 2D tensor with shape `(BS, K)`, supporting the data type `float32` and data format $ND$. Optional in non-cached mode, and must be `None` in cached mode.
- **handle** (`EPHandle`): optional parameter, indicates the handle object returned by the previous dispatch, used in cached mode. When handle is passed in, `topk_idx` and `topk_weights` must be `None`, and `do_cpu_sync` must be `False`. The default value is `None`, indicating non-cached mode.
- **num_experts** (`int`): optional parameter, the total number of MoE experts. The value range is `[2, 2048]`, and `num_experts % ep_world_size = 0` must be satisfied. It is a required parameter in non-cached mode; in cached mode, the value in `handle` is used and the input parameter is ignored.
- **num_max_tokens_per_rank** (`int`): optional parameter, indicates the maximum quantity of tokens on each device. If passed in, it overrides the value set during ElasticBuffer initialization. By default, the value passed in during initialization is used. In cached mode, the value in `handle` is used and the input parameter is ignored.
- **expert_alignment** (`int`): optional parameter, indicates the expert alignment value. Currently, only the value 1 is supported; in cached mode, the value in `handle` is used.
- **do_cpu_sync** (`bool`): optional parameter, indicates whether to wait for CPU synchronization. The default value is `True` in non-cached mode; it cannot be `True` in cached mode.

**Output description**:

- **recv_x** (`Tensor` or `Tuple[Tensor, Tensor]`): indicates the token data received by this card. The tensor shape is `(A, H)`, the data type is the same as `x`, and the data format is $ND$. After being processed by the expert network, it is used as the `x` input of [combine](#combine). When the `x` input contains scales, the output is `(recv_x, recv_scales)`.
- **recv_topk_idx** (`None`): always `None` in the current version. This is a reserved parameter.
- **recv_topk_weights** (`Tensor | None`): indicates the topK weights received by this card. Returned only when the input `topk_weights` is not `None`; otherwise, it is `None`. Must be a 1D tensor with shape `(A,)`, data type `float32`, and data format $ND$. Used as the `topk_weights` input of [combine](#combine).
- **handle** (`EPHandle`): Handle object generated in the dispatch phase, containing slot indices, metadata, and other information. It must be passed to [combine](#combine). The attributes of handle are as follows:
  - **dst_buffer_slot_idx** (`Tensor`): Sender-side index, with data type `int32`.
  - **recv_src_metadata** (`Tensor`): Received metadata, with shape `(A_alloc, 4)` and data type `int32`; only the first `sum(num_recv_tokens_per_expert)` rows are valid.
  - **recv_rank_offsets** (`Tensor`): Exclusive prefix sum of the number of metadata rows corresponding to each source EP rank, with shape `(ep_world_size + 1,)` and data type `int32`; the metadata range of rank `r` is `[recv_rank_offsets[r], recv_rank_offsets[r + 1])`, and the last element equals the number of valid metadata rows.
  - **recv_metadata_buffer** (`Tensor`): Internal buffer for received metadata, with data type `int32`, generated by dispatch and maintained as part of the handle. The base address must be 512-byte aligned. Callers should pass the complete handle unchanged and should not split or replace this buffer separately. Upgrade the operator package and the Torch extension/wheel together and refresh the JIT cache; re-dispatch or explicitly repack legacy handles. When copying to another device, copy the complete buffer first and then rebuild the views from the capacity.
  - **num_recv_tokens_per_rank** (`Tensor`): Number of tokens received by each device, with shape `(ep_world_size,)` and data type `int32`.
  - **num_recv_tokens_per_expert** (`Tensor`): Number of tokens received by each local expert, with shape `(num_local_experts,)` and data type `int64`.
  - **num_experts** (`int`): Total number of experts.
  - **expert_alignment** (`int`): Expert alignment value.
  - **num_max_tokens_per_rank** (`int`): Maximum quantity of tokens per device.
  - **topk_idx** (`Tensor`): Original topK indices.

### combine

**Function**: Must be used in conjunction with [dispatch](#dispatch). It returns data along the path used by the dispatch operator for data collection. Based on the metadata recorded in the dispatch phase, this API combines the token data processed by experts into the original sequence order through reverse routing and weighted aggregation.

- Supports two modes: weighted aggregation with `topk_weights` and pure accumulation.
- The bias parameter is not supported in the current version.
- Graph mode is not supported in the current version.

**Formula**:

When `topk_weights` is provided:

$$combined\_x_i = \sum_{k=0}^{K-1} topk\_weights_{i,k} \times x_{slot(i,k)}$$

When `topk_weights` is not provided (pure accumulation):

$$combined\_x_i = \sum_{k=0}^{K-1} x_{slot(i,k)}$$

**Function Prototype**:

```python
ElasticBuffer.combine(x, handle, *, topk_weights=None, bias=None) -> (Tensor, Tensor?)
```

**Input parameters**:

- **x** (`Tensor`): Required parameter, indicates the token data after expert computation, that is, the result of processing the `recv_x` output of [dispatch](#dispatch) through the expert network. Must be a 2-dimensional tensor with shape `(A, H)`. The supported data types are `bfloat16` and `float16`, and the data format is $ND$.
- **handle** (`EPHandle`): Required parameter. Pass the handle object returned by the corresponding [dispatch](#dispatch) call directly. For handle attributes, see the output description of [dispatch](#dispatch).
- <strong>*</strong>: Variables before it are positional; variables after it are optional parameters, which must be assigned as key-value pairs. If not assigned, the default value is used.
- **topk_weights** (`Tensor`): Optional parameter, indicates the topK expert weight corresponding to each token, used for weighted aggregation. Must be a 1-dimensional tensor with shape `(A,)`. The supported data type is `float32`, and the data format is $ND$, corresponding to the `recv_topk_weights` output of [dispatch](#dispatch). If not provided, pure accumulation combine is performed, and the output `combined_topk_weights` is `None`.
- **bias** (`Tensor` or `Tuple[Tensor, Tensor]`): Optional parameter. bias is not supported in the current version; pass `None`. Reserved for supporting a single bias tensor or the dual-tensor mode with `bias_0` and `bias_1`.

**Output description**:

- **combined_x** (`Tensor`): Indicates the token data after combine, restored to the original sequence order. Must be a 2-dimensional tensor with shape `(BS, H)`, with the same data type as `x` (`bfloat16` or `float16`) and the $ND$ data format. Non-contiguous tensors are not supported.
- **combined_topk_weights** (`Tensor | None`): Indicates the topK expert weight after combine. When the `topk_weights` input is not `None`, must be a 2-dimensional tensor with shape `(BS, K)`, with the `float32` data type and the $ND$ data format; when the `topk_weights` input is `None`, `None` is returned.

### get_moe_ep_ccl_buffer_size (static method)

**Function**: Returns the default CCL communication buffer size (unit: MB). Currently, the communication buffer required by dispatch and combine no longer needs to be manually configured through the `HCCL_BUFFSIZE` environment variable; therefore, this API does not need to be called.

**Function Prototype**:

```python
ElasticBuffer.get_moe_ep_ccl_buffer_size(world_size, num_max_tokens_per_rank, hidden, num_experts, topk) -> int
```

**Input Parameters**:

- **world_size** (`int`): Required parameter. Indicates the size of the EP communicator (that is, the number of devices participating in EP communication). Value range: `[2, 1024]`.
- **num_max_tokens_per_rank** (`int`): Required parameter. Indicates the maximum quantity of tokens on each device.
- **hidden** (`int`): Required parameter. Indicates the hidden layer size. Value range: `(0, 8192]`.
- **num_experts** (`int`): Required parameter. Total number of MoE experts. Value range: `[2, 2048]`, and `num_experts % ep_world_size = 0` must be satisfied.
- **topk** (`int`): Required parameter. Indicates that topK experts are selected. Value range: `[1, 32]`.

**Output Description**:

- **ccl_buffer_size** (`int`): Default CCL communication buffer size, fixed at 200, in MB. The communication buffer is currently automatically calculated and allocated inside ElasticBuffer based on operator parameters, so this value has no reference significance.

### destroy

**Function**: Releases the resources held by the ElasticBuffer instance, including the Dispatch/Combine communication context. Note: The host pinned memory for Engram communication (including the self-created host pinned memory in inference mode and the registered mapping of the caller's zero-copy storage in training mode) is registered into the HCCL engine context at creation, and no deregistration API is available. After destroy, this part of memory is retained by the process-level shared pool by communicator and released uniformly when the process exits; destroy does not release this part of memory (the caller's storage memory itself is never released). When an ElasticBuffer is destroyed and then recreated on the same communicator, the shared pool memory is reused directly. Reuse constraints: the capacity of a rebuilt self-created buffer must not exceed the value at initial creation; when an external zero-copy buffer is rebuilt, its address and size must be the same as those registered at initial creation; otherwise, the API reports an error, in which case a new communicator is required. In training mode, the default HCCL communication buffer is managed by the framework, and no manual intervention is required for its release. When constructed with `explicitly_destroy=False` (default), this method is automatically called when the instance is garbage collected; when `explicitly_destroy=True`, it must be explicitly called by the caller.

**Input parameters**: None.

**Output**: No return value. The resources of the instance are released.

## Constraints

- **Parameter alignment constraints**:
  - `num_cpu_bytes` must be 2MB-aligned (that is, divisible by `2 × 1024 × 1024`).
  - `hidden` must be greater than 0.
  - The return value of [get_engram_storage_size_hint](#get_engram_storage_size_hint-static-method) is automatically 2MB-aligned.

- **Engram dimension constraints**:
  - `storage` must be a 2D tensor.
  - `indices` must be a 1D tensor.

- **Engram dtype constraints**:
  - `storage.dtype` supports only `bfloat16`, `float16`, and `float32`.
  - `indices.dtype` must be `int32`.

- **Engram device constraints**:
  - `storage` must be on the CPU.
  - `indices` must be on the NPU.

- **Engram call order constraints**:
  - [engram_write](#engram_write) must be called at least once before [engram_fetch](#engram_fetch) is called.
  - Concurrent [engram_fetch](#engram_fetch) calls are not allowed on the same ElasticBuffer instance (the callable of the previous fetch must complete execution).
  - In training mode, call [engram_fetch](#engram_fetch) (training mode) first to obtain `fetch_ctx` before calling [engram_fetch_grad](#engram_fetch_grad).

- **Engram Training Mode Constraints**:
  - If `with_grad=True`, `num_max_tokens_per_rank` must be a positive integer.
  - If `with_grad=True`, the `fetch_ctx` returned by [engram_fetch](#engram_fetch) is a local variable; multiple fetches do not overwrite each other (each call returns an independent ctx).
  - The `grad_fetched` shape of [engram_fetch_grad](#engram_fetch_grad) must match the forward `fetched`.
  - In training mode, the communication buffer uses the HCCL default buffer (size controlled by the `HCCL_BUFFSIZE` environment variable; default value is 200 MB) and is managed by the framework; no manual intervention required for allocation or release.
  - In training mode, [engram_write](#engram_write) directly maps the `storage` pinned memory (zero-copy): `storage` must be **pinned memory** (allocated by `pin_memory()`); ordinary (non-pinned) CPU memory is not supported. `storage` must be non-empty and remain alive with an unchanged address throughout the ElasticBuffer lifecycle; repeated calls must pass the same `storage`.

- **Engram Numerical Constraints**:
  - `num_cpu_bytes` and `num_entries` must be non-negative.
  - `hidden` must be greater than 0.
  - `storage.nbytes()` must be less than or equal to `num_cpu_bytes`. `storage.nbytes()` indicates the actual number of bytes occupied by the tensor, that is, `num_entries × hidden × dtype_size` (where
  `dtype_size` is the byte size of a single element, for example, 2 bytes for `bfloat16` and 4 bytes for `float32`).
  - The total number of global entries `world_size × num_entries` must be less than 2^31 (the maximum value of int32) to ensure the indices do not overflow.

- **Dispatch/Combine constraints**:
  - [dispatch](#dispatch) and [combine](#combine) must be used in conjunction.
  - The values of the `num_experts` and `num_max_tokens_per_rank` parameters used when calling the APIs must be consistent across all devices, and the corresponding parameters of [dispatch](#dispatch) and [combine](#combine) must also be consistent.
  - The bias parameter is not supported in the current version. `bias` must be set to `None`.
  - In cached mode, `topk_idx` and `topk_weights` must be `None`, and `do_cpu_sync` must be `False`; in non-cached mode, `topk_idx` is a required parameter, and `topk_weights` is an optional parameter.

- **Dispatch/Combine shape variable descriptions**:
  - `A`: indicates the maximum number of tokens received by this card. `A = ep_world_size * num_max_tokens_per_rank * MIN(K, num_local_experts)`.
  - `H`: indicates the hidden layer size. The value range is `(0, 8192]`.
  - `BS`: indicates the batch sequence size, that is, the number of tokens on this card.
  - `K`: indicates the number of topK experts selected. The value range is `1 ≤ K ≤ min(32, num_experts)`.
  - `num_local_experts`: indicates the number of experts on this card, `num_local_experts = num_experts / ep_world_size`, where the value range of `num_experts` is `[2, 2048]` and `num_experts` must be divisible by `ep_world_size`, that is, `0 < num_local_experts * ep_world_size ≤ 2048`.

- **HCCL communicator buffer size**:
  - The CCL communication buffers required by [dispatch](#dispatch) and [combine](#combine) are automatically calculated and allocated on demand within ElasticBuffer, so the `HCCL_BUFFSIZE` environment variable does not need to be configured.
  - a2a send and receive in Engram training mode (`with_grad=True`) still uses the default HCCL communication buffer. The `HCCL_BUFFSIZE` environment variable only takes effect for this path (the default value is 200MB if the variable is not configured).

- **Communicator constraints**:
  - The `world_size` range of the Engram communicator is `[2, 1024]`, and multi-card distributed use cases are supported.
  - The [dispatch](#dispatch) and [combine](#combine) operators in a model only support the same EP communicator, and no other operators are allowed in the communicator.

- **Special Case Handling**:
  - Supports `num_entries = 0`.
  - Supports `num_tokens = 0`.
  - Binary consistency: engram_write and engram_fetch perform pure data movement throughout. The output must be byte-for-byte identical to the source, with no tolerance.

## Example

### Engram Single-Operator Mode Invocation (Multi-Card Distributed)

```python
import os
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
from cann_ops_transformer import ElasticBuffer

num_entries = 10000
hidden = 4096
dtype = torch.bfloat16
world_size = 2


def set_device(rank):
    torch_npu.npu.set_device(rank)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank, world_size):
    print(f"[INFO] device_{rank} create HCCL communication link")
    master_ip = "127.0.0.1"
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50001",
    )
    print(f"device_{rank} init_process_group success")

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    return group


def run_elastic_buffer(queue, rank, world_size, storage, indices):
    print(f"{os.getpid()=}{rank=}")
    set_device(rank)
    group = init_hccl_comm(rank, world_size)

    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries, hidden, dtype
    )
    print(f"[INFO] device_{rank} num_cpu_bytes={num_cpu_bytes}")

    buffer = ElasticBuffer(group, num_cpu_bytes=num_cpu_bytes, explicitly_destroy=True)

    print(f"[INFO] device_{rank} run engram_write")
    buffer.engram_write(storage)

    print(f"[INFO] device_{rank} run engram_fetch")
    indices_npu = indices.npu()
    wait_callable = buffer.engram_fetch(indices_npu)
    fetched = wait_callable()

    torch.npu.synchronize()
    print(f"[INFO] device_{rank} fetched shape: {fetched.shape}")
    buffer.destroy()
    dist.destroy_process_group()

    queue.put([rank, fetched.cpu()])


if __name__ == "__main__":
    storage = torch.randn(num_entries, hidden, dtype=dtype)
    indices = torch.randint(
        0,
        num_entries * world_size,
        (1000,),
        dtype=torch.int32,
    )

    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)

    proc_list = []
    for rank in range(world_size):
        p = Process(
            target=run_elastic_buffer,
            args=(result_queue, rank, world_size, storage.clone(), indices.clone()),
        )
        p.start()
        proc_list.append(p)

    results = [None] * world_size
    for _ in range(world_size):
        rank_id, fetched = result_queue.get()
        results[rank_id] = fetched
        print(f"[INFO] rank_{rank_id} result collected")

    for p in proc_list:
        p.join()

    if all(result is not None for result in results):
        print("All ranks finished successfully")
        for rank, result in enumerate(results):
            print(f"Rank {rank} fetched shape: {result.shape}")
    else:
        print("[ERROR] Task failed. Please check the detailed error logs.")
        exit(1)
```

### Engram Training Mode Invocation (Multi-Card Distributed)

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from cann_ops_transformer import ElasticBuffer

num_entries = 4
hidden = 128
dtype = torch.bfloat16
world_size = 2
num_tokens = 6


def set_device(rank):
    torch_npu.npu.set_device(rank)


def init_hccl_comm(rank, world_size):
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:50003",
    )
    return dist.new_group(backend="hccl", ranks=list(range(world_size)))


def train_worker(rank):
    set_device(rank)
    group = init_hccl_comm(rank, world_size)
    device = f"npu:{rank}"

    # ==================== Generate embedding and indices ====================
    torch.manual_seed(42 + rank)
    storage = torch.randn(num_entries, hidden, dtype=dtype, pin_memory=True)
    total_entries = num_entries * world_size

    torch.manual_seed(100 + rank)
    indices = torch.randint(0, total_entries, (num_tokens,), dtype=torch.int32)

    # ==================== Create ElasticBuffer (with_grad=True) ====================
    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries, hidden, dtype
    )
    buffer = ElasticBuffer(
        group=group,
        num_cpu_bytes=num_cpu_bytes,
        num_max_tokens_per_rank=num_tokens,
        with_grad=True,
        explicitly_destroy=True,
    )

    # ==================== Write to the storage ====================
    buffer.engram_write(storage)
    dist.barrier()
    torch.npu.synchronize()

    # ==================== Training forward pass ====================
    indices_npu = indices.to(device)
    fetched, fetch_ctx = buffer.engram_fetch(indices_npu)()
    print(f"[rank {rank}] forward: fetched shape={fetched.shape}")

    # ==================== Training backward pass ====================
    # Assume grad_fetched is obtained from loss.backward().
    grad_fetched = fetched.clone()
    grad_unique, unique_local_entry = buffer.engram_fetch_grad(grad_fetched, fetch_ctx)
    print(f"[rank {rank}] backward: grad_unique shape={grad_unique.shape}, dtype={grad_unique.dtype}")
    print(f"[rank {rank}] backward: unique_local_entry={unique_local_entry.cpu().tolist()}")

    # ==================== Optimizer update (sparse) ====================
    # storage[unique_local_entry] -= lr * grad_unique
    # Note: grad_unique and grad_fetched have the same dtype; perform type conversion before updating the storage.

    # ==================== Cleanup ====================
    dist.barrier()
    torch.npu.synchronize()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver", force=True)
    proc_list = []
    for rank in range(world_size):
        proc = Process(target=train_worker, args=(rank,))
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    print("All processes finished.")
```

### Dispatch/Combine Single-Operator Mode Invocation (Multi-Card Distributed)

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from cann_ops_transformer import ElasticBuffer

master_ip = "127.0.0.1"
world_size = 2
num_experts = world_size * 4
num_max_tokens_per_rank = 128
hidden = 4096
top_k = 4


def run_dispatch_combine(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50002",
    )

    group = dist.new_group(backend="hccl")
    buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        explicitly_destroy=True,
    )

    num_tokens = 64
    x = torch.randn(
        num_tokens,
        hidden,
        dtype=torch.bfloat16,
        device=f"npu:{rank}",
    )
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=f"npu:{rank}",
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.float32,
        device=f"npu:{rank}",
    )

    recv_x, _, recv_topk_weights, handle = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
    )
    torch.npu.synchronize()

    expert_output = recv_x
    combined_x, combined_topk_weights = buffer.combine(
        expert_output,
        handle,
        topk_weights=recv_topk_weights,
    )

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    assert combined_x.shape == (num_tokens, hidden)
    print(f"[rank {rank}] dispatch_combine PASS")

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    processes = []
    for rank in range(world_size):
        p = Process(target=run_dispatch_combine, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished.")
```
