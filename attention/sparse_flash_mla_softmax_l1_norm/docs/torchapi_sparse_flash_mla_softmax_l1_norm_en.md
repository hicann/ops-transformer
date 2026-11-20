# sparse_flash_mla_softmax_l1_norm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:28:32.151Z -->

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

- Description: Computes the Softmax L1Norm result of the `SparseFlashMla` attention. Supports the Compressed Attention and Sparse Compressed Attention use cases. This API is the companion API of the `dense_lightning_indexer_kl_loss_grad` backward operator, and its output can be used for backward gradient computation. During the call, call the `sparse_flash_mla_softmax_l1_norm_metadata` API first to complete load balancing computation, and then call the `sparse_flash_mla_softmax_l1_norm` API to perform the operator computation.
    - `sparse_flash_mla_softmax_l1_norm_metadata` API: Based on the shape, layout, mask, and other information of the main operator, evenly splits tasks to available AIC cores in a strided manner, and outputs metadata for the main operator.
    - `sparse_flash_mla_softmax_l1_norm` API: Computes the Softmax L1Norm of Q and K based on the core splitting information in metadata.

- Formula:

    Stage 1: Select the input key based on whether the use case is sparse.

    - When sparse:

    $$
    selectedKv\text{ }=\text{ }Gather \left( K, sparseIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{  }selectBlockCount\right. \right. \right. \right.
    $$

    - else:

    $$
    selectedKv\text{ }=\text{ }K
    $$

    Stage 2: Calculate P (SimpleSoftmax).

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    Stage 3: Calculate Softmax L1Norm.

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    where $G$ is the number of groups ($G = N1 / N2$). $ReduceSum$ sums the softmax probability $P$ over the G dimension (query head group dimension) and then averages the sum.

## Function Prototype

Before calling the `sparse_flash_mla_softmax_l1_norm` API, call the prerequisite API `sparse_flash_mla_softmax_l1_norm_metadata` first to complete the load balancing computation.

```python
cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    topk=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None
) -> Tensor
```

```python
cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q,
    k,
    softmax_lse,
    *,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    metadata=None,
    softmax_scale=1.0,
    max_seqlen_k=0,
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND"
) -> Tensor
```

## Parameters

### sparse_flash_mla_softmax_l1_norm

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | Required | Input Q of the attention structure, corresponding to $Q$ in the formula. When `layout_q` is "BSND", the shape is (B,S1,N1,D); when `layout_q` is "TND", the shape is (T1,N1,D). | float16, bfloat16 | (B,S1,N1,D), (T1,N1,D) |
| k | Tensor | Required | Input K of the attention structure, corresponding to $K$ in the formula. B remains consistent with that of q, S2 supports generalization, and N2=1. A2/A3 supports D=256 or 512, and A5 supports only D=512. | float16, bfloat16 | (B,S2,N2,D), (T2,N2,D) |
| softmax_lse | Tensor | Required | Output softmax_lse of the forward attention computation. B remains consistent with that of q, N2=1, S1 remains consistent with S1 of q, and G=N1/N2. | float32 | (B,N2,S1,G), (N2,T1,G) |
| sparse_indices | Tensor | Optional | Attention indices with higher weights selected from k in sparse use cases. B/S1 remain consistent with those of q, N2=1, and K supports generalization. | int32 | (B,S1,N2,K), (T1,N2,K) |
| cu_seqlens_q | Tensor | Optional | Cumulative sum of the number of valid tokens of q in each batch. This parameter is mandatory when `layout_q` is TND. The length remains consistent with B+1, and the accumulated sum remains consistent with T1. | int32 | (B+1,) |
| cu_seqlens_k | Tensor | Optional | Cumulative sum of the number of valid tokens of k in each batch. This parameter is mandatory when `layout_k` is TND. The length remains consistent with B+1, and the accumulated sum remains consistent with T2. | int32 | (B+1,) |
| seqused_q | Tensor | Optional | Indicates the number of tokens of q that actually participate in the computation in different batches. | int32 | (B,) |
| seqused_k | Tensor | Optional | Indicates the number of tokens of k that actually participate in the computation in different batches. | int32 | (B,) |
| cmp_residual_k | Tensor | Optional | Indicates the remainder of S2 // cmpRatio for each batch. Must be passed in when k is not empty and mask_mode=3. | int32 | (B,) |
| topk_length | Tensor | Optional | Indicates the actual selectable topk length of k corresponding to each row of q. Must be passed in when mask_mode=0 and sparse_indices is not empty, and the value must be accurate. | int32 | (B,S1,N2), (T1,N2) |
| metadata | Tensor | Required | Indicates the output result of the aicpu operator with tiling sunk, generated by the `sparse_flash_mla_softmax_l1_norm_metadata` operator. | int32 | (x) |
| softmax_scale | float | Optional | Indicates the scaling coefficient. The default value is 1.0. Recommended value: reciprocal of sqrt(head_dim). | float32 | - |
| max_seqlen_k | int | Optional | Indicates the maximum sequence length of k. Used for output shape derivation in TND dense use cases. The default value is 0. | int | - |
| cmp_ratio | int | Optional | Indicates the compression ratio of k. Value range: 1-128. The default value is 1. | int | - |
| mask_mode | int | Optional | Indicates the mask mode for the computation of q and k. 0: No mask; 3: rightDownCausal mode. The default value is 0. | int | - |
| layout_q | str | Optional | Indicates the data layout format of input q. Supported values: "BSND" and "TND". The default value is "BSND". Must remain consistent with layout_k. | string | - |
| layout_k | str | Optional | Indicates the data layout format of input k. Supported values: "BSND" and "TND". The default value is "BSND". Must remain consistent with layout_q. | string | - |

### sparse_flash_mla_softmax_l1_norm_metadata

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Indicates the number of heads of $Q$ in the formula (N1). Currently supports 1-128. | int | - |
| num_heads_k | int | Required | Indicates the number of heads of $key$ in the formula (N2). Currently only supports 1. | int | - |
| head_dim | int | Required | Indicates the head dimension (D). A2/A3 supports 256 or 512, and A5 supports only 512. | int | - |
| cu_seqlens_q | Tensor | Optional | Indicates the accumulated sum of the number of valid tokens of q in each batch. This parameter is mandatory when `layout_q` is TND. The accumulated sum must remain consistent with T1. | int32 | (B+1,) |
| cu_seqlens_k | Tensor | Optional | Indicates the accumulated sum of the number of valid tokens of k in each batch. This parameter is mandatory when `layout_k` is TND. The accumulated sum must remain consistent with T2. | int32 | (B+1,) |
| seqused_q | Tensor | Optional | Indicates the number of tokens of q that actually participate in the operation in different batches. | int32 | (B,) |
| seqused_k | Tensor | Optional | Reserved parameter. Indicates the number of tokens of k that actually participate in the operation in different batches. Currently not used by the kernel path. | int32 | (B,) |
| cmp_residual_k | Tensor | Optional | Reserved parameter. Indicates the residual of the sequence length of k in different batches related to cmpRatio. Currently not used by the kernel path. | int32 | (B,) |
| topk_length | Tensor | Optional | Indicates the actual selectable topk length of k corresponding to each row of q. | int32 | (B,S1,N2), (T1,N2) |
| batch_size | int | Optional | Indicates the batch size of input samples (B). The default value is None (a positive number is required for the BSND use case; it can be None and automatically derived for the TND use case). | int | - |
| max_seqlen_q | int | Optional | Indicates the maximum sequence length of q. The default value is None. It must be a positive number for the BSND use case. | int | - |
| max_seqlen_k | int | Optional | Indicates the maximum sequence length of k. The default value is None. It must be a positive number for the BSND use case. | int | - |
| topk | int | Optional | Indicates the number of key tokens selected from k. 0 indicates no sparsity. The default value is None. | int | - |
| layout_q | str | Optional | Indicates the data layout format of q. Supported values are "BSND" and "TND". The default value is None (internally converted to "BSND"). | string | - |
| layout_k | str | Optional | Indicates the data layout format of k. Supported values are "BSND" and "TND". The default value is None (internally converted to "BSND"). | string | - |
| mask_mode | int | Optional | Indicates the mask mode for computing q and k. 0 indicates No mask, and 3 indicates rightDownCausal mode. The default value is None (internally converted to 0). | int | - |
| cmp_ratio | int | Optional | Indicates the compression ratio of k. The value range is 1-128. The default value is None (internally converted to 1). | int | - |

## Return Values

### sparse_flash_mla_softmax_l1_norm

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| softmax_l1_norm | Tensor | Required | Softmax L1Norm result computed from q and k. The formula is reduceG(softmax)/G. This output is not empty when sparse_indices exists and empty in other use cases. | float32 | (B,S1,N2,S2), (T1,N2,T2) |

### sparse_flash_mla_softmax_l1_norm_metadata

| Name | Type | Required/Optional | Description | data type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Output of the load balancing result, split in a strided manner. The seq indices processed by each core are arranged with strides across cores. | int32 | (64,) |

Metadata layout:

| Field | Index | Description |
|:---|:---|:---|
| totalNum | 0 | Total number of seqs involved in load balancing. |
| formerCoreProcessNum | 1 | Number of seqs processed by regular cores, that is, ceil(totalNum / totalCoreNum). |
| remainCoreProcessNum | 2 | Number of seqs processed by tail cores, that is, floor(totalNum / totalCoreNum). |
| remainCoreNum | 3 | Number of tail cores. |
| totalCoreNum | 4 | Number of AIC cores actually used, that is, min(totalNum, aicCoreNum, 36). |

## Constraints

- This API supports training use cases.
- This API supports single-operator mode and TorchAir graph mode calls.
- The data types of q and k must remain consistent.
- Empty input handling: If q is an empty tensor, the API returns directly.

- Mask mode support:

    | Mode | Meaning | Note |
    | :--- | :--- | :--- |
    | 0 | No mask operation | Supported |
    | 3 | rightDownCausal | Supported |

- Specification constraints:

    | Specification Item | Specification | Specification Description |
    | :--- | :--- | :--- |
    | B | Support generalization | - |
    | S1, S2 | Support generalization | S1 and S2 of different lengths are supported. |
    | N1 | 1-128 | num_heads_q must be divisible by num_heads_k. |
    | N2 | 1 | Currently only supports N2=1. |
    | D | A2/A3: 256, 512; A5: 512 | The last dimension of q and k must remain consistent. D=256 is supported by the native A2/A3 implementation. |
    | layout_q/k | BSND / TND, must be the same | - |
    | cmp_ratio | 1-128 | - |

  - BSND use case: The batch_size, max_seqlen_q, and max_seqlen_k parameters are mandatory.
  - TND use case: The cu_seqlens_q and cu_seqlens_k parameters are mandatory, and batch_size can be None (deduced from cu_seqlens_q).
  - metadata must be passed in, and is generated by the `sparse_flash_mla_softmax_l1_norm_metadata` operator.

## Deterministic Computation

Defaults to a deterministic implementation.

## Invocation Instructions

- Single-operator mode call:

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    S1 = 16
    S2 = 2048
    cmp_ratio = 128
    actual_seq_q = [S1]
    actual_seq_k = [S2]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_k)

    B = 1
    N1 = 128
    N2 = 1
    D = 512
    scale_value = 1.0 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = (T1, N1, D)
    k_shape = (T2, N2, D)
    softmax_lse_shape = (N2, T1, N1 // N2)
    softmax_l1_norm_shape = (T1, N2, T2)

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_klen = [0] + [sum(actual_seq_k[:x+1]) for x in range(len(actual_seq_k))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    k = (torch.rand(k_shape).to(dtype)) * 2
    softmax_lse = (torch.rand(softmax_lse_shape).to(torch.float32))

    cu_seq_qlen_tensor = torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_klen_tensor = torch.tensor(cu_seq_klen).to(torch.int32).npu()

    cmp_residual_k = torch.zeros(B, dtype=torch.int32, device="npu")

    # Call sparse_flash_mla_softmax_l1_norm_metadata to complete load balancing computation.
    metadata = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
        N1, N2, D,
        cu_seqlens_q=cu_seq_qlen_tensor,
        cu_seqlens_k=cu_seq_klen_tensor,
        cmp_residual_k=cmp_residual_k,
        max_seqlen_q=S1,
        max_seqlen_k=S2,
        topk=0,
        cmp_ratio=cmp_ratio,
        mask_mode=3,
        layout_q=input_layout,
        layout_k=input_layout,
    )

    # Call sparse_flash_mla_softmax_l1_norm to perform operator computation.
    softmax_l1_norm = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
        q.npu(),
        k.npu(),
        softmax_lse.npu(),
        cu_seqlens_q=cu_seq_qlen_tensor,
        cu_seqlens_k=cu_seq_klen_tensor,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        softmax_scale=scale_value,
        max_seqlen_k=S2,
        cmp_ratio=cmp_ratio,
        mask_mode=3,
        layout_q=input_layout,
        layout_k=input_layout,
    )

    torch_npu.npu.synchronize()
    assert softmax_l1_norm.shape == softmax_l1_norm_shape
    assert softmax_l1_norm.dtype == torch.float32
    assert torch.isfinite(softmax_l1_norm.float()).all().item()
    ```

- TorchAir graph mode call:

    ```python
    import math
    import torch
    import torch_npu
    import torchair
    import cann_ops_transformer
    from torchair.configs.compiler_config import CompilerConfig

    class SparseFlashMlaSoftmaxL1NormModel(torch.nn.Module):
        def __init__(self):
            super(SparseFlashMlaSoftmaxL1NormModel, self).__init__()

        def forward(self, m_inputs, npu_inputs):
            # Call sparse_flash_mla_softmax_l1_norm_metadata to complete load balancing computation.
            metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
                **m_inputs
            )
            # Explicitly migrate the metadata output Tensor to the NPU, then use it as an input parameter of the main operator.
            npu_inputs["metadata"] = metadata.npu()
            # Call sparse_flash_mla_softmax_l1_norm to perform operator computation.
            return torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
                **npu_inputs
            )

    def sparse_flash_mla_softmax_l1_norm_acl_graph(m_inputs, npu_inputs):
        npu_mode = SparseFlashMlaSoftmaxL1NormModel().npu()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        torch._dynamo.reset()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)
        return npu_mode(m_inputs, npu_inputs)

    torch_npu.npu.set_device(0)

    S1 = 16
    S2 = 2048
    cmp_ratio = 128
    actual_seq_q = [S1]
    actual_seq_k = [S2]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_k)

    B = 1
    N1 = 128
    N2 = 1
    D = 512
    scale_value = 1.0 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = (T1, N1, D)
    k_shape = (T2, N2, D)
    softmax_lse_shape = (N2, T1, N1 // N2)

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_klen = [0] + [sum(actual_seq_k[:x+1]) for x in range(len(actual_seq_k))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    k = (torch.rand(k_shape).to(dtype)) * 2
    softmax_lse = (torch.rand(softmax_lse_shape).to(torch.float32))

    cu_seq_qlen_tensor = torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_klen_tensor = torch.tensor(cu_seq_klen).to(torch.int32).npu()

    cmp_residual_k = torch.zeros(B, dtype=torch.int32, device="npu")

    # Input parameters of the preceding metadata API
    m_inputs = {
        "num_heads_q": N1,
        "num_heads_k": N2,
        "head_dim": D,
        "cu_seqlens_q": cu_seq_qlen_tensor,
        "cu_seqlens_k": cu_seq_klen_tensor,
        "cmp_residual_k": cmp_residual_k,
        "max_seqlen_q": S1,
        "max_seqlen_k": S2,
        "topk": 0,
        "cmp_ratio": cmp_ratio,
        "mask_mode": 3,
        "layout_q": input_layout,
        "layout_k": input_layout,
    }

    # Input parameters of the main operator, with metadata generated in the graph
    npu_inputs = {
        "q": q.npu(),
        "k": k.npu(),
        "softmax_lse": softmax_lse.npu(),
        "cu_seqlens_q": cu_seq_qlen_tensor,
        "cu_seqlens_k": cu_seq_klen_tensor,
        "cmp_residual_k": cmp_residual_k,
        "softmax_scale": scale_value,
        "max_seqlen_k": S2,
        "cmp_ratio": cmp_ratio,
        "mask_mode": 3,
        "layout_q": input_layout,
        "layout_k": input_layout,
    }

    softmax_l1_norm = sparse_flash_mla_softmax_l1_norm_acl_graph(m_inputs, npu_inputs)

    torch_npu.npu.synchronize()
    assert softmax_l1_norm.shape == (T1, N2, T2)
    assert softmax_l1_norm.dtype == torch.float32
    assert torch.isfinite(softmax_l1_norm.float()).all().item()
    ```
