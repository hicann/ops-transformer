# ChunkKdaFwd

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: Performs chunked forward KDA computation without CP splitting, and computes the attention output, the final state, and optional intermediate quantities for backward computation.

- Computation formula:

  Each sequence is divided into $M$ chunks with $C=chunk\_size$. Taking the c-th chunk as an example, $i$ and $j$ are token indices within the chunk, and $l_c$ is the index of the last valid token in the chunk. The formulas omit batch and head indices. When $HV>H$, the scenario is grouped value attention (GVA): each group of $HV/H$ consecutive Value heads shares one Query/Key head. When head numbering starts from 0, the Value head numbered $h_v$ corresponds to the Query/Key head numbered $\lfloor h_v/(HV/H)\rfloor$.

  Let $x_{c,i,d}=g_{c,i,d}+dt\_bias_d$. If `dt_bias` is not passed, set $dt\_bias_d=0$. The gate after activation is:

  $$
  \gamma_{c,i,d}=
  \begin{cases}
  g_{c,i,d}, & use\_gate\_in\_kernel=false,\\
  -\exp(A_{log})\operatorname{softplus}(x_{c,i,d}),
      & use\_gate\_in\_kernel=true,\ safe\_gate=false,\\
  lower\_bound\operatorname{sigmoid}(\exp(A_{log})x_{c,i,d}),
      & use\_gate\_in\_kernel=true,\ safe\_gate=true.
  \end{cases}
  $$

  `gk` is the chunk-local log2 cumulative gate:

  $$
  gk_{c,i,d}=\frac{1}{\ln 2}\sum_{t=0}^{i}\gamma_{c,t,d}.
  $$

  `qg` and `kg` are the gate-scaled Query and Key respectively:

  $$
  qg_{c,i,d}=q_{c,i,d}2^{gk_{c,i,d}},\qquad
  kg_{c,i,d}=k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}}.
  $$

  For tokens $i$ and $j$ in the same chunk, `Aqk` is the lower-triangular Query-Key coefficient matrix including the diagonal:

  $$
  Aqk_{c,i,j}=\mathbb{1}_{j\le i}\cdot scale
  \sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}}.
  $$

  The strictly lower-triangular matrix $L_c$ and `Akk` are:

  $$
  L_{c,i,j}=\mathbb{1}_{j<i}\cdot\beta_{c,i}
  \sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\qquad
  Akk_c=(I+L_c)^{-1}.
  $$

  Let $w^{seed}$ and $u^{seed}$ be:

  $$
  w^{seed}_{c,i,d}=\beta_{c,i}k_{c,i,d}2^{gk_{c,i,d}},\qquad
  u^{seed}_{c,i,e}=\beta_{c,i}v_{c,i,e}.
  $$

  `w` and `u` are:

  $$
  w_c=Akk_c\,w^{seed}_c,\qquad u_c=Akk_c\,u^{seed}_c.
  $$

  Let $h_c$ be the state before the computation of the c-th chunk. If `initial_state` is passed, $h_0=initial\_state$; otherwise, $h_0=0$. `v_new` and the state of the next chunk are:

  $$
  v^{new}_c=u_c-w_c\,h_c,
  $$

  $$
  h_{c+1}=2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,
  $$

  where $2^{gk_{c,l_c}}\odot h_c$ denotes scaling the state matrix along the $K$ dimension. The `h` output stores the input state of each chunk, and `final_state` is the state after the last chunk is updated:

  $$
  h[c]=h_c,\qquad final\_state=h_M.
  $$

  `attn_out` is:

  $$
  attn\_out_c=scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c.
  $$

  The 12th return value of the Python interface is the pass-through of the input object:

  $$
  initial\_state\_out=initial\_state.
  $$

  The optional output parameters only determine whether the corresponding intermediate quantities are returned to the caller, and do not change the computation semantics described above.

## Parameter Description

This document uses the following dimension symbols:

- B: batch size of input samples.
- S: sequence length of rank-4 inputs.
- T: total number of tokens of rank-3 inputs.
- H: number of Query and Key heads.
- HV: number of Value and gate heads.
- K: head dim of Query and Key.
- V: head dim of Value.
- N: number of logical sequences.
- C: chunk size.
- NC: number of chunks per sequence ($\lceil S/C\rceil$) when `cu_seqlens` is not passed; total number of chunks across all logical sequences when `cu_seqlens` is passed.

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| q | Input | Query in the formula. `BSND` is (B, S, H, K), `BNSD` is (B, H, S, K), `TND` is (T, H, K), `NTD` is (H, T, K). | FLOAT16, BFLOAT16 | ND |
| k | Input | Key in the formula. The shape and data type must be the same as those of `q`. | FLOAT16, BFLOAT16 | ND |
| v | Input | Value in the formula. `BSND` is (B, S, HV, V), `BNSD` is (B, HV, S, V), `TND` is (T, HV, V), `NTD` is (HV, T, V). The data type must be the same as that of `q`. | FLOAT16, BFLOAT16 | ND |
| g | Input | Raw gate or activated natural-log gate. `BSND` is (B, S, HV, K), `BNSD` is (B, HV, S, K), `TND` is (T, HV, K), `NTD` is (HV, T, K). | FLOAT, BFLOAT16 | ND |
| beta | Input | Delta coefficient in the formula. `BSND` is (B, S, HV), `BNSD` is (B, HV, S), `TND` is (T, HV), `NTD` is (HV, T). | FLOAT, BFLOAT16 | ND |
| a_log | Optional input | Gate decay parameter with shape (HV). It must be passed when `use_gate_in_kernel=true`. | FLOAT | ND |
| dt_bias | Optional input | Gate bias with shape (HV×K). | FLOAT | ND |
| initial_state | Optional input | Initial state. The shape is (N, HV, K, V) when `state_v_first=false`, and (N, HV, V, K) otherwise. | FLOAT | ND |
| cu_seqlens | Optional input | Cumulative lengths of variable-length sequences, with shape (N+1). The first element is 0, the last element is T or S, and elements are monotonically non-decreasing. | INT64 | ND |
| chunk_indices | Optional input | Chunk indices with shape (2×NC), stored as `(seq_id, chunk_id)`, in sequence-major canonical order. When this parameter is passed, `cu_seqlens` must also be passed. | INT64 | ND |
| attn_out | Output | Attention output. rank-4 inputs always output the BSND format (B, S, HV, V); rank-3 inputs always output the TND format (T, HV, V). | FLOAT16, BFLOAT16 | ND |
| final_state | Optional output | Final state. The shape is (N, HV, K, V) when `state_v_first=false`, and (N, HV, V, K) otherwise. | FLOAT | ND |
| gk | Optional output | Chunk-local log2 cumulative gate. rank-4 inputs: (B, HV, S, K); rank-3 inputs: (HV, T, K). | FLOAT | ND |
| Aqk | Output | Query-Key coefficient matrix within a chunk. rank-4 inputs: (B, HV, S, C); rank-3 inputs: (HV, T, C). | FLOAT16, BFLOAT16 | ND |
| Akk | Output | Lower-triangular inversion result within a chunk. rank-4 inputs: (B, HV, S, C); rank-3 inputs: (HV, T, C). | FLOAT16, BFLOAT16 | ND |
| w | Optional output | W intermediate quantity for backward computation. rank-4 inputs: (B, HV, S, K); rank-3 inputs: (HV, T, K). | FLOAT16, BFLOAT16 | ND |
| u | Optional output | U intermediate quantity for backward computation. rank-4 inputs: (B, HV, S, V); rank-3 inputs: (HV, T, V). | FLOAT16, BFLOAT16 | ND |
| qg | Optional output | Gate-scaled Query for backward computation. rank-4 inputs: (B, HV, S, K); rank-3 inputs: (HV, T, K). | FLOAT16, BFLOAT16 | ND |
| kg | Optional output | Gate-scaled Key for backward computation. rank-4 inputs: (B, HV, S, K); rank-3 inputs: (HV, T, K). | FLOAT16, BFLOAT16 | ND |
| v_new | Optional output | Value intermediate quantity for backward computation. rank-4 inputs: (B, HV, S, V); rank-3 inputs: (HV, T, V). | FLOAT16, BFLOAT16 | ND |
| h | Optional output | Exposed chunk state. rank-4 inputs: (B, NC, HV, K, V); rank-3 inputs: (NC, HV, K, V). The last two dimensions are swapped when `state_v_first=true`. | FLOAT16, BFLOAT16 | ND |
| layout | Optional attribute | Input layout. `BSND`, `BNSD`, `TND`, and `NTD` are supported, with the default value `BSND`. This attribute only describes the input layout. | STRING | - |
| scale | Required attribute | Query scale coefficient. Usually $K^{-0.5}$. | FLOAT | - |
| chunk_size | Required attribute | Chunk size. 64 and 128 are supported, with the default value 64. | INT | - |
| safe_gate | Required attribute | Whether to use the bounded gate. The default value is false. | BOOL | - |
| lower_bound | Optional attribute | Lower bound of the bounded gate. When `safe_gate=true` and `use_gate_in_kernel=true`, the value range is [-5, 0), and the default value is -5.0. | FLOAT | - |
| use_gate_in_kernel | Required attribute | Whether to compute the activation from the raw gate inside the kernel. The default value is false. | BOOL | - |
| state_v_first | Optional attribute | Whether the last two dimensions of the state tensor are arranged as (V, K). The default value is false. | BOOL | - |

## Constraints

- H and HV must be positive integers and satisfy HV≥H, HV≤128, and HV % H=0.
- K and V are in the range [16, 256] and must be multiples of 16.
- Only 64 and 128 are supported for `chunk_size`.
- The data types of `q`, `k`, and `v` must be the same.
- When `use_gate_in_kernel=true`, `a_log` must be passed, and `dt_bias` does not need to be passed.
- When `safe_gate=true` and `use_gate_in_kernel=true`, the value range of `lower_bound` is [-5, 0).
- rank-4 variable-length inputs require B=1.
- Variable-length inputs support at most 1024 logical sequences.
- `layout` describes only the input. `attn_out` is always sequence-major; intermediate outputs for subsequent backward computation are always head-major; `final_state` and the exposed `h` are output as specified in the parameter table.
- Deterministic computation is supported by default.

## Invocation Description

| Invocation Mode | Sample Code | Description |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_chunk_kda_fwd.cpp](./examples/test_aclnn_chunk_kda_fwd.cpp) | Calls the ChunkKdaFwd operator through [aclnnChunkKdaFwd](docs/aclnnChunkKdaFwd_en.md). For compilation and running steps, see [Invocation Sample Description](./examples/README_en.md). |
| PyTorch API | [torchapi_chunk_kda_fwd](./docs/torchapi_chunk_kda_fwd.md) | Calls the ChunkKdaFwd operator through `cann_ops_transformer.ops.chunk_kda_fwd`. |

For stage division, platform templates, and pipeline design, refer to [ChunkKdaFwd Operator Design Introduction](./docs/ChunkKdaFwd_design_en.md).
