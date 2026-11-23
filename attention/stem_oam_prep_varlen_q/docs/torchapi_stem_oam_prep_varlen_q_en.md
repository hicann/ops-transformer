# stem_oam_prep_varlen_q

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:40:50.118Z -->

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

- **Description**:

  Performs preprocessing computation on the Q side in Stem OAM block-sparse attention. Converts the variable-length Q tensor stored in token-concatenated layout ([total_tokens, H_q, D]) into flattened qFlat output grouped by stem block, for subsequent OAM score computation.

- **Formula**:

  Stage 1 Scale Fusion:
  $$q\_scale[b,h,pos] = qScale[cu\_seqlens\_q[b]+pos,\ h]$$ (pos is the token index within the batch sequence)

  Stage 2 Varlen Positioning:
  $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

  Stage 3 Weighted Group Sum (natural order, NO flip):
  $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times qScale[cu\_seqlens\_q[b]+position(qb,r,g),\ h]$$ (where position(qb,r,g) = qb*stemBlockSize + r*stemStride + g; positions beyond qSeqLens[b] are zero-filled by the kernel and contribute zero)

  Stage 4 Flatten:
  $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

  Stage 5 Cast output:
  $$qflat\_out = qflat.to(\text{BF16})$$

- **Key features**: The stride dimensions on the Q side are in natural order (no flip), unlike the flip processing on the K side.

## Function Prototype

```python
cann_ops_transformer.stem_oam_prep_varlen_q(q, q_seq_lens, cu_seq_lens_q, *, q_scale=None, stem_block_size=128, stem_stride=16) -> Tensor
```

## Parameters

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 150px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 350px">
<col style="width: 180px">
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
        <td>q</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Indicates the variable-length Q tensor. Tokens of all batches are concatenated in storage. Corresponds to q in the formula. The last dimension must be equal to 128.</td>
        <td>float8_e4m3fn</td>
        <td>(total_tokens, H_q, D), D=128</td>
    </tr>
    <tr>
        <td>q_seq_lens</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Indicates the Q sequence length of each batch. Corresponds to q_seq_lens in the formula. The length must be equal to batch. Batch value range: [0, 1024]. Each value ≥ 0. If the length is 0, an empty output is returned directly.</td>
        <td>int64</td>
        <td>(batch,)</td>
    </tr>
    <tr>
        <td>cu_seq_lens_q</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Indicates the cumulative sequence length offsets of Q, used for varlen indexing. It corresponds to cu_seqlens_q in the corresponding formula. The length is batch+1. cu_seq_lens_q[0] must be 0, cu_seq_lens_q[batch] must be equal to total_tokens, and the prefix-sum relation cu_seq_lens_q[b+1] - cu_seq_lens_q[b] equals q_seq_lens[b] (equal adjacent values are allowed for zero-length sequences).</td>
        <td>int64</td>
        <td>(batch+1,)</td>
    </tr>
    <tr>
        <td>q_scale</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Indicates the per-token scale factor of Q, corresponding to qscale in the formula. Required when q is of the float8_e4m3fn type, and the data type must be float32.</td>
        <td>float32</td>
        <td>(total_tokens, H_q)</td>
    </tr>
    <tr>
        <td>stem_block_size</td>
        <td>int</td>
        <td>Optional</td>
        <td>Indicates the stem block size, corresponding to B in the formula. Controls the number of tokens in each stem block and determines the grouping granularity of Q Processing. Default value: 128.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>stem_stride</td>
        <td>int</td>
        <td>optional</td>
        <td>Indicates the stem stride size, corresponding to S in the formula. Controls the number of tokens in each stride group within a stem block and determines the dimension granularity of qFlat. Default value: 16.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody>
</table>

## Outputs

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 150px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 350px">
<col style="width: 180px">
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
        <td>q_flat</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Indicates the flattened Q output, used for OAM score computation. It is qflat_out in the corresponding formula, where max_Qb=ceil(max(q_seq_lens)/stem_block_size) and kflat_dim=stem_stride×D=2048.</td>
        <td>bfloat16</td>
        <td>(batch, H_q, max_Qb, kflat_dim)</td>
    </tr>
</tbody>
</table>

## Constraints

- This API supports both training and inference use cases.
- This API can be called in single-operator mode and TorchAir graph mode.
- The last dimension D of q must be equal to 128.
- Only 128 is supported for stem_block_size, and only 16 is supported for stem_stride. Derived values: R = stem_block_size / stem_stride = 8, kflat_dim = stem_stride × D = 2048.
- Only the float8_e4m3fn data type is supported for q. q_scale is required and must be of the float32 data type. The q_flat output is fixed to bfloat16.
- The value range of the q_seq_lens length (batch) is [0, 1024]. If the length is 0, an empty output is returned directly.
- When a batch value in q_seq_lens is 0, the q_flat output of the corresponding batch is padded with zeros.
- The stride dimension on the Q side is in natural order (g ∈ [0, S-1]) and is not flipped (unlike the K-side processing).
- Empty tensor inputs are supported: if q is empty or the length of q_seq_lens is 0, the API directly returns.

## Deterministic Computation

Defaults to a deterministic implementation.

## API call

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import stem_oam_prep_varlen_q

  total_tokens = 256
  H_q = 32
  D = 128
  batch = 2

  q = torch.randn(total_tokens, H_q, D, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
  q_scale = torch.randn(total_tokens, H_q, dtype=torch.float32).npu()

  q_seq_lens = [128, 128]
  cu_seq_lens_q = [0, 128, 256]

  q_flat = stem_oam_prep_varlen_q(
      q, q_seq_lens, cu_seq_lens_q, q_scale=q_scale
  )
  ```

- Call in graph mode

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import stem_oam_prep_varlen_q

  torch_npu.npu.set_device(0)

  total_tokens = 256
  H_q = 32
  D = 128
  batch = 2

  class StemOamPrepVarlenQModel(torch.nn.Module):
      def forward(self, q, q_scale):
          return stem_oam_prep_varlen_q(
              q, [128, 128], [0, 128, 256], q_scale=q_scale
          )

  model = StemOamPrepVarlenQModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  q = torch.randn(total_tokens, H_q, D, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
  q_scale = torch.randn(total_tokens, H_q, dtype=torch.float32).npu()

  output = model(q, q_scale)
  ```
