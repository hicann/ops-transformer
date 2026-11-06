# StemOamPrepVarlenQ

English | [简体中文](./README.md)

## Product Support

|Product      | Supported or Not |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- **Operator function**: In the inference scenario, this operator performs the Q-side preprocessing computation in Stem OAM block-sparse attention. It converts the variable-length Q tensor from the paged storage format into the flattened qFlat output grouped by stem block (stemBlockSize tokens) for the subsequent OAM score computation. The main computation process is as follows:
    - First, perform Scale Fusion on the Q tensor by applying the per-token scale factor to Q
    - Then perform De-page Varlen based on cuSeqLensQ to extract the variable-length Q tensor from the paged storage
    - Perform Weighted Group Sum on Q (natural order; the stride dimension is not flipped)
    - Flatten the output to qFlat: the shape is [batch, H_q, max_Qb, kflat_dim]
    - Cast the output to the BFLOAT16 format
- **Computation formula**:

    Stage 1 Scale Fusion

    $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

    Stage 2 De-page Varlen

    $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

    Stage 3 Weighted Group Sum(natural order, NO flip)

    $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

    Stage 4 Flatten

    $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

    Stage 5 Cast output

    $$qflat\_out = qflat.to(\text{BF16})$$

- **Key feature**: The stride dimension on the Q side is in natural order (g ∈ [0, S-1]) and is not flipped (different from the K-side processing).

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 150px">
<col style="width: 100px">
<col style="width: 650px">
<col style="width: 180px">
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
    <td>q</td>
    <td>Input</td>
    <td>aclTensor on the device. Variable-length Q tensor with the tokens of all batches stored in a concatenated manner. The shape is [total_tokens, H_q, D], where D must be 128.</td>
    <td>FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qScale</td>
    <td>Optional Input</td>
    <td>aclTensor on the device. Per-token scale factor of Q. The shape is [total_tokens, H_q].</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qSeqLens</td>
    <td>Input</td>
    <td>aclTensor on the device. Q sequence length of each batch. The shape is [batch]. The length equals batch, each value must be ≥0, and empty arrays are not supported. The maximum value of batch is 1024.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cuSeqLensQ</td>
    <td>Input</td>
    <td>aclTensor on the device. Cumulative sequence length offsets of Q, used for varlen indexing. The shape is [batch+1]. cuSeqLensQ[0] must be 0, cuSeqLensQ[batch] must equal total_tokens, and the values must increase monotonically.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stemBlockSize</td>
    <td>Attribute</td>
    <td>Stem block size, which controls the number of tokens in each stem block and determines the grouping granularity of Q Processing. The default value is 128. Currently, only 128 is supported.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stemStride</td>
    <td>Attribute</td>
    <td>Stem stride size, which controls the number of tokens in each stride group within a stem block and determines the dimension granularity of qFlat. The default value is 16. Currently, only 16 is supported.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qFlat</td>
    <td>Output</td>
    <td>aclTensor on the device. Flattened Q output for the OAM score computation. The shape is [batch, H_q, max_Qb, kflat_dim], where max_Qb=ceil(max(qSeqLens)/stemBlockSize) and kflat_dim=stemStride×D (default value: 16×128=2048). The data type is fixed to BFLOAT16.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- **Deterministic computation**: aclnnStemOamPrepVarlenQ uses a deterministic implementation by default.
- **Meanings of and constraints on the shape fields**
    - total_tokens: total_tokens indicates the total number of tokens of all batches
    - H_q: H_q indicates the number of heads of Q. The value range is 1~128
    - stemBlockSize: stem block size. Currently, only 128 is supported
    - stemStride: stem stride size. Currently, only 16 is supported
    - D: last dimension of the Q matrix, which must be 128
    - max_Qb: maximum number of Q blocks. max_Qb=ceil(max(qSeqLens)/stemBlockSize)
    - kflat_dim: output dimension of qFlat. kflat_dim=stemStride×D (default value: 16×128=2048)
- **Stride dimension constraint**: The stride dimension on the Q side is in natural order (g ∈ [0, S-1]) and is not flipped (different from the K-side processing)
- **Handling of zero values**: When the value of a batch in qSeqLens is 0, the corresponding qFlat output of the batch is filled with zeros
- **Group sum without weights**: When a null pointer is passed for qScale, the kernel performs the group sum without weights (Σ_r Q_blocks[:, r, :, :]) during Q Processing, instead of the weighted group sum (Σ_r Q_blocks×qScale).
- **FP8 input constraint**: For FP8 input, the last dimension of qScale must cover max(qSeqLens) positions (padded to be aligned with stemBlockSize)

## Calling Instructions

| Calling Method | Sample Code                                                  | Description                                                  |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_stem_oam_prep_varlen_q](./examples/test_aclnn_stem_oam_prep_varlen_q.cpp) | The StemOamPrepVarlenQ operator is called through [aclnnStemOamPrepVarlenQ](docs/aclnnStemOamPrepVarlenQ_en.md) |
| Graph mode | [test_geir_stem_oam_prep_varlen_q](./examples/test_geir_stem_oam_prep_varlen_q.cpp) | The StemOamPrepVarlenQ operator is called through the [operator IR](./op_graph/stem_oam_prep_varlen_q_proto.h) |
| PyTorch API | - | The stem_oam_prep_varlen_q operator is called through the [stem_oam_prep_varlen_q](docs/torchapi_stem_oam_prep_varlen_q_en.md) API. |
