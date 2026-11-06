# NormRopeConcat

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: In the attention mechanism of a (multimodal) transformer, performs normalization (Norm), rotary position encoding (Rope), and feature concatenation (Concat) on query, key, and Value:

    - Normalization (Norm) currently supports LayerNorm, AFFINE LayerNorm, RmsNorm, and AFFINE RmsNorm types.
    - Rotary position encoding (Rope) supports the Interleave and Half types.
    - Feature concatenation (Concat) supports concatenation along the sequence dimension, and the concatenation order matters.

- Computation formula (taking Query (video) and EncoderQuery (text) as an example):

	  $$
    hiddenState_q = \text{Norm}(query, normQueryWeight, normQueryBias, eps) \\
    hiddenState_{eq} = \text{Norm}(encoderQuery, normEncoderQueryWeight, normEncoderQueryBias, eps) \\
    concatenatedHiddenState = \text{Concat}(hiddenState_q, hiddenState_{eq}) \\
    transposedHiddenState = \text{Transpose}(concatenatedHiddenState, (0, 2, 1, 3)) \\
    hiddenState = \text{RoPE}(concatenatedHiddenState, ropeSin, ropeCos)
    $$

- Notes:

    1. The input and output layouts are as follows: the shape of the input `query` is `(B, S, N, D)`, and the shape of the output `hiddenState` is `(B, N, S, D)`, where
    B is batch, S is sequenceLen, N is headNum, and D is headDim.
    2. Norm has five modes (`normType`): `NONE(0), LAYER_NORM(1), LAYER_NORM_AFFINE(2), RMS_NORM(3), RMS_NORM_AFFINE(4)`, where:
        When `normType = NONE`:

        $$
        hiddenState_q = query
        $$

        When `normType = LAYER_NORM`

        $$
        queryMean_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D-1}query_{b,s,n} \\
        queryVar_{b,s,n} = \frac{1}{D}\sum_{i=0}^{D-1}(query-queryMean_{b,s,n})^2 \\
        queryRstd_{b,s,n}=  \frac{1}{\sqrt{queryVar_{b,s,n}+\epsilon}} \\
        hiddenState_q = (query-queryMean)*queryRstd$$
        When `normType = LAYER_NORM_AFFINE`, based on the preceding formulas:

        $$
        hiddenState_q = normQueryWeight*hiddenState_q + normQueryBias
        $$

        When `normType = RMS_NORM`:

        $$
        queryMs = \frac{1}{D}\sum_{i=0}^{D-1}(query_{b,s,n})^2 \\
        queryRms = \frac{1}{\sqrt{queryMs+\epsilon}} \\
        hiddenState_q = query * queryRms
        $$

        When `normType = RMS_NORM_AFFINE`, based on the preceding formulas:

        $$
        hiddenState_q = normQueryWeight*hiddenState_q
        $$

    3. Concat means concatenation along the sequence dimension, and the concatenation order matters (`concatOrder`). When `concatOrder=0`, $hiddenState_q$ comes before $hiddenState_{eq}$; when `concatOrder=1`, $hiddenState_q$ comes after $hiddenState_{eq}$.
    4. RoPE has three modes (`ropeType`): `NONE(0), INTERLEAVE(1), HALF(2)`. When `ropeType=NONE`, the output is returned directly without transformation. For other cases, refer to the following:

        ```python
          def image_rotary_emb(hidden_states, rope_sin, rope_cos, mode=1):
              out = torch.empty_like(hidden_states)
              if mode == 1: # interleave
                  x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                  x1, x2 = x[..., 0], x[..., 1]
                  rotated_x = torch.stack([-x2, x1],dim=-1).flatten(3)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
              else: # half
                  x1, x2 = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)
                  rotated_x = torch.cat([-x2, x1],dim=-1)
                  out = hidden_states.float() * rope_cos + rotated_x.float()*rope_sin
                  return out.type_as(hidden_states)
        ```

    5. The shape of the RoPE input `ropeSin` is `(seqRope, D)`, where

    $$
    seqRope <= min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)
    $$

    6. In the training scenario, `queryMean, queryRstd, encoderQueryMean, encoderQueryRstd` are output for subsequent backward computation.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>query</td>
      <td>Input</td>
      <td>Represents the Query in the attention mechanism</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Represents the Key in the attention mechanism</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Represents the Value in the attention mechanism</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>encoderQuery</td>
      <td>Input</td>
      <td>Represents the Query in the attention mechanism, from EncoderHiddenState, and can be a null pointer</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>encoderKey</td>
      <td>Input</td>
      <td>Represents the Key in the attention mechanism, from EncoderHiddenState, and can be a null pointer</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>encoderValue</td>
      <td>Input</td>
      <td>Represents the Value in the attention mechanism, from EncoderHiddenState, and can be a null pointer</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryWeight</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to Query</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryBias</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to Query</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyWeight</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to Key</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyBias</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to Key</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedQueryWeight</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to encoderQuery</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedQueryBias</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to encoderQuery</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedKeyWeight</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to encoderKey</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedKeyBias</td>
      <td>Input</td>
      <td>Represents the affine transformation parameters of LayerNorm, applied to encoderKey</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ropeSin</td>
      <td>Input</td>
      <td>Represents the sine encoding of RoPE</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ropeCos</td>
      <td>Input</td>
      <td>Represents the cosine encoding of RoPE</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normType</td>
      <td>Attribute</td>
      <td>Represents the normalization type applied to q and k. 0: no normalization, 1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normAddedType</td>
      <td>Attribute</td>
      <td>Represents the normalization type applied to encoderQuery and encoderKey. 0: no normalization, 1: LayerNorm, 2: LayerNormAffine, 3: RmsNorm, 4: RmsNormAffine</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ropeType</td>
      <td>Attribute</td>
      <td>Represents the RoPE mode, of the int64 type. 0: no RoPE, 1: Interleave, 2: Half</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>concatOrder</td>
      <td>Attribute</td>
      <td>Represents the concatenation order, of the int64 type. 0: query first, 1: query last</td>
      <td>int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Represents the epsilon value in normalization</td>
      <td>float32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>isTraining</td>
      <td>Attribute</td>
      <td>Indicates whether it is the training phase, and determines whether the values used for backward computation are output</td>
      <td>bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>queryOutput</td>
      <td>Output</td>
      <td>Represents the query output in the attention mechanism</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyOutput</td>
      <td>Output</td>
      <td>Represents the key output in the attention mechanism</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>valueOutput</td>
      <td>Output</td>
      <td>Represents the value output in the attention mechanism</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryMean</td>
      <td>Output</td>
      <td>Query mean output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normQueryRstd</td>
      <td>Output</td>
      <td>Query standard deviation output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyMean</td>
      <td>Output</td>
      <td>Key mean output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normKeyRstd</td>
      <td>Output</td>
      <td>Key standard deviation output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderQueryMean</td>
      <td>Output</td>
      <td>encoderQuery mean output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderQueryRstd</td>
      <td>Output</td>
      <td>encoderQuery standard deviation output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderKeyMean</td>
      <td>Output</td>
      <td>encoderKey mean output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normEncoderKeyRstd</td>
      <td>Output</td>
      <td>encoderKey standard deviation output in LayerNorm, used for backward computation</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation:
  - aclnnNormRopeConcat uses a deterministic implementation by default.
- The data types of query, key, value, encoderQuery, encoderKey, and encoderValue must be the same.
- The headDim length is in the range [1~1024] and must be an even number.
- The seqRope length is in the range [1~Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)].

## Calling Instructions

| Calling Method           | Calling Sample                                                   | Description                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_norm_rope_concat.cpp](./examples/test_aclnn_norm_rope_concat.cpp)                     |Calls the NormRopeConcat operator through the [aclnnNormRopeConcat](docs/aclnnNormRopeConcat_en.md) API.                   |
