# ApplyRotaryPosEmb

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    x     |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Description

- Operator function: Performs the rotary position embedding computation. To improve performance, inference networks fuse the query and key operators into one operator. The computation result is updated in place.
- Computation formula:
  (1) rotaryMode is "half":

  $$
  query\_q1 = query[..., : query.shape[-1] // 2]
  $$

  $$
  query\_q2 = query[..., query.shape[-1] // 2 :]
  $$

  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1)
  $$

  $$
  key\_k1 = key[..., : key.shape[-1] // 2]
  $$

  $$
  key\_k2 = key[..., key.shape[-1] // 2 :]
  $$

  $$
  key\_rotate = torch.cat((-key\_k2, key\_k1), dim=-1)
  $$

  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$

  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

  (2) rotaryMode is "quarter":

  $$
  query\_q1 = query[..., : query.shape[-1] // 4]
  $$

  $$
  query\_q2 = query[..., query.shape[-1] // 4 : query.shape[-1] // 2]
  $$

  $$
  query\_q3 = query[..., query.shape[-1] // 2 : query.shape[-1] // 4 * 3]
  $$

  $$
  query\_q4 = query[..., query.shape[-1] // 4 * 3 :]
  $$

  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1, -query\_q4, query\_q3), dim=-1)
  $$

  $$
  key\_q1 = key[..., : key.shape[-1] // 4]
  $$

  $$
  key\_q2 = key[..., key.shape[-1] // 4 : key.shape[-1] // 2]
  $$

  $$
  key\_q3 = key[..., key.shape[-1] // 2 : key.shape[-1] // 4 * 3]
  $$

  $$
  key\_q4 = key[..., key.shape[-1] // 4 * 3 :]
  $$

  $$
  key\_rotate = torch.cat((-key\_q2, key\_q1, -key\_q4, key\_q3), dim=-1)
  $$

  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$

  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

  (3) rotaryMode is "interleave":

  $$
  query\_q1 = query[..., ::2].view(-1, 1)
  $$

  $$
  query\_q2 = query[..., 1::2].view(-1, 1)
  $$

  $$
  query\_rotate = torch.cat((-query\_q2, query\_q1), dim=-1).view(query.shape[0], query.shape[1], query.shape[2], query.shape[3])
  $$

  $$
  key\_q1 = key[..., ::2].view(-1, 1)
  $$

  $$
  key\_q2 = key[..., 1::2].view(-1, 1)
  $$

  $$
  key\_rotate = torch.cat((-key\_q2, key\_q1), dim=-1).view(key.shape[0], key.shape[1], key.shape[2], key.shape[3])
  $$

  $$
  q\_embed = (query * cos) + query\_rotate * sin
  $$

  $$
  k\_embed = (key * cos) + key\_rotate * sin
  $$

## Parameter Description

| Parameter | Input/Output/Attribute | Description            | Data Type                   | Data Format |
|-----|----------|---------------|------------------------|------|
| query      | Input/Output | Input and output tensor, that is, the input tensor query and output tensor q_embed in the formula; a 4D tensor | BFLOAT16, FLOAT16, FLOAT | ND   |
| key        | Input/Output | Input and output tensor, that is, the input tensor key and output tensor k_embed in the formula; a 4D tensor   | BFLOAT16, FLOAT16, FLOAT | ND   |
| cos        | Input     | Input tensor cos in the formula; a 4D tensor   | BFLOAT16, FLOAT16, FLOAT | ND   |
| sin        | Input     | Input tensor sin in the formula; a 4D tensor   | BFLOAT16, FLOAT16, FLOAT | ND   |
| layout     | Attribute     | Indicates the layout format of the input tensors | int64: 1-BSND, 2-SBND, 3-BNSD, 4-TND | - |
| rotary_mode| Attribute     | Rotation mode in the formula       | string: "half", "interleave", "quarter" modes | - |

- <term>Atlas inference products</term>: BFLOAT16 is not supported

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>:
    - The input tensors query, key, cos, and sin support 4D and 3D shapes, and layout supports 1-BSND and 4-TND. The first two dimensions (BSND format) or the first dimension (TND format) and the last dimension of the shapes of the four inputs must be equal. The second-to-last dimension (N dimension) of the shapes of cos and sin must be 1, and the last dimension of the input shapes must be 128 or 64.
    - The dtypes of the input tensors query, key, cos, and sin must be the same.
    - When layout is 1, the shape of the input queryRef is represented as (q_b, q_s, q_n, q_d), the keyRef shape as (q_b, q_s, k_n, q_d), and the shapes of cos and sin as (q_b, q_s, 1, q_d). Here, b indicates batch_size, s indicates seq_length, n indicates head_num, and d indicates head_dim. When layout is 4, the shape of the input queryRef is represented as (q_t, q_n, q_d), the keyRef shape as (q_t, k_n, q_d), and the shapes of cos and sin as (q_t, 1, q_d). Here, t indicates the merged axis of b and s, n indicates head_num, and d indicates head_dim

      - When the input is BFLOAT16, cast is represented as 1, castSize is 4, and DtypeSize is 2
      - When the input is FLOAT16 or FLOAT32, cast is represented as 0, and castSize = DtypeSize (2 for FLOAT16 and 4 for FLOAT32)

      lastDim indicates the value of head_dim in the last dimension of the input shape. The required UB space size is calculated as follows:
      `ub_required = (q_n + k_n) * lastDim * castSize * 2 + lastDim * DtypeSize * 4 + (q_n + k_n) * lastDim * castSize + (q_n + k_n) * lastDim * castSize * 2 + cast * (lastDim * 4 * 2)`,
      When the calculated size of `ub_required` exceeds the total UB space size of the current AI processor, the fused operator is not supported.
    - rotary_mode supports only "half". If `"interleave"` or `"quarter"` is passed in, the API does not return an error code, prints a warning, and performs the computation with the `"half"` semantics.
    - The empty tensor scenario is not supported.

- <term>Ascend 950PR&950DT products</term>:
    - The input tensors query, key, cos, and sin support only 4D shapes. For any layout, query and key must be the same in all dimensions except the N dimension; the shapes of cos and sin must be the same; the B dimension of cos and sin must be the same as the B dimension of query and key or equal to 1; the N dimension of cos and sin must be 1; and the S and D dimensions of query, key, cos, and sin must be the same, with the D dimension less than or equal to 1024.
    - The dtypes of the input tensors query, key, cos, and sin must be the same.
    - When rotary_mode is "half" or "interleave", the last dimension of the input shape must be divisible by 2. When rotary_mode is "quarter", the last dimension of the input shape must be divisible by 4.

## Calling Instructions

| Calling Method | Calling Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn call | [test_aclnn_apply_rotary_pos_emb.cpp](examples/test_aclnn_apply_rotary_pos_emb.cpp) | Calls the ApplyRotaryPosEmb operator through the [aclnnApplyRotaryPosEmb](docs/aclnnApplyRotaryPosEmb_en.md) API.    |
| aclnn call | [test_aclnn_apply_rotary_pos_emb_v2.cpp](examples/test_aclnn_apply_rotary_pos_emb_v2.cpp) | Calls the ApplyRotaryPosEmb operator through the [aclnnApplyRotaryPosEmbV2](docs/aclnnApplyRotaryPosEmbV2_en.md) API.    |
| PyTorch API | - | Calls the apply_rotary_pos_emb operator through the [apply_rotary_pos_emb](docs/torchapi_apply_rotary_pos_emb_en.md) API. |
