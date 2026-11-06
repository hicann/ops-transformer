# ApplyRotaryPosEmbGrad

English | [简体中文](./README.md)

## Product Support

| Product                                                         |  Supported   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×    |
| <term>Atlas A2 products</term> |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Description

- **Operator function**: Performs the backward computation of the dual-path rotary position embedding [ApplyRotaryPosEmb](../apply_rotary_pos_emb/README_en.md). Computes the rope backward gradients of **query** and **key** at the same time and fuses them into one kernel call, saving the cos/sin loading and kernel launch overhead.

- **Computation formula**:

    In the forward computation of the rotary position embedding, the list of broadcast axes is `dims`. The computation formula can be expressed as follows:

    (1) half mode:
    $$
    grad\_q_1, grad\_q_2 = chunk(grad\_query\_embed, chunks=2, dim=-1)
    $$

    $$
    grad\_k_1, grad\_k_2 = chunk(grad\_key\_embed, chunks=2, dim=-1)
    $$

    $$
    cos_1, cos_2 = chunk(cos, chunks=2, dim=-1)
    $$

    $$
    sin_1, sin_2 = chunk(sin, chunks=2, dim=-1)
    $$

    $$
    query\_rotate = cat((-query_2, query_1), dim=-1)
    $$

    $$
    key\_rotate = cat((-key_2, key_1), dim=-1)
    $$

    $$
    grad\_query = cat(cos_1 * grad\_q_1 + sin_2 * grad\_q_2, cos_2 * grad\_q_2 - sin_1 * grad\_q_1, dim=-1)
    $$

    $$
    grad\_key = cat(cos_1 * grad\_k_1 + sin_2 * grad\_k_2, cos_2 * grad\_k_2 - sin_1 * grad\_k_1, dim=-1)
    $$

    $$
    grad\_cos = sum(grad\_query\_embed * query + grad\_key\_embed * key, dims)
    $$

    $$
    grad\_sin = sum(grad\_query\_embed * query\_rotate + grad\_key\_embed * key\_rotate, dims)
    $$

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
|-----|----------|---------------|------------------------|------|
| grad_query_embed | Input | Derivative of the forward output query, corresponding to $grad\_q_{embed}$ in the formula. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| grad_key_embed | Input | Derivative of the forward output key, corresponding to $grad\_k_{embed}$ in the formula. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| cos | Input | Forward computation input cos, whose data type must be the same as that of grad_query_embed. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| sin | Input | Forward computation input sin, whose data type must be the same as that of grad_query_embed. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| query | Optional input | Forward computation input query. If it is a null pointer, grad_cos and grad_sin are not computed. It must be passed in or not passed in together with key. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| key | Optional input | Forward computation input key. If it is a null pointer, grad_cos and grad_sin are not computed. It must be passed in or not passed in together with query. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| rotary_mode | Attribute | Rotation mode. Only "half" is supported. | STRING | - |
| layout | Attribute | Layout format of the input tensor. 1=BSND, 2=SBND, 4=TND. The default value is 1. | INT64 | - |
| grad_query | Output | Derivative of the forward computation input query, with the same shape as grad_query_embed. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| grad_key | Output | Derivative of the forward computation input key, with the same shape as grad_key_embed. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| grad_cos | Output | Derivative of the forward computation input cos, which is valid only when query and key are not null. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| grad_sin | Output | Derivative of the forward computation input sin, which is valid only when query and key are not null. | BFLOAT16, FLOAT16, FLOAT32 | ND |

## Constraints

  - The constraints on each parameter are described as follows:
    - The input and output tensors support only 3D or 4D shapes: 4D when layout is 1 or 2, and 3D when layout is 4.
    - The dtypes of the input and output tensors must be the same.
    - Empty tensors are not supported for the input and output tensors (each dimension must be greater than 0).
    - The layouts of the input and output tensors must be the same.
    - The D axis of the input and output tensors must be the same. In half mode, it must be ≤1024 and divisible by 2.
    - The shapes of `grad_query_embed` and `grad_query` must be the same, and the shapes of `grad_key_embed` and `grad_key` must be the same.
    - For any `layout`, `grad_query_embed` and `grad_key_embed` must be the same in all dimensions except the N dimension.
    - The N dimension of `cos` and `sin` must be 1. When `layout` is 1 (BSND) or 2 (SBND), the B dimension of `cos` and `sin` can be 1 or the same as the B dimension of `grad_query_embed`. When `layout` is 4 (TND), the T dimension of `cos` and `sin` must be the same as the T dimension of `grad_query_embed`. Except for the N dimension (and the optional broadcast B dimension in the BSND and SBND layouts), the other dimensions must be the same as those of `grad_query_embed`.
    - The shapes of `cos`, `sin`, `grad_cos`, and `grad_sin` must be the same.
    - The dimensions of `query` must be the same as those of `grad_query_embed`, the dimensions of `key` must be the same as those of `grad_key_embed`, and `query` and `key` must be passed in or not passed in together.
    - `rotary_mode` supports only "half".
    - `layout` supports only {1, 2, 4}, corresponding to {BSND, SBND, TND}. 3=BNSD is reserved and is not supported yet.

## Calling Instructions

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_apply_rotary_pos_emb_grad](./examples/test_aclnn_apply_rotary_pos_emb_grad.cpp) | Calls the ApplyRotaryPosEmbGrad operator through the [aclnnApplyRotaryPosEmbGrad](docs/aclnnApplyRotaryPosEmbGrad_en.md) API.             |
| PyTorch API | - | Calls the apply_rotary_pos_emb_grad operator through the [apply_rotary_pos_emb_grad](docs/torchapi_apply_rotary_pos_emb_grad_en.md) API. |
