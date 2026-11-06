# AttentionWorkerCombine

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :---- | :----: |
|<term>Ascend 950PR&950DT products</term>| √ |
|<term>Atlas A3 products</term>| √ |
|<term>Atlas A2 products</term>| √ |
|<term>Atlas 200I/500 A2 inference products</term>| × |
|<term>Atlas inference products</term>| × |
|<term>Atlas training products</term>| × |

## Function Description

+ Operator function: Fuses the attention token data processed by multiple compute units, weights the results with expert weights, outputs the final attention fusion result, and updates the layer ID.

## Parameter Description

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Parameter name</th>
    <th class="tg-0pky">Input/Output/Attribute</th>
    <th class="tg-0pky">Description</th>
    <th class="tg-0pky">Data type</th>
    <th class="tg-0pky">Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">schedule_context</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Contains scheduling context information.</td>
    <td class="tg-0pky">INT8</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">expert_scales</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Expert weights. The type is FLOAT8_E8M0 in the MXFP dequantization scenario.</td>
    <td class="tg-0pky">FLOAT, FLOAT8_E8M0</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">layer_id</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Current model layer ID.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">y</td>
    <td class="tg-0pky">Output</td>
    <td class="tg-0pky">Final attention merge result.</td>
    <td class="tg-0pky">FLOAT16, BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">next_layer_id</td>
    <td class="tg-0pky">Output</td>
    <td class="tg-0pky">ID of the next layer to process.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">hidden_size</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Hidden dimension size of token_data, used to determine the size of the second dimension of the output y. Required attribute.</td>
    <td class="tg-0pky">Int</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">token_dtype</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Specifies the original precision type of the token data in schedule_context. 0 indicates FLOAT16, and 1 indicates BFLOAT16; <term>Ascend 950PR&950DT products</term> supports MXFP type dequantization. 2 indicates MXFP8_E5M2, 3 indicates MXFP8_E4M3FN, and 4 indicates MXFP4_E2M1.</td>
    <td class="tg-0pky">Int</td>
    <td class="tg-0pky">-</td>
  </tr>
    <tr>
    <td class="tg-0pky">need_schedule</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Specifies whether to wait until the token data is filled before execution. 0 indicates not waiting, and 1 indicates waiting.</td>
    <td class="tg-0pky">Int</td>
    <td class="tg-0pky">-</td>
  </tr>
</tbody></table>

## Constraint Description

* In the non-quantized scenario, expert_scales is a FLOAT 2D tensor, [BatchSize, K], where K ranges from [1, 64].
* In the MXFP dequantization scenario, expert_scales is an E8M0 3D tensor with the shape [BatchSize, K+1, S], where S = Ceil(H/32), and then S is rounded up to an even number.
* In the MXFP dequantization scenario, token_dtype=2/3/4 is supported; R/H must be positive, K ranges from [1, 64], and need_schedule is 0 or 1.
* y is a 2D tensor, [BatchSize, HiddenSize]. That is, the second dimension is determined by the attribute hidden_size.

## Invocation Description

| Invocation method | Sample code | Description |
| ---- | ---- | ---- |
| Graph mode invocation | [test_geir_attention_worker_combine.cpp](examples/test_geir_attention_worker_combine.cpp) | Invokes the AttentionWorkerCombine operator by constructing the graph through the [operator IR](op_graph/attention_worker_combine_proto.h). |
| aclnn interface | [test_aclnn_attention_worker_combine.cpp](examples/test_aclnn_attention_worker_combine.cpp) | Invokes the AttentionWorkerCombine operator through the [aclnnAttentionWorkerCombine](docs/aclnnAttentionWorkerCombine_en.md) interface. |
| PyTorch interface | [torchapi_attention_worker_combine.md](docs/torchapi_attention_worker_combine.md) | Invoked through `cann_ops_transformer.attention_worker_combine`, supporting the non-quantized and MXFP dequantization scenarios. |
