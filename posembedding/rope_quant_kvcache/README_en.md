# RopeQuantKvcache

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Description

- Operator function: Splits the trailing axis of the input tensor into q, k, and v, and applies rotary positional encoding to q and k. The current kernel writes out q and updates the quantized k and v to kCacheRef and vCacheRef according to indice. The reserved k and v outputs are currently not written.

## Parameter Description

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter Name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data Type</th>
      <th style="white-space: nowrap">Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">qkv</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the tensor to be split.</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cos</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the tensor used for rotary positional encoding.</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">sin</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the tensor used for rotary positional encoding.</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quant_scale</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the tensor of quantization scale parameters.</td>
      <td style="white-space: nowrap">FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">quant_offset</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">aclTensor on the device, the tensor of quantization offsets.</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">k_cache</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input used for in-place update.</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">v_cache</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Input used for in-place update.</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">indice</td>
      <td style="white-space: nowrap">Input</td>
      <td style="white-space: nowrap">Index used to update the quantized result</td>
      <td style="white-space: nowrap">INT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">size_splits</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Used to split qkv.</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layout</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Data layout mode of qkv.</td>
      <td style="white-space: nowrap">String</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kv_output</td>
      <td style="white-space: nowrap">Attribute</td>
      <td style="white-space: nowrap">Controls whether to output the original k and v.</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">q</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Result of the split q after rotary positional encoding.</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">k</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Result of the split k after rotary positional encoding.</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">v</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">The split v.</td>
      <td style="white-space: nowrap">FLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">k_cache</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Result of the split k after rotary positional encoding and quantization.</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">v_cache</td>
      <td style="white-space: nowrap">Output</td>
      <td style="white-space: nowrap">Result of the split v after quantization.</td>
      <td style="white-space: nowrap">INT8</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody></table>

## Constraints

- `qkv` is 3-dimensional `[B, S, H]`, where `H=(Nq+Nkv+Nkv)*D`. `cos` and `sin` are 4-dimensional `[B, S, 1, D]`, whose B, S, and D must be identical to those of the split k. Broadcast is not supported.
- Both `quant_scale` and `quant_offset` are 1-dimensional `[D]`. kCache and vCache are both 4-dimensional `[B, S_cache, Nkv, D]` with identical shapes, and `S_cache` must be able to hold the update positions. The element count of `indice` is B.
- `size_splits` contains 3 items `[Nq*D, Nkv*D, Nkv*D]`, and the sum of the three items must equal the trailing axis H of `qkv`. Neither `layout` nor `kv_output` is currently consumed by the implementation.
- Dynamic shape (-1) and dynamic rank (-2) are not supported.

## Calling Description

| Calling Method | Calling Sample | Description |
|---|---|---|
| Graph mode invocation | [test_geir_rope_quant_kvcache](examples/test_geir_rope_quant_kvcache.cpp) | Invokes the operator in graph mode. |
