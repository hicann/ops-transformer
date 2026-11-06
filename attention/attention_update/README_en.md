# AttentionUpdate

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

+ Operator function: Updates the intermediate results lse and localOut output by the PA operators in each SP domain into the global results.
+ Computation formula: Inputs $lse_i$ and $O_i$, and output $O$.

$$
lse_{max} = \text{max}lse_i
$$

$$
lse = \sum_i \text{exp}(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \text{log}(lse)
$$

$$
O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
$$

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
    <td class="tg-0pky">lse<sub>i</sub></td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Local lse of each SP domain.</td>
    <td class="tg-0pky">FLOAT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">O<sub>i</sub></td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">Local attentionout of each SP domain.</td>
    <td class="tg-0pky">FLOAT32, FLOAT16, BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">lse<sub>m</sub></td>
    <td class="tg-0pky">Output</td>
    <td class="tg-0pky">Updated global lse.</td>
    <td class="tg-0pky">FLOAT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">O</td>
    <td class="tg-0pky">Output</td>
    <td class="tg-0pky">Updated global attentionout.</td>
    <td class="tg-0pky">FLOAT32, FLOAT16, BFLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

## Constraint Description

* For <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: O<sub>i</sub> and O of the FLOAT32, FLOAT16, and BFLOAT16 types are supported.
* For <term>Ascend 950PR&950DT products</term>: O<sub>i</sub> and O of the FLOAT32, FLOAT16, and BFLOAT16 types are supported, and O<sub>i</sub> and O must have the same data type.
* The value range of the sequence parallelism degree sp:
  * <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: [1, 128].
  * <term>Ascend 950PR&950DT products</term>: [1, 16].
* headDim ranges from [8, 512] and must be a multiple of 8.
* localout supports non-contiguous tensors.
* Empty input with T=0 is supported. In this case, the output is empty and no computation is performed.

## Invocation Description

| Invocation method  | Sample code                                                                | Description                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn interface | [test_aclnn_attention_update](./examples/test_aclnn_attention_update.cpp) | Invokes the AttentionUpdate operator through the [aclnnAttentionUpdate](docs/aclnnAttentionUpdate_en.md) interface. |
