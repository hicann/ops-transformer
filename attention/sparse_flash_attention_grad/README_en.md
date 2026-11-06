# SparseFlashAttentionGrad

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

## Description

- Operator function: Rearranges blocks of size selectedBlockSize selected from key and value based on topkIndices, and then computes the backward output of attention in the training scenario.

- Formula: Rearranges selectedBlockCount blocks of size selectedBlockSize selected from keyIn and value based on the input topkIndices. The formulas are as follows:

  $$
   selectedKey\text{ }=\text{ }Gather \left( key,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

  $$
   selectedValue\text{ }=\text{ }Gather \left( value,topkIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

  Stage 1: Compute $dP$ and $dV$ according to the matrix multiplication derivative rules:

</div>

  $$
   dP\mathop{{}}\nolimits_{{t,:}}=dO\mathop{{}}\nolimits_{{t,:}}\text{@}V\mathop{{}}\nolimits^{{T}}
  $$

  $$
   dV \left[ u \left] =P\mathop{{}}\nolimits_{{T}}^{{t,:}}\text{@}dO\mathop{{}}\nolimits_{{t,:}}\right. \right.
  $$

<div style="padding-left:40px;">

   Stage 2: Compute $dS$:

</div>

  $$
   d\mathop{{S}}\nolimits_{{t,:}}= \left[ P\mathop{{}}\nolimits_{{t,:}}@ \left( dP\mathop{{}}\nolimits_{{t,:}}-FlashSoftmaxGrad \left( dO,O \left)  \left)  \right] \right. \right. \right. \right.
  $$

<div style="padding-left:40px;">

   Stage 3: Compute $dQ$ and $dK$:

</div>

  $$
   d\mathop{{Q}}\nolimits_{{t,:}}=d\mathop{{S}}\nolimits_{{t,:}}@K \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}/\sqrt{{d\mathop{{}}\nolimits_{{k,:}}}}\right. \right.
  $$

  $$
   dK \left[ u \left] \mathop{{}}\nolimits_{{:t,:}}=dS\mathop{{}}\nolimits_{{t,:t}}\mathop{{}}\nolimits^{{T}}\text{@}Q/\sqrt{{d\mathop{{}}\nolimits_{{t,:}}}}\right. \right.
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1080px">
    <colgroup>
        <col style="width: 170px">
        <col style="width: 120px">
        <col style="width: 300px">
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
        <td>Input Q of the attention structure.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K of the attention structure.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input v of the attention structure.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sparseIndices</td>
        <td>Input</td>
        <td>Attention indices with high weights selected in the sparse scenario.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dOut</td>
        <td>Input</td>
        <td>Gradient of the attention output matrix.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>out</td>
        <td>Input</td>
        <td>Attention output matrix.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxMax</td>
        <td>Input</td>
        <td>Intermediate output of the forward attention computation.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxSum</td>
        <td>Input</td>
        <td>Intermediate output of the forward attention computation.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthsQueryOptional</td>
        <td>Input</td>
        <td>Number of valid Query tokens in each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthskvOptional</td>
        <td>Input</td>
        <td>Number of valid Key and value tokens in each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>MLA rope part: output of the Query positional encoding.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>MLA rope part: output of the Key positional encoding.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>Attribute</td>
        <td>Scaling factor.</td>
        <td>FLOAT32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseBlockSize</td>
        <td>Attribute</td>
        <td>Size of the selected block.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout</td>
        <td>Attribute</td>
        <td>Layout format.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>Attribute</td>
        <td>Sparse mode.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
    <td>preTokens</td>
        <td>Attribute</td>
        <td>In the Attention operator, the start position of the sliding window on the S matrix.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
    <td>nextTokens</td>
        <td>Attribute</td>
        <td>In the Attention operator, the end position of the sliding window on the S matrix.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
    <td>deterministic</td>
        <td>Attribute</td>
        <td>Deterministic computation.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQuery</td>
        <td>Output</td>
        <td>Gradient of query.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKey</td>
        <td>Output</td>
        <td>Gradient of key.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dValue</td>
        <td>Output</td>
        <td>Gradient of value.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
      <tr>
        <td>dQueryRopeOptional</td>
        <td>Output</td>
        <td>Gradient of queryRope.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKeyRopeOptional</td>
        <td>Output</td>
        <td>Gradient of keyRope.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

## Constraints

- D of query and D of key and value must both be 512, and Dr of query_rope and Dr of key_rope must both be 64.
- The data types of query, key, and value must be the same.
- Currently, only the scenario where value is exactly the same as key is supported.
- Currently, only sparseMode=0 or 3 is supported (no mask, or the lower triangular scenario partitioned from the right vertex)
- Only BSND or TND layout is supported. The constraints on data shapes are as follows:
  - B: value range 1 to 256.
  - S1 supports 1 to 128K, and S2 supports 1 to 1M. S1 and S2 support different lengths.
  - N1 supports 1/2/4/8/16/32/64/128.
    - <term>Ascend 950PR&950DT products</term>:
      - Additionally, 48, 24, 12, 6, and 3 are supported.
  - N2: only 1 is supported.
  - D: only 512 is supported.
  - Drope: only 64 is supported.
  - topk: 1024, 2048, 3072, 4096, 5120, 6144, 7168, and 8192.
    - It is not recommended that topk * sparseBlockSize exceed 100k, because the hardware limitations of the internal algorithm may cause OOM.


## Invocation Example

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Invocation Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_sparse_flash_attention_grad.cpp">test_aclnn_sparse_flash_attention_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Calls the operator through the
    <a href="./docs/aclnnSparseFlashAttentionGrad_en.md">aclnnSparseFlashAttentionGrad
    </a>
    API
    </td>
  </tr>
</tbody></table>
