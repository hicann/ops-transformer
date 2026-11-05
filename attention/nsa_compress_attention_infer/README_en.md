# NsaCompressAttentionInfer

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

* Operator function: Computes Compress Attention during Native Sparse Attention inference.
* Computation formula:

<center>

  $$
  P_{cmp}= Softmax(scale * query · key^T) \\
  attentionOut = P_{cmp} · value\\
  P_{slc}[j] = \sum\limits_{m=0}^{l'/d -1} \sum\limits_{n = 0}^{l/d -1} P_{cmp} [l'/d * j -m - n]\\
  P_{slc'} = \sum\limits_{g=1}^{G}  P_{slc} ^g,\quad
  \text{where } G = \text{GroupSize (group size), that is: } G = \frac{\text{numHeads}}{\text{numKeyValueHeads}} \\
  topkIndices = topk(P_{slc'})\\
  $$

</center>

## Parameter Description

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Parameter Name</th>
    <th class="tg-0lax">Input/Output/Attribute</th>
    <th class="tg-0lax">Description</th>
    <th class="tg-0lax">Data Type</th>
    <th class="tg-0lax">Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">query</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">The input query in the formula.</td>
    <td class="tg-0lax">FLOAT16, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">key</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">The input key in the formula.</td>
    <td class="tg-0lax">FLOAT16, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">value</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">The input value in the formula.</td>
    <td class="tg-0lax">FLOAT16, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">scale</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">The input scale in the formula, indicating the scaling coefficient of the attention computation.</td>
    <td class="tg-0lax">DOUBLE</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">l'</td>
    <td class="tg-0lax">Attribute</td>
    <td class="tg-0lax">The input l' in the formula, indicating the block size in the select phase.</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">l</td>
    <td class="tg-0lax">Attribute</td>
    <td class="tg-0lax">The input l in the formula, indicating the block size in the compress phase.</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">d</td>
    <td class="tg-0lax">Attribute</td>
    <td class="tg-0lax">The input d in the formula, indicating the sliding window interval size between two compressions.</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">attentionOut</td>
    <td class="tg-0lax">Output</td>
    <td class="tg-0lax">attentionOut in the formula, the result of the attention computation.</td>
    <td class="tg-0lax">FLOAT16, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">topkIndices</td>
    <td class="tg-0lax">Output</td>
    <td class="tg-0lax">topkIndices in the formula, the indexes of the blocks with the highest importance scores.</td>
    <td class="tg-0lax">INT32</td>
    <td class="tg-0lax">-</td>
  </tr>
</tbody></table>

## Constraints

* The N in parameter query equals numHeads, the N of key and value equals numKeyValueHeads, and numHeads is a multiple of numKeyValueHeads.
* The D in parameter query equals the D (H/numKeyValueHeads) of key, and the D (H/numKeyValueHeads) of value equals the D of output.
* The functional usage restrictions of query, key, and value inputs are as follows:
  * The B axis supports values less than or equal to 3072.
  * The N axis of key/value supports values less than or equal to 256.
  * The ratio of the N axis of query to the N axis (H/D) of key/value (that is, the group size in GQA) supports values less than or equal to 16.
  * The D axis of query and key supports the value 192.
  * The D axis of value supports the value 128.
  * The blockSize of key and value supports 64 or 128.
  * In normal scenarios, only the value 1 is supported for the S axis of query.
  * In multi-token inference scenarios, only a maximum of 4 is supported for the S axis of query, and each batch must separately satisfy actualQSeqLen<=actualSelKvSeqLen.
  * Only paged attention is supported.
  * selectBlockSize supports only multiples of 16, with a maximum of 128.
  * The upper limit of selectBlockCount satisfies selectBlockCount*selectBlockSize<=MaxKvSeqlen, where MaxKvSeqlen=Max(actualSelKvSeqLenOptional).

## Calling Description

| Calling Method | Sample Code | Description |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_nsa_compress_attention_infer](./examples/test_aclnn_nsa_compress_attention_infer.cpp) | The NsaCompressAttentionInfer operator is called through the [`aclnnNsaCompressAttentionInfer`](docs/aclnnNsaCompressAttentionInfer_en.md) API. |
