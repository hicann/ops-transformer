# NsaCompressWithCache

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
|<term>Kirin X90 processor products</term> | √ |
|<term>Kirin 9030 processor products</term> | √ |

## Function Description

+ Operator function: The operator compresses KV in the Native-Sparse-Attention inference phase. Each inference produces a new token for each batch. Whenever the number of tokens of a batch fills a compress\_block, the operator compresses the last compress\_block tokens of the batch into one compress\_token. The algorithm flow is as follows:

  1. Check whether act\_seq\_lens contains a sequence length that satisfies $s \ge compressBlockSize$ and $(s - compressBlockSize) \% stride ==0$;
  2. Find the batchIdx that satisfies the sequence length, and compress the last compress\_block\_size tokens of the batch according to block\_table;
  3. Execute the compression algorithm;
  4. Write back to output\_cache according to slot\_mapping.
+ Computation formula:

$$
compressIdx=(s-compressBlockSize)/stride\\
outputCacheRef[slotMapping[i]] = input[compressIdx*stride : compressIdx*stride+compressBlockSize]*weight[:]
$$

## Parameter Description

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Parameter Name</th>
    <th class="tg-0pky">Input/Output/Attribute</th>
    <th class="tg-0pky">Description</th>
    <th class="tg-0pky">Data Type</th>
    <th class="tg-0pky">Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">s</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">The token length of the current batch.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">compressBlockSize</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Compression sliding window size.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">stride</td>
    <td class="tg-0pky">Attribute</td>
    <td class="tg-0pky">Interval size between two compression sliding windows.</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">weight</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">The compression weight of k/v values.</td>
    <td class="tg-0lax">BFLOAT16, FLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">input</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">The cache of k/v values.</td>
    <td class="tg-0pky">BFLOAT16, FLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">slotMapping</td>
    <td class="tg-0pky">Input</td>
    <td class="tg-0pky">The index of the storage location of the compressed data at the tail of each batch.</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">outputCacheRef</td>
    <td class="tg-0pky">Input/Output</td>
    <td class="tg-0pky">The output cache.</td>
    <td class="tg-0pky">BFLOAT16, FLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

## Constraints

* input and weight are in a broadcast relationship; the third dimension of input equals the second dimension of weight.
* compressBlockSize and stride must be multiples of 16, compressBlockSize>=stride, and compressBlockSize<=64.
* actSeqLenType currently supports only the value 1.
* layoutOptional can be BSH, SBH, BSND, BNSD, or TND, but it does not take effect.
* pageBlockSize can only be 64 or 128.
* headDim is a multiple of 16, and headDim<=256.
* Empty input for input/weight/outputCache is not supported.
* The values of slotMapping must be unique; otherwise, the computation result becomes unstable.
* The values of blockTableOptional must not exceed blockNum; otherwise, an out-of-bounds access occurs.
* The values of actSeqLenOptional must not exceed the maximum sequence length.
* headNum<=64, and headNum%2=0 when headNum>50.

## Calling Description

| Calling Method | Sample Code | Description |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_nsa_compress_with_cache](./examples/test_aclnn_nsa_compress_with_cache.cpp) | The NsaCompressWithCache operator is called through the [`aclnnNsaCompressWithCache`](docs/aclnnNsaCompressWithCache_en.md) API. |
