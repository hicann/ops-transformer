# SparseFlashMlaGrad

English | [简体中文](./README.md)

## Product Support Status

|Product   | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: Computes the backward output of attention in the `SparseFlashMla` training scenario, and supports Sliding Window Attention, Compressed Attention, and Sparse Compressed Attention.

- Computation formula:

    Stage 1: Selects the input ori_kv and cmp_kv according to different cmp_ratio scenarios

    * When cmp_ratio = 1 (SWA):

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

    * When cmp_ratio = 4 (SCFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else (CFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    Stage 2: Computes P, dP, and dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    Stage 3: Computes dQ, dKV, and dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
    $$

## Parameter Description

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
        <td>lse</td>
        <td>Input</td>
        <td>Output lse of the forward attention computation. For the computation formula, see the forward document.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriKvOptional</td>
        <td>Input</td>
        <td>Original input K(V) of the attention structure.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpKvOptional</td>
        <td>Input</td>
        <td>K(V) compressed by the Compressor.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriSparseIndicesOptional</td>
        <td>Input</td>
        <td>Attention indices with higher weights selected from oriKvOptional in the sparse scenario.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpSparseIndicesOptional</td>
        <td>Input</td>
        <td>Attention indices with higher weights selected from cmpKvOptional in the sparse scenario.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensQOptional</td>
        <td>Input</td>
        <td>Number of valid Query tokens in each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensOriKvOptional</td>
        <td>Input</td>
        <td>Number of valid oriKvOptional tokens in each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensCmpKvOptional</td>
        <td>Input</td>
        <td>Number of valid cmpKvOptional tokens in each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedQOptional</td>
        <td>Input</td>
        <td>Number of query tokens that actually participate in the computation in different batches.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedOriKvOptional</td>
        <td>Input</td>
        <td>Number of oriKvOptional tokens that actually participate in the computation in different batches.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedCmpKvOptional</td>
        <td>Input</td>
        <td>Number of cmpKvOptional tokens that actually participate in the computation in different batches.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpResidualKvOptional</td>
        <td>Input</td>
        <td>Remainder of S2 // cmpRatio for each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriTopkLengthOptional</td>
        <td>Input</td>
        <td>Actual selectable topk length of oriKvOptional corresponding to each row of query.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpTopkLengthOptional</td>
        <td>Input</td>
        <td>Actual selectable topk length of cmpKvOptional corresponding to each row of query.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sinksOptional</td>
        <td>Input</td>
        <td>Attention sink tensor.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadataOptional</td>
        <td>Input</td>
        <td>Output result of the AICPU operator with tiling offloaded.</td>
        <td>INT32</td>
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
        <td>cmpRatio</td>
        <td>Attribute</td>
        <td>Compression ratio of oriKvOptional.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>oriMaskMode</td>
        <td>Attribute</td>
        <td>Mask mode for the computation between query and oriKvOptional.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cmpMaskMode</td>
        <td>Attribute</td>
        <td>Mask mode for the computation between query and cmpKvOptional.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>oriWinLeft</td>
        <td>Attribute</td>
        <td>Number of past tokens computed by query in the computation between query and oriKvOptional.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>oriWinRight</td>
        <td>Attribute</td>
        <td>Number of future tokens computed by query in the computation between query and oriKvOptional.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutQOptional</td>
        <td>Attribute</td>
        <td>Data layout format of the input query.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutKvOptional</td>
        <td>Attribute</td>
        <td>Data layout format of the inputs ori_kv and cmp_kv.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQueryOut</td>
        <td>Output</td>
        <td>Gradient of query.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dOriKvOutOptional</td>
        <td>Output</td>
        <td>Gradient of oriKvOptional.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dCmpKvOptional</td>
        <td>Output</td>
        <td>Gradient of cmpKvOptional.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dSinksOutOptional</td>
        <td>Output</td>
        <td>Gradient of sinksOptional.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>oriSoftmaxL1NormOptional</td>
        <td>Output</td>
        <td>Softmax result computed from query and oriKvOptional.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpSoftmaxL1NormOptional</td>
        <td>Output</td>
        <td>Softmax result computed from query and cmpKvOptional.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The oriTopkLengthOptional, cmpTopkLengthOptional, and metadataOptional fields are not supported yet.

## Constraints

- Only the BSND or TND layout is supported. The constraints on the data shape are as follows:
  - B: Generalized support.
  - S1, S2: Generalized support; S1 and S2 can have different lengths.
  - N1: Supports 1 to 128.
  - N2: Supports only 1.
  - D: Supports only 512.

## Calling Example

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Calling Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_sparse_flash_mla_grad.cpp">test_aclnn_sparse_flash_mla_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    Calls the operator through the
    <a href="./docs/aclnnSparseFlashMlaGrad_en.md">aclnnSparseFlashMlaGrad
    </a>
    API
    </td>
  </tr>
  <tr>
    <td class="tg-9wq8">PyTorch API</td>
    <td class="tg-0pky">-</td>
    <td class="tg-lboi">Calls the sparse_flash_mla_grad operator through the <a href="../../attention/sparse_flash_mla_grad/docs/torchapi_sparse_flash_mla_grad_en.md">sparse_flash_mla_grad</a> API.</td>
  </tr>
</tbody></table>
