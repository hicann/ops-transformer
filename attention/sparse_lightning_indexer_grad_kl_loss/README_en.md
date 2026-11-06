# SparseLightningIndexerGradKLLoss

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|     √     |
|<term>Atlas A3 products</term>|    √     |
|<term>Atlas A2 products</term>|    √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Operator Function

- Operator function: The SparselightningIndexerGradKlLoss operator is the backward operator of LightningIndexer and additionally integrates the loss calculation function. The LightningIndexer operator selects the TopK tokens with the highest intrinsic relevance between QueryToken and KeyToken and stores them in SparseIndices, thereby reducing the Attention computation in long sequence scenarios and accelerating the inference and training performance of long sequence networks.

- Calculation formula:
   The calculation formula for the values used to obtain Top-k can be expressed as:

   $$
   I_{t,:}=W_{t,:}@ReLU(q_{t,:}@(K_{:t,:})^T)
   $$

   where $W$ is the weights corresponding to the $t$-th token, $q$ is the matrix obtained by merging the $G$ query heads of the $t$-th token along one axis, and $K$ is the $K$ matrix with $t$ rows.

   LightningIndexer is trained separately, and the corresponding loss function is:

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
   $$

   where $p$ is the target distribution, obtained by summing the main attention score over all heads and then performing L1 normalization on the sum along the context direction. $D_{KL}$ is the KL divergence, whose expression is:

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the loss is obtained by derivation:

   $$
   dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
   $$

   The gradients of the weights, query, and key matrices can be computed using the chain rule:

   $$
   dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left)  \left) \mathop{{}}\nolimits^{{T}}\right. \right. \right. \right.
   $$

   $$
   d\mathop{{q}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@K\mathop{{}}\nolimits_{{:t,:}}
   $$

   $$
   dK\mathop{{}}\nolimits_{{:t,:}}= \left( dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{{T}}@q\mathop{{}}\nolimits_{{:t,:}}\right. \right.
   $$

   where S is the softmax result of the QK matrix.

<!-- - **Description**:
   <blockquote>The query, key, and value data layout formats can be interpreted from multiple dimensions, where B (Batch) indicates the batch size of the input samples, S (Seq-Length) indicates the sequence length of the input samples, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the size of the smallest unit of the hidden layer, and D=H/N is satisfied; T indicates the sum of sequence lengths of the input samples over all batches.
   </blockquote> -->

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
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
        <td>Input Q of the attention structure.</td>
        <td>FLOAT16, BFLOAT16 </td>
        <td>ND</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K of the attention structure.</td>
        <td>FLOAT16, BFLOAT16 </td>
        <td>ND</td>
    </tr>
    <tr>
        <td>queryIndex</td>
        <td>Input</td>
        <td>Input queryIndex of the lightingIndexer structure.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>keyIndex</td>
        <td>Input</td>
        <td>Input keyIndex of the lightingIndexer structure.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>weights</td>
        <td>Input</td>
        <td>Weights.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sparseIndices</td>
        <td>Input</td>
        <td>topk_index, used to select the key and value corresponding to each query.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxMax</td>
        <td>Input</td>
        <td>Intermediate output of the attention forward computation.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxSum</td>
        <td>Input</td>
        <td>Intermediate output of the attention forward computation.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>queryRope</td>
        <td>Input</td>
        <td>MLA rope part: output of the Query position encoding.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>keyRope</td>
        <td>Input</td>
        <td>MLA rope part: output of the Key position encoding.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthsQuery</td>
        <td>Input</td>
        <td>Number of valid Query tokens in each batch.</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKey</td>
        <td>Input</td>
        <td>Number of valid Key tokens in each batch.</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Scaling factor.</td>
        <td>DOUBLE</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout</td>
        <td>Input</td>
        <td>Layout format.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
        <td>sinks</td>
        <td>Optional input</td>
        <td>Additional softmax term used to correct the normalization factor of the target distribution (p). It is supported only on Ascend 950 and received only by the aclnnSparseLightningIndexerGradKLLossV2 API. It must be 1-dimensional with a length equal to the N dimension of query.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
    <td>deterministic</td>
        <td>Input</td>
        <td>Deterministic computation.</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQueryIndex</td>
        <td>Output</td>
        <td>Gradient of QueryIndex.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKeyIndex</td>
        <td>Output</td>
        <td>Gradient of KeyIndex.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dWeights</td>
        <td>Output</td>
        <td>Gradient of Weights.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>loss</td>
        <td>Output</td>
        <td>Value of the loss function.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - T1 supports values greater than or equal to the cumulative sum of actualSeqLengthsQuery, and T2 supports values greater than or equal to the cumulative sum of actualSeqLengthsKey.

## Constraints

- Common constraints
    - The data types of query, key, queryIndex, and keyIndex must be consistent.
    - If weights is not float32, the data types of query, key, queryIndex, keyIndex, and weights must be consistent.
    - Handling of scenarios where input parameters are empty:
        - If query is an empty Tensor, the operator returns directly.
        - For scenarios where input parameters are empty, the common constraints are the same as those of FAG.
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 900px">
        <col style="width: 150px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask mode. If attenmask is not passed in, no mask operation is performed, and preTokens and nextTokens are ignored; if attenmask is passed in, the complete attenmask matrix must be passed in, indicating that the part between preTokens and nextTokens needs to be computed</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask. The complete attenmask matrix must be passed in</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal mode mask. The optimized attenmask matrix must be passed in</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal mode mask, corresponding to the lower triangle scenario partitioned by the right vertex. The optimized attenmask matrix must be passed in</td>
            <td>Supported</td>
  </tr>
        <tr>
            <td>4</td>
            <td>band mode mask. The optimized attenmask matrix must be passed in</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>Not supported</td>
        </tr>
        </tbody>
    </table>

- Specification constraints
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 550px">
        <col style="width: 500px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Specification</th>
                <th>Specification Description</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>B</td>
            <td>Supports 1~256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1, S2</td>
            <td>S1 supports 1~32K, and S2 supports 1~1M</td>
            <td>S1 and S2 support unequal lengths; S1 must be less than or equal to S2</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>
            32, 64, 128
            </td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>Nidx1</td>
            <td>
            8, 16, 32, 64
            </td>
            <td>SparseFA is MQA.</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA is MQA, and Nidx2=1.</td>
        </tr>
        <tr>
            <td>Nidx2</td>
            <td>1</td>
            <td>SparseFA is MQA, and N2=1.</td>
        </tr>
        <tr>
            <td>DQuery</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>DQueryIndex</td>
            <td>128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>DRope</td>
            <td>64</td>
            <td>-</td>
        </tr>
        <tr>
            <td>K</td>
            <td>1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

    <term>Ascend 950PR&950DT products</term>: N1 additionally supports 48, and Nidx1 additionally supports 24. Only the (48,24) combination is allowed, and pairing with other values is prohibited.

    <term>Ascend 950PR&950DT products</term>: B, S1, and S2 all support generalization.

## Calling Description

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn API | [test_aclnn_sparse_lightning_indexer_grad_kl_loss](./examples/test_aclnn_sparse_lightning_indexer_grad_kl_loss.cpp) | Calls the SparseLightningIndexerGradKLLoss operator through the [aclnnSparseLightningIndexerGradKLLoss](docs/aclnnSparseLightningIndexerGradKLLoss_en.md) API.             |
