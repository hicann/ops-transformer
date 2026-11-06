# SparseLightningIndexerKLLossGrad

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|     √     |
|<term>Atlas A3 products</term>|    √     |
|<term>Atlas A2 products</term>|    √     |
|<term>Atlas 200I/500 A2 inference products</term>|      x     |
|<term>Atlas inference products</term>|      x     |
|<term>Atlas training products</term>|      x     |

## Operator Function

- Operator function: The SparseLightningIndexerKLLossGrad operator is the backward computation operator of the LightningIndexer KL loss. The operator receives q, k, w, and sparseIndices of the Lightning Indexer branch and attnSoftmaxL1Norm precomputed by the main Attention branch, and computes and outputs dq, dk, dw, and softmaxOut. Compared with SparseLightningIndexerGradKLLoss, this operator no longer recomputes the main Attention `Q @ K -> softmax -> reduce/l1 norm` inside the kernel and no longer outputs the loss; the target distribution of the main Attention is provided by the attnSoftmaxL1Norm input, and softmaxOut can be used for subsequent loss calculation.

- Calculation formula:
   The Indexer logits used to obtain the Top-k values can be expressed as:

   $$
   S_{t,:}=q_{t,:}@K_{\operatorname{topk}(t),:}^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   where $q$ and $K$ correspond to q and k of this operator, $W$ corresponds to w of this operator, and $\operatorname{topk}(t)$ is given by sparseIndices. The softmax output of the Indexer branch is:

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   This operator writes $y$ to softmaxOut. The target distribution $p$ is provided by the attnSoftmaxL1Norm input, obtained by averaging the main attention score over heads, without L1 normalization along the context. Let $Z_{t}$ be the sum of $p$ along the context, and $t$ be the L1-normalized target distribution:

   $$
   Z_{t}=\sum_{j}p_{t,j},\quad t_{t,:}=\frac{p_{t,:}}{Z_{t}}
   $$

   If the KL loss is further calculated, the corresponding loss function is:

   $$
   L_{t}=Z_{t}\,D_{KL}(t_{t,:}||y_{t,:})=\sum_{j}p_{t,j}(\log t_{t,j}-\log y_{t,j})
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the loss is obtained by derivation:

   $$
   dI_{t,:}=y_{t,:}\cdot Z_{t}-p_{t,:}
   $$

   The gradients of the w, q, and k matrices can be computed using the chain rule:

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K_{\operatorname{topk}(t),:}
   $$

   $$
   dK_{\operatorname{topk}(t),:}=\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   When dK is written back, scatter-add is performed on the key positions pointed to by sparseIndices, and invalid top-k positions do not participate in the calculation.

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
        <td>q</td>
        <td>Input</td>
        <td>Query input of the Lightning Indexer branch.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>k</td>
        <td>Input</td>
        <td>Key input of the Lightning Indexer branch.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>w</td>
        <td>Input</td>
        <td>Head weights of the Indexer logits.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sparseIndices</td>
        <td>Input</td>
        <td>Top-k key indices. Valid values in each row are placed first and invalid values are placed last; valid key indices are stored consecutively starting from column 0 of the last dimension, and only the tail is filled with -1 for invalid positions.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>attnSoftmaxL1Norm</td>
        <td>Input</td>
        <td>Target distribution precomputed by the main Attention branch, used to replace the main Attention recomputation result inside the old kernel.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqLensQOptional</td>
        <td>Optional input</td>
        <td>Cumulative sequence lengths of q under the TND layout, with shape (B+1,).</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqLensKOptional</td>
        <td>Optional input</td>
        <td>Cumulative sequence lengths of k under the TND layout, with shape (B+1,).</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>seqUsedQOptional</td>
        <td>Optional input</td>
        <td>Indicates the actually used q length of each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>seqUsedKOptional</td>
        <td>Optional input</td>
        <td>Indicates the actually used k length of each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpResidualKOptional</td>
        <td>Optional input</td>
        <td>Residual length of the compressed key. It must be passed in when maskMode=3 and cmpRatio!=1, with shape (B,).</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadataOptional</td>
        <td>Optional input</td>
        <td>Core splitting information generated by SparseLightningIndexerKLLossGradMetadata.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>layoutQ</td>
        <td>Attribute</td>
        <td>Layout format on the q side.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutK</td>
        <td>Attribute</td>
        <td>Layout format on the k side.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>maskMode</td>
        <td>Attribute</td>
        <td>Mask mode. Currently 0 and 3 are supported.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cmpRatio</td>
        <td>Attribute</td>
        <td>Compression ratio. The value range is [1,128].</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dq</td>
        <td>Output</td>
        <td>Gradient of q.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dk</td>
        <td>Output</td>
        <td>Gradient of k.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dw</td>
        <td>Output</td>
        <td>Gradient of w.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxOut</td>
        <td>Output</td>
        <td>Softmax output of the Indexer branch.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - T1 supports values greater than or equal to the last element of cuSeqLensQOptional, and T2 supports values greater than or equal to the last element of cuSeqLensKOptional.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: seqUsedQOptional, seqUsedKOptional, and cmpResidualKOptional are supported. When passed in, they must all be 1-dimensional INT32 tensors with shape (B,). When seqUsedQOptional or seqUsedKOptional is passed in, metadataOptional generated by the companion metadata operator must be passed in at the same time.

## Constraints

- Deterministic computation:
  - The external API of SparseLightningIndexerKLLossGrad does not provide a deterministic input parameter, and the kernel prefers the deterministic configuration in the runtime context.
- Common constraints
    - The data types of q, k, dq, and dk must be consistent, and FLOAT16 and BFLOAT16 are supported.
    - The data types of w, dw, attnSoftmaxL1Norm, and softmaxOut must be FLOAT32.
    - The data types of sparseIndices, cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional, and metadataOptional must be INT32.
    - layoutQ and layoutK currently support BSND and TND.
    - When layoutQ is TND, cuSeqLensQOptional must be passed in; when layoutK is TND, cuSeqLensKOptional must be passed in.
    - The valid positions in sparseIndices must be within the key sequence range of the current batch; valid key indices in each row are stored consecutively starting from column 0 of the last dimension, invalid positions are filled with -1 only at the tail, and filling -1 before or between valid positions is not supported.
    - It is recommended that invalid top-k positions in attnSoftmaxL1Norm be set to zero and be consistent with the valid positions of sparseIndices.
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 900px">
        <col style="width: 150px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask mode. No causal mask is applied.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask. The complete attenmask matrix must be passed in.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal mode mask.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal mode mask, corresponding to the lower triangle scenario partitioned by the right vertex.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band mode mask.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated.</td>
            <td>Not supported</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local.</td>
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
            <td>N2</td>
            <td>1</td>
            <td>Currently only N2=1 is supported.</td>
        </tr>
        <tr>
            <td>D</td>
            <td>128</td>
            <td>The last dimensions of q and k must be consistent.</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpRatio</td>
            <td>1~128</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

  - Parameter B support:
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: B supports 1~256.
    - <term>Ascend 950PR&950DT products</term>: B>0.
  - Parameter S1, S2 support:
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: S1 supports 1~8K, and S2 supports 1~512K.
    - <term>Ascend 950PR&950DT products</term>: S1>0 and S2>0.
  - Parameter N1 support:
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: N1 supports 8, 16, 32, 64.
    - <term>Ascend 950PR&950DT products</term>: N1 supports 1~128.
  - Parameter K support:
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: K supports 1~8192.
    - <term>Ascend 950PR&950DT products</term>: K supports 0~2048.

- Typical values
    <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
        <col style="width: 100px">
        <col style="width: 800px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Typical Value</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>q</td>
            <td>N1=64/32/16/8, D=128</td>
        </tr>
        <tr>
            <td>k</td>
            <td>N2=1, D=128</td>
        </tr>
        <tr>
            <td>w</td>
            <td>N1=64/32/16/8</td>
        </tr>
        <tr>
            <td>sparseIndices</td>
            <td>K=512/1024/2048</td>
        </tr>
        <tr>
            <td>attnSoftmaxL1Norm</td>
            <td>The shape is consistent with that of sparseIndices.</td>
        </tr>
        </tbody>
    </table>

## Calling Description

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn API | [aclnnSparseLightningIndexerKLLossGrad](docs/aclnnSparseLightningIndexerKLLossGrad_en.md) | Calls the SparseLightningIndexerKLLossGrad operator through the [aclnnSparseLightningIndexerKLLossGrad](docs/aclnnSparseLightningIndexerKLLossGrad_en.md) API. |
| pytest | [test_sparse_lightning_indexer_kl_loss_grad](./tests/pytest/test_sparse_lightning_indexer_kl_loss_grad.py) | Performs function and accuracy verification through the torch extension API. |
| PyTorch API | - | Calls the sparse_lightning_indexer_kl_loss_grad operator through the [sparse_lightning_indexer_kl_loss_grad](docs/torchapi_sparse_lightning_indexer_kl_loss_grad_en.md) API. |
