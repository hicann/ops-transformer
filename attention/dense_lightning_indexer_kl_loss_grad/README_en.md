# DenseLightningIndexerKLLossGrad

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|     √     |
|<term>Atlas A3 products</term>|    x     |
|<term>Atlas A2 products</term>|    x     |
|<term>Atlas 200I/500 A2 inference products</term>|      x     |
|<term>Atlas inference products</term>|      x     |
|<term>Atlas training products</term>|      x     |

## Operator Function

- Operator function: The DenseLightningIndexerKLLossGrad operator is the backward computation operator of LightningIndexer KL Loss (Dense mode). This API receives q, k, and w of the Lightning Indexer branch, as well as attnSoftmaxL1Norm and softmaxLse precomputed by the main Attention branch, and computes and outputs dq, dk, dw, and softmaxOut. This API performs dense attention computation over all keys and no longer relies on sparseIndices for top-k filtering. The API also does not need to recompute the main Attention `Q @ K -> softmax -> reduce/l1 norm` inside the kernel and no longer outputs the loss. The target distribution of the main Attention is provided by the attnSoftmaxL1Norm input, and softmaxOut can be used for subsequent loss computation.

- Calculation formulas:
    The Indexer logits can be expressed as:

   $$
   S_{t,:}=q_{t,:}@K^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   Here, $q$ and $K$ correspond to q and k of this API respectively, and $W$ corresponds to w of this API. The softmax output of the Indexer branch is:

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   This API writes $y$ to softmaxOut. The target distribution $p$ is provided by the attnSoftmaxL1Norm input, which is equivalent to the result obtained inside the kernel of the old version by summing the main attention score over heads and applying L1 normalization. If the KL Loss is subsequently computed, its form remains the same as that of the old version:

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the Loss is obtained by differentiation:

   $$
   dI_{t,:}=\operatorname{Softmax}(I_{t,:})-p_{t,:}
   $$

   The gradients of the w, q, and k matrices can be computed using the chain rule:

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K
   $$

   $$
   dK=\sum_t\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK is directly accumulated to the corresponding key positions, without scatter-add.

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
        <th>Parameter</th>
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
        <td>attn_softmax_l1_norm</td>
        <td>Input</td>
        <td>Target distribution precomputed by the main Attention branch, used to replace the main Attention recomputation result inside the old kernel.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>Input</td>
        <td>Softmax log-sum-exp value, precomputed by DenseLightningIndexerSoftmaxLse.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cu_seqLens_q</td>
        <td>Optional input</td>
        <td>Cumulative sequence lengths of q under the TND layout, with shape (B+1,).</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cu_seqLens_k</td>
        <td>Optional input</td>
        <td>Cumulative sequence lengths of k under the TND layout, with shape (B+1,).</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>seqused_q</td>
        <td>Optional input</td>
        <td>Actually used q length of each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>seqused_k</td>
        <td>Optional input</td>
        <td>Actually used k length of each batch.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmp_residual_k</td>
        <td>Optional input</td>
        <td>Used to compute the S2 length before compression when maskMode=3 and cmpRatio!=1.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>Optional input</td>
        <td>Core-splitting information generated by DenseLightningIndexerKLLossGradMetadata.</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>layout_q</td>
        <td>Attribute</td>
        <td>Layout format on the q side.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout_k</td>
        <td>Attribute</td>
        <td>Layout format on the k side.</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mask_mode</td>
        <td>Attribute</td>
        <td>Mask mode. Currently 0 and 3 are supported.</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cmp_ratio</td>
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
        <td>softmax_out</td>
        <td>Output</td>
        <td>Softmax output of the Indexer branch.</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

## Constraints

- Common constraints
    - The data types of parameters q, k, dq, and dk should be the same. FLOAT16 and BFLOAT16 are supported.
    - The data types of parameters w, dw, attnSoftmaxL1Norm, and softmaxOut should be FLOAT32.
    - The data types of parameters cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional, and metadataOptional should be INT32.
    - layoutQ and layoutK currently support BSND and TND.
    - When layoutQ is TND, cuSeqLensQOptional must be passed in. When layoutK is TND, cuSeqLensKOptional must be passed in.
  - The dimensions of attnSoftmaxL1Norm are (B,S1,N2,S2) or (T1,N2,maxSeqLenK), covering the target distribution of all keys.
  - softmaxLse is precomputed and provided by the DenseLightningIndexerSoftmaxLse operator, with dimensions (B,N2,S1) or (N2,T1), used for numerically stable computation.
  - The metadata input must be passed in and is used to compute the core-splitting information.
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
            <td>rightDownCausal mode mask, corresponding to the lower triangular scenario partitioned by the right vertex.</td>
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

- Specification constraints:
    <table style="undefined;table-layout: fixed; width: 942px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 300px">
            <col style="width: 360px">
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
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>S1, S2</td>
                <td>-</td>
                <td>S1 and S2 support unequal lengths</td>
            </tr>
            <tr>
                <td>N1</td>
                <td>1~128</td>
                <td>MQA scenario</td>
            </tr>
            <tr>
                <td>N2</td>
                <td>1</td>
                <td>Currently only N2=1 is supported.</td>
            </tr>
            <tr>
                <td>D</td>
                <td>128</td>
                <td>-</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>1~128</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

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
        </tbody>
    </table>

## Invocation Description

| Invocation Method | Invocation Sample                                                                            | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dense_lightning_indexer_kl_loss_grad](./examples/test_aclnn_dense_lightning_indexer_kl_loss_grad.cpp) | Calls the DenseLightningIndexerKLLossGrad operator through the [aclnnDenseLightningIndexerKLLossGrad](docs/aclnnDenseLightningIndexerKLLossGrad_en.md) API. |
| PyTorch API | - | Calls the dense_lightning_indexer_kl_loss_grad operator through the [dense_lightning_indexer_kl_loss_grad](docs/torchapi_dense_lightning_indexer_kl_loss_grad_en.md) API. |
