# DenseLightningIndexerKLLossGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|     √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    x     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    x     |
|<term>Atlas 200I/500 A2 推理产品</term>|      x     |
|<term>Atlas 推理系列产品</term>|      x     |
|<term>Atlas 训练系列产品</term>|      x     |

## 算子功能

- 算子功能：DenseLightningIndexerKLLossGrad算子是LightningIndexer KL Loss的反向计算算子（Dense模式）。该接口接收Lightning Indexer分支的q、k、w，以及由主Attention分支预先计算得到的attnSoftmaxL1Norm和softmaxLse，计算并输出dq、dk、dw和softmaxOut。本接口对全部key做dense注意力计算，不再依赖sparseIndices做top-k筛选；也不需要在kernel内部重算主Attention的`Q @ K -> softmax -> reduce/l1 norm`，不再输出loss；主Attention的目标分布由attnSoftmaxL1Norm输入提供，softmaxOut可用于后续loss计算。

- 计算公式：
    Indexer logits可表示为：

   $$
   S_{t,:}=q_{t,:}@K^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   其中，$q$和$K$分别对应本接口的q和k，$W$对应本接口的w。Indexer分支的softmax输出为：

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   本接口将$y$写出到softmaxOut。目标分布$p$由attnSoftmaxL1Norm输入提供，等价于旧版kernel内部由main attention score经head求和和L1归一化得到的结果。若后续继续计算KL Loss，其形式与旧版保持一致：

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   通过求导可得Loss的梯度表达式：

   $$
   dI_{t,:}=\operatorname{Softmax}(I_{t,:})-p_{t,:}
   $$

   利用链式法则可以进行w、q和k矩阵的梯度计算：

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K
   $$

   $$
   dK=\sum_t\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK直接累积到对应key位置上，无需scatter-add。

## 参数说明

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
        <th>参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>q</td>
        <td>输入</td>
        <td>Lightning Indexer分支的query输入。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>k</td>
        <td>输入</td>
        <td>Lightning Indexer分支的key输入。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>w</td>
        <td>输入</td>
        <td>Indexer logits的head权重。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>attn_softmax_l1_norm</td>
        <td>输入</td>
        <td>主Attention分支预先计算出的目标分布，用于替代旧kernel内部的主Attention重算结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>输入</td>
        <td>softmax log-sum-exp值，由DenseLightningIndexerSoftmaxLse预先计算。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cu_seqLens_q</td>
        <td>可选输入</td>
        <td>TND layout下q的累积序列长度，shape为(B+1,)。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cu_seqLens_k</td>
        <td>可选输入</td>
        <td>TND layout下k的累积序列长度，shape为(B+1,)。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>seqused_q</td>
        <td>可选输入</td>
        <td>表示每个batch实际使用的q长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>seqused_k</td>
        <td>可选输入</td>
        <td>表示每个batch实际使用的k长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmp_residual_k</td>
        <td>可选输入</td>
        <td>maskMode=3且cmpRatio!=1时用于计算压缩前S2长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>可选输入</td>
        <td>由DenseLightningIndexerKLLossGradMetadata生成的分核信息。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>layout_q</td>
        <td>属性</td>
        <td>q侧layout格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout_k</td>
        <td>属性</td>
        <td>k侧layout格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mask_mode</td>
        <td>属性</td>
        <td>mask模式，当前支持0和3。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cmp_ratio</td>
        <td>属性</td>
        <td>压缩比，取值范围为[1,128]。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dq</td>
        <td>输出</td>
        <td>q的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dk</td>
        <td>输出</td>
        <td>k的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dw</td>
        <td>输出</td>
        <td>w的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmax_out</td>
        <td>输出</td>
        <td>Indexer分支softmax输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

## 约束说明

- 公共约束
    - 参数q、k、dq、dk的数据类型应保持一致，支持FLOAT16和BFLOAT16。
    - 参数w、dw、attnSoftmaxL1Norm、softmaxOut的数据类型应为FLOAT32。
    - 参数cuSeqLensQOptional、cuSeqLensKOptional、seqUsedQOptional、seqUsedKOptional、cmpResidualKOptional、metadataOptional的数据类型应为INT32。
    - layoutQ和layoutK当前支持BSND和TND。
    - 当layoutQ为TND时，需要传入cuSeqLensQOptional；当layoutK为TND时，需要传入cuSeqLensKOptional。
  - attnSoftmaxL1Norm的维度为(B,S1,N2,S2)或(T1,N2,maxSeqLenK)，覆盖全部key的目标分布。
  - softmaxLse由DenseLightningIndexerSoftmaxLse算子预先计算提供，维度为(B,N2,S1)或(N2,T1)，用于数值稳定计算。
  - metadata输入必须传入，用于计算分核信息。
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 900px">
        <col style="width: 150px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask模式，不做causal mask。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵。</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal模式的mask。</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask。</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix。</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global。</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated。</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local。</td>
            <td>不支持</td>
        </tr>
        </tbody>
    </table>

- 规格约束：
    <table style="undefined;table-layout: fixed; width: 942px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 300px">
            <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>B</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>S1、S2</td>
                <td>-</td>
                <td>S1、S2支持不等长</td>
            </tr>
            <tr>
                <td>N1</td>
                <td>1~128</td>
                <td>MQA场景</td>
            </tr>
            <tr>
                <td>N2</td>
                <td>1</td>
                <td>当前仅支持N2=1。</td>
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

- 典型值
    <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
        <col style="width: 100px">
        <col style="width: 800px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>典型值</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>q</td>
            <td>N1=64/32/16/8，D=128</td>
        </tr>
        <tr>
            <td>k</td>
            <td>N2=1，D=128</td>
        </tr>
        <tr>
            <td>w</td>
            <td>N1=64/32/16/8</td>
        </tr>
        </tbody>
    </table>

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dense_lightning_indexer_kl_loss_grad](./examples/test_aclnn_dense_lightning_indexer_kl_loss_grad.cpp) | 通过[aclnnDenseLightningIndexerKLLossGrad](./docs/aclnnDenseLightningIndexerKLLossGrad.md)接口方式调用DenseLightningIndexerKLLossGrad算子。 |
| PyTorch API | [test_torch_dense_lightning_indexer_kl_loss_grad](./examples/test_torch_dense_lightning_indexer_kl_loss_grad.py) | 通过torch extension接口进行功能和精度验证。 |
