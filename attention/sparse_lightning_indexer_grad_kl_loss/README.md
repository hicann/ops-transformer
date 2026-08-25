# SparseLightningIndexerGradKLLoss

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|     √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 算子功能

- 算子功能：SparselightningIndexerGradKlLoss算子是LightningIndexer的反向算子，再额外融合了Loss计算功能。LightningIndexer算子将QueryToken和KeyToken之间的最高内在联系的TopK个筛选出来，存放在SparseIndices中，从而减少长序列场景下Attention的计算量，加速长序列的网络的推理和训练的性能。

- 计算公式：
   用于取Top-k的value的计算公式可以表示为：

   $$
   I_{t,:}=W_{t,:}@ReLU(q_{t,:}@(K_{:t,:})^T)
   $$

   其中，$W$是第$t$个token对应的weights，$q$是第$t$个token对应的$G$个query头合轴后的矩阵，$K$为$t$行$K$矩阵。

   LightningIndexer会单独训练，对应的loss function为：

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||Softmax(I_{t,:}))
   $$

   其中，$p$是target distribution，通过对main attention score进行所有的head的求和，然后把求和结果沿着上下文方向进行L1正则化得到。$D_{KL}$为KL散度，其表达式为：

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   通过求导可得Loss的梯度表达式：

   $$
   dI\mathop{{}}\nolimits_{{t,:}}=Softmax \left( I\mathop{{}}\nolimits_{{t,:}} \left) -p\mathop{{}}\nolimits_{{t,:}}\right. \right.
   $$

   利用链式法则可以进行weights，query和key矩阵的梯度计算：

   $$
   dW\mathop{{}}\nolimits_{{t,:}}=dI\mathop{{}}\nolimits_{{t,:}}\text{@} \left( ReLU \left( S\mathop{{}}\nolimits_{{t,:}} \left)  \left) \mathop{{}}\nolimits^{{T}}\right. \right. \right. \right.
   $$

   $$
   d\mathop{{q}}\nolimits_{{t,:}}=dS\mathop{{}}\nolimits_{{t,:}}@K\mathop{{}}\nolimits_{{:t,:}}
   $$

   $$
   dK\mathop{{}}\nolimits_{{:t,:}}= \left( dS\mathop{{}}\nolimits_{{t,:}} \left) \mathop{{}}\nolimits^{{T}}@q\mathop{{}}\nolimits_{{:t,:}}\right. \right.
   $$

   其中，S为QK矩阵softmax的结果。

<!-- - **说明**：
   <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
   </blockquote> -->

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
        <td>query</td>
        <td>输入</td>
        <td>attention结构的输入Q。</td>
        <td>FLOAT16、BFLOAT16 </td>
        <td>ND</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>attention结构的输入K。</td>
        <td>FLOAT16、BFLOAT16 </td>
        <td>ND</td>
    </tr>
    <tr>
        <td>queryIndex</td>
        <td>输入</td>
        <td>lightingIndexer结构的输入queryIndex。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>keyIndex</td>
        <td>输入</td>
        <td>lightingIndexer结构的输入keyIndex。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>weights</td>
        <td>输入</td>
        <td>权重。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sparseIndices</td>
        <td>输入</td>
        <td>topk_index，用来选择每个query对应的key和value。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxMax</td>
        <td>输入</td>
        <td>注意力正向计算的中间输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxSum</td>
        <td>输入</td>
        <td>注意力正向计算的中间输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>queryRope</td>
        <td>输入</td>
        <td>MLA rope部分：Query位置编码的输出。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>keyRope</td>
        <td>输入</td>
        <td>MLA rope部分：Key位置编码的输出。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthsQuery</td>
        <td>输入</td>
        <td>每个Batch中，Query的有效token数。</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKey</td>
        <td>输入</td>
        <td>每个Batch中，Key的有效token数。</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>缩放系数。</td>
        <td>DOUBLE</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout</td>
        <td>输入</td>
        <td>layout格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>sparse的模式。</td>
    <td>INT64</td>
    <td>-</td>
    </tr>
    <tr>
        <td>sinks</td>
        <td>可选输入</td>
        <td>用于修正target distribution（p）归一化因子的附加softmax项，仅Ascend 950支持，仅aclnnSparseLightningIndexerGradKLLossV2接口接收，需为1维且长度等于query的N维度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
    <td>deterministic</td>
        <td>输入</td>
        <td>确定性计算。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQueryIndex</td>
        <td>输出</td>
        <td>QueryIndex的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKeyIndex</td>
        <td>输出</td>
        <td>KeyIndex的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dWeights</td>
        <td>输出</td>
        <td>Weights的梯度。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>loss</td>
        <td>输出</td>
        <td>损失函数值。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - T1支持大于等于actualSeqLengthsQuery的累加和，T2支持大于等于actualSeqLengthsKey的累加和。

## 约束说明

- 确定性计算：
  - SparseLightningIndexerGradKLLoss默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- 公共约束
    - 参数query、key、queryIndex、keyIndex的数据类型应保持一致。
    - 参数weights不为float32时，参数query、key、queryIndex、keyIndex、weights的数据类型应保持一致。
    - 入参为空的场景处理：
        - query为空Tensor：直接返回。
        - 公共约束里入参为空的场景和FAG保持一致。
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 900px">
        <col style="width: 150px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens；如果传入，则需要传入完整的attenmask矩阵，表示preTokens和nextTokens之间的部分需要计算</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>2</td>
            <td>leftUpCausal模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
            <td>支持</td>
  </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，需要传入优化后的attenmask矩阵</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>5</td>
            <td>prefix</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>6</td>
            <td>global</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>7</td>
            <td>dilated</td>
            <td>不支持</td>
        </tr>
        <tr>
            <td>8</td>
            <td>block_local</td>
            <td>不支持</td>
        </tr>
        </tbody>
    </table>

- 规格约束
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 550px">
        <col style="width: 500px">
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
            <td>支持1~256</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>S1支持1~32K，S2支持1~1M</td>
            <td>S1、S2支持不等长；S1必须小于等于S2</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>
            32、64、128
            </td>
            <td>SparseFA为MQA。</td>
        </tr>
        <tr>
            <td>Nidx1</td>
            <td>
            8、16、32、64
            </td>
            <td>SparseFA为MQA。</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>SparseFA为MQA，Nidx2=1。</td>
        </tr>
        <tr>
            <td>Nidx2</td>
            <td>1</td>
            <td>SparseFA为MQA，N2=1。</td>
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
            <td>1024、2048、3072、4096、5120、6144、7168、8192</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

    <term>Ascend 950PR/Ascend 950DT</term>：N1额外支持48，Nidx1额外支持24，二者仅允许(48,24)组合，禁止其余数值配对。

    <term>Ascend 950PR/Ascend 950DT</term>：B、S1、S2均支持泛化。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sparse_lightning_indexer_grad_kl_loss](./examples/test_aclnn_sparse_lightning_indexer_grad_kl_loss.cpp) | 通过[aclnnSparseLightningIndexerGradKLLoss](./docs/aclnnSparseLightningIndexerGradKLLoss.md)接口方式调用SparseLightningIndexerGradKLLoss算子。             |
