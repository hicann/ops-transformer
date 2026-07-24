# LightningIndexerGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      x     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`lightning_indexer_grad`为`lightning_indexer`的反向算子，基于正向算子的输出`sparseIndices`计算`query`、`key`、`weights`的梯度。

- 计算公式：
  LightningIndexer反向计算公式如下：

  $$
  S = Relu(Matmul(Query, Gather(Key, Indices)))
  $$

  $$
  Y = Dy*Weights
  $$

  $$
  dW = Reduce(S * dy)
  $$

  $$
  dQ = Matmul(ReluGrad(Y, S), Gather(Key, Indices))
  $$

  $$
  dK = ScatterAdd(Matmul(ReluGrad(Y, S), Q), Indices)
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 150px">
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
      <td>公式中的输入Q，不支持空tensor和非连续。layout为BSND时，shape为(B,S1,N1,D)；layout为TND时，shape为(T1,N1,D)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入K，不支持空tensor和非连续。layout为BSND时，shape为(B,S2,N2,D)；layout为TND时，shape为(T2,N2,D)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的输入dY，表示输出梯度，不支持空tensor和非连续。layout为BSND时，shape为(B,S1,N1,D)；layout为TND时，shape为(T1,N1,D)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>输入</td>
      <td>公式中的输入Indices，为LightningIndexer正向算子的sparseIndicesOut输出，不支持空tensor和非连续。layout为BSND时，shape为(B,S1,N2,K)；layout为TND时，shape为(T1,N2,K)，其中K为topK保留的block数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>公式中的输入W，不支持空tensor和非连续。layout为BSND时，shape为(B,S1,N1)；layout为TND时，shape为(T1,N1)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery</td>
      <td>输入</td>
      <td>每个Batch中Query的有效token数，不支持空tensor和非连续。可传入None表示与query的S长度相同；支持长度为B的一维tensor，且每个Batch的有效token数不超过query中的维度S大小且不小于0。layout为TND时该入参必须传入，并以元素数量作为B值；每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKey</td>
      <td>输入</td>
      <td>每个Batch中Key的有效token数，不支持空tensor和非连续。可传入None表示与key的S长度相同；支持长度为B的一维tensor，且每个Batch的有效token数不超过key中的维度S大小且不小于0。layout为TND时该入参必须传入，每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>输入</td>
      <td>用于标识输入Query/Key的数据排布格式，默认值为"BSND"，当前支持BSND、TND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>输入</td>
      <td>代表head个数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse的模式。sparse_mode为0时代表defaultMask模式；sparse_mode为3时代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dQuery</td>
      <td>输出</td>
      <td>公式中的dQ输出，表示query的梯度，不支持空tensor和非连续。数据类型与query一致，shape与query一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dKey</td>
      <td>输出</td>
      <td>公式中的dK输出，表示key的梯度，不支持空tensor和非连续。数据类型与key一致，shape与key一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dWeights</td>
      <td>输出</td>
      <td>公式中的dW输出，表示weights的梯度，不支持空tensor和非连续。数据类型与weights一致，shape与weights一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>deterministic</td>
      <td>输入</td>
      <td>表示当前是否支持确定性计算，默认值为False。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- inputLayout支持TND/BSND。
- 关于数据shape的约束，以Layout的BSND举例。其中：
  
  - B（Batchsize）：取值范围为1\~1024。
  - N（Head-Num）：取值为1\~64。
  - G（Group）：取值为N。
  - S1（Seq-LengthQ）：取值范围为1\~128K。
  - S2（Seq-LengthK）：取值范围为topK\~128K。
  - D（Head-Dim）：取值为128。
  - TopK：取值为2048。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_lightning_indexer_grad.cpp">test_aclnn_lightning_indexer_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnLightningIndexerGrad.md">aclnnLightningIndexerGrad
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
