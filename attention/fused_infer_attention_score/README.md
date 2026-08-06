# FusedInferAttentionScore

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（[PromptFlashAttention](../prompt_flash_attention/README.md)），也可支持增量计算场景（[IncreFlashAttention](../incre_flash_attention/README.md)）。

- 计算公式：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q$、$K$、$V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中：$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的平方根进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>公式中的输入Q。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入K。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的输入V。</td>
    <td>FLOAT16、BFLOAT16、INT8、INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>输出</td>
    <td>公式中的输出。</td>
    <td>FLOAT16、BFLOAT16、INT8</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空Tensor，参数key、value为空tensor(即S2为0)，则attentionOut填充为全零。attentionOut为空Tensor时，AscendCLNN框架会处理。其余在上述参数说明中标注了“可传入nullptr”的入参为空指针时，不进行处理。

- 参数key、value中对应Tensor的shape需要完全一致；非连续场景下key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。由于tensorlist限制，非连续场景下B不能大于256。

- 当Q_S大于1时，query，key，value输入，功能使用限制如下：
    - 支持B轴小于等于65536。

  - 如果输入类型为INT8且D轴不是32字节对齐，则B轴的最大支持值为128。若输入类型为FLOAT16或BFLOAT16且D轴不是16字节对齐，B轴同样仅支持到128。

  - 支持N轴小于等于256，支持D轴小于等于512。inputLayout为BSH或者BSND时，建议N*D小于65535。

  - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（aicore error类型报错，errorStr为:timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 550px"><colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 150px">
    </colgroup><thead>
    <tr>
    <th>B</th>
    <th>Q_N</th>
    <th>Q_S</th>
    <th>D</th>
    <th>KV_N</th>
    <th>KV_S</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>1</td>
    <td>20</td>
    <td>2097152</td>
    <td>256</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    <tr>
    <td>1</td>
    <td>2</td>
    <td>20971520</td>
    <td>256</td>
    <td>2</td>
    <td>20971520</td>
    </tr>
    <tr>
    <td>20</td>
    <td>1</td>
    <td>2097152</td>
    <td>256</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    <tr>
    <td>1</td>
    <td>10</td>
    <td>2097152</td>
    <td>512</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    </tbody>
    </table>
    </div>

  - D轴限制：query、key、value或attentionOut类型包含INT8时，D轴需要32对齐；query、key、value或attentionOut类型包含INT4时，D轴需要64对齐；类型全为FLOAT16、BFLOAT16时，D轴需16对齐。

- 当Q_S等于1时：query，key，value输入，功能使用限制如下：

  - 支持B轴小于等于65536，支持N轴小于等于256，支持D轴小于等于512。
  - query、key、value输入类型均为INT8的场景暂不支持。
  - 在INT4伪量化场景下，aclnn单算子调用支持KV INT4输入或者INT4拼接成INT32输入（建议通过dynamicQuant生成INT4格式的数据，因为dynamicQuant就是一个INT32包括8个INT4）。
  - 在INT4伪量化场景下，若KV INT4拼接成INT32输入，那么KV的N、D或者H是实际值的八分之一（prefix同理）。
  - key、value在特定数据类型下存在对于D轴的限制
    - key、value输入类型为INT4（INT32）时，D轴需要64对齐（INT32仅支持D 8对齐）。

- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。

## 调用说明

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、 <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  | 调用方式  | 样例代码                                                     | 说明                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn接口 | [test_aclnn_FusedInferAttentionScoreV4](./examples/test_aclnn_fused_infer_attention_score_v4_gqa_noquant.cpp) | 通过[aclnnFusedInferAttentionScoreV4](./docs/aclnnFusedInferAttentionScoreV4.md)调用FusedInferAttentionScore算子 |

- <term>Ascend 950PR/Ascend 950DT</term>：

  | 调用方式  | 样例代码                                                     | 说明                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn接口 | [test_aclnn_FusedInferAttentionScoreV5](./examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp) | 通过[aclnnFusedInferAttentionScoreV5](./docs/aclnnFusedInferAttentionScoreV5.md)调用FusedInferAttentionScore算子 |
