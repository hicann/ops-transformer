# BlockSparseAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品</term>|      √     |
|<term>Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
|<term>Atlas 推理系列产品</term>                                                 |    ×    |
|<term>Atlas 训练系列产品</term>                                                 |    ×    |

## 功能说明

* ​算子功能​：aclnnBlockSparseAttention稀疏注意力反向计算，支持灵活的块级稀疏模式，通过BlockSparseMask指定每个Q块选择的KV块，实现高效的稀疏注意力计算。
* ​计算公式​：

  稀疏块大小：$blockShapeX×blockShapeY$，BlockSparseMask指定稀疏模式。

 blockShapeX 和 blockShapeY 必须为 16 的正整数倍。BlockSparseMask 在逻辑块粒度上定义稀疏 pattern，每个 mask 元素控制一个 blockShapeX × blockShapeY 的连续块是否参与计算。
  已知正向计算公式为：

  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$

  为方便表达，以变量$S$和$P$表示计算公式：

  $$
  S = Mask(scale⋅query⋅key_{sparse}^{T},atten\_mask)
  $$

  $$
  P = SoftMax(S)
  $$

  $$
  V = value_{sparse}
  $$

  $$
  Out = PV
  $$

  则反向计算公式为：

  $$
  softmax\_grad = softmaxGrad(dOut, attentionOut)
  $$

  $$
  dP=dOut * V^T
  $$

  $$
  dS = P * (dP-softmax\_grad)
  $$

  $$
  dV=P^T * dOut
  $$

  $$
  dQ=(dS*K)*scale
  $$

  $$
  dK=(dS^T*Q)*scale
  $$

BlockSparseAttentionGrad输入dout、query、key、value, attentionOut的数据排布格式支持从多种维度排布解读，可通过qInputLayout和kvInputLayout传入。为了方便理解后续支持的具体排布格式（如 BNSD、TND 等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

* B：表示输入样本批量大小（Batch）
* T：B和S合轴紧密排列的长度（Total tokens）
* S：表示输入样本序列长度（Seq-Length）
* H：表示隐藏层的大小（Head-Size）
* N：表示多头数（Head-Num）
* D：表示隐藏层最小的单元尺寸，需满足D=H/N（Head-Dim）

当前支持的布局：

* qInputLayout: "TND" "BNSD" "BSND"
* kvInputLayout: "TND" "BNSD" "BSND"

- <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>:
不支持"BSND"。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 145px">
</colgroup>
<thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>dout（aclTensor*）</td>
    <td>输入</td>
    <td>反向输出梯度，代表最终输出对当前算子的梯度信息。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>query（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的查询向量，即公式中的query。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的键向量，即公式中的key。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value（aclTensor*）</td>
    <td>输入</td>
    <td>注意力计算中的值向量，即公式中的value。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attentionOut（aclTensor*）</td>
    <td>输入</td>
    <td>正向 BlockSparseAttention 计算的输出结果，即公式中的attentionOut。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>softmaxLse（aclTensor*）</td>
    <td>输入</td>
    <td>Softmax计算的log-sum-exp中间结果。用于反向计算梯度的对数和指数逆推。</td>
    <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, 1]。</li><li>BNSD: [batch, headNum, maxQSeqLength, 1]。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockSparseMaskOptional（aclTensor*）</td>
    <td>输入</td>
    <td>块状稀疏掩码，表示实际的稀疏pattern。决定哪些block实际参与注意力计算。</td>
    <td>不支持空Tensor。<br>可选输入（当前版本为必选）：<ul><li>shape为[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]。</li><li>表示按block划分后哪些block需要参与计算（为1），哪些block不参与计算（为0）。</li><li>如传入nullptr，则视为不开启块稀疏计算，即所有token之间的注意力分数都会被计算。</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional（aclTensor*）</td>
    <td>输入</td>
    <td>注意力掩码，即公式中的atten_mask。用于屏蔽不应参与计算的特定token。</td>
    <td>支持空Tensor。<br>当前不支持，应传入nullptr。</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td rowspan="3">blockShapeOptional（aclIntArray*）</td>
    <td rowspan="3">输入</td>
    <td rowspan="3">稀疏块形状数组。指定每个稀疏块的二维尺寸（行数和列数）。</td>
    <td> <ul><li>当配置了blockSparseMaskOptional时：如配置此输入，算子会从中获取稀疏块尺寸；如不配置此输入，算子将默认稀疏块尺寸为[128,128]。</li></ul></td>
    <td rowspan="3">INT64</td>
    <td rowspan="3">-</td>
    <td rowspan="3">1</td>
    <td rowspan="3">-</td>
    </tr>
    <tr>
    <td><ul><li>当未配置blockSparseMaskOptional时：无论此项如何配置，算子均将忽略。</li></ul></td>
    </tr>
    <tr>
    <td>当配置此输入时的元素要求：<ul><li>必须包含至少两个元素 [blockShapeX, blockShapeY]。</li><li>blockShapeX: Q方向块大小，值必须大于0。</li><li>blockShapeY: KV方向块大小，值必须大于0。</li></ul></td>
    </tr>
    <tr>
    <td>actualSeqLengthsOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>query的实际序列长度数组。<br>用于描述变长序列场景下（即含有 Padding 填充数据的场景），每个 Batch 中实际有效的 query token 数量。</td>
    <td> 变长序列场景（当 qInputLayout 为 "TND" 时）：该项输入必须配置。因为 TND 格式为一维连续排布，算子需要依赖该数组来准确切分界定各个序列的真实边界。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKvOptional（aclIntArray*）</td>
    <td>输入</td>
    <td>key/value的实际序列长度数组。<br>用于描述变长序列场景下（即含有 Padding 填充数据的场景），每个 Batch 中实际有效的 key/value token 数量。</td>
    <td> 变长序列场景（当 kvInputLayout 为 "TND" 时）：该项输入必须配置。因为 TND 格式为一维连续排布，算子需要依赖该数组来准确切分界定各个序列的真实边界。</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout（char*）</td>
    <td>输入</td>
    <td>query的数据排布格式。指示输入张量在内存中的具体排布（如连续或合轴排列）。</td>
    <td>当前仅支持"TND"、"BNSD"、"BSND"，qInputLayout与kvInputLayout需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout（char*）</td>
    <td>输入</td>
    <td>key和value的数据排布格式。指示输入张量在内存中的具体排布。</td>
    <td>当前仅支持"TND"、"BNSD"、"BSND"，qInputLayout与kvInputLayout需要保持一致。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads（int64_t）</td>
    <td>输入</td>
    <td>key/value的注意力头数。用于支持GQA（分组查询注意力）机制下的头数比例映射。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType（int64_t）</td>
    <td>输入</td>
    <td>注意力计算中的掩码类型。指定采用何种预设规则的掩码逻辑。</td>
    <td>当前只支持传 0：代表不加mask场景。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue（double）</td>
    <td>输入</td>
    <td>缩放系数，即公式中的scale。用于注意力分数的归一化处理。</td>
    <td>一般设置为D^-0.5。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens（int64_t）</td>
    <td>输入</td>
    <td>滑窗向前包含的token数量。限制当前token只能与前方的多少个历史token计算注意力。</td>
    <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens（int64_t）</td>
    <td>输入</td>
    <td>滑窗向后包含的token数量。限制当前token只能与后方的多少个未来token计算注意力。</td>
    <td>用于滑窗attention场景，当前不支持滑窗attention，只支持传入2147483647。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dq（aclTensor*）</td>
    <td>输出</td>
    <td>query的梯度输出结果，即公式中的dq。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入query保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dk（aclTensor*）</td>
    <td>输出</td>
    <td>key的梯度输出结果，即公式中的dk。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入key保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dv（aclTensor*）</td>
    <td>输出</td>
    <td>value的梯度输出结果，即公式中的dv。</td>
    <td>不支持空Tensor。<br>数据类型和shape与输入value保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
</tbody>
</table>

<ul>
- <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>:
不支持FLOAT8_E5M2、FLOAT8_E4M3FN。
</ul>

## 约束说明

* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
* actualSeqLengthsOptional在qInputLayout为“TND”时必选；actualSeqLengthsKvOptional在kvInputLayout为“TND”时必选。
* softmaxLse的layout需要与query的layout保持一致。如果query的layout为"BSND"时，softmaxLse的layout应传入"BNS1"。
* HeadDim必须等于128。
* 根据算子支持的输入 Layout，query 张量 Shape 中对应的 head 维度大小记为 N1，key 和 value 张量 Shape 中对应的 head 维度大小记为 N2。必须满足N1 % N2 == 0。
  - <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>：当前只支持MHA，即N1等于N2。
  - <term>Ascend 950PR/Ascend 950DT</term>：支持MQA、MHA、GQA场景。
* actualSeqLengthsOptional与actualSeqLengthsKvOptional相关约束：

  - <term>Atlas A2 训练产品</term>、<term>Atlas A3 训练产品</term>：当 qInputLayout或kvInputLayout 为 "BNSD" 时，如配置该项，算子会按指定的有效长度处理，忽略 Padding 部分的数据，提升性能；如不配置（传 nullptr），算子将默认把 query shape 中的 S 维度作为有效长度进行全量处理。

  - <term>Ascend 950PR/Ascend 950DT</term>：当qInputLayout或kvInputLayout为非"TND"时，会忽略这两个入参。
* 不支持确定性计算场景。

## 调用说明

| 调用方式           | 调用样例                                                                                                          | 说明                                                                                                                  |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_block_sparse_attention_grad](examples/test_aclnn_block_sparse_attention_grad.cpp)                     | 非TND（TND代表Total sequence length, Num heads, Head dimension，通常用于表示变长序列场景下的连续内存排布格式）场景，通过[aclnnBlockSparseAttentionGrad](docs/aclnnBlockSparseAttentionGrad.md)接口方式调用BlockSparseAttentionGrad算子。                   |
