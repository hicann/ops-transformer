# MlaPreprocessV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **算子功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程如下：
    - 首先对输入$x$ RmsNormQuant后乘以$W^{DQKV}$进行下采样后分为通路1和通路2。
    - 通路1做RmsNormQuant后乘以$W^{UQ}$后再分为通路3和通路4。
    - 通路3后乘以$W^{uk}$后输出$q^N$。
    - 通路4后经过旋转位置编码后输出$q^R$。
    - 通路2拆分为通路5和通路6。
    - 通路5经过RmsNorm后传入Cache中得到$k^N$。
    - 通路6经过旋转位置编码后传入另一个Cache中得到$k^R$。
- **计算公式**：

    RmsNormQuant公式

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    RmsNormQuant(x) = ({RmsNorm}(x) + bias) * deqScale
    $$
  
    Query计算公式，包括W^{DQKV}矩阵乘、W^{UK}矩阵乘、RmsNormQuant和ROPE旋转位置编码处理

    $$
    q^N =  RmsNormQuant(x) \cdot W^{DQKV} \cdot W^{UK}
    $$

    $$
    q^R = ROPE(x^Q)
    $$

    Key计算公式，包括RmsNorm和rope，将计算结果存入cache

    $$
    k^N = Cache({RmsNorm}(RmsNormQuant(x)))
    $$

    $$
    k^R = Cache(ROPE(RmsNormQuant(x)))
    $$

    ACLNN接口的`cos`和`sin`同时传入空指针时跳过Q-RoPE和K-RoPE数学运算，$q^R$和$k^R$分别保存未经旋转的原始Q-rope和K-rope分量；二者均为非空Tensor时保持原RoPE行为，禁止仅传入一个空指针。其他计算、输出形状、`qDownOut`语义和Cache布局不变。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 194px">
<col style="width: 146px">
<col style="width: 721px">
<col style="width: 230px">
<col style="width: 136px">
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
    <td>input</td>
    <td>输入</td>
    <td>Device侧的aclTensor，用于计算Query和Key的x，shape为[tokenNum,hiddenSize]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma0</td>
    <td>输入</td>
    <td>Device侧的aclTensor，首次RmsNorm计算中的γ参数，shape为[hiddenSize]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta0</td>
    <td>输入</td>
    <td>Device侧的aclTensor，首次RmsNorm计算中的β参数，shape为[hiddenSize]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantScale0</td>
    <td>输入</td>
    <td>Device侧的aclTensor，首次RmsNorm公式中量化缩放的参数，shape为[1]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantOffset0</td>
    <td>输入</td>
    <td>Device侧的aclTensor，首次RmsNorm公式中的量化偏移参数，shape为[1]</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wdqkv</td>
    <td>输入</td>
    <td>Device侧的aclTensor，与输入首次做矩阵乘的降维矩阵，shape为[qLoraDim + keyTotalDim,hiddenSize]</td>
    <td>INT8, FLOAT16, BFLOAT16</td>
    <td>NZ</td>
  </tr>
  <tr>
    <td>deScale0</td>
    <td>输入</td>
    <td>Device侧的aclTensor，输入首次做矩阵乘的降维矩阵中的系数，shape为[qLoraDim + keyTotalDim]。input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT</td>
    <td>INT32, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias0</td>
    <td>输入</td>
    <td>Device侧的aclTensor，输入首次做矩阵乘的降维矩阵中的系数，shape为[qLoraDim + keyTotalDim]。支持传入空tensor，quantMode为1、3时不传入</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma1</td>
    <td>输入</td>
    <td>Device侧的aclTensor，第二次RmsNorm计算中的γ参数，shape为[qLoraDim]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>输入</td>
    <td>Device侧的aclTensor，第二次RmsNorm计算中的β参数，shape为[qLoraDim]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantScale1</td>
    <td>输入</td>
    <td>Device侧的aclTensor，第二次RmsNorm公式中量化缩放的参数，shape为[1]。仅在quantMode为0时传入</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantOffset1</td>
    <td>输入</td>
    <td>Device侧的aclTensor，第二次RmsNorm公式中的量化偏移参数，shape为[1]。仅在quantMode为0时传入</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wuq</td>
    <td>输入</td>
    <td>Device侧的aclTensor，权重矩阵，shape为[headNum * (qNoRopeDim + qRopeDim),qLoraDim]</td>
    <td>INT8, FLOAT16, BFLOAT16</td>
    <td>NZ</td>
  </tr>
  <tr>
    <td>deScale1</td>
    <td>输入</td>
    <td>Device侧的aclTensor，参与wuq矩阵乘的系数，shape为[headNum * (qNoRopeDim + qRopeDim)]。input输入dtype为FLOAT16支持INT64，输入BFLOAT16时支持FLOAT</td>
    <td>INT64, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias1</td>
    <td>输入</td>
    <td>Device侧的aclTensor，参与wuq矩阵乘的系数，shape为[headNum * (qNoRopeDim + qRopeDim)]。quantMode为1、3时不传入</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma2</td>
    <td>输入</td>
    <td>Device侧的aclTensor，参与RmsNormAndreshapeAndCache计算的γ参数，shape为[512]。</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>Device侧的aclTensor，表示用于计算旋转位置编码的余弦参数矩阵，启用RoPE时shape为[tokenNum,64]，其中tokenNum等于input的第0维；与sin同时传入空指针时关闭RoPE。</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>Device侧的aclTensor，表示用于计算旋转位置编码的正弦参数矩阵，启用RoPE时shape为[tokenNum,64]，其中tokenNum等于input的第0维；与cos同时传入空指针时关闭RoPE。</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wuk</td>
    <td>输入</td>
    <td>Device侧的aclTensor，表示计算Key的上采样权重，shape为[headNum,qNoRopeDim,512]。</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kvCache</td>
    <td>输入</td>
    <td>Device侧的aclTensor，与输出的kvCacheOut为同一tensor，输入格式随cacheMode变化。<br><br>cacheMode为0：shape为[blockNum,blockSize,1,576]<br>cacheMode为1：shape为[blockNum,blockSize,1,512]<br>cacheMode为2：shape为[blockNum,headNum*512/32,block_size,32]<br>cacheMode为3：shape为[blockNum,headNum*512/16,block_size,16]</td>
    <td>cacheMode为0：与input一致<br>cacheMode为1：与input一致<br>cacheMode为2：INT8<br>cacheMode为3：与input一致</td>
    <td>ND<br>ND<br>NZ<br>NZ</td>
  </tr>
  <tr>
    <td>kvCacheRope</td>
    <td>输入</td>
    <td>Device侧的aclTensor，可选参数，支出传入空指针。与输出的krCacheOut为同一tensor，输入格式随cacheMode变化。<br><br>cacheMode为0：不传入。<br>cacheMode为1：shape为[blockNum,blockSize,1,64]<br>cacheMode为2或3：shape为[blockNum, headNum*64 / 16 ,block_size, 16]</td>
    <td>与input一致</td>
    <td><br>ND<br>NZ</td>
  </tr>
  <tr>
    <td>slotmapping</td>
    <td>输入</td>
    <td>Device侧的aclTensor，表示用于存储kv_cache和kr_cache的索引，shape为[tokenNum]</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ctkvScale</td>
    <td>输入</td>
    <td>Device侧的aclTensor，输出量化处理中参与计算的系数，仅在cacheMode为2时传入，shape为[1]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qNopeScale</td>
    <td>输入</td>
    <td>Device侧的aclTensor，输出量化处理中参与计算的系数，仅在cacheMode为2时传入，shape为[1]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wdqDim</td>
    <td>输入</td>
    <td>表示经过matmul后拆分的dim大小。预留参数，目前只支持1536</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qRopeDim</td>
    <td>输入</td>
    <td>表示q传入rope的dim大小。预留参数，目前只支持64。</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>kRopeDim</td>
    <td>输入</td>
    <td>表示k传入rope的dim大小。预留参数，目前只支持64。</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>表示加在分母上防止除0</td>
    <td>float</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qRotaryCoeff</td>
    <td>输入</td>
    <td>表示q旋转系数。预留参数，目前只支持2</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>kRotaryCoeff</td>
    <td>输入</td>
    <td>表示k旋转系数。预留参数，目前只支持2</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transposeWdq</td>
    <td>输入</td>
    <td>表示wdq是否转置。预留参数，目前只支持true</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transposeWuq</td>
    <td>输入</td>
    <td>表示wuq是否转置。预留参数，目前只支持true</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transposeWuk</td>
    <td>输入</td>
    <td>表示wuk是否转置。预留参数，目前只支持true</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cacheMode</td>
    <td>输入</td>
    <td>表示指定cache的类型，取值范围[0, 3]<br><br>0：kcache和q均经过拼接后输出<br>1：输出的kvCacheOut拆分为kvCacheOut和krCacheOut，qOut拆分为qOut和qRopeOut<br>2：krope和ctkv转为NZ格式输出，ctkv和qnope经过per_head静态对称量化为int8类型<br>3：krope和ctkv转为NZ格式输出</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>表示指定RmsNorm量化的类型，取值范围[0, 3]<br>0：per_tensor静态非对称量化，默认量化类型<br>1：per_token动态对称量化，未实现<br>2：per_token动态非对称量化，未实现<br>3：不量化，浮点输出，未实现</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>doRmsNorm</td>
    <td>输入</td>
    <td>表示是否对input输入进行RmsNormQuant操作，false表示不操作，true表示进行操作。预留参数，目前只支持true</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>wdkvSplitCount</td>
    <td>输入</td>
    <td>表示指定wdkv拆分的个数，支持[1-3]，分别表示不拆分、拆分为2个、拆分为3个降维矩阵。预留参数，目前只支持1</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qOut</td>
    <td>输出</td>
    <td>表示Query的输出tensor，对应计算流图中右侧经过NOPE和矩阵乘后的输出，shape和dtype随cacheMode变化<br><br>cacheMode为0：shape为[tokenNum, headNum, 576]<br>cacheMode为1或3：shape为[tokenNum, headNum, 512]<br>cacheMode为2：shape为[tokenNum, headNum, 512]</td>
    <td>cacheMode为0：与input一致<br>cacheMode为1或3：与input一致<br>cacheMode为2：INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kvCacheOut</td>
    <td>输出</td>
    <td>表示Key经过ReshapeAndCache后的输出，shape和dtype随cacheMode变化<br><br>cacheMode为0：shape为[blockNum, blockSize, 1, 576]<br>cacheMode为1：shape为[blockNum, blockSize, 1, 512]<br>cacheMode为2：shape为[blockNum, headNum*512/32, block_size, 32]<br>cacheMode为3：shape为[blockNum, headNum*512/16, block_size, 16]</td>
    <td>cacheMode为0：与input一致<br>cacheMode为1：与input一致<br>cacheMode为2：INT8<br>cacheMode为3：与input一致</td>
    <td>ND<br>ND<br>NZ<br>NZ</td>
  </tr>
  <tr>
    <td>qRopeOut</td>
    <td>输出</td>
    <td>表示Query经过旋转编码后的输出，shape和dtype随cacheMode变化<br><br>cacheMode为0：不输出<br>cacheMode为1或3：shape为[tokenNum, headNum, 64]<br>cacheMode为2：shape为[tokenNum, headNum, 64]</td>
    <td><br>cacheMode为1或3：与input一致<br>cacheMode为2：与input一致</td>
    <td><br>ND<br>ND</td>
  </tr>
  <tr>
    <td>krCacheOut</td>
    <td>输出</td>
    <td>表示Key经过ROPE和ReshapeAndCache后的输出，shape和dtype随cacheMode变化，<br><br>cacheMode为0：不输出<br>cacheMode为1：shape为[blockNum, blockSize, 1, 64]<br>cacheMode为2或3：shape为[blockNum, headNum*64 / 16 ,block_size, 16]</td>
    <td><br>cacheMode为1：与input一致<br>cacheMode为2或3：与input一致</td>
    <td><br>ND<br>NZ</td>
  </tr>
</tbody></table>

## 约束说明

- shape格式字段含义及约束
    - tokenNum：tokenNum表示输入样本批量大小，取值范围：0~256
    - hiddenSize：hiddenSize表示隐藏层的大小，取值固定为：2048~10240，为256的倍数
    - headNum：表示多头数，取值范围：1~128
    - blockNum：PagedAttention场景下的块数，取值范围：192
    - blockSize：PagedAttention场景下的块大小，取值范围：128
    - qloraDim：表示Q矩阵的LoRA输入维度，取值范围：32~4096，为32的倍数
    - keyTotalDim：表示Key部分的总维度，取值固定为：576（512主维度+64 rope维度）
    - qRopeDim：表示Q矩阵中旋转编码部分的维度，取值固定为：64
    - qNoRopeDim：表示Q矩阵中无旋转编码部分的维度，取值范围：16~256，为16的倍数
- rope模式约束
    - mla_preprocess算子中的Rotary Embedding（RoPE）操作采用half模式，暂不支持interleave模式

## 调用说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_preprocess_v2.cpp">test_aclnn_mla_preprocess_v2
    </a>
    </td>
    <td class="tg-lboi">
    通过
    <a href="./docs/aclnnMlaPreprocessV2.md">aclnnMlaPreprocessV2
    </a>
    接口方式调用MlaPreprocessV2算子
    </td>
  </tr>
</tbody></table>
