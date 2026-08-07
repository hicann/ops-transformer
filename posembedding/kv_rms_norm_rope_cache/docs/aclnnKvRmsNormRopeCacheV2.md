# aclnnKvRmsNormRopeCacheV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/kv_rms_norm_rope_cache)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算（对应$rms\_size$）与RoPE（Rotary Position Embedding）位置编码（对应$rope\_size$)，以及更新KVCache的ScatterUpdate操作。本接口支持两种场景，向下兼容aclnnKvRmsNormRopeCache。

- 支持场景：

    | 场景类型 | kv分量来源 | 说明 |
    | :------: | :------: | :------ |
    |V1|rms_size=Dv=512<br>rope_size=Dk=64<br>vOptional=None|kv合轴模式：对输入张量kv的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。<li>与DeepSeekV3网络结构强相关，仅支持N=1的场景。</li><li>rms_norm计算所需数据Dv和rope计算所需数据Dk由输入kv的D切分而来，Dk、Dv大小需满足Dk+Dv=Dkv。</li>|
    |V2|Dv=128<br>rms_size=Dk=Dkv=192<br>rope_size=64<br>vOptional的shape为[Bkv, Nkv, Skv, Dv]|kv分离模式：对输入张量kv进行rms_norm计算，之后对尾轴前64维进行rope计算并覆盖写回对应元素，最终结果scatter写入到kCacheRef中；对输入张量vOptional进行中间处理，最终结果scatter写入到ckvCacheRef中。<li>支持N=1/2/4/8</li><li>此场景下k与v尾轴分离，kv仅存储k分量尾轴，vOptional则存储v分量尾轴。</li>|

    * <term>Ascend 950PR/Ascend 950DT</term>：仅支持V1场景。
    * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持V1和V2场景。

- 计算公式：

  定义输入张量kv的shape为$[Bkv, N, Skv, Dkv]$以及张量vOptional的shape为$[Bkv, N, Skv, Dv]$。

  (1) RmsNorm：

  $$
  x=kv[...,:rms\_size]
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  y = \operatorname{RmsNorm}(x)
  $$

  (2) interleaveRope：

  $$
  x=\begin{cases}
  kv[...,Dv:], \quad vOptional = None\\
  {\operatorname{RmsNorm}(\mathbf{x})}[...,:rope\_size],  \quad vOptional != None
  \end{cases}
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y\_rope=x\_part1*cos+x\_part2*sin
  $$

  $$
  rope\_out=\begin{cases}y\_rope, \quad vOptional = None \\
  concat(y\_rope, {\operatorname{RmsNorm}(\mathbf{x})}[...,rope\_size:]),  \quad vOptional != None
  \end{cases}
  $$

  (3) 量化计算：

  x表示将要写入到kCacheRef和ckvCacheRef上的原始数据，作为量化过程的输入。

  $$
  x = x * scale,\ if\ scale\ !=\ None
  $$

  $$
  x = x + offset,\ if\ offset\ !=\ None
  $$

  $$
  y = \begin{cases}x, \quad scale == None \space and \space offset == None \\ round(x).clamp(-128,127), \quad others
  \end{cases}
  $$

  (4) Scatter写出：

  输入张量index对应输入kv缓存中各元素的索引映射表，取x中具体元素的索引$b \in Bkv$以及$s \in Skv$，$n$为注意力头索引，

  $$
  scatter\_idx = index(b, s) \\
  $$

   Quant表示前述量化计算过程，对原地更新参数k\_cache和ckv\_cache：

  $$
  k\_cache[scatter\_idx, ...] = Quant[x = rope\_out, scale = k\_scale, offset = k\_offset](b, n, s)
  $$

  $$
  ckv\_cache[scatter\_idx, ...] = \begin{cases} Quant(x = \operatorname{RmsNorm}(x), scale = v\_scale, offset = v\_offset)[b, n, s], \quad vOptional = None \\
  Quant[x = vOptional, scale = v\_scale, offset = v\_offset](b, n, s),  \quad vOptional != None
  \end{cases}
  $$

  (5) 原始结果写出：

  当$is\_output\_kv=True$且有效时：

  $$
  k\_rope = rope\_out
  $$

  $$
  c\_kv = \begin{cases}\operatorname{RmsNorm}(x), \quad vOptional = None \\
  vOptional,  \quad vOptional != None
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnKvRmsNormRopeCacheV2GetWorkspaceSize”接口获得入参并根据流程计算所需workspace大小，再调用“aclnnKvRmsNormRopeCacheV2”接口执行计算。

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheV2GetWorkspaceSize(
  const aclTensor* kv,
  const aclTensor* gamma,
  const aclTensor* cos,
  const aclTensor* sin,
  const aclTensor* index,
  aclTensor*       kCacheRef,
  aclTensor*       ckvCacheRef,
  const aclTensor* kRopeScaleOptional,
  const aclTensor* cKvScaleOptional,
  const aclTensor* kRopeOffsetOptional,
  const aclTensor* cKvOffsetOptional,
  const aclTensor* vOptional,
  double           epsilon,
  char*            cacheModeOptional,
  bool             isOutputKv,
  const aclTensor* kRopeOut,
  const aclTensor* cKvOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnKvRmsNormRopeCacheV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 403px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
  <col style="width: 146px">
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
      <td>kv</td>
      <td>输入</td>
      <td>公式中用于切分出rms_norm计算所需数据Dv和RoPE计算所需数据Dk的输入数据。</td>
      <td><ul><li>shape仅支持4维[Bkv,N,Skv,D]。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>公式中用于rms_norm计算的输入数据。</td>
      <td><ul><li>与kv数据类型一致，shape为1维[Dv,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>公式中用于RoPE计算的输入数据，对输入张量Dk进行余弦变换</td>
      <td><ul><li>与kv数据类型一致，shape为4维[Bkv,N,Skv,Dk]或[Bkv,N,1,Dk]。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>可选输入</td>
      <td>公式中用于RoPE计算的输入数据，对输入张量Dk进行正弦变换。</td>
      <td><ul><li>与kv数据类型一致，shape为4维[Bkv,N,Skv,Dk]或[Bkv,N,1,Dk]，与cos的shape保持一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>用于指定写入cache的具体索引位置，当index的value数值为-1时，代表跳过更新。</td>
      <td><ul><li>当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]。</li><li>当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]。</li><li>当cacheModeOptional为PA_BLK_BNSD、PA_BLK_NZ时，shape为1维[Bkv\*ceil_div(Skv,BlockSize)]。</li><li>不支持空Tensor。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCacheRef</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td><ul><li>非量化场景下，数据类型与输入kv一致。</li><li>量化场景下，数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。</li><li>当cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dk]。</li><li>当cacheModeOptional为Norm场景时，shape为4维[Bcache,N,Scache,Dk]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvCacheRef</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td><ul><li>非量化场景下，数据类型与输入kv一致。</li><li>量化场景下，数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。</li><li>当cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）时，shape为4维[BlockNum,BlockSize,N,Dv]。</li><li>当cacheModeOptional为Norm场景时，shape为4维[Bcache,N,Scache,Dv]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeScaleOptional</td>
      <td>输入</td>
      <td>当kCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN时需要此输入参数。</td>
      <td><ul><li>shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvScaleOptional</td>
      <td>输入</td>
      <td>当ckvCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN时需要此输入参数。</td>
      <td><ul><li>shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeOffsetOptional</td>
      <td>输入</td>
      <td>当kCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN且对应的kRopeScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td><ul><li>shape为2维[N,Dk]；或者shape为1维[Dk,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOffsetOptional</td>
      <td>输入</td>
      <td>当ckvCacheRef数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN且对应的ckvScaleOptional输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td><ul><li>shape为2维[N,Dv]；或者shape为1维[Dv,]；或者shape为1维[1,]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOptional</td>
      <td>输入</td>
      <td>仅限kv分离场景(V2)中，作为immediate scatter的Dv分量输入来源。<br>shape的前三维度必须与kv保持一致，数据类型必须与kv保持一致。</td>
      <td><ul><li>shape为4维[Bkv,N,Skv,Dv]。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>rms_norm计算防止除0。</td>
      <td>建议设为1e-5。</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>属性</td>
      <td>cache格式的选择标记。</td>
      <td>类型有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ，建议设为Norm。</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isOutputKv</td>
      <td>属性</td>
      <td>kRopeOut和cKvOut输出控制标记。</td>
      <td>当isOutputKv为true时，表示需输出kRopeOut和cKvOut。建议设为false。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeOut</td>
      <td>输出</td>
      <td>由isOutputKv控制，对应interleaveRope计算公式中的`rope_out`。<br>当isOutputKv为true时，需输出。数据类型与输入kv一致。</td>
      <td>shape为4维[Bkv,N,Skv,Dk]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOut</td>
      <td>输出</td>
      <td>由isOutputKv控制，对应rmsNorm计算公式中的`y`。<br>当isOutputKv为true时，需输出。数据类型与输入kv一致。</td>
      <td>shape为4维[Bkv,N,Skv,Dv]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 325px">
  <col style="width: 126px">
  <col style="width: 809px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>输入和输出Tensor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>参数kv、gamma、cos、sin、index、kCacheRef、ckvCacheRef、kRopeScaleOptional、ckvScaleOptional、kRopeOffsetOptional、cKvOffsetOptional的shape校验非法。</td>
    </tr>
    <tr>
      <td>参数kv、gamma、cos、sin、index、kCacheRef、ckvCacheRef、kRopeScaleOptional、ckvScaleOptional、kRopeOffsetOptional、cKvOffsetOptional的dtype校验非法。</td>
    </tr>
    <tr>
      <td>PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ）下，cache的BlockSize维度值校验非法。</td>
    </tr>
    <tr>
      <td>NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ）下，Dk、Dv的维度值校验非法。</td>
    </tr>
  </tbody>
  </table>

## aclnnKvRmsNormRopeCacheV2

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnKvRmsNormRopeCacheV2GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  * 本算子默认确定性实现。
  * 输入shape限制：

      * kv为四维张量，shape为[Bkv,N,Skv,D]，Bkv为输入kv的batch size，Skv为输入kv的sequence length，大小由用户输入场景决定，无明确限制。
      * N为输入kv的head number。V1场景与DeepSeekV3网络结构强相关，仅支持N=1的场景。V2场景支持N=1/2/4/8。
      * D为输入kv的head dim。根据rope规则，Dk为偶数。若cacheModeOptional为NZ场景（cacheModeOptional为PA_NZ、PA_BLK_NZ），Dk、Dv需32B对齐。该规则适用于所有场景和计算类型中。
      * 若cacheModeOptional为PA场景（cacheModeOptional为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），block_size需32B对齐。
      * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以block_size为例，若cache的数据类型为int8，则需block_size%32=0；若cache的数据类型为float16，则需block_size%16=0；若kCacheRef与ckvCacheRef参数的dtype不一致，block_size需同时满足block_size%32=0和block_size%16=0。
      * block_num为写入cache的内存块数，大小由用户输入场景决定，无明确限制。
      * 旋转位置编码（RoPE）参数项(cos, sin)需要满足shape约束：
        * shape允许为4维[Bkv,N,Skv,Dk]或[Bkv,N,1,Dk]。
        * cos与sin的shape必须保持一致。
      * 量化参数项(kRopeScaleOptional, kRopeOffsetOptional, ckvScaleOptional, cKvOffsetOptional)需要满足shape约束：
        * 所有量化参数项的维度数量和N轴尺寸（如果存在）必须保持一致。
        * 不同场景下，量化参数的合法shape约束存在差异：

          | 场景类型 | 量化参数Shape |
          | :------: | :------ |
          |V1|<li>kRopeScaleOptional和kRopeOffsetOptional的shape支持：[1, Dk]、[Dk,]、[1,]。</li><li>ckvScaleOptional和cKvOffsetOptional的shape支持：[1, Dv]、[Dv,]、[1,]。</li>|
          |V2|<li>kRopeScaleOptional和kRopeOffsetOptional的shape支持：[N, Dk]。</li><li>ckvScaleOptional和cKvOffsetOptional的shape支持：[N, Dv]。</li>|

          * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：V1场景不支持[1,]，V2场景不支持量化参数项广播。

      * 输入张量均不支持空Tensor。
      * 所有输入均不支持无效值，包括且不限于：±inf，nan。

  * cache的数据类型支持：

    * 非量化模式：cache类型必须与kv保持一致。
      * <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：可支持BFLOAT16、FLOAT16。
      * <term>Kirin X90/Kirin 9030 处理器系列产品</term>：仅支持FLOAT16。
    * 量化模式：
      * <term>Ascend 950PR/Ascend 950DT</term>：可支持INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。
      * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Kirin X90/Kirin 9030 处理器系列产品</term>：仅支持INT8。

  * 参数说明：

    * 输入参数中kv, gamma, cos, sin, vOptional的数据类型必须完全一致。
    * kCacheRef和ckvCacheRef是<b>原地更新参数</b>，它们的数据类型取决于相应的输入分量，以及相应的scale和offset。详情见下：

      | cache_type | offset==None | offset!=None |
      | :-------------------------: | :-----------: | :----------------------------------------------: |
      | scale==None | 与kv保持一致 | 非法输入，拦截 |
      | scale!=None | INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN | INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN |

      * 非量化模式时，量化参数(kRopeScaleOptional, kRopeOffsetOptional, ckvScaleOptional, cKvOffsetOptional)必须设为None，且kCacheRef和ckvCacheRef的dtype必须与kv保持一致。
      * 量化模式时，kCacheRef和ckvCacheRef的dtype应为相应产品上支持的数据类型。

    * 输入分量关于量化因子scale与量化偏移scale的对应关系如下：

      | kv分量 | 分量输入来源 | 对应scale输入 | 对应offset输入 | 对应输出cache |
      | :------: | :------: | :------: | :------: | :------: |
      | k分量 | V1场景下，对应kv[..., Dv:]。<br>V2场景下，对应kv[..., :]。 | kRopeScaleOptional | kRopeOffsetOptional | kCacheRef |
      | v分量 | V1场景下，对应kv[..., :Dv]。<br>V2场景下，对应vOptional[..., :]。 |ckvScaleOptional | cKvOffsetOptional | ckvCacheRef |

      * kCacheRef：量化系数为kRopeScaleOptional和kRopeOffsetOptional。
      * ckvCacheRef：对应量化系数为ckvScaleOptional和cKvOffsetOptional。

    * 输出参数中，k_rope和c_kv的类型必须与kv保持一致。

  * 量化模式约束：

    <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
      <col style="width: 400px">
      <col style="width: 160px">
      <col style="width: 448px">
      </colgroup>
      <thead>
        <tr>
          <th>量化模式说明<br>(scale和offset输入之间为'与'关系)</th>
          <th>支持的cache模式</th>
          <th>可用性说明</th>
        </tr></thead>
      <tbody>
      <tr>
          <td rowspan="5">无量化模式<li>scale输入：kRopeScaleOptional==None && ckvScaleOptional==None </li><li>offset输入：kRopeOffsetOptional==None && cKvOffsetOptional==None</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5">V1和V2都支持。</td>
      </tr>
      <tr>
          <td>PA/PA_BNSD</td>
      </tr>
      <tr>
          <td>PA_NZ</td>
      </tr>
      <tr>
          <td>PA_BLK_BNSD</td>
      </tr>
      <tr>
          <td>PA_BLK_NZ</td>
      </tr>
      <tr>
          <td rowspan="5">静态量化模式<li>scale输入：kRopeScaleOptional和ckvScaleOptional至少一个非空。</li><li>offset输入：仅限对应scale为非空时，offset输入合法。相应offset如果为空，则为<b>静态对称量化</b>；相应offset如果非空，则为<b>静态非对称量化</b>。</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5"><li><b>静态对称量化</b>和<b>静态非对称量化</b>，支持存在差异。</li><li>支持K和V独立选择不同量化模式。</li></td>
      </tr>
      <tr>
          <td>PA/PA_BNSD</td>
      </tr>
      <tr>
          <td>PA_NZ</td>
      </tr>
      <tr>
          <td>PA_BLK_BNSD</td>
      </tr>
      <tr>
          <td>PA_BLK_NZ</td>
      </tr>
      </tbody>
    </table>

    * 静态量化模式支持细节：
      * Ascend 950PR/Ascend 950DT产品：仅支持V1场景，支持<b>静态对称量化</b>和<b>静态非对称量化</b>。
      * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - V1场景：对除`Norm`以外的cachemode，仅支持<b>静态对称量化</b>。即使传入合法的offset，也不会被算子处理，仍视为<b>静态对称量化</b>。对`Norm`模式的cachemode，不支持任何静态量化。
        - V2场景：对所有cachemode，支持<b>静态对称量化</b>和<b>静态非对称量化</b>。

  * cache与index相关约束：

    | cachemode| kCacheRef 形状 | ckvCacheRef 形状 | index 形状 | 说明 |
    | :------: | :------: | :------: | :------: | :------ |
    |Norm|[Bkv, N, Scache, Dk]|[Bkv, N, Scache, Dv]|[Bkv, Skv]|KV-Cache 更新模式，index 表示每个 Batch 下的偏移。<br>要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。<br>$Scache \ge Skv$|
    |PA/PA_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|PagedAttention 模式，index 表示每个 token 的偏移。<br>要求index的value值范围为[-1,block_num * block_size)。value数值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|Cache 数据格式为 FRACTAL_NZ 的 PagedAttention 模式，index表示每个 token 的偏移。<br>要求index的value值范围为[-1,block_num * block_size)。value数值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_BLK_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|特殊 PagedAttention 模式，index 表示每个 block 的起始偏移（不与 token逐一对应）。<br>要求index的value的数值范围为[-1,block_num * block_size)。value/block_size的值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_BLK_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|Cache 数据格式为 FRACTAL_NZ 的特殊的 PagedAttention 模式，index 表示每个 block 的起始偏移。<br>要求index的value的数值范围为[-1,block_num * block_size)。value/block_size的值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|

    * Scache为输入cache的sequence length，大小由用户输入场景决定，无明确限制。
    * 当cacheModeOptional为Norm时，shape为2维[Bkv,Skv]，要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。

    * 当cacheModeOptional为PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ时，cache中的数据排布方式为：
      * 非量化模式下：kCacheRef 为 [block_num, Dk//16, block_size, 1, 16]；ckvCacheRef 为 [block_num, Dv//16, block_size, 1, 16]。
      * 静态量化模式下：kCacheRef 为 [block_num, Dk//32, block_size, 1, 32]；ckvCacheRef 为 [block_num, Dv//32, block_size, 1, 32]。
    * 当cacheModeOptional为PA_BNSD、PA_NZ时，shape为1维[Bkv *Skv]，要求index的value值范围为[-1,block_num* block_size)。value数值不能重复。
    * 当cacheModeOptional为PA_BLK_BNSD、PA_BLK_NZ时，shape为1维[Bkv *ceil_div(Skv,block_size)]，要求index的value的数值范围为[-1,block_num* block_size)。value/block_size的值不能重复。

  * isOutputKv约束：
    * 作用是输出具体场景的中间处理结果，使能 kRopeOut 和 cKvOut 两项输出。
    * 在cacheModeOptional为PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, PA_BLK_NZ模式时有效。
    * 在cacheModeOptional为Norm时，仅在V2场景中使能量化模式时有效。

  * 输入组合约束：

    * 本约束内条目，仅适用于<b>Atlas A3 训练系列产品/Atlas A3 推理系列产品</b>、<b>Atlas A2 训练系列产品/Atlas A2 推理系列产品</b>。
    * 在所有cacheModeOptional下，必定支持`非广播模式`：即`[B, N, S]`三个维度与`kv`严格一致的`旋转位置编码（RoPE）参数`和`量化参数`。
    * 在各种cacheModeOptional下，支持的合法输入shape模式如下表：

      <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
        <col style="width: 160px">
        <col style="width: 280px">
        <col style="width: 280px">
        <col style="width: 320px">
        </colgroup>
        <thead>
          <tr>
            <th>cache模式</th>
            <th>kv Shape</th>
            <th>旋转位置编码（RoPE）参数<br>Shape</th>
            <th>其他参数</th>
          </tr>
          </thead>
        <tbody>
        <tr>
            <td>Norm</td>
            <td rowspan="6"><li>V1场景：[Bkv, 1, Skv, Dv+Dk]（kv）</li><li>V2场景：<ul><li>[Bkv, N, Skv, Dk]（kv）</li><li>[Bkv, N, Skv, Dv]（vOptional）</li></ul></li></td>
            <td rowspan="2"><li>支持S轴无广播：Srope=Skv。</li><li>支持S轴广播：Srope=1。</li></td>
            <td>在<b>无量化模式</b>下：isOutputKv无效，kRopeOut 和 cKvOut 无效。<li>V1场景：<ul><li>仅支持<b>无量化模式</b>，所有量化参数项皆为非法。</li></ul></li><li>V2场景：<ul><li>支持<b>无量化模式</b>和<b>静态量化模式</b>。</li><li>k和v对应的量化参数项必须为2维[N, D]。</li></ul></li></td>
        </tr>
        <tr>
            <td>PA</td>
            <td><b>无量化模式</b>下，支持S轴广播和S轴无广播。<li><b>静态量化模式</b>下，仅支持S轴无广播，否则为非法输入。</li><li>量化参数项：<ul><li>V1场景：支持shape为2维[1,D]或1维[D]。仅支持<b>静态对称量化</b>。</li><li>V2场景：支持shape为2维[N,D]。支持<b>静态对称量化</b>和<b>静态非对称量化</b>。</li></ul></li></td>
        </tr>
        <tr>
            <td>PA_BNSD</td>
            <td rowspan="4"><li>不支持S轴广播：必须满足Srope=Skv。</li></td>
            <td rowspan="4"><li>支持无量化模式。</li><li>量化参数项：<ul><li>V1场景：支持shape为2维[1,D]或1维[D]。仅支持<b>静态对称量化</b>。</li><li>V2场景：支持shape为2维[N,D]。支持<b>静态对称量化</b>和<b>静态非对称量化</b>。</li></ul></li></td>
        </tr>
        <tr>
            <td>PA_NZ</td>
        </tr>
        <tr>
            <td>PA_BLK_BNSD</td>
        </tr>
        <tr>
            <td>PA_BLK_NZ</td>
        </tr>
        </tbody>
      </table>

    * <b>未在上表覆盖范围内的输入组合，将导致算子的未定义行为</b>。

  * vOptional：
    * 该参数仅限aclnnKvRmsNormRopeCacheV2接口，aclnnKvRmsNormRopeCache接口不支持该参数！
    * 该参数仅限<b>Atlas A3 训练系列产品/Atlas A3 推理系列产品</b>、<b>Atlas A2 训练系列产品/Atlas A2 推理系列产品</b>。
      * 该参数仅在<b>kv分离场景(V2)</b>中作为必须入参，在其他类型中会作为无效参数被忽略。
      * 当vOptional存在时，它的类型必须与kv一致，`[B, N, S]`维度也必须与kv一致。

    * Ascend 950PR/Ascend 950DT：不会拦截该参数，但实际功能不支持，也不会处理该参数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache_v2.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, aclFloatToFloat16(0.0f));
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. 固定写法，device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口定义构造
  uint64_t totalBatch = 32;
  uint64_t totalHeads = 2;
  uint64_t seqLength = 64;
  uint64_t hDimK = 192;
  uint64_t hDimV = 128;
  uint64_t hDimRope = 64;
  uint64_t idxSlotNum = totalBatch * seqLength;
  std::vector<int64_t> kvShape = {totalBatch, totalHeads, seqLength, hDimK};
  std::vector<int64_t> gammaShape = {hDimK,};
  std::vector<int64_t> cosShape = {totalBatch, totalHeads, seqLength, hDimRope};
  std::vector<int64_t> sinShape = {totalBatch, totalHeads, seqLength, hDimRope};
  std::vector<int64_t> indexShape = {idxSlotNum,1};
  std::vector<int64_t> kpeCacheShape = {totalBatch, totalHeads, seqLength, hDimK};
  std::vector<int64_t> ckvCacheShape = {totalBatch, totalHeads, seqLength, hDimV};
  std::vector<int64_t> kRopeShape = {totalBatch, totalHeads, seqLength, hDimK};
  std::vector<int64_t> cKvShape = {totalBatch, totalHeads, seqLength, hDimV};
  std::vector<int64_t> vOptionalShape = {totalBatch, totalHeads, seqLength, hDimV};

  uint64_t totalEleHeads = totalBatch * totalHeads * seqLength;
  std::vector<int16_t> kvHostData(totalEleHeads*hDimK,0);
  std::vector<int16_t> gammaHostData(hDimK,0);
  std::vector<int16_t> cosHostData(totalEleHeads*hDimRope,0);
  std::vector<int16_t> sinHostData(totalEleHeads*hDimRope,0);
  std::vector<int64_t> indexHostData(idxSlotNum*1,0);           // Bkv * Skv
  std::vector<int16_t> kpeCacheHostData(totalEleHeads*hDimK,0);
  std::vector<int16_t> ckvCacheHostData(totalEleHeads*hDimV,0);
  std::vector<int16_t> kRopeHostData(totalEleHeads*hDimK,0);
  std::vector<int16_t> cKvHostData(totalEleHeads*hDimV,0);
  std::vector<int16_t> vOptionalHostData(totalEleHeads*hDimV,0);

  void* kvDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* kpeCacheDeviceAddr = nullptr;
  void* ckvCacheDeviceAddr = nullptr;
  void* vOptionalDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* cKvDeviceAddr = nullptr;

  aclTensor* kv = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* index = nullptr;
  aclTensor* kpeCache = nullptr;
  aclTensor* ckvCache = nullptr;
  aclTensor* vOpt = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* cKv = nullptr;

  double epsilon = 1e-5;
  char cacheMode[] = "Norm";
  bool isOutputKv = false;

  ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT16, &kv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kpeCacheHostData, kpeCacheShape, &kpeCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kpeCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(ckvCacheHostData, ckvCacheShape, &ckvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &ckvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vOptionalHostData, vOptionalShape, &vOptionalDeviceAddr, aclDataType::ACL_FLOAT16, &vOpt);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cKvHostData, cKvShape, &cKvDeviceAddr, aclDataType::ACL_FLOAT16, &cKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnKvRmsNormRopeCacheV2第一段接口
  ret = aclnnKvRmsNormRopeCacheV2GetWorkspaceSize(kv,gamma,cos,sin,index,
                                                kpeCache,ckvCache,nullptr,nullptr,nullptr,nullptr,vOpt,epsilon,cacheMode,isOutputKv,kRope,cKv,&workspaceSize,&executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnKvRmsNormRopeCacheV2第二段接口
  ret = aclnnKvRmsNormRopeCacheV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheV2 failed. ERROR: %d\n", ret); return ret);

  // 4. 固定写法，同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(kpeCacheShape, &kpeCacheDeviceAddr);
  PrintOutResult(ckvCacheShape, &ckvCacheDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(kv);
  aclDestroyTensor(gamma);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(index);
  aclDestroyTensor(kpeCache);
  aclDestroyTensor(ckvCache);
  aclDestroyTensor(vOpt);
  aclDestroyTensor(kRope);
  aclDestroyTensor(cKv);

  // 7. 释放device资源
  aclrtFree(kvDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(kpeCacheDeviceAddr);
  aclrtFree(ckvCacheDeviceAddr);
  aclrtFree(vOptionalDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(cKvDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
