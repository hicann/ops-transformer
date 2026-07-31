# aclnnFusedInferAttentionScoreV5

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：适配decode & prefill场景的FlashAttention算子，既可以支持prefill计算场景（PromptFlashAttention），也可支持decode计算场景（IncreFlashAttention）。

  相比于FusedInferAttentionScoreV4，本接口新增qStartIdxOptional、kvStartIdxOptional、pseType参数。

  **说明：**

  decode场景下特有KV Cache：KV Cache是大模型推理性能优化的一个常用技术。采样时，Transformer模型会以给定的prompt/context作为初始输入进行推理（可以并行处理），随后逐一生成额外的token来继续完善生成的序列（体现了模型的自回归性质）。在采样过程中，Transformer会执行自注意力操作，为此需要给当前序列中的每个项目（无论是prompt/context还是生成的token）提取键值（KV）向量。这些向量存储在一个矩阵中，通常被称为kv缓存（KV Cache）。
- 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为：

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

  **说明**：

  <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
    <br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnFusedInferAttentionScoreV5GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedInferAttentionScoreV5”接口执行计算。

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShiftOptional,
    const aclTensor     *attenMaskOptional,
    const aclIntArray   *actualSeqLengthsOptional,
    const aclIntArray   *actualSeqLengthsKvOptional,
    const aclTensor     *deqScale1Optional,
    const aclTensor     *quantScale1Optional,
    const aclTensor     *deqScale2Optional,
    const aclTensor     *quantScale2Optional,
    const aclTensor     *quantOffset2Optional,
    const aclTensor     *antiquantScaleOptional,
    const aclTensor     *antiquantOffsetOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *queryPaddingSizeOptional,
    const aclTensor     *kvPaddingSizeOptional,
    const aclTensor     *keyAntiquantScaleOptional,
    const aclTensor     *keyAntiquantOffsetOptional,
    const aclTensor     *valueAntiquantScaleOptional,
    const aclTensor     *valueAntiquantOffsetOptional,
    const aclTensor     *keySharedPrefixOptional,
    const aclTensor     *valueSharedPrefixOptional,
    const aclIntArray   *actualSharedPrefixLenOptional,
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    const aclTensor     *keyRopeAntiquantScaleOptional,
    const aclTensor     *dequantScaleQueryOptional,
    const aclTensor     *learnableSinkOptional,
    const aclIntArray   *qStartIdxOptional,
    const aclIntArray   *kvStartIdxOptional,
    int64_t             numHeads,
    double              scaleValue,
    int64_t             preTokens,
    int64_t             nextTokens,
    char                *inputLayout,
    int64_t             numKeyValueHeads,
    int64_t             sparseMode,
    int64_t             innerPrecise,
    int64_t             blockSize,
    int64_t             antiquantMode,
    bool                softmaxLseFlag,
    int64_t             keyAntiquantMode,
    int64_t             valueAntiquantMode,
    int64_t             queryQuantMode,
    int64_t             pseType,
    const aclTensor     *attentionOut,
    const aclTensor     *softmaxLse,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnFusedInferAttentionScoreV5(
    void                *workspace,
    uint64_t            workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream   stream)
```

## aclnnFusedInferAttentionScoreV5GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
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
        <td>query</td>
        <td>输入</td>
        <td>公式中的输入Q。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的输入K。</td>
        <td>
        <ul>
            <li>不支持非连续tensor。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN、INT4(INT32)、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的输入V。</td>
        <td>
        <ul>
            <li>不支持非连续tensor。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8_E4M3FN、INT4(INT32)、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>见参数inputLayout</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShiftOptional</td>
        <td>输入</td>
        <td>位置编码</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>建议shape输入(B,Q_N,Q_S,KV_S)、(1,Q_N,Q_S,KV_S)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>mask矩阵</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>BOOL、INT8、UINT8</td>
        <td>ND</td>
        <td>见<a href="#MaskChecker">Attention Mask参数组中的综合限制</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengthsOptional</td>
        <td>输入</td>
        <td>不同Batch中query的有效序列长度。</td>
        <td>
        <ul>
            <li>不指定序列长度可传入nullptr，表示和query的shape的S长度相同。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>输入</td>
        <td>不同Batch中key/value的有效序列长度。</td>
        <td>
        <ul>
            <li>不指定序列长度可传入nullptr，表示和key/value的shape的S长度相同。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）或（B）或（>B）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1Optional</td>
        <td>输入</td>
        <td>BMM1后面的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。</li>
            <li>使用全量化功能时，该参数由实际量化过程计算得来。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#DequantChecker">伪量化/全量化参数组中的综合限制</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1Optional</td>
        <td>输入</td>
        <td>BMM2前面的量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。 </li>
            <li>支持MxFP8。 </li>
            <li>支持范围应满足[1, 448]。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#DequantChecker">伪量化/全量化参数组中的综合限制</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2Optional</td>
        <td>输入</td>
        <td>BMM2后面的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor。 </li>
            <li>使用全量化功能时，该参数由实际量化过程计算得来。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>UINT64、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#DequantChecker">伪量化/全量化参数组中的综合限制</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2Optional</td>
        <td>输入</td>
        <td>输出的量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>见<a href="#PostQuantChecker">后量化参数组中的综合限制</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2Optional</td>
        <td>输入</td>
        <td>输出的量化偏移。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>支持per-tensor，per-channel。 </li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>与quantScale2Optional保持一致</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantScaleOptional</td>
        <td>输入</td>
        <td>伪量化因子</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantOffsetOptional</td>
        <td>输入</td>
        <td>伪量化偏移</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockTableOptional</td>
        <td>输入</td>
        <td>PageAttention中KV存储使用的block映射表。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为不同batch中最大actualSeqLengthsKv对应的block数量）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryPaddingSizeOptional</td>
        <td>输入</td>
        <td>表示Query中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>仅支持Q_S大于1，其余场景该参数无效。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvPaddingSizeOptional</td>
        <td>输入</td>
        <td>表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantScaleOptional</td>
        <td>输入</td>
        <td>表示key的反量化因子，用于KV伪量化参数分离场景。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>不支持非连续tensor。</li>
            <li>支持per-tensor，per-channel，per-token，per-token-group，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per-head并使用page attention模式管理scale/offset和per-token-group。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyAntiquantOffsetOptional</td>
        <td>输入</td>
        <td>KV伪量化参数分离时表示key的反量化偏移。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>使用时，shape必须与keyAntiquantScaleOptional保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-channel-group，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantScaleOptional</td>
        <td>输入</td>
        <td>表示value的反量化因子，用于KV伪量化参数分离场景。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>不支持非连续tensor。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset和per-token-group。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueAntiquantOffsetOptional</td>
        <td>输入</td>
        <td>KV伪量化参数分离时表示value的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>使用时，shape必须与valueAntiquantScaleOptional保持一致。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>支持per-tensor，per-channel，per-token，per-tensor叠加per-head，per-token叠加per-head，per-token叠加使用page attention模式管理scale/offset、per-token叠加per head并使用page attention模式管理scale/offset。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>keySharedPrefixOptional</td>
        <td>输入</td>
        <td>attention结构中Key的系统前缀部分的参数。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4/INT32</td>
        <td>ND</td>
        <td>
        <ul>
            <li>input_layout为BSH时，shape为（1，prefix_S，H=KV_N*KV_D）</li>
            <li>input_layout为BSND时，shape为（1，prefix_S，KV_N，KV_D）</li>
            <li>input_layout为BNSD、BNSD_BSND时，shape为（1，KV_N，prefix_S，KV_D）</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>valueSharedPrefixOptional</td>
        <td>输入</td>
        <td>attention结构中Value的系统前缀部分的输入。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16、INT8、INT4/INT32</td>
        <td>ND</td>
        <td>
        <ul>
            <li>input_layout为BSH时，shape为（1，prefix_S，H=KV_N*KV_D）</li>
            <li>input_layout为BSND时，shape为（1，prefix_S，KV_N，KV_D）</li>
            <li>input_layout为BNSD、BNSD_BSND时，shape为（1，KV_N，prefix_S，KV_D）</li>
        </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSharedPrefixLenOptional</td>
        <td>输入</td>
        <td>keySharedPrefix/valueSharedPrefix的有效Sequence Length。</td>
        <td>
        <ul>
            <li>不使用该功能时可传入nullptr，表示和keySharedPrefix/valueSharedPrefix的s长度相同。</li>
            <li>该入参中的有效Sequence Length应该不大于keySharedPrefix/valueSharedPrefix中的Sequence Length。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>（1）</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryRopeOptional</td>
        <td>输入</td>
        <td>MLA结构中的query的rope信息。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>queryRope的shape中d为64，其余维度与query一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeOptional</td>
        <td>输入</td>
        <td>MLA结构中的key的rope信息。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>不支持非连续tensor。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>keyRope的shape中d为64，其余维度与key一致</td>
        <td>×</td>
    </tr>
    <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>输入</td>
        <td>MLA结构中的key的rope信息的反量化因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>预留参数，当前版本不生效，传入nullptr即可。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dequantScaleQueryOptional</td>
        <td>输入</td>
        <td>对query进行反量化的因子。</td>
        <td>
        <ul>
            <li>不支持空Tensor。</li>
            <li>全量化场景涉及。量化模式支持per-token-group，per-token叠加per-head模式。</li>
            <li>不使用该功能时可传入nullptr。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>learnableSinkOptional</td>
        <td>输入</td>
        <td>表示通过可学习的"Sink Token"起到吸收Attention Score的作用。</td>
        <td>
        <ul>
            <li>仅支持非量化场景。</li>
            <li>不使用该功能时可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>(Q_N)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>qStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的query的sequence在全局中的起始索引。</td>
        <td>
        <ul>
            <li>可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvStartIdxOptional</td>
        <td>输入</td>
        <td>代表外切场景，当前分块的key和value的sequence在全局中的起始索引。</td>
        <td>
        <ul>
            <li>可传入nullptr。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>见<a href="#约束说明">约束说明</a></td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>输入</td>
        <td>query的head个数。</td>
        <td>在BNSD、BSND、BNSD_BSND、BSND_BNSD、BNSD_NBSD、BSND_NBSD、TND、NTD、NTD_TND、TND_NTD场景下，需要与shape中的query的N轴shape值相同，否则执行异常。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>输入</td>
        <td>公式中d开根号的倒数。</td>
        <td>
        <ul>
            <li>数据类型与query的数据类型需满足数据类型推导规则。 </li>
            <li>用户不特意指定时建议传入1.0。 </li>
        </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>输入</td>
        <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入2147483647。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>nextTokens</td>
        <td>输入</td>
        <td>表示attention需要和后几个Token计算关联。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入2147483647。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>输入</td>
        <td>标识输入query、key、value的数据排布格式。</td>
        <td>
        <ul>
            <li>当前支持BSH、BSND、BNSD、BNSD_BSND（输入为BNSD时，输出格式为BSND）、BSND_BNSD（输入为BSND时，输出格式为BNSD）、BSH_BNSD（输入为BSH时，输出格式为BNSD）、BNSD_NBSD（输入为BNSD时，输出格式为NBSD）、BSND_NBSD（输入为BSND时，输出格式为NBSD）、BSH_NBSD（输入为BSH时，输出格式为NBSD）、TND（TND相关场景综合约束见<a href="#约束说明">约束说明</a>）、NTD、NTD_TND（输入为NTD时，输出格式为TND）、TND_NTD（输入为TND时，输出格式为NTD）。不特意指定时建议传入"BSH"。</li>
            <li>注意排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出output的格式。</li>
            <li>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。</li>
            <li>inputLayout=BSH_BNSD、BSND_BNSD、NTD、NTD_TND仅支持Q_D=K_D=V_D都等于64或128，或Q_D=K_D等于192，V_D等于128。</li>
            <li>inputLayout=BNSD_BSND仅支持Q_D=K_D=V_D都16对齐(output dtype为int8时为32对齐)，或Q_D=K_D等于192，V_D等于128<br></li>
        </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>输入</td>
        <td>key、value中head个数。</td>
        <td>
        <ul>
            <li>用户不特意指定时建议传入0，表示key/value和query的head个数相等。</li>
            <li>需要满足numHeads整除numKeyValueHeads，GQA非量化场景和Prefill MLA非量化场景下，numHeads与numKeyValueHeads的比值无限制; Decode MLA场景仅支持numHeads与numKeyValueHeads的比值为1、2、4、8、16、32、64、128。</li>
            <li>在BNSD、BSND、BNSD_BSND、BSND_BNSD、BNSD_NBSD、BSND_NBSD、TND、NTD、NTD_TND、TND_NTD场景下，还需要与shape中的key/value的N轴shape值相同，否则执行异常</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>输入</td>
        <td>sparse的模式。</td>
        <td>
        <ul>
            <li>inputLayout为TND、TND_NTD、NTD_TND时，综合约束请见<a href="#约束说明">约束说明</a>。</li>
            <li>sparseMode为0时，代表defaultMask模式，如果attenmask未传入则不做mask操作，忽略preTokens和nextTokens（内部赋值为INT_MAX）；如果传入，则需要传入完整的attenmask矩阵（S1 * S2），表示preTokens和nextTokens之间的部分需要计算；要求preTokens + nextTokens >= 0。 </li>
            <li>sparseMode为1时，代表allMask，必须传入完整的attenmask矩阵（S1 * S2）。</li>
            <li>sparseMode为2时，代表leftUpCausal模式的mask，需要传入优化后的attenmask矩阵（2048*2048）。</li>
            <li>sparseMode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵（2048*2048）。</li>
            <li>sparseMode为4时，代表band模式的mask，需要传入优化后的attenmask矩阵（2048*2048）；要求preTokens + nextTokens >= 0。</li>
            <li>sparseMode为5、6、7、8时，分别代表prefix、global、dilated、block_local，均暂不支持。</li>
            <li>用户不特意指定时建议传入0。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>innerPrecise</td>
        <td>输入</td>
        <td>表示高精度或者高性能选择。</td>
        <td>
        <ul>
            <li>innerPrecise为0时，代表开启高精度模式，且不做行无效修正。</li>
            <li>innerPrecise为1时，代表高性能模式，且不做行无效修正。</li>
            <li>innerPrecise为2时，代表开启高精度模式，且做行无效修正。</li>
            <li>innerPrecise为3时，代表高性能模式，且做行无效修正。</li>
            <li>sparse_mode为0或1，并传入用户自定义mask的情况下，建议开启行无效修正。</li>
            <li>BFLOAT16和INT8不区分高精度和高性能，行无效修正对FLOAT16、BFLOAT16和INT8均生效。</li>
            <li>当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来开启行无效功能以提升精度，但是该配置会导致性能下降。</li>
            <li>如果算子可判断出存在无效行场景，会自动开启无效行计算，例如sparse_mode为3，Sq > Skv场景。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize</td>
        <td>输入</td>
        <td>PageAttention中KV存储每个block中最大的token个数。</td>
        <td>不传时按照0处理。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>antiquantMode</td>
        <td>输入</td>
        <td>伪量化的方式</td>
        <td>不支持</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLseFlag</td>
        <td>输入</td>
        <td>是否输出softmax_lse。</td>
        <td>
        <ul>
            <li>支持S轴外切（增加输出）。</li>
            <li>用户不特意指定时建议传入false。</li>
        </ul>
        </td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>keyAntiquantMode</td>
        <td>输入</td>
        <td>key的反量化的方式。</td>
        <td>
        <ul>
            <li>不特意指定时建议传入0。</li>
            <li>除了keyAntiquantMode为0并且valueAntiquantMode为1的场景外，需要与valueAntiquantMode一致。</li>
            <li>keyAntiquantMode为0时，代表per-channel模式（per-channel包含per-tensor）。</li>
            <li>keyAntiquantMode为1时，代表per-token模式。</li>
            <li>keyAntiquantMode为2时，代表per-tensor叠加per-head模式。</li>
            <li>keyAntiquantMode为3时，代表per-token叠加per-head模式。</li>
            <li>keyAntiquantMode为4时，代表per-token叠加使用page attention模式管理scale/offset模式。</li>
            <li>keyAntiquantMode为5时，代表per-token叠加per head并使用page attention模式管理scale/offset模式。</li>
            <li>keyAntiquantMode为6时，代表per-token-group模式。</li>
            <li>传入0-6之外的其他值会执行异常。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>valueAntiquantMode</td>
        <td>输入</td>
        <td>value的反量化的方式。</td>
        <td>
        <ul>
            <li>valueAntiquantMode模式编号0-6与keyAntiquantMode一致。</li>
            <li>valueAntiquantMode为8时，代表per-channel-group模式。</li>
            <li>用户不特意指定时建议传入0。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>queryQuantMode</td>
        <td>输入</td>
        <td>query反量化的模式。</td>
        <td>
        <ul>
            <li>模式编号与keyAntiquantMode一致。</li>
            <li>用户不特意指定时建议传入0。</li>
            <li>综合约束请见<a href="#约束说明">约束说明</a>。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>pseType</td>
        <td>输入</td>
        <td>pse的方式。</td>
        <td>
        <ul>
            <li>支持配置值为0（pseType = 1推理场景不支持）。</li>
            <li>pseType为0时，外部传入pse，先mul再add。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attentionOut</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、HIFLOAT8</td>
        <td>ND</td>
        <td>该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>输出</td>
        <td>ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max,再取exp，接着求sum，得到softmax_sum。最后对softmax_sum取log，再加上softmax_max得到的结果。</td>
        <td>
        <ul>
            <li>用户不特意指定时建议传入nullptr。</li>
            <li>数据为inf的代表无效数据；softmaxLseFlag为False时，如果softmaxLse传入的Tensor非空，则直接返回该Tensor数据，如果softmaxLse传入的是nullptr，则返回shape为{1}全0的Tensor。</li>
        </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>softmaxLseFlag为True时，一般情况下，shape必须为[B, N, Q_S, 1]，当inputLayout为TND/NTD/NTD_TND/TND_NTD时，shape必须为[T, N, 1]。</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </thead>
    <tbody>
        <tr>
            <td>ACLNN_ERR_PARAM_NULLPTR</td>
            <td>161001</td>
            <td>传入的query、key、value、attentionOut是空指针。</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_PARAM_INVALID</td>
            <td>161002</td>
            <td>query、key、value、pseShiftOptional、attenMaskOptional、attentionOut的数据类型和数据格式不在支持的范围内。</td>
        </tr>
        <tr>
            <td>ACLNN_ERR_RUNTIME_ERROR</td>
            <td>361001</td>
            <td>API内存调用npu runtime的接口异常。</td>
        </tr>
    </tbody>
    </table>

## aclnnFusedInferAttentionScoreV5

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedInferAttentionScoreV5GetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

### 约束类型说明

FusedInferAttentionScore算子约束分为4个档位，按约束复杂程度递增分为单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容和示例如下：

- 单参数约束：对于单个接口参数的约束，包含FusedInferAttentionScore算子接口中的Tensor、TensorList、Array和Attributes
  - 对于Tensor、TensorList、Array，单参数约束中包含如下校验
    - 校验shape，包括shape维度dim、每一维度dim value
    - 校验dtype
    - 校验format
  - 对于属性Attribute
    - 校验属性取值
- 存在性约束：约束特定场景下，特性参数组内，必须传入某参数，或不支持传入某参数
- 一致性约束：特性参数组内，各个参数间约束。
  - Example 1：属性sparseMode和输入tensor attenMask均属于Attention Mask参数组，sparseMode取值为2/3/4时，attenMask shape必须为（2048，2048），此类约束即为参数组内的一致性约束
- 特性交叉约束：涉及多个参数组，不同参数组间交叉约束
  - Example 1：输入tensor blockTable和属性blockSize属于Paged Attention（同PA）参数组，输入tensor attenMask属于Mask参数组；在PA场景下，attenMask输入最后一维（attenMaskS2）需要大于等于maxBlockNumPerSeq * blockSize，此类约束即为多参数组间的交叉约束；**且为保证风格统一，此约束会放在入参顺序靠后的Paged Attention参数组中**

### 特性参数组

<table><thead>
  <tr>
    <th>特性参数组</th>
    <th>参数字段名称</th>
    <th>字段分组</th>
    <th>字段类型</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="9">公共参数组</td>
    <td>query</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>key</td>
    <td>INPUT</td>
    <td>TensorList</td>
  </tr>
  <tr>
    <td>value</td>
    <td>INPUT</td>
    <td>TensorList</td>
  </tr>
  <tr>
    <td>numHeads</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>ATTR(OPTIONAL)</td>
    <td>double</td>
  </tr>
  <tr>
    <td>inputLayout</td>
    <td>ATTR(OPTIONAL)</td>
    <td>string</td>
  </tr>
  <tr>
    <td>numKeyValueHeads</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>innerPrecise</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="4">PSE参数组</td>
    <td>pseShift</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>qStartIdxOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td>kvStartIdxOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td>pseType</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="4">Mask参数组</td>
    <td>attenMaskOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>preTokens</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>nextTokens</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>sparseMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="2">ActualSeqLens参数组</td>
    <td>actualSeqLengthsOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td>actualSeqLengthsKvOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td rowspan="14">伪量化&amp;全量化参数组</td>
    <td>deqScale1Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>quantScale1Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>deqScale2Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>antiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>antiquantOffsetOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyAntiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyAntiquantOffsetOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>valueAntiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>valueAntiquantOffsetOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>dequantScaleQueryOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>antiquantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>keyAntiquantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>valueAntiquantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>queryQuantMode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="3">PostQuant参数组</td>
    <td>quantScale2Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>quantOffset2Optional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>outType</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="2">Paged Attention参数组</td>
    <td>blockTableOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>blockSize</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td rowspan="2">LeftPadding参数组</td>
    <td>queryPaddingSizeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>kvPaddingSizeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="3">SystemPrefix参数组</td>
    <td>keySharedPrefixOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>valueSharedPrefixOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>actualSharedPrefixLenOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>IntArray</td>
  </tr>
  <tr>
    <td rowspan="3">Rope参数组</td>
    <td>queryRopeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyRopeOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>keyRopeAntiquantScaleOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>LearnableSink参数组</td>
    <td>learnableSinkOptional</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="2">SoftmaxLSE参数组</td>
    <td>softmaxLseFlag</td>
    <td>ATTR(OPTIONAL)</td>
    <td>bool</td>
  </tr>
  <tr>
    <td>softmaxLse</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
</tbody></table>

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|     GQA     |           在资料约束中，泛指不传入ROPE的所有场景，包含G=1和G>1的场景           |
| Prefill MLA | 传入ROPE（包含合并和分离2种模式，headdim为64）：分离模式下，输入Q/K/V headdim为128，ROPE单独传入；合并模式下，输入Q/K headdim为192（与ROPE合并），V headdim为128，ROPE不单独传入 |
| Decode MLA |             输入Q/K/V headdim为512，key/value复用，其中value的不生效，且ROPE单独传入             |
| GQA全量化 | query/key/value数据类型为FLOAT8_E4M3FN的GQA全量化场景，queryQuantMode为3（per-token叠加per-head模式），keyAntiquantMode为3（per-token叠加per-head模式），valueAntiquantMode为2（per-tensor叠加per-head模式） |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入query tensor的头数，对应query shape中的N        |
|    KV_N    |    输入key/value tensor的头数，对应key/value shape中的N    |
|     Q_S     |      输入query tensor的序列长度，对应query shape中的S      |
|    KV_S    |  输入key/value tensor的序列长度，对应key/value shape中的S  |
|     Q_T     |          输入query tensor所有batch序列长度的累加和          |
|      G     |        Group，表示查询头的分组数，对应Q_N与KV_N的比值         |

### 参数组约束

<!--
#### 整体规则
1. 按照特性参数组 -> 四层约束 -> 量化类型（非量化/伪量化/全量化）-> FA场景（GQA/Prefill MLA/Decode MLA）-> 芯片代际（Atlas A2/Ascend 950PR）层级维护
2. 每个特性组，缩进风格一致，表格宽高无固定约束，保持简洁清晰即可
3. 约束中参数命名，第一次出现必须和接口中参数名一致，后续可通过“attenMask（同mask/后称mask）”方式缩写
4. 对于输入各轴缩写，缩写需要具有自解释性，如QueryS、KeyNumHead、PseShiftS2等，禁止出现S1、B、n2等字眼；最好在基准信息说明中补充各个变量含义
-->

#### 公共参数组（CommonChecker）

- 单参数约束

  - 公共约束

    <table style="undefined;table-layout: fixed; width: 1100px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 250px">
            <col style="width: 200px">
            <col style="width: 450px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>Query/Key/Value</th>
                <th>numHeads</th>
                <th>inputLayout</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td rowspan="3"><ul>
                                    <li>数据类型仅支持均为FLOAT16/BFLOAT16</li>
                                    <li>数据格式均仅支持ND</li>
                                </ul>
                </td>
                <td>须为整数，小于2的32次方</td>
                <td>支持BNSD、BSND、BSH、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND、BNSD_BSND</td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>须为整数，小于2的32次方</td>
                <td>支持BNSD、BSND、BSH、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND、BNSD_BSND</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>仅支持1,2,4,8,16,32,64,128</td>
                <td>支持BNSD、BSND、BSH、TND、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND_NTD</td>
            </tr>
        </tbody>
    </table>

- 存在性约束

  <table style="undefined;table-layout: fixed; width: 700px">
        <colgroup>
            <col style="width: 400px">
            <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>结果预期</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>query,key,value,attentionOut存在nullptr</td>
                <td>正常拦截</td>
            </tr>
            <tr>
                <td>query,attentionOut的tensor的shapeSize为0</td>
                <td>attentionOut为返回空tesor</td>
            </tr>
            <tr>
                <td>query,attentionOut的tensor的shapeSize不为0,且key,value的tensor的shapeSize为0</td>
                <td>attentionOut返回全0</td>
            </tr>
        </tbody>
    </table>

- 一致性约束

  <table style="undefined;table-layout: fixed; width: 1500px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 300px">
            <col style="width: 300px">
            <col style="width: 350px">
            <col style="width: 350px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>B</th>
                <th>N</th>
                <th>D</th>
                <th>numHeads/numKeyValueHeads</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>GQA</td>
                <td rowspan="3">
                    <ul>
                        <li>支持B轴小于等于65536</li>
                        <li>非连续场景下key、value的tensorlist中的batch只能为1,B不能大于256</li>
                    </ul>
                </td>
                <td rowspan="3">
                    query的layout不为BSH时，query/attentionOut的N轴与numHeads保持一致，key/value的N轴与numKeyValueHeads保持一致
                </td>
                <td>
                    <ul>
                        <li>支持D轴小于等于512</li>
                        <li>非量化场景下，当query/key/value三组HeadDim均小于等于128且layout不为NTD/NTD_TND/BSH_BNSD/BSND_BNSD/BNSD_BSND时，支持参数query/key的HeadDim与value的HeadDim不相等
                        </li>
                        <li>MxFP8场景下，HeadDim仅支持64或128</li>
                        <li>GQA全量化场景下，HeadDim仅支持128</li>
                        <li>layout为NTD/NTD_TND/BSH_BNSD/BSND_BNSD时，HeadDim仅支持64或128</li>
                        <li>layout为BNSD_BSND时，HeadDim需要16对齐</li>
                        <li>layout为BNSD_BSND，且AttentionOut数据类型为INT8时，HeadDim需要32对齐</li>
                    </ul>
                </td>
                <td rowspan="2">
                    numHeads需可整除numKeyValueHeads，numHeads与numKeyValueHeads的比值无限制
                </td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>支持参数query/key的HeadDim为192，value的HeadDim为128场景，其余场景query/key/value/attentionOut的HeadDim需保持一致</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>query/key/value/attentionOut的HeadDim需保持一致</td>
                <td>仅numHeads为1、2、4、8、16、32、64、128, numKeyValueHeads为1。</td>
            </tr>
        </tbody>
    </table>

- 特性交叉约束

  - 非量化场景
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 400px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>不支持场景</th>
                <th>PagedAttention场景</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>公共</td>
                <td>
                    query的layout为TND/NTD时，不支持pseShift、tensorlist
                </td>
                <td rowspan="4">
                    <ul>
                        <li>当inputLayout为BNSD、TND、BSH、BSND,Key/Value排布支持BnBsH（blockNum, blockSize, H）、BnNBsD（blockNum, KV_N,
                            blockSize, D）和NZ（blockNum，KV_N，D/16，blockSize，16）三种格式</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>GQA</td>
                <td>query/key的HeadDim与value的HeadDim不相等时，除attenMask参数组外，其余均不支持</td>
            </tr>
            <tr>
                <td>Prefill MLA</td>
                <td>不支持全量化、伪量化</td>
            </tr>
            <tr>
                <td>Decode MLA</td>
                <td>不支持tensorlist、左padding、伪量化、prefix</td>
            </tr>
        </tbody>
    </table>

      <!-- npu="A3,910b" id7 -->
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - Prefill MLA场景下，不支持tensorlist、左padding
        - MLA场景下，不支持后量化

      <!-- end id7 -->

#### PSE参数组

- 单参数约束
  - 公共
    - 入参pseType应满足以下条件：
      - pseType必须为0
      - pseType为1不支持FA推理场景，仅支持FA训练场景
    - 入参pseShift应满足以下条件：
      - tensor的数据类型应满足以下条件：
        - pseType为0时，且query的数据类型为FLOAT16或者INT8时，tensor的数据类型必须为FLOAT16
        - pseType为0时，且query的数据类型为BFLOAT16时，tensor的数据类型必须为BFLOAT16
      - tensor shape应满足以下条件：
        - pseType为0时:
          - tensor shape的维度必须为4
          - P_S1(tensor shape的第3维) > 1时：
            - tensor shape的第1维应等于1或者B
            - tensor shape的第2维应等于Q_N
            - tensor shape的第3为应大于等于Q_S
            - 非prefix场景时，tensor shape的第4维应大于等于KV_S
            - prefix场景时，tensor shape的第4维应大于等于KV_S + actualSharedPrefixLen
          - P_S1(tensor shape的第3维) = 1时：
            - tensor shape的第1维应等于1或者B
            - tensor shape的第2维应等于Q_N
            - 非prefix场景时，tensor shape的第4维应大于等于KV_S
            - prefix场景时，tensor shape的第4维应大于等于KV_S + actualSharedPrefixLen
- 特性交叉约束
  - 公共
    - PagedAttention场景下，入参pseShift应满足以下条件：
      - tensor shape的最后一维应大于等于maxBlockNumPerBatch * blockSize
    - alibi场景下，Q_S应等于KV_S
    - MLA场景下，入参pseShift应满足以下条件：
      - 不支持pse，不能传入pseShift
    - D不等长场景下，入参pseShift应满足以下条件：
      - 不支持pse，不能传入pseShift
    - 伪量化场景下，当pseType为0且Q_S为1时，pseShift的第三维仅支持1
    - GQA全量化场景下，不支持pse，不能传入pseShift

#### <span id="MaskChecker">Attention Mask参数组</span>

- 单参数约束
  - 公共
    - 入参attenMask需要满足以下条件：

      - tensor dtype为INT8/UINT8/BOOL类型
      - tensor format为ND/NCHW/NHWC/NCDHW类型
      - 如果输入attenMask shape中的Q_S、KV_S非32B对齐，可以向上取到对齐的Q_S、KV_S
    - 入参sparseMode需要满足以下条件：

      - sparseMode支持输入范围为0-4，默认值为0
      - sparseMode在不开启mask时，仅支持输入为0
      - sparseMode含义如下表所示（注：attenMask矩阵示例部分中的1 = masked out，0 = keep）

      | sparseMode |                                                                               含义                                                                               |      attenMask矩阵示例      |                                             备注                                             |
      | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------: | :-------------------------------------------------------------------------------------------: |
      |     0     | defaultMask模式，如果attenMask未传入则不做mask操作，忽略preTokens和nextTokens；<br />如果传入，则需要传入完整的attenMask矩阵，计算preTokens和nextTokens之间的部分 | 11111<br />01111<br />00111 |          完整的attenMask矩阵，即全量的Q_S*KV_S矩阵，<br />因此可以自定义不同mask场景          |
      |     1     |                                                allMask模式，必须传入完整的attenMask矩阵，忽略preTokens和nextTokens                                                | 00101<br />10111<br />10101 |                                             同上                                             |
      |     2     |                                            leftUpCausal模式，需要传入优化后的attenMask矩阵，忽略preTokens和nextTokens                                            | 01111<br />00111<br />00011 | 优化后的attenMask矩阵，固定为2048*2048的下三角矩阵，<br />以左上顶点为参数起点划分，对角线全0 |
      |     3     |                                           rightDownCausal模式，需要传入优化后的attenMask矩阵，忽略preTokens和nextTokens                                           | 00011<br />00001<br />00000 | 优化后的attenMask矩阵，固定为2048*2048的下三角矩阵，<br />以右下顶点为参数起点划分，对角线全0 |
      |     4     |                                           band模式，需要传入优化后的attenMask矩阵，计算preTokens和nextTokens之间的部分                                           | 00011<br />10001<br />11000 | 优化后的attenMask矩阵，固定为2048*2048的下三角矩阵，<br />以右下顶点为参数起点划分，对角线全0 |
      |     5     |                                                                              prefix                                                                              |              -              |                                            不支持                                            |
      |     6     |                                                                              global                                                                              |              -              |                                            不支持                                            |
      |     7     |                                                                              dilated                                                                              |              -              |                                            不支持                                            |
      |     8     |                                                                            block_local                                                                            |              -              |                                            不支持                                            |

- 存在性约束
  - 无
- 一致性约束
  - 公共
    - 入参attenMask的输入维度仅支持2/3/4

      <!-- npu="950" id8 -->
      - <term>Ascend 950PR/Ascend 950DT</term>：
        - 维度为2时，不支持sparseMode为0/1模式
      <!-- end id8 -->
      <!-- npu="A3,910b" id9 -->
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - sparseMode为0/1模式时，若传入query_rope和key_rope，或者query与key的D不等于value的D，则不支持attenMask输入维度为2
        - sparseMode为0/1模式时，若attenMask输入维度为2，则layout仅支持为BSH、BSND、BNSD、BNSD_BSND
      <!-- end id9 -->

    - 入参sparseMode为0/1模式时，attenMask矩阵的shape应满足 [batchSize/1，>=Q_S，>=KV_S + actualSharedPrefixLen]
    - 入参sparseMode为2/3/4模式时，attenMask矩阵的shape最后两维应等于2048，即支持传入(2048,2048)或(1,2048,2048)或(1,1,2048,2048)
    - 非伪量化或Q_S大于1时，preTokens与nextTokens应满足nextTokens * (-1) <= preTokens，以确保具有有效数据
  - 伪量化
    - Q_S等于1时，attenMask输入维度仅支持3/4，且attenMask输入的shape应满足，第一维等于batchSize或1，倒数第二维应大于等于Q_S，最后一维应大于等于blockTable的第二维 * blockSize + actualSharedPrefixLen
    - Q_S大于1时，若sparseMode为4模式，且attentionOut为int8类型时，则preTokens与nextTokens均不能为负数
- 特性交叉约束
  - 非量化
    - Decode MLA场景下，sparseMode仅支持0/3/4模式
    - GQA场景下，当query，key及value的head dim不相等时，sparseMode仅支持0/2/3模式
  - 全量化
    - Decode MLA场景下，如果query/key/value的类型为INT8，当layout为TND/TND_NTD时，sparseMode仅支持0/3模式，且0模式下不支持传入attenMask矩阵
    - Decode MLA场景下，如果query/key/value的类型为INT8，当layout不为TND/TND_NTD时，Q_S大于1时，sparseMode仅支持3模式，且传入attenMask矩阵；Q_S等于1时，sparseMode仅支持0模式，且不支持传入attenMask矩阵
    - Decode MLA场景下，如果query/key/value的类型为FLOAT8_E4M3FN/HIFLOAT8时，sparseMode仅支持0/3模式，且0模式下不支持传入attenMask矩阵
    - MxFP8场景下，sparseMode仅支持0/3模式，且0模式下不支持传入attenMask矩阵
    - GQA全量化场景下，sparseMode仅支持0/3模式，且0模式下不支持传入attenMask矩阵

#### ActualSeqLen参数组

- 单参数约束
  - 公共
    - 入参actualSeqLengths(query的actualSeqLengths)应满足以下条件：
      - 长度应满足以下条件：
        - 当query的layout为TND/NTD时，长度应等于batch数
        - 当query的layout为非TND/NTD时，长度应等于1或者大于等于query的batch值
      - 入参中的数值应满足以下条件：
        - 当query的layout为TND/NTD时，其值应递增(大于等于前一个值)排列
        - 当query的layout为TND/NTD是，最后一个元素应等于T
        - 当query的layout为非TND/NTD时，其值应不大于Q_S
        - 其值应为非负数
    - 入参actualSeqLengthsKv(key/value的actualSeqLengths)应满足以下条件：
      - 长度应满足以下条件：
        - 当key/value的layout为TND/NTD时，长度应等于batch数
        - 当key/value的layout为非TND/NTD时，长度应等于key/value的batch值
      - 入参中的数值应满足以下条件：
        - 当key/value的layout为TND/NTD时，最后一个元素应等于T
        - 当key/value的layout为TND/NTD时，其值应递增(大于等于前一个值)排列
        - 当key/value的layout为非TND/NTD时，其值应不大于KV_S
        - 其值应为非负数
- 存在性约束
  - 公共
    - 入参actualSeqLengths(query的actualSeqLengths)应满足以下条件:
      - 当query的layout为TND/NTD时，必须传入actualSeqLengths
    - 入参actualSeqLengthsKv(key/value的actualSeqLengths)应满足以下条件：
      - 当key/value的layout为TND/NTD时，必须传入actualSeqLengthsKv
      - PagedAttention场景下，必须传入actualSeqLengthsKv
- 一致性约束
  - 无
- 特性交叉约束
  - 全量化
    - Decode MLA场景下，若传入actualSeqLengths，query的layout必须为TND/NTD

#### <span id="DequantChecker">伪量化/全量化参数组（DequantChecker）</span>

- 单参数约束
  - 伪量化场景
    - 入参keyAntiquantMode和valueAntiquantMode应满足以下条件：
      - 入参中的数值应满足以下条件：
        - 其值应为0(per-channel/per-tensor)、1(per-token)、2(per-tensor叠加per-head)、3(per-token叠加per-head)、
          4(per-token模式使用page attention管理scale/offset)、
          5(per-token叠加per-head模式并使用page attenion管理scale/offser)、6(per-token-group)
        - 除key支持per-channel叠加value支持per-token，keyAntiquantMode和valueAntiquantMode应相等
    - 入参keyAntiquantScale和valueAntiquantScale应满足以下条件：
      - 入参的数据类型应满足以下条件：
        - per-channel(per-tensor)，数据类型应与query相同
        - per-token，数据类型仅支持FLOAT32
        - per-tensor叠加per-head，数据类型应与query相同
        - per-token叠加per-head，数据类型仅支持FLOAT32
        - per-token模式使用page attention管理scale/offset，数据类型仅支持FLOAT32
        - key支持per-channel叠加value支持per-token，数据类型仅支持FLOAT32
        - per-token-group，数据类型仅支持FLOAT8_E8M0
      - 入参的shape应满足以下条件：
        - per-channel，shape应为(1, N, 1, D)、(1, 1, N, D)、(1, N, D)、(1, H)、(N, 1, D)、(N, D)、(H)
        - per-tensor，shape应为(1)
        - per-token，shape应为(1, B, >=KV_S)、(B, >=KV_S)
        - per-tensor叠加per-head，shape应为(N)
        - per-token叠加per-head，shape应为(B, N, >=KV_S)
        - per-token模式使用page attention管理scale/offset，shape应为(blockNum, blockSize)
        - ter-token叠加per-head模式并使用page attention管理scale/offset, shape应为(blockNum, N, blockSize)
        - key支持per-channel叠加value支持per-token，keyAntiquantScale的shape应为(1, N, 1, D)、(1, N, D)、(1, H)、(N, 1, D)、(N, D)、(H)
        - key支持per-channel叠加value支持per-token，valueAntiquantScale的shape应为(1, B, >=KV_S)、(B, >=KV_S)
        - per-token-group，shape应为(1, B, N, >=KV_S, D/32)
    - 入参key和value应满足以下条件：
      - 入参的数据类型应满足以下条件：
        - per-tensor模式，其支持的数据类型为INT8
        - per-channel模式，其支持的数据类型为INT8、INT4(INT32)、HIFLOAT8、FLOAT8_E4M3FN
        - per-token模式，其支持的数据类型为INT8、INT4(INT32)、FLOAT8_E4M3FN
        - per-tensor叠加per-head模式，其支持的数据类型为INT8
        - per-token叠加per-head模式，其支持的数据类型为INT8、INT4(INT32)
        - per-token模式使用page attenion管理scale/offset，其支持的数据类型为INT8、FLOAT8_E4M3FN
        - key支持per-channel叠加value支持per-token，其支持的数据类型为INT8、INT4(INT32)
        - per-token-group，其支持的数据类型为FLOAT4_E2M1
  - 全量化场景
    - Decode MLA全量化
      - 入参dequantScaleQuery、keyAntiquantScale、valueAntiquantScale的dtype为FLOAT32类型
      - 入参keyAntiquantScale、valueAntiquantScale的shape为（1）
      - 入参keyAntiquantMode、valueAntiquantMode为0（per-tensor模式），queryQuantMode为3（per-token叠加per-head模式）
    - MxFP8全量化
      - 入参dequantScaleQuery、keyAntiquantScale、valueAntiquantScale的dtype为FLOAT8_E8M0类型
      - 当Q_S *G > 80场景，入参dequantScaleQuery的shape推荐为（Q_T, Q_N, D/64, 2），当Q_S* G <= 80场景，入参dequantScaleQuery的shape推荐为（KV_N, Q_T, G, D/64, 2）；非PagedAttention场景，keyAntiquantScale的shape为（KV_T, KV_N, D/64, 2），valueAntiquantScale的shape为（KV_T/64, KV_N, D, 2）；PagedAttention场景下，kv cache排布为BnNBsD时，keyAntiquantScale的shape为（Bn, KV_N, Bs, D/64, 2），valueAntiquantScale的shape为（Bn, KV_N, Bs/64, D, 2）；kv cache排布为NZ时，keyAntiquantScale的shape为（Bn, KV_N, Bs/16, D/64, 16, 2），valueAntiquantScale的shape为（Bn, KV_N, D/16, Bs/64, 16, 2）
      - 入参queryQuantMode、keyAntiquantMode为6（per-token-group模式），valueAntiquantMode为8（per-channel-group模式）
      - 入参quantScale1的shape为（1），dtype支持FLOAT32，复用quantScale1作为pScale
    - GQA全量化
      - 入参dequantScaleQuery、keyAntiquantScale、valueAntiquantScale的dtype为FLOAT32类型
      - 入参dequantScaleQuery的shape为（Q_N, Q_T），排布为NT；入参keyAntiquantScale的shape为（Bn, KV_N, Bs），排布为BnNBs；入参valueAntiquantScale的shape为（KV_N），排布为N
      - 入参queryQuantMode为3（per-token叠加per-head模式），keyAntiquantMode为3（per-token叠加per-head模式），valueAntiquantMode为2（per-tensor叠加per-head模式）
      - 入参quantScale1的shape为（1），dtype为FLOAT32，复用quantScale1作为pScale
      - kv cache排布仅支持BnNBsD（blocknum, KV_N, blocksize, D）
- 存在性约束
  - 伪量化场景
    - 入参keyAntiquantScale和valueAntiquantScale应满足以下条件：
      - 必须传入keyAntiquantScale
      - 必须传入valueAntiquantScale
    - 入参keyAntiquantOffset和valueAntiquantOffset应满足以下条件：
      - 传入keyAntiquantOffset时，必须传入valueAntiquantOffset
      - 传入valueAntiquantOffset时，必须传入keyAntiquantOffset
      - 当key/value的数据类型为FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1时，不支持offset，不能传入keyAntiquantOffset
      - 当key/value的数据类型为FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1时，不支持offset，不能传入valueAntiquantOffset
  - 全量化场景
    - Decode MLA全量化
      - dequantScaleQuery、keyAntiquantScale、valueAntiquantScale需同时存在
      - 不支持传入keyAntiquantOffset、valueAntiquantOffset
      - 不支持传入deqScale1、quantScale1、deqScale2
      - 不支持传入antiquantScale、antiquantOffset
      - 不支持传入keyRopeAntiquantScale
    - MxFP8全量化
      - dequantScaleQuery、keyAntiquantScale、valueAntiquantScale需同时存在
      - 不支持传入keyAntiquantOffset、valueAntiquantOffset
      - 不支持传入deqScale1、deqScale2
      - 不支持传入antiquantScale、antiquantOffset
      - 不支持传入keyRopeAntiquantScale
    - GQA全量化
      - dequantScaleQuery、keyAntiquantScale、valueAntiquantScale需同时存在
      - quantScale1需存在（复用quantScale1作为pScale）
      - 不支持传入keyAntiquantOffset、valueAntiquantOffset
      - 不支持传入deqScale1、deqScale2
      - 不支持传入antiquantScale、antiquantOffset
      - 不支持传入keyRopeAntiquantScale
- 一致性约束
  - 无
- 特性交叉约束
  - 伪量化场景
    - inputLayout仅支持BSH, BNSD, BSND, BNSD_BSND, TND
    - key/value的数据类型为INT8时，inputLayout不支持TND
    - Q_S = 1时：
      - inputLayout不支持BNSD_BSND
      - 当key/value的数据类型为INT8, keyAntiquantMode = 0且valueAntiquantMode = 1时，query和output的数据类型仅支持FLOAT16
    - Q_S > 1时：
      - key/value的数据类型为INT8时，keyAntiquantMode不支持2，3，4，5
      - key/value的数据类型为INT8，且keyAntiquantMode为0或1时，query和output的数据类型仅支持BF16
      - key/value的数据类型为INT8，且keyAntiquantMode为0或1时，Q_S长度不能大于16
      - key/value的数据类型为INT8，且keyAntiquantMode为0或1时，不支持tensor list
      - key/value的数据类型为INT8，且keyAntiquantMode为0或1时，不支持左padding
      - key/value的数据类型为INT8，且keyAntiquantMode为0或1时，不支持page attention
      - key/value的数据类型不支持INT4、INT32
    - page attention场景下，入参keyAntiquantScale和valueAntiquantScale应满足以下条件：
      - tensor shape应满足以下条件：
        - per-token模式，shape的最后一维应大于等于maxBlockNumPerBatch * blockSize
        - per-token叠加per-head模式，shape的最后一维应大于等于maxBlockNumPerBatch * blockSize
        - per-token-group模式，shape的倒数第二维应大于等于maxBlockNumPerBatch * blockSize
    - 不支持合并rope
  - 全量化场景
    - Decode MLA全量化
      - query、key、value的dtype为FLOAT8_E4M3FN/INT8/HIFLOAT8
      - attenOut的dtype为BFLOAT16
      - queryRope、keyRope的dtype为BFLOAT16
      - 当query/key/value是FLOAT8_E4M3FN/HIFLOAT8时， inputLayout仅支持BSH、BSND、BNSD、TND、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD；当query/key/value类型为INT8时, inputLayout仅支持BSH、BSND、TND、BSH_NBSD、BSND_NBSD、TND_NTD
      - Q_N支持1、2、4、8、16、32、64、128
      - KV_N必须为1；G支持 [1, 128]；Q_S支持[1,16]
      - 当query的inputLayout为BSH时，dequantScaleQuery的shape应该为（B, Q_S, Q_N）；当query的inputLayout为BSND、BNSD、TND时，dequantScaleQuery的shape相比query仅少一个维度D，且每一维需要和query的对应维度保持一致
      - 不支持公共前缀场景、不支持pse场景、不支持alibi场景、不支持左padding场景
      - 当query/key/value类型为INT8时，仅支持PagedAttention场景，且kv cache排布为NZ格式
    - MxFP8全量化
      - query、key、value的dtype为FLOAT8_E4M3FN
      - attentionOut的dtype支持FLOAT16、BFLOAT16
      - inputLayout仅支持TND
      - 支持PagedAttention场景和非PagedAttention场景，PagedAttention场景仅支持BnNBsD（blocknum, KV_N, blocksize, D）和NZ (blocknum，KV_N，D/D0，blocksize，D0)
      - 支持传入独立Rope，若传入Rope，Rope和attentionOut的dtype仅支持BFLOAT16
      - 不支持alibi场景、不支持Rope存在、不支持后量化、不支持D不等长场景、不支持公共前缀场景
    - GQA全量化
      - query、key、value的dtype仅支持FLOAT8_E4M3FN
      - attentionOut的dtype支持FLOAT16、BFLOAT16
      - inputLayout仅支持NTD_TND
      - 仅支持PagedAttention场景，kv cache排布仅支持BnNBsD（blocknum, KV_N, blocksize, D）
      - D仅支持128
      - Q_S支持[1, 64K]，KV_S支持[0, 256K]
      - 不支持公共前缀场景、不支持pse场景、不支持alibi场景、不支持左padding场景、不支持tensorlist场景、不支持rope场景、不支持后量化场景

#### <span id="PostQuantChecker">后量化参数组（PostQuantChecker）</span>

- 单参数约束
  - 公共
    - 入参quantScale2需要满足以下条件：
      - tensor dtype为BF16/FP32类型
    - 入参quantOffset2需要满足以下条件：
      - tensor dtype为BF16/FP32类型
    - PostQuant场景下，输出attenOut的数据类型仅支持INT8/FP8_E4M3FN/HIFLOAT8
- 存在性约束
  - 公共
    - PostQuant场景下，必须传入quantScale2
- 一致性约束
  - 公共
    - PostQuant场景下，当quantScale2维度大于1且量化方式为per-channel时，若layout为BSH/BSND/BNSD/BNSD_BSND，则quantScale2仅支持shape为queryN*valueD，否则仅支持 [numHeads, vHeadDim]
    - PostQuant场景下，当quantScale2维度等于1且量化方式为per-tensor时，quantScale2仅支持shape为(1)
    - PostQuant场景下，当quantOffset2存在时，quantOffset2应与quantScale2保持相同shape及数据类型
- 特性交叉约束
  - 公共
    - PostQuant场景下，当query输入类型不为BF16时，quantScale2仅支持FP32类型
  - 非量化
    - PostQuant场景下，当存在prefix时，仅支持输出attenOut的数据类型为INT8
  - 伪量化
    - PostQuant场景下，输出attenOut的数据类型仅支持与输入Key、Value数据类型相同
    - 当keyAntiquantMode和valueAntiquantMode为per-token或per-token使用page attention管理scale/offset，query数据类型为FP16/BF16且key/value数据类型为FLOAT8_E4M3FN时，不支持叠加后量化

#### Paged Attention参数组

- 单参数约束
  - 公共
    - 入参blockTable需要满足以下条件：
      - tensor dtype为int32类型
      - tensor shape为2维，每一维dim value取值均不能为0，第一维长度需等于Batch size，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）
    - 入参blockSize需要满足以下条件：
      - blockSize需要大于0
      - blockSize是用户自定义的参数，该参数的取值会影响PagedAttention的性能，通常情况下，PagedAttention可以提高吞吐量，但会带来性能上的下降，调大blockSize会有一定性能收益
  - 非量化
    - 入参blockSize需要满足以下条件：
      - Decode MLA/Prefill MLA场景：blockSize 16对齐，最大支持1024
      - GQA场景：QueryHeadDim/KeyHeadDim/ValueHeadDim均为64或128时，blockSize 16对齐，最大支持1024；其他情况下，若Q_S> 1，blockSize 128对齐，最大支持1024，若Q_S= 1，blockSize 16对齐，最大支持512
  - 伪量化
    - 入参blockSize需要满足以下条件：需要根据key、value dtype size 32B对齐，最大支持512。即当key、value dtype为INT8/HIFLOAT8/FLOAT8_E4M3FN时，blockSize需要32对齐，即当key、value dtype为INT4(INT32)、FLOAT4_E2M1时，blockSize需要64对齐
  - 全量化
    - Decode MLA全量化场景下，仅支持blockSize取值128
    - MxFP8全量化场景下，仅支持blockSize取值512或1024
    - GQA全量化场景下，仅支持blockSize取值128
- 存在性约束
  - 公共
    - PagedAttention开启情况下，必须传入actualSeqLengthsKv
    - PagedAttention不支持tensorlist场景，不支持左padding场景，不支持公共前缀场景，不支持D不等长场景
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - PagedAttention的开启场景下，若同时开启attenMask，传入attenMask的最后一维需要大于等于maxBlockNumPerSeq * blockSize
    - PagedAttention场景下，kv cache排布为BnNBsD时性能通常优于kv cache排布为BnBsH时的性能，建议优先选择BnNBsD格式。
    - PagedAttention场景下，当输入kv cache排布格式为BnBsH（blocknum, blocksize, H），且KV_N * D超过65535时，受硬件指令约束，会被拦截报错。可通过开启GQA（减小KV_N）或调整kv cache排布格式为BnNBsD（blocknum, KV_N, blocksize, D）解决。
  - 伪量化
    - PagedAttention场景下，kv cache的layout为4维时，inputLayout必须为BNSD、BNSD_BSND或TND
  - 全量化
    - Decode MLA场景下：
      - 当query的数据类型为FP8_E4M3FN/HIFLOAT8，且inputLayout为BSH、BSND、BSH_NBSD、BSND_NBSD时，kv cache排布只支持BnBsH（blocknum, blocksize, H）和NZ (blocknum，KV_N，D/D0，blocksize，D0)两种格式
      - 当query的数据类型为FP8_E4M3FN/HIFLOAT8，且inputLayout为BNSD、TND、BNSD_NBSD、TND_NTD时，kv cache排布支持BnBsH（blocknum, blocksize, H）、BnNBsD（blocknum, KV_N, blocksize, D）和NZ（blocknum，KV_N，D/D0，blocksize，D0）三种格式
      - 当query的数据类型为INT8时，kv cache排布仅支持NZ，且kv cache排布为(blocknum，KV_N，D/D0，blocksize，D0)
      - 当kv cache排布为NZ时，最后一维D0是32, keyRope最后一维D0是16
    - GQA全量化场景下，除MxFP8和GQA全量化（queryQuantMode = 3，keyAntiquantMode = 3，valueAntiquantMode = 2）外均不支持PagedAttention
    - MxFP8全量化支持非PagedAttention场景，kv cache排布支持BnNBsD（blocknum, KV_N, blocksize, D）和NZ (blocknum，KV_N，D/D0，blocksize，D0)

#### 左padding参数组

- 单参数约束
  - 非量化
    - Query左padding场景下，queryPaddingSize的shape应为(1)
    - Key、value左padding场景下，kvPaddingSize的shape应为(1)
- 存在性约束
  - 公共
    - Query左padding场景下，必须传入queryPaddingSize
    - Key、value左padding场景下，必须传入kvPaddingSize
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - 左padding场景下，不支持PagedAttention
    - 左padding场景下，不支持BSH_BNSD、BSND_BNSD、TND、NTD、NTD_TND、TND_NTD场景
    - 左padding场景下，必须传入actualSeqLengths/actualSeqLengthsKv

#### 公共前缀参数组

- 单参数约束
  - 入参keySharedPrefix和valueSharedPrefix应满足以下条件：
    - tensor shape应满足以下条件：
      - shape应为(1)
      - layout为BNSD和BSND时，N轴和D轴应与key/value的N轴和D轴相等
      - layout为BSH时，H轴应与key/value的H轴相等
      - keySharedPrefix和valueSharedPrefix的S轴应相等
    - tensor数据类型应满足以下条件：
      - 数据类型应与key/value的数据类型相同
  - 入参actualSharedPrefixLen应满足以下条件：
    - shape应满足以下条件：
      shape应为1
    - 入参中的数值应满足以下条件：
      - 其值不能大于keySharedPrefix和valueSharedPrefix的shape的S轴
- 存在性约束
  - 公共
    - 入参keySharedPrefix和valueSharedPrefix应满足以下条件：
      - 传入keySharedPrefix时，必须传入valueSharedPrefix
      - 传入valueSharedPrefix时，必须传入keySharedPrefix
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - 不支持PagedAttention场景
    - 不支持tensorlist场景
    - 不支持左padding场景
    - 不支持alibi场景
    - 不支持TND场景
    - 不支持Prefill MLA (包括D不等长和ROPE独立输入)场景
    - 不支持Decode MLA场景
  - 全量化
    - 全量化（包括MLA全量化和GQA全量化）场景，不支持prefix
    - 后量化场景，仅支持数据类型INT8
  - 伪量化
    - 伪量化key/value合成场景所有量化模式prefix均支持
    - 伪量化key/value分离场景，prefix仅支持以下量化模式：
      - Q_S > 1时，伪量化方式为per-channel(per-tensor)、per-token时，key/value数据类型仅支持INT8
      - Q_S = 1时，伪量化方式为per-tensor、per-tensor叠加per-head、per-token叠加使用page attention模式管理
        scale/offset、per-token叠加per-head并使用page attention模式管理scale/offset，key/value数据类型仅支持INT8
      - Q_S = 1时，伪量化方式为per-channel、per-token、per-token叠加per-head、key支持per-channel
        叠加value支持per-token，key/value数据类型支持INT8、INT4(INT32)

#### Rope参数组

- 单参数约束
  - 公共
    - 入参queryRope和keyRope需要满足以下条件
      - tensor dtype为FLOAT16/BFLOAT16
      - tensor shape中D维为64
- 存在性约束
  - 公共
    - 入参queryRope和keyRope必须同时存在
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - query shape的D仅支持128、512
    - 非tensorlist场景，  queryRope shape维度需要和query保持一致，除了queryRope shape的D为64外,其余维度需要和query一致；keyRope shape维度需要和key保持一致，除了keyRope shape的D为64外,其余维度需要和key一致
    - 非量化场景，入参queryRope和keyRope的dtype需要和query、key保持一致
    - 不支持公共前缀场景、不支持pse场景、不支持alibi场景
    - 不支持伪量化场景
  - Decode MLA
    - Layout仅支持BNSD、BSND、BSH、TND、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND_NTD
    - Q_N支持1/2/4/8/16/32/64/128；KV_N仅支持1
    - 非量化场景，Q_S无限制；全量化场景，Q_S支持 [1,16]
    - 不支持左padding场景、不支持tensorlist场景
  - Prefill MLA
    - Layout仅支持BNSD、BSND、BSH、TND、NTD、BSH_BNSD、BSND_BNSD、NTD_TND、BNSD_BSND
    - 在tensorlist场景下，传入keyRope shape中D为64，B需要和key的tensorlist长度保持一致，N、S需要与key的tensorlist中每个tensor的N、S相等
    - 不支持全量化场景
  - GQA全量化
    - 不支持rope场景

#### LearnableSink参数组

- 单参数约束
  - 公共
    - 入参learnableSink需要满足以下条件
      - tensor dtype为FLOAT16/BFLOAT16
      - tensor shape为(Q_N)
- 存在性约束
  - 无
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - LearnableSink开启场景下，tensor dtype需要和query dtype保持一致
    - LearnableSink开启场景下，QueryHeadDim/KeyHeadDim仅支持64、128、192，ValueHeadDim仅支持64、128
    - LearnableSink开启场景下，innerPrecise必须为高精度模式(0)
    - LearnableSink不支持左padding场景、不支持公共前缀场景、不支持pse场景、不支持alibi场景、不支持后量化场景
    - LearnableSink不支持全量化场景、不支持伪量化场景、不支持Decode MLA场景

#### SoftmaxLSE参数组

- 单参数约束
  - 公共
    - 输出lseOut仅支持数据类型FP32
- 存在性约束
  - 公共
    - softmaxLSE场景下，输出lseOut不应为空
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - softmaxLSE且非空tensor场景下，如输出layout为TND或NTD，则lseOut输入维度应为3，且shape匹配 [Q_T，Q_N，1]
    - softmaxLSE且非空tensor场景下，如输出layout不为TND或NTD，则lseOut输入维度应为4，且shape匹配 [B，Q_N，Q_S，1]

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
  #include <iostream>
  #include <vector>
  #include <math.h>
  #include <cstring>
  #include "acl/acl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

  using namespace std;

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
      auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
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
      // 1.（固定写法）device/stream初始化，acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int32_t batchSize = 1;
      int32_t numHeads = 2;
      int32_t sequenceLengthQ = 1;
      int32_t headDims = 16;
      int32_t keyNumHeads = 2;
      int32_t sequenceLengthKV = 16;
      std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
      std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
      std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
      void *queryDeviceAddr = nullptr;
      void *keyDeviceAddr = nullptr;
      void *valueDeviceAddr = nullptr;
      void *attenDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *queryTensor = nullptr;
      aclTensor *keyTensor = nullptr;
      aclTensor *valueTensor = nullptr;
      aclTensor *attenTensor = nullptr;
      aclTensor *outTensor = nullptr;
      std::vector<float> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
      std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
      std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
      std::vector<float> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);

      // 创建query aclTensor
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建key aclTensor
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int kvTensorNum = 1;
      aclTensor *tensorsOfKey[kvTensorNum];
      tensorsOfKey[0] = keyTensor;
      auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
      // 创建value aclTensor
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      aclTensor *tensorsOfValue[kvTensorNum];
      tensorsOfValue[0] = valueTensor;
      auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
      // 创建atten aclTensor
      ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
      auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

      int64_t numKeyValueHeads = numHeads;
      double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
      int64_t preTokens = 65535;
      int64_t nextTokens = 65535;
      string sLayerOut = "BNSD";
      char layerOut[sLayerOut.length()+1];
      strcpy(layerOut, sLayerOut.c_str());
      int64_t sparseMode = 0;
      int64_t innerPrecise = 1;
      int blockSize = 0;
      int antiquantMode = 0;
      bool softmaxLseFlag = false;
      int keyAntiquantMode = 0;
      int valueAntiquantMode = 0;
      int queryAntiquantMode = 0;
      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      int64_t pseType = 0;
      aclOpExecutor* executor;
      // 调用第一段接口
      ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryAntiquantMode, pseType, outTensor, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用第二段接口
      ret = aclnnFusedInferAttentionScoreV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV5 failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<op::fp16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
      }

      // 6. 释放资源
      aclDestroyTensor(queryTensor);
      aclDestroyTensor(keyTensor);
      aclDestroyTensor(valueTensor);
      aclDestroyTensor(attenTensor);
      aclDestroyTensor(outTensor);
      aclDestroyIntArray(actualSeqLengths);
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(attenDeviceAddr);
      aclrtFree(outDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
```
