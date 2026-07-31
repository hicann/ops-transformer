# aclnnSparseFlashMlaGrad

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

- **接口功能**：计算`SparseFlashMla`训练场景下注意力的反向输出，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

- **计算公式**：

    阶段一：根据不同cmp_ratio场景，对输入ori_kv与cmp_kv进行选择

    * 当cmp_ratio = 1 (SWA)：

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

    * 当cmp_ratio = 4 (SCFA)：

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else (CFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    阶段二：计算P、dP、dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    阶段三：计算dQ, dKV, dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSparseFlashMlaGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseFlashMlaGrad”接口执行计算。

```c++
aclnnStatus aclnnSparseFlashMlaGradGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *dOut,
    const aclTensor   *out,
    const aclTensor   *lse,
    const aclTensor   *oriKvOptional,
    const aclTensor   *cmpKvOptional,
    const aclTensor   *oriSparseIndicesOptional,
    const aclTensor   *cmpSparseIndicesOptional,
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedOriKvOptional,
    const aclTensor   *sequsedCmpKvOptional,
    const aclTensor   *cmpResidualKvOptional,
    const aclTensor   *oriTopkLengthOptional,
    const aclTensor   *cmpTopkLengthOptional,
    const aclTensor   *sinksOptional,
    const aclTensor   *metadataOptional,
    double             scaleValue,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    char              *layoutQOptional,
    char              *layoutKvOptional,
    const aclTensor   *dQueryOut,
    const aclTensor   *dOriKvOutOptional,
    const aclTensor   *dCmpKvOutOptional,
    const aclTensor   *dSinksOutOptional,
    const aclTensor   *oriSoftmaxL1NormOptional,
    const aclTensor   *cmpSoftmaxL1NormOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnSparseFlashMlaGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream);
```

## aclnnSparseFlashMlaGradGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 290px">
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
        </tr></thead>
        <tbody>
        <tr>
            <td>query（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的输入Q。</td>
            <td>
            query、oriKvOptional、cmpKvOptional、dOut、outOptional、lse、oriSparseIndicesOptional、cmpSparseIndicesOptional的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            B：支持泛化；S1：支持泛化；N1：支持1~128；D：512；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dOut（aclTensor*）</td>
            <td>输入</td>
            <td>注意力输出矩阵的梯度。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            B：与query的B保持一致；S1：与query的S1保持一致；N1：与query的N1保持一致；D：512；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>out（aclTensor*）</td>
            <td>输入</td>
            <td>注意力输出矩阵。</td>
            <td>
            -
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            Shape与dOut保持一致
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>lse（aclTensor*）</td>
            <td>输入</td>
            <td>注意力正向计算的输出lse，计算公式详见sparse_flash_mla文档。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,N2,S1,G)、(N2,T1,G)<br>
            B：与query的B保持一致；N2：1；S1：与query的S1保持一致；G：N1/N2；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>attention结构的原始输入K(V)。</td>
            <td>
            当前暂不支持空tensor。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)<br>
            B：与query的B保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>经过Compressor压缩后的K(V)。</td>
            <td>
            支持空tensor，此时按计算公式中的SWA场景计算。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S3,N2,D)、(T3,N2,D)<br>
            B：与query的B保持一致；S3 = S2 // cmpRatio；N2：1；D：512；T3：B × S3
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriSparseIndicesOptional（aclTensor*）</td>
            <td>输入</td>
            <td>稀疏场景下选择的oriKvOptional中权重较高的注意力索引。</td>
            <td>
            目前暂不支持对ori_kv进行稀疏计算，故设置此参数无效。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K1)、(T1,N2,K1)<br>
            B：与query的B保持一致；S1：与query的S1保持一致；N2：1；K1：支持泛化；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpSparseIndicesOptional（aclTensor*）</td>
            <td>输入</td>
            <td>稀疏场景下选择的cmpKvOptional中权重较高的注意力索引。</td>
            <td>
            支持空tensor：
            <ul>
                <li>若此时cmpKvOptional非空则按CFA场景计算。</li>
                <li>若此时cmpKvOptional为空则按SWA场景计算。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K2)、(T1,N2,K2)<br>
            B：与query的B保持一致；S1：与query的S1保持一致；N2：1；K2：支持泛化；T1：B × S1
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensQOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，Query的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T1保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensOriKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，oriKvOptional的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T2保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>cuSeqlensCmpKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>每个Batch中，cmpKvOptional的有效token数。</td>
            <td>
            <ul>
                <li>可选项：当layout为TND，该变量存在。</li>
                <li>长度与B+1保持一致。</li>
                <li>累加和与T3保持一致。</li>
            </ul>
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B+1,)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedQOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中query实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedOriKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中oriKvOptional实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sequsedCmpKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示不同batch中cmpKvOptional实际参与运算的token数。</td>
            <td>
            长度为B。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpResidualKvOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示每个batch S2 // cmpRatio后的余数。</td>
            <td>
            当cmpKvOptional不为空且cmpMaskMode=3时必须传入。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriTopkLengthOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示每行query对应的oriKvOptional实际可选的topk长度。</td>
            <td>
            oriMaskMode=0且oriSparseIndicesOptional不为空时需要传，且必须为准确值。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2)、(T1,N2)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpTopkLengthOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示每行query对应的cmpKvOptional实际可选的topk长度。</td>
            <td>
            cmpMaskMode=0且cmpSparseIndicesOptional不为空时需要传，且必须为准确值。
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(B,S1,N2)、(T1,N2)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>sinksOptional（aclTensor*）</td>
            <td>输入</td>
            <td>注意力下沉tensor。</td>
            <td>
            -
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1)<br>
            N1：与query的N1保持一致
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>metadataOptional（aclTensor*）</td>
            <td>输入</td>
            <td>表示tiling下沉的aicpu算子输出结果。</td>
            <td>
            -
            </td>
            <td>INT32</td>
            <td>ND</td>
            <td>(x)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>scaleValue（double）</td>
            <td>输入</td>
            <td>缩放系数。</td>
            <td>
            建议值：公式中d开根号的倒数。
            </td>
            <td>FLOAT32</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>cmpRatio（int64_t）</td>
            <td>输入</td>
            <td>表示对oriKvOptional的压缩率。</td>
            <td>
            取值范围：1~128。
            </td>
            <td>INT64</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>oriMaskMode（int64_t）</td>
            <td>输入</td>
            <td>表示query和oriKvOptional计算的mask模式。</td>
        <td>
              <ul>
                <li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
            <td>cmpMaskMode（int64_t）</td>
            <td>输入</td>
            <td>表示query和cmpKvOptional计算的mask模式。</td>
        <td>
              <ul>
                <li>表示sparse的模式。sparse不同模式的详细说明请参见<a href="#约束说明">约束说明</a>。</li>
              </ul>
        </td>
        <td>INT64</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
        </tr>
        <tr>
        <td>oriWinLeft（int64_t）</td>
            <td>输入</td>
            <td>表示query和oriKvOptional计算中query对过去token计算的数量。</td>
            <td>
            <ul>
                <li>仅支持取值：127</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
        <td>oriWinRight（int64_t）</td>
            <td>输入</td>
            <td>表示query和oriKvOptional计算中query对未来token计算的数量。</td>
            <td>
            <ul>
                <li>仅支持取值：0</li>
            </ul>
            </td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutQOptional（char*）</td>
            <td>输入</td>
            <td>表示输入query的数据排布格式。</td>
            <td>
            支持"BSND"、"TND"。
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layoutKvOptional（char*）</td>
            <td>输入</td>
            <td>表示输入ori_kv和cmp_kv的数据排布格式。</td>
            <td>
            支持"BSND"、"TND"。
            </td>
            <td>STRING</td>
            <td>N/A</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dQueryOut（aclTensor*）</td>
            <td>输出</td>
            <td>表示query的梯度。</td>
            <td>
            与输入query的Shape维度保持一致
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S1,N1,D)、(T1,N1,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dOriKvOutOptional（aclTensor*）</td>
            <td>输出</td>
            <td>表示oriKvOptional的梯度。</td>
            <td>
            与输入oriKvOptional的Shape维度保持一致
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S2,N2,D)、(T2,N2,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dCmpKvOptional（aclTensor*）</td>
            <td>输出</td>
            <td>表示cmpKvOptional的梯度。</td>
            <td>
            与输入cmpKvOptional的Shape维度保持一致。
            </td>
            <td>BFLOAT16、FLOAT16</td>
            <td>ND</td>
            <td>(B,S3,N2,D)、(T3,N2,D)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>dSinksOutOptional（aclTensor*）</td>
            <td>输出</td>
            <td>表示sinksOptional的梯度。</td>
            <td>
            与输入sinksOptional的Shape维度保持一致。
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(N1)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>oriSoftmaxL1NormOptional（aclTensor*）</td>
            <td>输出</td>
            <td>表示query与oriKvOptional计算得出的softmax L1Norm结果，公式为reduceG(softmax)/G。</td>
            <td>
            若存在oriSparseIndicesOptional，则该输出不为空；其他场景下该参数输出为空。
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K1)、(T1,N2,K1)<br>
            </td>
            <td>√</td>
        </tr>
        <tr>
            <td>cmpSoftmaxL1NormOptional（aclTensor*）</td>
            <td>输出</td>
            <td>表示query与cmpKvOptional计算得出的softmax L1Norm结果，公式为reduceG(softmax)/G。</td>
            <td>
            若存在cmpSparseIndicesOptional，则该输出不为空；其他场景下该参数输出为空。
            </td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S1,N2,K2)、(T1,N2,K2)<br>
            </td>
            <td>√</td>
        </tr>
        </tbody>
    </table>

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：暂不支持sequsedQOptional、sequsedOriKvOptional、sequsedCmpKvOptional、oriTopkLengthOptional、cmpTopkLengthOptional、metadataOptional字段。

    <!-- end id7 -->

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
                <td>必选参数或者输出是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入变量，如query、oriKvOptional、cmpKvOptional……的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_INNER_TILING_ERROR</td>
                <td>561002</td>
                <td>输入参数（如layout、maskMode）的取值超出支持范围，或输入Tensor的shape维度不符合约束要求。</td>
            </tr>
        </tbody>
    </table>

## aclnnSparseFlashMlaGrad

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashMlaGradGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream流。</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：aclnnSparseFlashMlaGrad默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。

    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：aclnnSparseFlashMlaGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
    <!-- end id9 -->

- 公共约束
    - 入参为空的场景处理：
        - query为空Tensor：直接返回。

- Mask
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
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
            <td>不做mask操作</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
            <td>支持</td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask，滑窗范围由oriWinLeft、oriWinRight控制，参数起点为右下角。</td>
            <td>ori支持</td>
        </tr>
        </tbody>
    </table>

- 规格约束
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
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
            <td>支持泛化</td>
            <td>-</td>
        </tr>
        <tr>
            <td>S1、S2</td>
            <td>支持泛化</td>
            <td>支持S1、S2支持不等长</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1~128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1</td>
            <td>-</td>
        </tr>
        <tr>
            <td>D</td>
            <td>512</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>BSND/TND</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

  - 参数cmpMaskMode的支持情况:

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：cmpMaskMode支持3。
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：cmpMaskMode支持0、3。

    <!-- end id11 -->

  - 参数oriMaskMode的支持情况:

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：oriMaskMode支持4。
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：oriMaskMode支持0、3、4。

    <!-- end id13 -->

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <numeric>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad.h"
#include "aclnnop/aclnn_sparse_flash_mla_grad_metadata.h"

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
  std::vector<short> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %e\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，AscendCL初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
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
  // 1. （固定写法）device/context/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {1, 16, 512};                // T1, N1, D
  std::vector<int64_t> oriKvShape = {2048, 1, 512};          // T2, N2, D
  std::vector<int64_t> cmpKvShape = {16, 1, 512};            // T3, N2, D
  std::vector<int64_t> dOutShape = {1, 16, 512};             // T1, N1, D
  std::vector<int64_t> outShape = {1, 16, 512};              // T1, N1, D
  std::vector<int64_t> lseShape = {1, 1, 16};                // N2, T1, G
  std::vector<int64_t> cuSeqQLenshape = {2};                 // B + 1
  std::vector<int64_t> cuSeqOriKvLenshape = {2};             // B + 1
  std::vector<int64_t> cuSeqCmpKvLenshape = {2};             // B + 1
  std::vector<int64_t> cmpResidualKvShape = {1};             // B
  std::vector<int64_t> sinksShape = {16};                    // N1
  std::vector<int64_t> metadataShape = {1024};               // 固定大小

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* dOutDeviceAddr = nullptr;
  void* lseDeviceAddr = nullptr;
  void* cuSeqQLenDeviceAddr = nullptr;
  void* cuSeqOriKvLenDeviceAddr = nullptr;
  void* cuSeqCmpKvLenDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dOriKvDeviceAddr = nullptr;
  void* dCmpKvDeviceAddr = nullptr;
  void* dSinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* oriSoftmaxL1NormDeviceAddr = nullptr;
  void* cmpSoftmaxL1NormDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* out = nullptr;
  aclTensor* dOut = nullptr;
  aclTensor* lse = nullptr;
  aclTensor* cuSeqQLen = nullptr;
  aclTensor* cuSeqOriKvLen = nullptr;
  aclTensor* cuSeqCmpKvLen = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* sinks = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dOriKv = nullptr;
  aclTensor* dCmpKv = nullptr;
  aclTensor* dSinks = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* oriSoftmaxL1Norm = nullptr;
  aclTensor* cmpSoftmaxL1Norm = nullptr;

  std::vector<short> qHostData(1 * 16 * 512, 1.0);
  std::vector<short> oriKvHostData(2048 * 1 * 512, 1.0);
  std::vector<short> cmpKvHostData(16 * 1 * 512, 1.0);
  std::vector<short> outHostData(1 * 16 * 512, 1.0);
  std::vector<short> dOutHostData(1 * 16 * 512, 1.0);
  std::vector<float> lseHostData(16, 3.0);
  std::vector<int32_t> cuSeqQLenHostData = {0, 1};
  std::vector<int32_t> cuSeqOriKvLenHostData = {0, 2048};
  std::vector<int32_t> cuSeqCmpKvLenHostData = {0, 16};
  std::vector<int32_t> cmpResidualKvHostData = {0};
  std::vector<float> sinksHostData(16, 128.0);
  std::vector<short> dqHostData(1 * 16 * 512, 0);
  std::vector<short> dOriKvHostData(2048 * 1 * 512, 0);
  std::vector<short> dCmpKvHostData(16 * 1 * 512, 0);
  std::vector<float> dSinksHostData(16, 0);
  std::vector<int32_t> metadataHostData(1024, 0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT16, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpKvHostData, cmpKvShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &cmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOutHostData, dOutShape, &dOutDeviceAddr, aclDataType::ACL_FLOAT16, &dOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lseHostData, lseShape, &lseDeviceAddr, aclDataType::ACL_FLOAT, &lse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqQLenHostData, cuSeqQLenshape, &cuSeqQLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqQLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqOriKvLenHostData, cuSeqOriKvLenshape, &cuSeqOriKvLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqOriKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqCmpKvLenHostData, cuSeqCmpKvLenshape, &cuSeqCmpKvLenDeviceAddr, aclDataType::ACL_INT32, &cuSeqCmpKvLen);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cmpResidualKvHostData, cmpResidualKvShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dOriKvHostData, oriKvShape, &dOriKvDeviceAddr, aclDataType::ACL_FLOAT16, &dOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dCmpKvHostData, cmpKvShape, &dCmpKvDeviceAddr, aclDataType::ACL_FLOAT16, &dCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dSinksHostData, sinksShape, &dSinksDeviceAddr, aclDataType::ACL_FLOAT, &dSinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double scaleValue = 0.088388;
  int64_t cmpRatio = 128;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 3;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;
  char layoutQ[5] = {'T', 'N', 'D', 0};
  char layoutKv[5] = {'T', 'N', 'D', 0};

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 3.1 调用SparseFlashMlaGradMetadata获取metadata
  uint64_t metaWorkspaceSize = 0;
  aclOpExecutor* metaExecutor;
  ret = aclnnSparseFlashMlaGradMetadataGetWorkspaceSize(
      cuSeqQLen, cuSeqOriKvLen, cuSeqCmpKvLen,
      nullptr, nullptr, nullptr,
      cmpResidualKv,
      nullptr, nullptr,
      16, 1, 512,
      0, 0, 0, 0,
      0, 0,
      cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, true,
      metadata, &metaWorkspaceSize, &metaExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metaWorkspaceAddr = nullptr;
  if (metaWorkspaceSize > 0) {
    ret = aclrtMalloc(&metaWorkspaceAddr, metaWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseFlashMlaGradMetadata(metaWorkspaceAddr, metaWorkspaceSize, metaExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradMetadata failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 3.2 调用SparseFlashMlaGrad计算梯度
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSparseFlashMlaGradGetWorkspaceSize(q, dOut, out, lse, oriKv, cmpKv, nullptr, nullptr, cuSeqQLen, cuSeqOriKvLen, cuSeqCmpKvLen,
            nullptr, nullptr, nullptr, cmpResidualKv, nullptr, nullptr, sinks, metadata, scaleValue, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
            layoutQ, layoutKv, dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSparseFlashMlaGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SparseFlashMlaGrad failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(qShape, &dqDeviceAddr);
  PrintOutResult(cmpKvShape, &dCmpKvDeviceAddr);
  PrintOutResult(sinksShape, &dSinksDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
  aclDestroyTensor(out);
  aclDestroyTensor(dOut);
  aclDestroyTensor(lse);
  aclDestroyTensor(cuSeqQLen);
  aclDestroyTensor(cuSeqOriKvLen);
  aclDestroyTensor(cuSeqCmpKvLen);
  aclDestroyTensor(cmpResidualKv);
  aclDestroyTensor(sinks);
  aclDestroyTensor(dq);
  aclDestroyTensor(dOriKv);
  aclDestroyTensor(dCmpKv);
  aclDestroyTensor(dSinks);
  aclDestroyTensor(metadata);
  aclDestroyTensor(oriSoftmaxL1Norm);
  aclDestroyTensor(cmpSoftmaxL1Norm);

  // 7. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(oriKvDeviceAddr);
  aclrtFree(cmpKvDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(dOutDeviceAddr);
  aclrtFree(lseDeviceAddr);
  aclrtFree(cuSeqQLenDeviceAddr);
  aclrtFree(cuSeqOriKvLenDeviceAddr);
  aclrtFree(cuSeqCmpKvLenDeviceAddr);
  aclrtFree(cmpResidualKvDeviceAddr);
  aclrtFree(sinksDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dOriKvDeviceAddr);
  aclrtFree(dCmpKvDeviceAddr);
  aclrtFree(dSinksDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(oriSoftmaxL1NormDeviceAddr);
  aclrtFree(cmpSoftmaxL1NormDeviceAddr);
  if (metaWorkspaceSize > 0) {
    aclrtFree(metaWorkspaceAddr);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
