# NsaSelectedAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      √     |
|<term>Atlas A2推理系列产品</term>|      ×     |

## 功能说明

- 算子功能：根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。

- 计算公式：

    根据传入的topkIndices对key和value选取数量为selectedBlockCount个大小为selectedBlockSize的数据重排，公式如下：

    $$
    selectedKey = Gather(key, topkIndices[i]),0<=i<selectedBlockCount \\
    selectedValue = Gather(value, topkIndices[i]),0<=i<selectedBlockCount
    $$

    接着，进行注意力机制的反向计算，计算公式为：

    $$
    V=P^TdY
    $$

    $$
    Q=\frac{((dS)*K)}{\sqrt{d}}
    $$

    $$
    K=\frac{((dS)^T*Q)}{\sqrt{d}}
    $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| query | 输入 | 公式中的输入Q。 | BFLOAT16、FLOAT16 | ND |
| key | 输入 | 公式中的输入K。 | BFLOAT16、FLOAT16 | ND |
| value | 输入 | 公式中的输入V。 | BFLOAT16、FLOAT16 | ND |
| attentionOut | 可选输入 | 注意力正向计算的最终输出。 | BFLOAT16、FLOAT16 | ND |
| attentionOutGrad | 输入 | 公式中的输入dY。 | BFLOAT16、FLOAT16 | ND |
| softmaxMax | 输入 | 注意力正向计算的中间输出。 | FLOAT | ND |
| softmaxSum | 输入 | 注意力正向计算的中间输出。 | FLOAT | ND |
| topkIndices | 输入 | 公式中的所选择KV的索引。 | INT32 | ND |
| scaleValue | 属性 | 公式中d开根号的倒数，表示缩放系数。 | DOUBLE | - |
| dqOut | 输出 | 公式中的dQ，表示query的梯度。 | BFLOAT16、FLOAT16 | ND |
| dkOut | 输出 | 公式中的dK，表示key的梯度。 | BFLOAT16、FLOAT16 | ND |
| dvOut | 输出 | 公式中的dV，表示value的梯度。 | BFLOAT16、FLOAT16 | ND |

## 约束说明

- 该接口与pytorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 输入query、key、value、attentionOut、attentionOutGrad的B(batchsize)必须相等。
- 输入key、value的N(numHead)必须一致。
- 输入query、attentionOut、attentionOutGrad的N(numHead)必须一致。
- 输入value、attentionOut、attentionOutGrad的D(HeadDim)必须一致。
- 输入query、key、value、attentionOut、attentionOutGrad的inputLayout必须一致。
- 关于数据shape的约束，以inputLayout的TND举例。其中：
  - T1：取值范围为1\~2M。T1表示query所有batch下S的和。
  - T2：取值范围为1\~2M。T2表示key、value所有batch下S的和。
  - B：取值范围为1\~2M。
  - N1：取值范围为1\~128。表示query的headNum。N1必须为N2的整数倍。
  - N2：取值范围为1\~128。表示key、value的headNum。
  - G：取值范围为1\~32。 `G = N1 / N2`。
  - S：取值范围为1\~128K。对于key、value的S必须大于等于selectedBlockSize * selectedBlockCount,且必须为selectedBlockSize的整数倍。
  - D：取值范围为192或128，支持K和V的D(HeadDim)不相等。
  - selectedBlockSize支持<=128且满足16的整数倍。
  - selectBlockCount：支持[1~128]。总计选择的大小`selectBlockCount * selectBlockSize` < 128*64(8K)
  - Layout为TND时，每个Batch的S2都要大于总计选择的大小`selectBlockCount * selectBlockSize`
- 关于softmaxMax与softmaxSum参数shape的约束：\[T1, N1, 8\]。
- 关于topkIndices参数shape的约束：[T1, N2, selectedBlockCount]。

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                 |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_nsa_selected_attention_grad](./examples/test_aclnn_nsa_selected_attention_grad.cpp) | 非TND场景，通过[aclnnNsaSelectedAttentionGrad](./docs/aclnnNsaSelectedAttentionGrad.md)接口方式调用NsaSelectedAttentionGrad算子。             |
