# NsaCompressGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      √     |
|<term>Atlas A2推理系列产品</term>|      ×     |

## 功能说明

- 算子功能：aclnnNsaCompress算子的反向计算。

- 计算公式：
  选择注意力的正向计算公式如下：

    $$
    \text{dw} = \text{dk\_cmp} \cdot K^\top
    $$

    $$
    \text{dk} = W^\top \cdot \text{dk\_cmp}
    $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| outputGrad | 输入 | 正向算子输出的反向梯度，shape支持[T, N, D]。 | BFLOAT16、FLOAT16 | ND |
| input | 输入 | 待压缩张量，shape支持[T, N, D]。 | BFLOAT16、FLOAT16 | ND |
| weight | 输入 | 压缩权重，shape为[compressBlockSize, N]，与input满足broadcast关系。 | BFLOAT16、FLOAT16 | ND |
| actSeqLenOptional | 输入 | 每个Batch对应的S大小，batch序列长度不等时需输入。 | INT64 | ND |
| compressBlockSize | 输入 | 压缩滑窗大小。 | INT64 | - |
| compressStride | 输入 | 两次压缩滑窗间隔大小。 | INT64 | - |
| actSeqLenType | 输入 | 序列长度类型，0表示cumsum结果，1表示每个batch序列大小，当前仅支持0。 | INT64 | - |
| layoutOptional | 输入 | 输入数据排布格式，支持TND。 | String | - |
| inputGrad | 输出 | input的梯度，shape与input保持一致。 | BFLOAT16、FLOAT16 | ND |
| weightGrad | 输出 | weight的梯度，shape与weight保持一致。 | BFLOAT16、FLOAT16 | ND |

## 约束说明

- compressBlockSize和compressStride要是16的整数倍，且compressBlockSize > compressStride。

## 调用说明

| 调用方式        | 调用样例        | 说明                                                 |
|----------------|----------------|------------------------------------------------------|
| aclnn调用 | [test_aclnn_nsa_compress_grad](./examples/test_aclnn_nsa_compress_grad.cpp) | 非TND场景，通过[aclnnNsaCompressGrad](./docs/aclnnNsaCompressGrad.md)接口方式调用NsaCompressGrad算子。|
