# NsaCompressGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3系列产品</term>|      √     |
|<term>Atlas A2训练系列产品</term>|      √     |
|<term>Atlas A2推理系列产品</term>|      ×     |

## 功能说明

- 算子功能：aclnnNsaCompress算子的反向计算。

- 计算公式：

  设 \(a_b\) 为 batch \(b\) 在 input 上的起点，\(o_b\) 为该 batch 在 outputGrad 上的起点，\(s\) 为 compressStride，\(L\) 为 compressBlockSize。\(i\) 只遍历该 batch 的完整窗口。

    $$
    \text{weightGrad}[r,n] = \sum_{b,i,d} \text{outputGrad}[o_b+i,n,d] \cdot \text{input}[a_b+i\cdot s+r,n,d], \quad 0 \le r < L
    $$

    $$
    \text{inputGrad}[a_b+t,n,d] = \sum_{i} \text{outputGrad}[o_b+i,n,d] \cdot \text{weight}[t-i\cdot s,n]
    $$

  inputGrad 只对满足 \(0 \le t-i\cdot s < L\) 的完整窗口求和。权重梯度沿特征维求和后，再跨 batch 和窗口累加。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| outputGrad | 输入 | 正向算子输出的反向梯度，shape为[T_compressed, N, D]。T_compressed = Σ_b max(0, floor((S_b - compressBlockSize) / compressStride) + 1)，S_b 为第 b 个 batch 的序列长度。 | BFLOAT16、FLOAT16 | ND |
| input | 输入 | 待压缩张量，shape为[T, N, D]。T 为各 batch 序列长度之和，与 T_compressed 不一定相等。 | BFLOAT16、FLOAT16 | ND |
| weight | 输入 | 压缩权重，shape为[compressBlockSize, N]，与input满足broadcast关系。 | BFLOAT16、FLOAT16 | ND |
| actSeqLenOptional | 输入 | 每个Batch对应的S大小。当前必须显式传入非nullptr参数，单batch长度为S时传入[S]。 | INT64 | ND |
| compressBlockSize | 输入 | 压缩滑窗大小。 | INT64 | - |
| compressStride | 输入 | 两次压缩滑窗间隔大小。 | INT64 | - |
| actSeqLenType | 输入 | 序列长度类型，取值为0或1。0表示各batch序列长度的累积和，1表示各batch的独立序列长度，当前建议使用0。例如batch长度为64、80时，取值0对应[64, 144]，取值1对应[64, 80]。 | INT64 | - |
| layoutOptional | 输入 | 输入数据排布格式，仅支持"TND"。传入nullptr时使用默认值"TND"。 | String | - |
| inputGrad | 输出 | input的梯度，shape为[T, N, D]，与input保持一致。 | BFLOAT16、FLOAT16 | ND |
| weightGrad | 输出 | weight的梯度，shape与weight保持一致。 | BFLOAT16、FLOAT16 | ND |

## 约束说明

- compressBlockSize和compressStride要是16的整数倍，且compressBlockSize >= compressStride。
- outputGrad和input的末维 D 必须是16的整数倍。

## 调用说明

| 调用方式        | 调用样例        | 说明                                                 |
|----------------|----------------|------------------------------------------------------|
| aclnn调用 | [test_aclnn_nsa_compress_grad](./examples/test_aclnn_nsa_compress_grad.cpp) | TND场景，通过[aclnnNsaCompressGrad](./docs/aclnnNsaCompressGrad.md)接口方式调用NsaCompressGrad算子。|
