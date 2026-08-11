# CompressorGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：CompressorGrad是Compressor算子的反向算子，用于计算输入$X$、权重$W^{KV}$/$W^{Gate}$与位置编码$Ape$的梯度。前向在gradEnabled为true时导出softmax\_score（分组softmax结果）与kv（softmax结果与kv\_state的Hadamard乘积）中间结果，作为本算子的输入。主要计算过程为：
    1. 逐块计算Hadamard积反向：将上游梯度$dC$与softmax\_score、kv逐元素相乘，得到$dK$与$dS^\prime$；
    2. softmax反向：对$dS^\prime$沿压缩轴做softmax反向，得到$dZ$；
    3. APE梯度计算：按token位置累加$dZ$，得到$dApe$；
    4. matmul反向：将$dK$、$dZ$与权重做矩阵乘法反向，得到$dX$、$dW^{KV}$、$dW^{Gate}$。

- 计算公式：

    1. 计算Hadamard乘积反向，$N$为压缩块总数，$i$为压缩块序号，$dC_i$为第$i$块的上游梯度，$S_i$、$K_i$分别为softmax\_score、kv第$i$块：

        $$
        dK_i = dC_i \odot S_i,~ i=1,\cdots,N
        $$

        $$
        dS^\prime_i = dC_i \odot K_i,~ i=1,\cdots,N
        $$

    2. 计算softmax反向（沿压缩轴求和），$k$为块内行序号：

        $$
        dZ_i = S_i \odot \left(dS^\prime_i - \sum_{k=1}^{coff \cdot cmp\_ratio} \left(S_i \odot dS^\prime_i\right)_{k,:}\right),~ i=1,\cdots,N
        $$

    3. 计算APE梯度，$pos$为$dZ$各行对应token的全局位置：

        $$
        dApe = ScatterAdd\left(dZ,~ pos \% cmp\_ratio\right)
        $$

    4. 计算矩阵乘法反向，$dNewKv$、$dNewScore$为$dK$、$dZ$按压缩块映射回全局token行的结果（coff=2时prev/cur半区分别对应上一块与本块的token行，与正向的$W^{aKV}$/$W^{bKV}$对应）：

        $$
        dX = dNewKv @ W^{KV} + dNewScore @ W^{Gate}
        $$

        $$
        dW^{KV} = dNewKv^T @ X,~ dW^{Gate} = dNewScore^T @ X
        $$
## 参数说明

| 参数名                      | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| x | 输入 | 公式中的$X$，前向输入的原始数据。 | FLOAT16、BFLOAT16 | ND         |
| wkv | 输入 | 公式中的$W^{KV}$，前向kv压缩权重。  | FLOAT16、BFLOAT16 | ND |
| wgate | 输入 | 公式中的$W^{Gate}$，前向gate压缩权重。 | FLOAT16、BFLOAT16 | ND |
| d\_cmp\_kv | 输入 | 公式中的$dC$，前向输出cmp\_kv的上游梯度。 | FLOAT16、BFLOAT16 | ND         |
| softmax\_score | 输入 | 公式中的$S$，前向在gradEnabled为true时导出的分组softmax中间结果。 | FLOAT32     | ND         |
| kv | 输入 | 公式中的$K$，前向在gradEnabled为true时导出的softmax结果与kv\_state的Hadamard乘积中间结果。 | FLOAT32       | ND         |
| cu\_seqlens | 可选输入 | 表示不同Batch中的有效token数。<br>当x的shape为[T,H]时必传；当x的shape为[B,S,H]时，参数必须为空。 | INT32 | ND         |
| seqused | 可选输入 | 表示不同Batch中实际参与压缩的token数。<br>如果指定为None时，表示和每个Batch上的Sequence Length长度相同。 | INT32          | ND         |
| start\_pos | 可选输入 | 表示计算起始位置。<br>如果指定为None时，表示从0开始进行计算。 | INT32          | ND         |
| cmp\_ratio | 属性 | 用于稀疏计算，表示数据压缩率，取值范围为[2, 128]内的整数。 | INT32          | -         |
| coff | 可选属性 | 表示是否进行overlap数据重排。 <br>coff=1：无需进行overlap数据重排，coff=2：需要进行overlap数据重排。<br>默认值为1。 | INT32          | -         |
| d\_x | 输出 | 公式中的$dX$，输入x的梯度。 | FLOAT16、BFLOAT16         | ND          |
| d\_wkv | 输出 | 公式中的$dW^{KV}$，权重wkv的梯度。 | FLOAT16、BFLOAT16         | ND          |
| d\_wgate | 输出 | 公式中的$dW^{Gate}$，权重wgate的梯度。 | FLOAT16、BFLOAT16         | ND          |
| d\_ape | 输出 | 公式中的$dApe$，APE位置编码的梯度。 | FLOAT32         | ND          |

## 约束说明

- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff* D,H]
    - wgate支持输入shape[coff* D,H]
    - softmax\_score支持输入shape：BS合轴时为[min(T,T//cmp_ratio+B), coff*cmp_ratio, D]；BS非合轴时为[B,ceil(S/cmp_ratio),coff*cmp_ratio,D]
    - kv支持输入shape：同softmax\_score
    - d\_cmp\_kv支持输入shape：BS合轴时为[min(T,T//cmp_ratio+B),D]；BS非合轴时为[B,ceil(S/cmp_ratio),D]
    - d\_x支持输出shape：与x相同，BS合轴时为[T,H]、BS非合轴时为[B,S,H]
    - d\_wkv、d\_wgate支持输出shape[coff* D,H]
    - d\_ape支持输出shape[cmp_ratio,coff* D]
    - start\_pos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - cu\_seqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须为0。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - cu\_seqlens，参数必须为空。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即要求seqused[n] <= S，且不小于0。
- 输入值域限制：
  - 该接口支持B、S泛化，且存在如下场景限制：
      - **不支持B、S、T为0的空Tensor**：与正向Compressor不同，CompressorGrad所有输入/输出均不支持空Tensor，shapeSize必须大于0。
      - 部分长序列场景下，如果计算量过大可能会导致出现超过NPU内存的报错，注：这里计算量会受x输入shape的影响，值越大计算量越大。
- 输入属性限制：
  - 支持D为128/512。
  - 支持H为1K~10K，512对齐。
  - 支持coff为1/2。
  - 支持cmp\_ratio为2~128。

## 调用说明

| 调用方式  | 样例代码                                                                          | 说明                                                                          |
| --------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| aclnn接口 | -| 通过[aclnnCompressorGrad](./docs/aclnnCompressorGrad.md)调用CompressorGrad算子。 |
| PyTorch API | - | 通过[cann_ops_transformer.compressor](../../torch_extension/cann_ops_transformer/docs/zh/compressor.md)调用Compressor算子，反向经其autograd自动调用CompressorGrad。 |
