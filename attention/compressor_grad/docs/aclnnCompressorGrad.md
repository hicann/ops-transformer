# aclnnCompressorGrad

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

- 接口功能：CompressorGrad是Compressor算子的反向算子，用于计算输入$X$、权重$W^{KV}$/$W^{Gate}$与位置编码$Ape$的梯度。前向在gradEnabled为true时导出softmax\_score（分组softmax结果）与kv（softmax结果与kv\_state的Hadamard乘积）中间结果，作为本算子的输入。主要计算过程为：
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
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnCompressorGradGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnCompressorGrad”接口执行计算。

```cpp
aclnnStatus aclnnCompressorGradGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    const aclTensor *dCmpKv,
    const aclTensor *softmaxScore,
    const aclTensor *kv,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t          cmpRatio,
    int64_t          coff,
    const aclTensor *dXOut,
    const aclTensor *dWkvOut,
    const aclTensor *dWgateOut,
    const aclTensor *dApeOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

``` cpp
aclnnStatus aclnnCompressorGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCompressorGradGetWorkspaceSize

- **参数说明**

    | 参数名                      | 输入/输出 | 描述  |  使用说明  | 数据类型       | 数据格式   | 维度（shape） | 非连续Tensor |
    |----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|-|-|-|
    | x | 输入 | 公式中的$X$，前向输入的原始数据。 | 不支持空Tensor。  | FLOAT16、BFLOAT16 | ND         | BS合轴：[T,H]、BS非合轴：[B,S,H]|×|
    | wkv | 输入 | 公式中的$W^{KV}$，前向kv压缩权重。  |不支持空Tensor。| FLOAT16、BFLOAT16 | ND |[coff* D,H]|×|
    | wgate | 输入 | 公式中的$W^{Gate}$，前向gate压缩权重。 |不支持空Tensor。| FLOAT16、BFLOAT16 | ND |[coff* D,H]|×|
    | dCmpKv | 输入 | 公式中的$dC$，前向输出cmp\_kv的上游梯度。 | 不支持空Tensor。 | FLOAT16、BFLOAT16 | ND |BS合轴：[min(T,T//cmp_ratio+B),D]、BS非合轴：[B,ceil(S/cmp_ratio),D]|×|
    | softmaxScore | 输入 | 公式中的$S$，前向在gradEnabled为true时导出的分组softmax中间结果。 | 不支持空Tensor。 | FLOAT32 | ND |BS合轴：[min(T,T//cmp_ratio+B), coff*cmp_ratio, D]、BS非合轴：[B,ceil(S/cmp_ratio),coff*cmp_ratio,D]|×|
    | kv | 输入 | 公式中的$K$，前向在gradEnabled为true时导出的softmax结果与kv\_state的Hadamard乘积中间结果。 | 不支持空Tensor。 | FLOAT32 | ND |同softmaxScore|×|
    | cuSeqlensOptional | 可选输入 | 表示不同Batch中的有效token数。 | 当x的shape为[T,H]时必传，输入shape为[B+1,]；当x的shape为[B,S,H]时，参数必须为空。不支持空Tensor。| INT32          | ND         |当x的shape为[T,H]时，输入shape为[B+1,]|×|
    | sequsedOptional | 可选输入 | 表示不同Batch中实际参与压缩的token数。 | 为None时，表示和每个Batch上的Sequence Length长度相同；要求seqused[n]不超过对应Sequence Length，且不小于0。| INT32          | ND         |[B,]|×|
    | startPosOptional | 可选输入 | 表示计算起始位置。 | 为None时，表示从0开始进行计算。| INT32          | ND         |[B,]|×|
    | cmpRatio | 输入 | 用于稀疏计算，表示数据压缩率，与前向一致。 |取值范围为[2, 128]内的整数。| INT32          | -         |-|-|
    | coff | 可选输入 | 表示是否进行overlap数据重排，与前向一致。 |取值范围为[1, 2]。当coff=1时，无需进行overlap数据重排。当coff=2时，需要进行overlap数据重排。| INT32          | -         |-|-|
    | dXOut | 输出 | 公式中的$dX$，输入x的梯度。 | 不支持空Tensor。| FLOAT16、BFLOAT16         | ND          |与x相同：BS合轴：[T,H]、BS非合轴：[B,S,H]|×|
    | dWkvOut | 输出 | 公式中的$dW^{KV}$，权重wkv的梯度。 | 不支持空Tensor。| FLOAT16、BFLOAT16         | ND          |[coff* D,H]|×|
    | dWgateOut | 输出 | 公式中的$dW^{Gate}$，权重wgate的梯度。 | 不支持空Tensor。| FLOAT16、BFLOAT16         | ND          |[coff* D,H]|×|
    | dApeOut | 输出 | 公式中的$dApe$，APE位置编码的梯度。 | 不支持空Tensor。| FLOAT32         | ND          |[cmp_ratio,coff* D]|×|

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
      <thead>
        <tr>
          <th>返回值</th>
          <th>错误码</th>
          <th>描述</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>必须传入的参数（如接口核心依赖的输入/输出参数）中存在空指针。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>输入参数的shape（维度/尺寸）、dtype（数据类型）不在接口支持的范围内。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_RUNTIME_ERROR</td>
          <td>361001</td>
          <td>API内存调用NPU Runtime接口时发生异常（如Runtime服务未启动、内存申请失败等）。</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_TILING_ERROR</td>
          <td>561002</td>
          <td>tiling发生异常，入参的dtype类型或者shape错误。</td>
        </tr>
      </tbody>
    </table>

## aclnnCompressorGrad

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCompressorGradGetWorkspaceSize获取。</td>
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

- x参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、D（Head Dim）表示hidden层的最小单元大小、T表示所有Batch输入样本序列长度的累加和。
- 输入shape限制：
    - wkv支持输入shape[coff* D,H]
    - wgate支持输入shape[coff* D,H]
    - softmaxScore支持输入shape：BS合轴时为[min(T,T//cmp_ratio+B), coff*cmp_ratio, D]；BS非合轴时为[B,ceil(S/cmp_ratio),coff*cmp_ratio,D]
    - kv支持输入shape：同softmaxScore
    - dCmpKv支持输入shape：BS合轴时为[min(T,T//cmp_ratio+B),D]；BS非合轴时为[B,ceil(S/cmp_ratio),D]
    - dX支持输出shape：与x相同，BS合轴时为[T,H]、BS非合轴时为[B,S,H]
    - dWkv、dWgate支持输出shape[coff* D,H]
    - dApe支持输出shape[cmp_ratio,coff* D]
    - startPos支持输入shape[B,]
    - 若x的维度采用BS合轴，即x的输入shape为[T,H]
        - cuSeqlens输入shape必须为[B+1,]。该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值，且第一位必须为0。
        - seqused，支持输入shape[B,]，要求每个Batch的有效token数要求小于等于对应Sequence Length长度，即seqused[n] <= cu\_seqlens[n+1] - cu\_seqlens[n]，且不小于0。
    - 若x的维度不采用BS合轴，即x的输入shape为[B,S,H]
        - cuSeqlens，参数必须为空。
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

## 调用示例

  无
