# BlockSparseAttentionGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="310b" id6 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id7 -->
<!-- npu="910" id8 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id8 -->

## 功能说明

- **接口功能**：计算块稀疏注意力的反向梯度 dq、dk 和 dv。通过 blockSparseMaskOptional 指定每个 Q 块参与计算的 KV 块，Q/KV 方向独立分块。

  相比于 [aclnnBlockSparseAttentionGrad](aclnnBlockSparseAttentionGrad.md)，本接口新增 qDequantScaleOptional、kDequantScaleOptional、vDequantScaleOptional、doDequantScaleOptional、pQuantScaleOptional、dsQuantScaleOptional 六个 Tensor 参数和 quantMode 模式参数。原接口的参数列表保持不变，仅支持 FP16/BF16；所有 HiF8 量化功能通过 V2 提供。

  quantMode=0 使用 FP16/BF16，六个 scale 均传 nullptr；quantMode=5 使用 HiF8 per-tensor，六个 scale 均必传。模式必须与输入 dtype 一致。quantMode 经 V2 ACLNN、L0 传入底层 BlockSparseAttentionGrad 的同名属性（默认值 0）。Tiling 校验模式与 dtype 一致，并据此选择 HiF8 或非量化内核；六个 scale 作为 Tensor 输入传入。原 ACLNN 内部固定传 quantMode=0，原参数列表保持不变。HiF8 仅支持 <term>Ascend 950PR&950DT系列产品</term>。

- **计算公式**：记 $Q$、$K$、$V$ 为 query、key、value，$O$ 为前向输出 attentionOut，$dO$ 为上游梯度 dout，$\alpha$ 为 scaleValue。在掩码保留的位置，使用前向输出的 softmaxLse 重构注意力概率 $P$。

  $$
  S = \alpha Q K^{T}, \quad P_{ij} = \exp(S_{ij} - \mathrm{softmaxLse}_i)
  $$

  被掩蔽的位置上 $P_{ij}=0$。对每个 Q token，沿 headDim 求和得到 $\delta_i$，再计算 softmax 的输入梯度 $dS$。

  $$
  \delta_i = \sum_{h} dO_{ih} O_{ih}, \quad dP = dO V^{T}, \quad dS_{ij} = P_{ij}(dP_{ij} - \delta_i)
  $$

  $$
  dQ = \alpha dS K, \quad dK = \alpha dS^{T}Q, \quad dV = P^{T}dO
  $$

在 GQA 和 MQA 场景中，多个 Q 头共享同一个 KV 头，dK 和 dV 在共享该 KV 头的 Q 头之间累加。

### HiF8 量化计算

本节仅适用于 V2 的 `quantMode=5`。原接口不能传入 HiF8 或量化参数。

以下用Q8、K8、V8、dO8表示HiF8编码对应的数值（尚未乘反量化因子），O表示BF16的attentionOut。qDequantScaleOptional、kDequantScaleOptional、vDequantScaleOptional、doDequantScaleOptional是输入反量化因子，满足：

```text
Q = Q8 * qDequantScaleOptional
K = K8 * kDequantScaleOptional
V = V8 * vDequantScaleOptional
dOut = dO8 * doDequantScaleOptional
```

在blockSparseMask保留的有效位置上，使用与本次输入及稀疏模式对应的softmaxLse重构P，并在FP32中计算softmax梯度：

```text
S = (Q8 @ K8^T) * (scaleValue * qDequantScaleOptional * kDequantScaleOptional)
P = exp(S - softmaxLse)
delta = rowsum((dO8 * doDequantScaleOptional) * O)
dP = (dO8 @ V8^T) * (doDequantScaleOptional * vDequantScaleOptional)
dS = (dP - delta) * P
```

其中`@`表示矩阵乘法，`*`表示逐元素乘法或标量乘法，rowsum沿HeadDim归约。P和dS在送入后续矩阵乘法前转换为HiF8；用于计算dS的P保留FP32数值：

```text
P8 = HiF8(P * pQuantScaleOptional)
dS8 = HiF8(dS * dsScaleUsed)
dV = (P8^T @ dO8) * (doDequantScaleOptional / pQuantScaleOptional)
dQ = (dS8 @ K8) * (scaleValue * kDequantScaleOptional / dsScaleUsed)
dK = (dS8^T @ Q8) * (scaleValue * qDequantScaleOptional / dsScaleUsed)
```

`pQuantScaleOptional`是量化倍率，而非反量化因子。`dsQuantScaleOptional > 0`时，`dsScaleUsed = dsQuantScaleOptional`；`dsQuantScaleOptional == 0`时，内核对每个有效稀疏任务的dS独立归约，当前动态缩放规则为：

```text
amax = max(abs(dS_task))
dsScaleUsed = 16384 / amax    # amax > 1e-12
dsScaleUsed = 1              # amax <= 1e-12
```

动态归约仅包含该任务的有效元素，不包含尾块补齐区域。同一任务的有效元素共用一个量化倍率，缩放粒度为稀疏任务。各任务的dQ/dK/dV贡献恢复尺度并累加，最终输出BF16。

### 数据排布

qInputLayout 指定 query、dout、attentionOut 和 dq 的轴顺序，kvInputLayout 指定 key、value、dk 和 dv 的轴顺序。B 表示批量大小，N 表示注意力头数，S 表示序列长度，D 表示每个头的维度，T 表示沿 token 轴拼接各样本后的总 token 数。

| 产品 | 支持的 Q/KV 布局 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | BNSD、BSND、TND |
| <term>Atlas A2训练系列产品</term>、<term>Atlas A3训练系列产品</term> | BNSD、TND |

qInputLayout 与 kvInputLayout 必须一致。softmaxLse 的轴顺序见参数表。

## 函数原型

本接口采用[两段式调用](../../../docs/zh/context/two_phase_api.md)。包含头文件 `aclnnop/aclnn_block_sparse_attention_grad_v2.h`，先调用 aclnnBlockSparseAttentionGradV2GetWorkspaceSize 获取 Device 侧 workspace 大小和执行器，再调用 aclnnBlockSparseAttentionGradV2 执行计算。

```c++
aclnnStatus aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
    const aclTensor* dout, const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* attentionOut, const aclTensor* softmaxLse, const aclTensor* blockSparseMaskOptional,
    const aclTensor* attenMaskOptional, const aclIntArray* blockShapeOptional,
    const aclIntArray* actualSeqLengthsOptional, const aclIntArray* actualSeqLengthsKvOptional,
    const aclTensor* qDequantScaleOptional, const aclTensor* kDequantScaleOptional,
    const aclTensor* vDequantScaleOptional, const aclTensor* doDequantScaleOptional,
    const aclTensor* pQuantScaleOptional, const aclTensor* dsQuantScaleOptional,
    char* qInputLayout, char* kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue,
    int64_t preTokens, int64_t nextTokens, int64_t quantMode, aclTensor* dq, aclTensor* dk, aclTensor* dv,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnBlockSparseAttentionGradV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
```

## aclnnBlockSparseAttentionGradV2GetWorkspaceSize

以下参数均属于 V2。名称中的 Optional 不表示在所有模式下都能省略：quantMode=5 时六个 scale 均必传；quantMode=0 时六个 scale 均传 nullptr。

- **参数说明**

  <table>
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
        <td>前向输出 attentionOut 的上游梯度。</td>
        <td>shape 与 query 一致。</td>
        <td>quantMode=0：FLOAT16、BFLOAT16；quantMode=5：HIFLOAT8</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>query（aclTensor*）</td>
        <td>输入</td>
        <td>注意力计算中的查询向量，即公式中的query。</td>
        <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalQTokens, headNum, headDim]。</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim]。</li><li>BSND: [batch, maxQSeqLength, headNum, headDim]。</li></ul></td>
        <td>quantMode=0：FLOAT16、BFLOAT16；quantMode=5：HIFLOAT8</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key（aclTensor*）</td>
        <td>输入</td>
        <td>注意力计算中的键向量，即公式中的key。</td>
        <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalKTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
        <td>quantMode=0：FLOAT16、BFLOAT16；quantMode=5：HIFLOAT8</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value（aclTensor*）</td>
        <td>输入</td>
        <td>注意力计算中的值向量，即公式中的value。</td>
        <td>不支持空Tensor。<br>支持的shape为：<ul><li>TND: [totalVTokens, numKeyValueHeads, headDim]。</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim]。</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim]。</li></ul></td>
        <td>quantMode=0：FLOAT16、BFLOAT16；quantMode=5：HIFLOAT8</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attentionOut（aclTensor*）</td>
        <td>输入</td>
        <td>前向块稀疏注意力的输出。</td>
        <td>shape 与 query 一致。HiF8 路径传 BFLOAT16，FP16/BF16 路径与 query 的数据类型相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxLse（aclTensor*）</td>
        <td>输入</td>
        <td>前向计算得到的 Softmax log-sum-exp，每个 Q token、每个 Q 头对应一个值。</td>
        <td>TND 传 [totalQTokens, headNum]，BNSD 传 [batch, headNum, maxQSeqLength]。BSND 中，FP16/BF16 路径仍按 [batch, headNum, maxQSeqLength] 传入，HiF8 路径按前向的 [batch, maxQSeqLength, headNum] 传入。上述 shape 均可在末尾增加一个大小为 1 的维度。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>2/3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>blockSparseMaskOptional（aclTensor*）</td>
        <td>输入</td>
        <td>块稀疏掩码，指定每个 Q 块参与计算的 KV 块。</td>
        <td>两个接口共用同一规格，必传，shape 为 [batch, headNum, qBlockNum, kvBlockNum]。headNum 是 Q 头数，GQA 时不是 KV 头数；TND 的 batch 是样本数。块数按对应方向最大单样本长度向上取整，见约束说明。只使用 0/1，1 保留、0 跳过；所有块参与时传全 1。它是块级布尔掩码，不是稀疏块索引列表。</td>
        <td>BOOL、UINT8</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional（aclTensor*）</td>
        <td>输入</td>
        <td>每个逻辑块的有效前缀长度，与 blockSparseMaskOptional 一起确定参与计算的 token。</td>
        <td>maskType 为 0 时必须传入 <code>nullptr</code>。maskType 为 1 时必须传入 [batch, headNum, M, 2]，其中 M ≥ max(qBlockNum, kvBlockNum)。<code>[b, n, i, 0]</code> 和 <code>[b, n, i, 1]</code> 分别为第 i 个 Q 块和第 i 个 KV 块的有效前缀长度，取值范围为 0 到对应方向的块大小。尾块仅计算有效 token，前缀长度为 0 的块跳过。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>blockShapeOptional（aclIntArray*）</td>
        <td>输入</td>
        <td>Q 和 KV 方向的稀疏块大小。</td>
        <td>必传。数组至少包含两个元素，前两个元素分别为 blockShapeX 和 blockShapeY。两者均须大于 0。Ascend 950 的 FP16/BF16 路径要求两者均为 16 的正整数倍，例如 [48, 80]。HiF8 路径中，两者均取 32、64、96 或 128，例如 [32, 64]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsOptional（aclIntArray*）</td>
        <td>输入</td>
        <td>TND 布局下的 Q 序列长度信息。</td>
        <td>TND 布局时必传。A2/A3 传各样本的实际长度，950 传从首个样本开始的累计长度。其他布局可传 <code>nullptr</code>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKvOptional（aclIntArray*）</td>
        <td>输入</td>
        <td>TND 布局下的 KV 序列长度信息。</td>
        <td>TND 布局时必传。编码规则与 actualSeqLengthsOptional 相同，数组元素数须一致。其他布局可传 <code>nullptr</code>。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qDequantScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>query 的反量化因子。</td>
        <td>quantMode=0 传 <code>nullptr</code>；quantMode=5 必传 Device Tensor。取有限正数，与产生 Q 的 HiF8 编码所用尺度一致，可复用同一次 BSA 前向的 Q 反量化因子。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[1]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>kDequantScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>key 的反量化因子。</td>
        <td>quantMode=0 传 <code>nullptr</code>；quantMode=5 必传 Device Tensor。取有限正数，与产生 K 的 HiF8 编码所用尺度一致，可复用同一次 BSA 前向的 K 反量化因子。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[1]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>vDequantScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>value 的反量化因子。</td>
        <td>quantMode=0 传 <code>nullptr</code>；quantMode=5 必传 Device Tensor。取有限正数，与产生 V 的 HiF8 编码所用尺度一致，可复用同一次 BSA 前向的 V 反量化因子。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[1]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>doDequantScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>dout 的反量化因子。</td>
        <td>quantMode=0 传 <code>nullptr</code>；quantMode=5 必传 Device Tensor。取有限正数，与产生 dO 的 HiF8 编码所用尺度一致。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[1]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>pQuantScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>反向重计算概率 P 的静态量化倍率。</td>
        <td>quantMode=0 传 <code>nullptr</code>；quantMode=5 必传 Device Tensor。取有限正数，由调用方指定。量化时对 P 乘此倍率。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[1]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>dsQuantScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>softmax 梯度 dS 的量化模式和静态倍率。</td>
        <td>quantMode=0 传 <code>nullptr</code>；quantMode=5 必传 Device Tensor。取有限非负数。大于 0 时按该值静态量化，等于 0 时由内核按稀疏计算任务动态计算倍率。动态模式传入 shape 为 <code>[1]</code>、唯一元素为 <code>0.0f</code> 的 Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[1]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>quantMode（int64_t）</td>
        <td>输入</td>
        <td>显式量化模式参数，透传为底层算子的 quantMode 属性。</td>
        <td>0：FP16/BF16，六个 scale 为空；5：HiF8 per-tensor，六个 scale 必传。其他值及 dtype 不匹配时报错。</td>
        <td>INT64</td><td>-</td><td>-</td><td>-</td>
      </tr>
      <tr>
        <td>qInputLayout（char*）</td>
        <td>输入</td>
        <td>query的数据排布格式。指示输入张量在内存中的具体排布（如连续或合轴排列）。</td>
        <td>与 kvInputLayout 一致，产品支持的取值见数据排布。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvInputLayout（char*）</td>
        <td>输入</td>
        <td>key和value的数据排布格式。指示输入张量在内存中的具体排布。</td>
        <td>与 qInputLayout 一致，产品支持的取值见数据排布。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads（int64_t）</td>
        <td>输入</td>
        <td>key 和 value 的注意力头数。</td>
        <td>传入与 key、value 的 N 轴大小相同的正整数。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maskType（int64_t）</td>
        <td>输入</td>
        <td>块内掩码类型。</td>
        <td><ul><li>0：仅使用块稀疏掩码，attenMaskOptional 传 <code>nullptr</code>。</li><li>1：在块稀疏掩码的基础上按块内有效前缀计算，仅 Atlas A2/A3 训练系列产品支持，必须传入 attenMaskOptional。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue（double）</td>
        <td>输入</td>
        <td>缩放系数，对应公式中的 α。</td>
        <td>与前向计算使用的缩放系数相同，通常取 1 / sqrt(headDim)。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens（int64_t）</td>
        <td>输入</td>
        <td>滑窗左侧范围。</td>
        <td>固定传入 2147483647。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens（int64_t）</td>
        <td>输入</td>
        <td>滑窗右侧范围。</td>
        <td>固定传入 2147483647。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dq（aclTensor*）</td>
        <td>输出</td>
        <td>query的梯度输出结果，即公式中的dq。</td>
        <td>shape 与 query 一致，数据类型与 attentionOut 一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dk（aclTensor*）</td>
        <td>输出</td>
        <td>key的梯度输出结果，即公式中的dk。</td>
        <td>shape 与 key 一致，数据类型与 attentionOut 一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3/4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dv（aclTensor*）</td>
        <td>输出</td>
        <td>value的梯度输出结果，即公式中的dv。</td>
        <td>shape 与 value 一致，数据类型与 attentionOut 一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3/4</td>
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

- **返回值**

  返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。下表列出 ACLNN 层直接返回的参数错误。布局支持范围和 Ascend 950 的块大小对齐要求会在 ACLNN 层校验，headDim、HiF8 块大小限制和序列长度等条件还会在 Tiling 阶段校验。

  <table>
    <thead>
      <tr>
        <th>返回码</th>
        <th>错误码</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>dout、query、key、value、attentionOut、softmaxLse、blockSparseMaskOptional、blockShapeOptional、dq、dk、dv、workspaceSize 或 executor 为空指针，或 HiF8 路径缺少任一缩放 Tensor。</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>输入输出的数据类型或 shape 校验失败，quantMode 不为 0/5、与 dtype 不匹配、mode 0 传入缩放 Tensor，或 mode 5 的缩放 Tensor 类型/shape 不符合要求，以及产品不支持 HiF8。</td>
      </tr>
      <tr>
        <td>qInputLayout 或 kvInputLayout 为空指针、取值非法、两者不一致，在 Atlas A2/A3 上使用 BSND，或 TND 布局缺少对应的序列长度数组。</td>
      </tr>
      <tr>
        <td>maskType、attenMaskOptional、preTokens 或 nextTokens 的配置不符合要求。</td>
      </tr>
      <tr>
        <td>blockShapeOptional 的元素少于两个，数组数据为空，前两个元素存在非正值，或 Ascend 950 上任一块大小不是 16 的整数倍。</td>
      </tr>
    </tbody>
  </table>

## aclnnBlockSparseAttentionGradV2

第二阶段应与同版本的 GetWorkspaceSize 配套使用。

- **参数说明**

  <table>
  <colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由同版本第一阶段 GetWorkspaceSize 接口获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的AscendCL stream流。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

| 数据 | quantMode=0：FP16/BF16 | quantMode=5：HiF8 |
| --- | --- | --- |
| dout、query、key、value | 均为 FLOAT16 或均为 BFLOAT16 | 均为 HIFLOAT8 |
| attentionOut、dq、dk、dv | 与 query 一致 | BFLOAT16 |
| softmaxLse | FLOAT（FP32） | FLOAT（FP32） |
| 六个缩放参数 | 均传 nullptr | 均为 Device 侧的 FLOAT（FP32）Tensor，shape 为 [1] |

- 张量的数据格式为 ND。HiF8 的六个缩放参数各自为一维 Tensor，唯一维度的长度为 1。Q/K/V/dO 各自使用一个 per-tensor 反量化因子，P 使用一个静态量化倍率；dS 为正数时使用一个静态倍率，为 0 时只表示动态模式开关，实际倍率由每个稀疏任务独立计算，不能将该 0 值理解为量化倍率。取值范围见参数表。
- Ascend 950 产品要求 headDim 取 128。query 的头数必须为 KV 头数的正整数倍。dq、dk、dv 由调用方按 query、key、value 的 shape 分别分配。
- attentionOut 和 softmaxLse 须来自同一次前向计算。反向使用相同的 Q、K、V、缩放系数和掩码设置。
- 在 <term>Ascend 950PR&950DT系列产品</term>上，BNSD/BSND 的 Q、K、V 序列长度须相等，TND 的 Q、K、V 总 token 数须相等。
- TND 布局时，两个长度数组均须包含 batch 个元素，各样本按 batch 顺序沿 token 轴拼接。例如，某一侧的两个样本分别有 64、96 个 token，Atlas A2/A3 训练系列产品传 [64, 96]，Ascend 950PR&950DT 系列产品传 [64, 160]。BNSD/BSND 布局直接使用 shape 中的 S 轴长度。
- 块稀疏掩码的 qBlockNum = ceilDiv(maxQSeqLength, blockShapeX)，kvBlockNum = ceilDiv(maxKvSeqLength, blockShapeY)。BNSD/BSND 取各自 shape 的 S 轴作为最大序列长度，TND 取对应方向的最大单样本长度，累计长度须先差分。序列长度无需整除块大小，尾块只覆盖剩余 token。

- 原接口与 V2 的块稀疏 mask 的 shape、0/1 语义及 Q 头维一致；HiF8 不要求更换 mask 规格。原接口与 V2 均只接受 BOOL、UINT8，不接受 INT8；BOOL 输入在 Host 调度中转换为 UINT8，底层算子注册类型保持 UINT8。
- ND 格式的输入 Tensor 可非连续，ACLNN 会先执行连续化；这可能增加拷贝开销。scale 仍须满足 FLOAT32 `[1]`。输出通过 ViewCopy 写入调用方提供的 Tensor。
- preTokens 和 nextTokens 均固定为 2147483647。仅 Atlas A2/A3 训练系列产品可通过 maskType = 1 配置块内有效前缀。
- 与 PyTorch 配合使用时，CANN 与 PyTorch 相关软件包的版本须匹配。

## 调用示例

### FP16/BF16 调用示例

示例使用 V2、quantMode=0、BNSD 布局、FLOAT16 和全 1 块稀疏掩码演示两段式调用。实际计算应传入配套的前向 attentionOut 和 softmaxLse。编译与运行步骤见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention_grad_v2.h"

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
    // 固定写法，AscendCL初始化
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
    // 检查shape是否有效
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // 检查hostData大小是否匹配
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // 调用aclrtMalloc申请device侧内存
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置核心参数(以BNSD Layout为例)
    int32_t batch = 1;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t qSeqlen = 161;
    int32_t kvSeqlen = 161;
    int32_t headDim = 128;
    // Q/KV方向采用不同的、均小于128且16对齐的稀疏粒度
    int32_t blockShapeX = 48;
    int32_t blockShapeY = 80;

    // 块数量计算
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilKv = (kvSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. 构建张量Shape
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kvShape = {batch, numKvHeads, kvSeqlen, headDim};
    std::vector<int64_t> lseShape = {batch, numHeads, qSeqlen}; // BNSD 的 LSE 按 [B, N, S] 排布
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilKv};

    // 4. 分配并初始化Host数据
    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);

    // 将Q, K, V初始化为0.1f等较小的数
    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kvSize, 0.1f);
    std::vector<op::fp16_t> vData(kvSize, 0.1f);

    // 梯度初始值可以给一个小正数
    std::vector<op::fp16_t> doutData(qSize, 0.01f);
    std::vector<op::fp16_t> outData(qSize, 0.1f);

    // 构造演示用 LSE。实际调用使用对应的前向输出。
    std::vector<float> lseData(GetShapeSize(lseShape), 5.0f);
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 1);

    // 创建所有的前向输入/输出aclTensor
    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr;
    void *lseAddr = nullptr, *maskAddr = nullptr;

    aclTensor *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *doutTensor = nullptr, *outTensor = nullptr;
    aclTensor *lseTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(vData, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CreateAclTensor(doutData, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &doutTensor);
    CreateAclTensor(outData, qShape, &outAddr, aclDataType::ACL_FLOAT16, &outTensor);

    CreateAclTensor(lseData, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lseTensor);     // 严格使用FP32
    CreateAclTensor(maskData, maskShape, &maskAddr, aclDataType::ACL_UINT8, &maskTensor); // 本示例使用 UINT8

    // 5. 创建反向输出梯度(dq, dk, dv)
    std::vector<op::fp16_t> dqData(qSize, 0.0f);
    std::vector<op::fp16_t> dkData(kvSize, 0.0f);
    std::vector<op::fp16_t> dvData(kvSize, 0.0f);

    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;

    CreateAclTensor(dqData, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dqTensor);
    CreateAclTensor(dkData, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dkTensor);
    CreateAclTensor(dvData, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dvTensor);

    // 6. 创建块大小数组
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    // 7. 标量与字符串参数配置
    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    // 强制规定滑动窗口极大值
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;

    // 8. 调用第一段接口: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnBlockSparseAttentionGradV2GetWorkspaceSize...\n");
    ret = aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
        doutTensor,
        qTensor,
        kTensor,
        vTensor,
        outTensor,
        lseTensor,
        maskTensor,                 // blockSparseMaskOptional
        nullptr,                    // attenMaskOptional必须为空
        blockShapeArr,
        nullptr,                    // BNSD 不需要 Q 序列长度数组
        nullptr,                    // BNSD 不需要 KV 序列长度数组
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, // quantMode=0：六个 scale 均为空
        qLayoutBuffer,
        kvLayoutBuffer,
        static_cast<int64_t>(numKvHeads),
        maskType,
        scaleValue,
        preTokens,
        nextTokens,
        0,                          // quantMode=0：FP16/BF16
        dqTensor,
        dkTensor,
        dvTensor,
        &workspaceSize,
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 9. 分配workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 10. 调用第二段接口: 执行计算
    LOG_PRINT("Calling aclnnBlockSparseAttentionGrad...\n");
    ret = aclnnBlockSparseAttentionGradV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 11. 同步Stream，等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 12. 将结果拷贝回Host侧打印
    ret = aclrtMemcpy(dqData.data(), qSize * sizeof(op::fp16_t), dqAddr, qSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! Output results (first 10 elements of dQ):\n");
    for (uint64_t i = 0; i < 10 && i < dqData.size(); i++) {
        LOG_PRINT("  dQ index %lu: %f\n", i, static_cast<float>(dqData[i]));
    }

    // 13. 释放所有资源
    LOG_PRINT("Cleaning up resources...\n");
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(doutAddr); aclrtFree(outAddr);
    aclrtFree(lseAddr); aclrtFree(maskAddr);
    aclrtFree(dqAddr); aclrtFree(dkAddr); aclrtFree(dvAddr);

    aclDestroyTensor(qTensor); aclDestroyTensor(kTensor); aclDestroyTensor(vTensor);
    aclDestroyTensor(doutTensor); aclDestroyTensor(outTensor);
    aclDestroyTensor(lseTensor); aclDestroyTensor(maskTensor);
    aclDestroyTensor(dqTensor); aclDestroyTensor(dkTensor); aclDestroyTensor(dvTensor);

    aclDestroyIntArray(blockShapeArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BlockSparseAttentionGrad Test completed successfully!\n");
    return 0;
}
```

### HiF8 调用示例

完整的数据加载、Device Tensor创建、结果比较及资源释放代码见[HiF8 AscendCL示例](../examples/test_aclnn_block_sparse_attention_grad_hif8.cpp)。示例使用BNSD布局，B=1、Nq=4、Nkv=2、Sq=Skv=128、HeadDim=128、blockShape=[32, 32]。

下面展示两段式调用的核心片段。假定已初始化ACL与stream，创建HIFLOAT8的dout/query/key/value、BF16的attentionOut/dq/dk/dv、FP32的softmaxLse，以及UINT8 或 BOOL 的 blockSparseMask。六个缩放 Tensor 均在 Device 侧创建，数据类型为 FLOAT32，shape 均为 [1]。dsQuantScaleOptional 的唯一元素为正数时使用静态倍率，写入 0.0f 时启用动态量化。

```c++
int64_t blockDims[] = {32, 32};
aclIntArray* blockShape = aclCreateIntArray(blockDims, 2);
char qLayout[] = "BNSD";
char kvLayout[] = "BNSD";
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;

aclnnStatus ret = aclnnBlockSparseAttentionGradV2GetWorkspaceSize(
    dout, query, key, value, attentionOut, softmaxLse,
    blockSparseMask, nullptr, blockShape, nullptr, nullptr,
    qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, doDequantScaleOptional, pQuantScaleOptional, dsQuantScaleOptional,
    qLayout, kvLayout, 2, 0, 1.0 / std::sqrt(128.0),
    2147483647, 2147483647, 5, dq, dk, dv, &workspaceSize, &executor);
if (ret != ACL_SUCCESS) {
    aclDestroyIntArray(blockShape);
    return ret;
}

void* workspace = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        aclDestroyIntArray(blockShape);
        return ret;
    }
}
ret = aclnnBlockSparseAttentionGradV2(workspace, workspaceSize, executor, stream);
// 等待stream完成后再读取结果、释放workspace和输入输出Tensor。
aclError syncRet = aclrtSynchronizeStream(stream);
if (workspace != nullptr) {
    aclrtFree(workspace);
}
aclDestroyIntArray(blockShape);
if (ret != ACL_SUCCESS) {
    return ret;
}
if (syncRet != ACL_SUCCESS) {
    return syncRet;
}
// 此时可将BF16的dq/dk/dv拷回Host；其shape分别与query/key/value一致。
```

编译上述片段所在程序时需包含`<cmath>`、ACL头文件和`aclnn_block_sparse_attention_grad_v2.h`，输出内存按BF16元素大小分配。

### PyTorch 配套调用

Python 函数仍为 `cann_ops_transformer.block_sparse_attention_grad`，没有新增名为 V2 的 Python 函数。当前封装的 FP16/BF16 路径调用原 ACLNN，HiF8 路径调用本 V2 接口。`quant_mode=5` 选择 HiF8，`quant_mode=0` 选择非量化，省略或传 None 时按 query dtype 推断；其他值或 dtype 不匹配时报错。HiF8 需要配套安装包含 V2 符号的算子包与更新后的 Torch 扩展。

以下 Q/K/V/dO 已编码为 HiF8 字节，Python dtype 为 `torch.uint8`；O 为 `torch.bfloat16`，LSE 为 `torch.float32`，均与本次前向相对应。六个 scale 为同 Device 的 `torch.float32`、shape `[1]`。本例使用 BSND，LSE 为 `[B,Sq,Nq]` 或 `[B,Sq,Nq,1]`；三个返回梯度均为 BF16，shape 分别与 Q/K/V 一致。

```python
dq, dk, dv = cann_ops_transformer.block_sparse_attention_grad(
    dout_hif8, query_hif8, key_hif8, value_hif8,
    attention_out, softmax_lse, block_sparse_mask,
    block_shape=block_shape,
    q_input_layout="BSND",
    kv_input_layout="BSND",
    num_key_value_heads=num_kv_heads,
    scale_value=softmax_scale,
    q_dequant_scale=bsa_q_scale,
    k_dequant_scale=bsa_k_scale,
    v_dequant_scale=bsa_v_scale,
    do_dequant_scale=dout_scale,
    p_quant_scale=backward_p_scale,
    ds_quant_scale=torch.zeros((1,), dtype=torch.float32, device=query_hif8.device),
    quant_mode=5,
)
```

`ds_quant_scale=[0]` 选择任务级动态 dS 量化；改为有限正数时选择静态倍率。`p_quant_scale` 始终显式传入有限正数，没有 BSA V3 的默认 16 行为。原 [Torch 接口文档](torchapi_block_sparse_attention_grad.md)保留非量化接口说明。
