# 算子名称：GateGeluQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                         | 是       |

## 功能说明

- 算子功能：在Transformer模型的FFN层中，对Gated GLU结构的输出进行融合计算与动态Per-Channel量化。将输入按列均分为Gate和Value两部分，对Gate部分施加GELU激活函数后与Value部分逐元素相乘，再经过可选的截断约束，最后乘以缩放因子量化输出为INT8，即实现 **GeGLU + Per-Channel Quantization** 的融合算子。

- 数学公式：

  $$intermediate = GELU(input[:, :W]) \odot input[:, W:]$$
  $$if \ constrait: \ intermediate = Clamp(intermediate, -clampValue, clampValue)$$
  $$output = Quantize(intermediate \times scale) \quad \in [-128, 127]$$

  其中 $W = gbW / 2$，$\odot$ 表示逐元素乘法，$Quantize$ 表示四舍五入取整并截断到INT8范围。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
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
      <td>in</td>
      <td>输入</td>
      <td>Gate和Value拼接的输入张量，shape为(gbH, gbW)，其中gbW = 2 × hidden_size，前半部分为Gate，后半部分为Value</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>Per-Channel量化缩放因子，shape为(gbW / 2, )</td>
      <td>float32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>GeGLU计算并量化后的INT8结果，shape为(gbH, gbW / 2)</td>
      <td>int8_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gbH</td>
      <td>输入</td>
      <td>输入张量的行数（token数量）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gbW</td>
      <td>输入</td>
      <td>输入张量的列数（Gate + Value拼接后的隐藏维度，必须为偶数）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>constrait</td>
      <td>输入</td>
      <td>是否在量化前对GeGLU的FP32中间结果进行截断约束，默认为false</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampValue</td>
      <td>输入</td>
      <td>截断约束的阈值，配合constrait使用，将值截断在 [-clampValue, clampValue] 内，默认为128.0</td>
      <td>float32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockDim</td>
      <td>输入</td>
      <td>AI Core的数量，如Ascend910B为40</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>Device端的stream</td>
      <td>AclrtStream</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量的列数`gbW`必须为偶数，以便均分为Gate和Value两部分。
- 数据类型约束：输入为float16，Scale为float32，输出为int8_t。

## 价值/作用

在LLaMA、Qwen、DeepSeek等大语言模型的W8A8量化推理部署中，GeGLU激活及其输出量化是FFN层计算的核心步骤。本算子将GELU激活、门控乘法、激活值截断及Per-Channel动态量化四个操作融合为一个Kernel，彻底消除了FP16/FP32中间结果的Global Memory写回与读入，大幅降低显存带宽压力和Kernel启动开销，显著提升端到端推理性能。

## 设计方案

### Tiling策略

- **分核策略**：
  按行（gbH）维度进行分核。每个Core处理`⌈gbH / blockNum⌉`行数据，采用向上取整方式分配，尾部行通过`i * blockNum_ + blockIdx_ < gbH`判断跳过。

- **分块策略**：
  在列方向（bkW = gbW / 2）上按可用UB容量进行分块。每个Tile的宽度`tlW_`根据UB最大可用字节数（184KB）和所需Buffer（2个half输入、1个float scale、1个int8输出）动态计算，并向下对齐到64元素（满足128字节对齐）。尾部Tile宽度`tlTailW_`为`bkW % tlW_`，向上对齐到64用于计算，实际有效宽度用于搬入搬出。

### Kernel侧设计

进行 **Init** 和 **Process** 两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、数据搬出（CopyOut）三个阶段。采用单缓冲（BUFFER_NUM = 1）机制。

- **初始化（Init）**
  - 计算分核参数：`bkLoop_`（每个Core处理的行数）、`blockIdx_`（当前Core编号）。
  - 计算分块参数：`bkW_ = gbW / 2`，根据UB容量计算`tlMaxW_`，进而确定`tlW_`、`tlTailW_`、`tlAlignTailW_`、`tlLoop_`。
  - 建立GM Tensor映射（inGm_、scaleGm_、outGm_）。
  - 初始化VECIN队列：`inQueIn_`（Gate和Value输入合并）、`inQueScale_`（缩放因子），以及VECOUT队列`outQueOut_`（INT8输出）。

- **计算流程（Process）**
  - FOR i = 0 TO bkLoop_（行方向循环）：
    - 判断当前行是否在有效范围内
    - **CopyIn**：
      - 从GM搬入拼接的Gate和Value数据到同一块本地内存`in_local[0]`和`in_local[tlW_]`。
      - 从GM搬入当前Tile对应的`scale`数据。
    - **Compute**（高度融合的量化计算流程）：
      1. `Gelu(in_one, in_one)`：对Gate部分计算GELU激活。
      2. `Mul(in_two, in_one, in_two)`：Gate和Value逐元素相乘，得到GeGLU结果。
      3. `Cast(infloat_local, in_two, CAST_NONE)`：将FP16结果转为FP32，以便进行高精度量化计算。
      4. **[可选约束]** 若`constrait_`为true，则使用`Mins`和`Maxs`将FP32结果截断在`[-clampValue_, clampValue_]`范围内。
      5. `Mul(infloat_local, infloat_local, scale_local)`：乘以Per-Channel量化缩放因子。
      6. `Cast(infloat_local, infloat_local, CAST_RINT)`：FP32四舍五入到整数（仍以FP32格式存储）。
      7. `Mins`和`Maxs`：将值截断到`[-128.0, 127.0]`的INT8表示范围。
      8. `Cast(in_one_local, infloat_local, CAST_NONE)`和`Cast(out_local, in_one_local, CAST_RINT)`：将结果从FP32经FP16转换为最终INT8输出。
    - **CopyOut**：将INT8计算结果从UB搬回GM，输出偏移量：`offset = (i * blockNum_ + blockIdx_) * bkW_ + j * tlW_`。
