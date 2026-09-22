# 算子名称：LayerNormAddAddQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2训练系列产品</term>                   | 是       |

## 功能说明

- 算子功能：在大语言模型的Transformer层中，将残差加法、层归一化和动态量化三个操作融合为一个Kernel。具体而言，先对两个输入进行逐元素相加（其中inTwo为按行广播的偏置），再与第三个输入相加，随后对相加结果执行LayerNorm（乘gamma加beta），最后对归一化结果乘以缩放因子scale并量化截断为int8输出。同时，保留相加后的fp16中间结果供后续网络使用。

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
      <td>inOne</td>
      <td>输入</td>
      <td>主输入张量，shape为(gbH, gbW)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inTwo</td>
      <td>输入</td>
      <td>偏置向量，shape为(gbW,)，在行方向广播与inOne相加</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inThr</td>
      <td>输入</td>
      <td>残差输入张量，shape为(gbH, gbW)，与(inOne + inTwo)相加</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>LayerNorm的缩放参数，shape为(gbW,)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>LayerNorm的平移参数，shape为(gbW,)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化缩放因子，shape为(gbW,)</td>
      <td>float32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outLynQuant</td>
      <td>输出</td>
      <td>量化后的int8结果，shape为(gbH, gbW)</td>
      <td>int8_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outAdd</td>
      <td>输出</td>
      <td>相加后的fp16中间结果(inOne + inTwo + inThr)，shape为(gbH, gbW)</td>
      <td>float16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gbH</td>
      <td>属性</td>
      <td>输入张量的行数（token数量）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gbW</td>
      <td>属性</td>
      <td>输入张量的列数（隐藏维度）</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>LayerNorm防除零极小值</td>
      <td>float32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>constrait</td>
      <td>属性</td>
      <td>是否在量化前对中间浮点结果进行 [-128, 128] 的截断约束</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

## 价值/作用

在LLM推理中，Attention输出与残差分支的相加、LayerNorm以及KV Cache的量化写入是极为频繁的操作。如果分步执行，相加结果需要写回GM再由LN算子读入，LN结果再写回GM供Quant算子读入。本算子将这三步合一：

避免了outAdd和LN中间结果的GM写回与重新搬运，极大节省显存带宽。
将LN计算提升到float32精度保证数值稳定性，但中间变量均在UB中流转，无需额外的GM显存分配。
一次性输出fp16残差结果和int8量化结果，完美契合推理引擎中“残差继续前向传播 + KV Cache量化存储”的双重需求。

## 设计方案

### Tiling策略

- **分核策略**：
  按行（gbH）维度进行分核。每个Core处理 ⌈gbH / blockNum⌉ 行数据。尾部行通过i * blockNum_ + blockIdx_ < gbH判断跳过。

- **分块策略**：
  当前列方向不进行分块（整行处理），bkW_ = gbW，按64元素向上对齐得到bkAlignW_。所有输入输出按整行搬入搬出。

### Kernel侧设计

进行 **Init** 和 **Process** 两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、数据搬出（CopyOut）三个阶段。采用单缓冲（BUFFER_NUM = 1）机制，1D参数（inTwo, gamma, beta, scale）在Init阶段搬入UB并常驻，所有行计算复用，减少GM读取次数。

- **初始化（Init）**
  - 计算分核与对齐参数。
  - 搬入1D参数（inTwo, gamma, beta, scale）到UB并出队保存引用，供后续行循环复用。
  - 对inTwo的尾部对齐补零。

- **计算流程（Process）**
  - FOR i = 0 TO bkLoop_（行方向循环）：
      - CopyIn：从GM搬入当前行的inOne和inThr到UB，尾部补零。
      - Compute：
      - 计算LayerNormNormAddAdd
      - 计算量化结果
      - 将out_add (half)和out_lyn_quant (int8_t)搬回GM对应行

- **数据搬入（CopyIn）**
  - 从GM搬入当前行的inOne和inThr到UB，尾部补零。

- **数据搬出（CopyOut）**
  - 将out_add (half)和out_lyn_quant (int8_t)搬回GM对应行。
