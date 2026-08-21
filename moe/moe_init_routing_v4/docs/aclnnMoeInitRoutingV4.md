# aclnnMoeInitRoutingV4

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_init_routing_v4)

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

- 接口功能：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持非量化、静态量化和动态量化模式。本接口针对V3接口[aclnnMoeInitRoutingV3](../../moe_init_routing_v3/docs/aclnnMoeInitRoutingV3.md)做出如下功能变更，请根据实际情况选择合适的接口：<br>
  <ol><li>active_num从属性改为可选输入，支持图模式下动态传入active_num值。</li><li>新增可选输入topkWeight和可选输出expandedTopkWeightOut，支持对topkWeight按排序后的索引进行重排，使得expandedTopkWeightOut与expandedXOut一一对应。topkWeight与expandedTopkWeightOut必须同时传入或同时不传入。</li></ol>

- 计算公式：

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
    - rowIdxType等于1时, 输出scatter索引

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - rowIdxType等于0时, 输出gather索引

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.如果quantMode不等于-1, 计算quant结果：
     - 静态quant

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$

    - 动态quant：
        - 若不输入scale：

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 若输入scale:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 当quantMode为13时，动态量化使用对称量化范围[-8, 7]，scale计算中的分母为7，量化结果沿H维每两个INT4值打包为1个字节。

  5.若活跃的expert范围为全专家范围时，按照Scatter索引搬运token；反之按照Gather索引搬运token。在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedXOut：
    - 非量化场景
      - 按照Scatter索引搬运

      $$
      expandedXOut[i]=x[scatterRowIdx[i] // K]
      $$

      - 按照Gather索引搬运

      $$
      expandedXOut[gatherRowIdx[i]]=x[i // K]
      $$

    - 量化场景
      - 按照Scatter索引搬运

      $$
      expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
      $$

      - 按照Gather索引搬运

      $$
      expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
      $$

  6.若输入topkWeight，按照排序后的索引对topkWeight进行重排，使得expandedTopkWeightOut与expandedXOut一一对应：
    - Dropless场景（dropPadMode=0）

      $$
      expandedTopkWeightOut[i] = topkWeight[sortedRowIdx[i]], \quad 0 \le i < effectiveNum
      $$

      其中effectiveNum为：

      $$
      effectiveNum = \min(activeNum, availableIdxNum)
      $$

    - DropPad场景（dropPadMode=1）

      $$
      expandedTopkWeightOut[dstIdx] = topkWeight[globalSortIdx], \quad dstIdx \ne -1
      $$

      其中globalSortIdx为expertIdx在flatten布局中的位置（即globalSortIdx = rowIdx * K + colIdx，对应topkWeight[rowIdx, colIdx]），dstIdx为expandedRowIdxOut中对应位置的目标索引，未被分配的专家容量位置expandedTopkWeightOut填充为0。

  7.expandedRowIdxOut的有效元素数量availableIdxNum，计算方式为expertIdx中activeExpertRangeOptional范围内的元素的个数

    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 "aclnnMoeInitRoutingV4GetWorkspaceSize"接口获取入参并计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeInitRoutingV4"接口执行计算。

```Cpp
aclnnStatus aclnnMoeInitRoutingV4GetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *expertIdx,
  const aclTensor   *scaleOptional,
  const aclTensor   *offsetOptional,
  const aclTensor   *activeNumOptional,
  const aclTensor   *topkWeightOptional,
  int64_t            expertCapacity,
  int64_t            expertNum,
  int64_t            dropPadMode,
  int64_t            expertTokensNumType,
  bool               expertTokensNumFlag,
  int64_t            quantMode,
  const aclIntArray *activeExpertRangeOptional,
  int64_t            rowIdxType,
  const aclTensor   *expandedXOut,
  const aclTensor   *expandedRowIdxOut,
  const aclTensor   *expertTokensCountOrCumsumOut,
  const aclTensor   *expandedScaleOut,
  const aclTensor   *expandedTopkWeightOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnMoeInitRoutingV4(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMoeInitRoutingV4GetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 158px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 400px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
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
      <td>x（aclTensor）</td>
      <td>输入</td>
      <td>MOE的输入，即token特征输入</td>
      <td>shape为(NUM_ROWS, H)</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8、HIFLOAT8、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdx（aclTensor）</td>
      <td>输入</td>
      <td>每一行特征对应的K个处理专家，里面元素专家id不能超过专家数</td>
      <td>shape为(NUM_ROWS, K)</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示用于计算量化结果的参数</td>
      <td><ul>
        <li>如果不输入表示计算时不使用scale;</li>
        <li>非量化场景下为可选输入，如果输入则要求为1D的Tensor，shape为(NUM_ROWS,)，类型为FLOAT32。当输入x数据类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2时，如果输入则要求3D的Tensor，shape为(NUM_ROWS, CeilDiv(H, 64), 2), 类型为FLOAT8_E8M0;</li>
        <li>静态量化场景必须输入，输入要求为1D的Tensor，shape为[1, ]；</li>
        <li>quantMode为1的INT8动态量化场景下为可选输入，如果输入则要求为2D的Tensor，shape为(expertEnd-expertStart, H)；quantMode为13的INT4动态量化场景下为可选输入，如果输入则要求shape为(1, H)，表示按H维广播的smooth scale。</li>
        <li>MXFP8量化场景下（quantMode为2、3）不输入。</li>
        <li>HIF8直转和HIF8 PERTOKEN量化场景下（quantMode为6、8）不输入。</li>
        <li>HIF8 PERTENSOR量化场景下（quantMode为7）,输入要求为1D的Tensor，shape为[1, ]。</li>
        <li>MXFP4量化场景下（quantMode为9）不输入。</li>
        <li>FP8 PerBlock量化场景下（quantMode为11、12）不输入。</li>
        </ul></td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示用于计算quant结果的偏移值</td>
      <td><ul>
        <li>在非量化场景下不输入;</li><li>静态量化场景必须输入，输入要求为1D的Tensor，shape为[1, ]；</li>
        <li>动态量化、MXFP8量化、HIF8量化、MXFP4量化场景下不输入。</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNumOptional（aclTensor）</td>
      <td>输入</td>
      <td>表示总的最大处理row数，输出expandedXOut只有这么多行是有效的</td>
      <td><ul>
        <li>可选输入，不输入时activeNum默认为NUM_ROWS*K；</li>
        <li>输入时为标量Tensor（shape为()或(1,)），数据类型为INT64，值大于等于0，0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>0-1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkWeightOptional（aclTensor）</td>
      <td>输入</td>
      <td>topk专家的路由权重，用于按排序索引重排以与expandedXOut一一对应</td>
      <td><ul>
        <li>可选输入，不输入表示不输出expandedTopkWeightOut；</li>
        <li>输入shape为(NUM_ROWS, K)，数据类型仅支持FLOAT32。</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity（int64_t）</td>
      <td>输入</td>
      <td>表示每个专家能够处理的tokens数</td>
      <td>Dropless场景下仅校验其值，不使用该参数；DropPad场景下取值范围为(0, NUM_ROWS]。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum（int64_t）</td>
      <td>输入</td>
      <td>表示专家数</td>
      <td>expertTokensNumType为key_value模式时，取值范围为[1, 5120]，其它模式取值范围为[1, 10240]</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode（int64_t）</td>
      <td>输入</td>
      <td>表示是否为DropPad场景</td>
      <td>取值为0和1
        <br>0：表示Dropless场景，该场景下不校验expertCapacity；
        <br>1：表示DropPad场景；
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumType（int64_t）</td>
      <td>输入</td>
      <td>表示直方图的不同模式</td>
      <td>取值为0、1和2
        <br>0：表示 comsum 模式；
        <br>1：表示 count 模式；
        <br>2：表示 key_value 模式；
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumFlag（bool）</td>
      <td>输入</td>
      <td>表示是否输出 expertTokensCountOrCumsumOut </td>
      <td>取值为false和true</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>表示不同量化场景</td>
      <td>取值为0、1、-1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17
        <br>0：表示静态 quant 场景;
        <br>1：表示动态 quant 场景，expandedXOut量化到INT8;
        <br>-1：表示非量化场景;
        <br>2：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E5M2;
        <br>3：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E4M3FN;
        <br>4：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），expandedXOut量化到FLOAT8_E5M2;
        <br>5：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），expandedXOut量化到FLOAT8_E4M3FN;
        <br>6：表示HIF8直转量化场景，expandedXOut量化到HIFLOAT8;
        <br>7：表示HIF8 PERTENSOR量化场景，expandedXOut按照pertensor模式量化到HIFLOAT8;
        <br>8：表示HIF8 PERTOKEN量化场景，expandedXOut按照pertoken模式量化到HIFLOAT8;
        <br>9：表示MXFP4量化场景，expandedXOut量化到FLOAT4_E2M1;
        <br>11：表示FP8 PerBlock量化场景（BlockSize=128），expandedXOut量化到FLOAT8_E5M2，expandedScaleOut为FLOAT32三维布局;
        <br>12：表示FP8 PerBlock量化场景（BlockSize=128），expandedXOut量化到FLOAT8_E4M3FN，expandedScaleOut为FLOAT32三维布局;
        <br>13：表示INT4动态量化场景，expandedXOut量化到INT4;
        <br>14：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），expandedXOut量化到FLOAT8_E5M2;
        <br>15：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），expandedXOut量化到FLOAT8_E4M3FN;
        <br>16：表示MXFP8 RoundScale+Amax量化场景，expandedXOut量化到FLOAT8_E5M2;
        <br>17：表示MXFP8 RoundScale+Amax量化场景，expandedXOut量化到FLOAT8_E4M3FN;
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeExpertRangeOptional（aclIntArray）</td>
      <td>输入</td>
      <td>表示活跃的expert范围</td>
      <td>长度为2，数组内的值为[expertStart, expertEnd]，左闭右开，要求值大于等于0，并且expertEnd不大于expertNum；Drop/Pad场景下，expertStart等于0, expertEnd等于expertNum。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIdxType（int64_t）</td>
      <td>输入</td>
      <td>表示expandedRowIdxOut使用的索引类型</td>
      <td>取值为0、1
        <br>0：表示gather类型的索引
        <br>1：表示scatter类型的索引</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut（aclTensor）</td>
      <td>输出</td>
      <td>根据expertIdx进行扩展过的特征</td>
      <td><ul>
        <li>Dropless场景shape为[NUM_ROWS * K, H]。</li>
        <li>Active场景shape为[min(activeNum, NUM_ROWS * K), H]。</li>
        <li>Drop/Pad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。</li>
        <li>非量化场景下数据类型同x，量化场景quantMode为0、1时数据类型支持INT8，quantMode为13且x数据类型为FLOAT32或BFLOAT16时数据类型支持INT4，quantMode为2、4、14、16时数据类型支持FLOAT8_E5M2，quantMode为3、5、15、17时数据类型支持FLOAT8_E4M3FN，quantMode为6、7、8时数据类型支持HIFLOAT8，quantMode为9时数据类型支持FLOAT4_E2M1，quantMode为11、12时数据类型分别支持FLOAT8_E5M2、FLOAT8_E4M3FN。</li>
      </ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut（aclTensor）</td>
      <td>输出</td>
      <td>expandedXOut和x的索引映射关系</td>
      <td>输出shape为(NUM_ROWS*K, )， 前availableIdxNum个元素为有效数据，其余无效数据由rowIdxType决定：
        <ul><li>当rowIdxType为0时，无效数据由-1填充</li>
        <li>当rowIdxType为1时，无效数据未初始化</li></ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut（aclTensor）</td>
      <td>输出</td>
      <td>输出每个专家处理的token数量的统计结果或累加值</td>
      <td><ul>
        <li>当expertTokensNumFlag为false时，输出为空tensor。</li>
        <li>在expertTokensNumType为0时，表示activeExpertRangeOptional范围内expert在排序后处理token总数的前缀和，输出shape为[expertEnd-expertStart]。</li>
        <li>在expertTokensNumType为1时，表示activeExpertRangeOptional范围内expert对应的处理token的总数，输出shape为[expertEnd-expertStart]。</li>
        <li>在expertTokensNumType为2时，表示activeExpertRangeOptional范围内token总数为非0的expert，以及对应expert处理token的总数，输出shape为[expertNum, 2]。</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedScaleOut（aclTensor）</td>
      <td>输出</td>
      <td>输出不同量化过程中scaleOptional的中间值。</td>
      <td> 输出shape为expandedXOut的shape去掉最后一维之后所有维度的乘积。
        <ul style="list-style-type: circle;">
        <li>非量化场景下，当scaleOptional输入时，shape为[NUM_ROWS*K, 1]，前availableIdxNum个元素为有效数据，输出FLOAT32类型。当输入x数据类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2时，如果scaleOptional输入，则expandedScaleOut的shape为[NUM_ROWS*K, CeilDiv(H, 64), 2]，输出FLOAT8_E8M0类型。当Drop/Pad场景输出是一个1D的Tensor，shape为[expertNum * expertCapacity]，输出FLOAT32类型。</li>
        <li>动态量化场景下，当scaleOptional输入时，前availableIdxNum个元素为有效数据。</li>
        <li>静态量化场景下（quantMode为0）、HIF8直转量化场景下（quantMode为6）、HIF8 PERTENSOR量化场景下（quantMode为7），expandedScaleOut不输出。</li>
        <li>MXFP8量化场景下（quantMode为2、3、16、17），输出FLOAT8_E8M0类型，Shape为[NUM_ROWS*K, M]，其中M=CeilAlign(CeilDiv(H,32),2)，NUM_ROWS*K的前availableIdxNum行为有效数据。</li>
        <li>按照直转方式量化到HIFLOAT8场景下，expandedScaleOut不输出。</li>
        <li>按照PERTENSOR模式量化到HIFLOAT8场景下，expandedScaleOut不输出。</li>
        <li>按照PERTOKEN模式量化到HIFLOAT8场景下，输出FLOAT32类型，Shape为[NUM_ROWS*K, 1]。</li>
        <li>MXFP4量化场景下，输出FLOAT8_E8M0类型，Shape为[NUM_ROWS*K, M, 2]，其中M=CeilDiv(H, 64)，NUM_ROWS*K的前availableIdxNum行为有效数据。</li>
        <li>FP8 PerGroup量化场景下（quantMode为4、5、14、15），输出FLOAT32类型，Shape为[NUM_ROWS*K, CeilDiv(H, 128)]，NUM_ROWS*K的前availableIdxNum行为有效数据。</li>
        <li>FP8 PerBlock量化场景下（quantMode为11、12），输出FLOAT32类型，Shape为[NUM_ROWS*K, CeilDiv(H,256), 2]，NUM_ROWS*K的前availableIdxNum行为有效数据。</li></ul>
      </td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedTopkWeightOut（aclTensor）</td>
      <td>输出</td>
      <td>按排序索引重排后的路由权重，与expandedXOut一一对应</td>
      <td><ul>
        <li>可选输出，topkWeight输入时必须同时输出；</li>
        <li>Dropless场景下shape为[min(activeNum, NUM_ROWS*K), 1]，前effectiveNum个元素为有效数据，其余未初始化，effectiveNum=min(activeNum, availableIdxNum)；</li>
        <li>DropPad场景下shape为[expertNum*expertCapacity, 1]，已分配容量的位置为有效数据，未分配容量的位置填充为0；</li>
        <li>数据类型仅支持FLOAT32。</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

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
      <td>计算输入和计算输出是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>
      输入、输出Tensor的shape不在支持的范围内。<br />
      输入的属性不在支持的范围内。<br />
      </td>
    </tr>
  </tbody></table>

## aclnnMoeInitRoutingV4

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeInitRoutingV4GetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnMoeInitRoutingV4默认确定性实现。

- 该算子在以下产品型号上支持性能模板，需要额外满足准入条件，否则进入通用模板：
  - 支持性能模板的产品：
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：支持
    <!-- end id7 -->
  - 性能模板的准入条件：
    <table>
      <tr align="center">
        <th style="text-align: center;">性能模板类型</th>
        <th style="text-align: center;">准入条件</th>
      </tr>
      <tr>
        <td align="center">全载性能模板</td>
        <td>在算子输入shape较小的场景，操作间的多核同步时间占比较高，成为性能瓶颈。因此，针对这种特化场景，添加性能模板。该模板中，搬入、排序、计算都在同一个kernel内完成。需要满足如下条件：<ul style="list-style-type: circle;"><li>属性要求：dropPadMode=0；非量化场景还支持dropPadMode=1，此时quantMode=-1、rowIdxType=0、activeExpertRange=[0, expertNum]</li></ul></td>
      </tr>
    </table>

- DropPad模式特殊约束（dropPadMode=1时）：
  - quantMode仅支持-1（非量化）。
  - rowIdxType仅支持0（gather索引）。
  - activeExpertRangeOptional必须为[0, expertNum]。

- quantMode为13的INT4动态量化场景，需同时满足：x数据类型为FLOAT32或BFLOAT16；H为偶数。

- 空tensor处理：
  - 当输入的x首个维度的值为0时，DropPadMode必须为0, 进入空tensor模板。expandedXOut、expandedRowIdxOut和expandedScaleOut的返回值为空tensor，expertTokensCountOrCumsumOut返回全0的tensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v4.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream *stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}
int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 2};
    std::vector<int64_t> expertIdxShape = {3, 4};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> offsetShape = {1};
    std::vector<int64_t> activeNumShape = {};
    std::vector<int64_t> topkWeightShape = {3, 4};

    std::vector<int64_t> expandedXOutShape = {12, 2};
    std::vector<int64_t> expandedRowIdxOutShape = {12};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalShape = {4};
    std::vector<int64_t> expandedScaleOutOptionalShape = {};
    std::vector<int64_t> expandedTopkWeightOutShape = {12, 1};

    std::vector<int64_t> activeExpertRangeArray = {0, 4};

    void *xDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;
    void *activeNumDeviceAddr = nullptr;
    void *topkWeightDeviceAddr = nullptr;

    void *expandedXOutDeviceAddr = nullptr;
    void *expandedRowIdxOutDeviceAddr = nullptr;
    void *expertTokensCountOrCumsumOutOptionalDeviceAddr = nullptr;
    void *expandedScaleOutOptionalDeviceAddr = nullptr;
    void *expandedTopkWeightOutDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;
    aclTensor *activeNumOptional = nullptr;
    aclTensor *topkWeightOptional = nullptr;

    int64_t expertCapacity = 0;
    int64_t expertNum = 256;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    bool expertTokensNumFlag = true;
    int64_t quantMode = 0;
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());
    int64_t rowIdxType = 1;

    aclTensor *expandedXOut = nullptr;
    aclTensor *expandedRowIdxOut = nullptr;
    aclTensor *expertTokensCountOrCumsumOutOptional = nullptr;
    aclTensor *expandedScaleOutOptional = nullptr;
    aclTensor *expandedTopkWeightOut = nullptr;

    std::vector<float> xHostData = {0.1, 0.1, 0.2, 0.2, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 3, 0, 2, 1, 3, 0, 1, 3, 2};
    std::vector<float> scaleHostData = {0.3423, 0.1652, 0.2652};
    std::vector<float> offsetHostData = {1.8369};
    std::vector<int64_t> activeNumHostData = {12};
    std::vector<float> topkWeightHostData = {0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65};

    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalHostData = {0, 0, 0, 0};
    std::vector<float> expandedScaleOutOptionalHostData = {};
    std::vector<float> expandedTopkWeightOutHostData(expandedTopkWeightOutShape[0] * expandedTopkWeightOutShape[1], 0.0f);

    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(activeNumHostData, activeNumShape, &activeNumDeviceAddr, aclDataType::ACL_INT64, &activeNumOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkWeightHostData, topkWeightShape, &topkWeightDeviceAddr, aclDataType::ACL_FLOAT, &topkWeightOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(
        expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData,
        expandedRowIdxOutShape,
        &expandedRowIdxOutDeviceAddr,
        aclDataType::ACL_INT32,
        &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensCountOrCumsumOutOptionalHostData,
        expertTokensCountOrCumsumOutOptionalShape,
        &expertTokensCountOrCumsumOutOptionalDeviceAddr,
        aclDataType::ACL_INT64,
        &expertTokensCountOrCumsumOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedScaleOutOptionalHostData,
        expandedScaleOutOptionalShape,
        &expandedScaleOutOptionalDeviceAddr,
        aclDataType::ACL_FLOAT,
        &expandedScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedTopkWeightOutHostData, expandedTopkWeightOutShape, &expandedTopkWeightOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedTopkWeightOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnMoeInitRoutingV4第一段接口
    ret = aclnnMoeInitRoutingV4GetWorkspaceSize(x,
        expertIdx,
        scale,
        offset,
        activeNumOptional,
        topkWeightOptional,
        expertCapacity,
        expertNum,
        dropPadMode,
        expertTokensNumType,
        expertTokensNumFlag,
        quantMode,
        activeExpertRange,
        rowIdxType,
        expandedXOut,
        expandedRowIdxOut,
        expertTokensCountOrCumsumOutOptional,
        expandedScaleOutOptional,
        expandedTopkWeightOut,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV4GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeInitRoutingV4第二段接口
    ret = aclnnMoeInitRoutingV4(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV4 failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(),
        expandedXData.size() * sizeof(expandedXData[0]),
        expandedXOutDeviceAddr,
        expandedXSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(expandedRowIdxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(),
        expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]),
        expandedRowIdxOutDeviceAddr,
        expandedRowIdxSize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokensCountOrCumsumOutOptionalShape);
    std::vector<int64_t> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(),
        expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]),
        expertTokensCountOrCumsumOutOptionalDeviceAddr,
        expertTokensBeforeCapacitySize * sizeof(int64_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %ld\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(expandedScaleOutOptionalShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(),
        dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]),
        expandedScaleOutOptionalDeviceAddr,
        dynamicQuantScaleSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dynamicQuantScaleSize; i++) {
        LOG_PRINT("dynamicQuantScaleData[%ld] is: %f\n", i, dynamicQuantScaleData[i]);
    }
    auto expandedTopkWeightSize = GetShapeSize(expandedTopkWeightOutShape);
    std::vector<float> expandedTopkWeightData(expandedTopkWeightSize, 0);
    ret = aclrtMemcpy(expandedTopkWeightData.data(),
        expandedTopkWeightData.size() * sizeof(float),
        expandedTopkWeightOutDeviceAddr,
        expandedTopkWeightSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedTopkWeightSize; i++) {
        LOG_PRINT("expandedTopkWeightData[%ld] is: %f\n", i, expandedTopkWeightData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(activeNumOptional);
    aclDestroyTensor(topkWeightOptional);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensCountOrCumsumOutOptional);
    aclDestroyTensor(expandedScaleOutOptional);
    aclDestroyTensor(expandedTopkWeightOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(activeNumDeviceAddr);
    aclrtFree(topkWeightDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokensCountOrCumsumOutOptionalDeviceAddr);
    aclrtFree(expandedScaleOutOptionalDeviceAddr);
    aclrtFree(expandedTopkWeightOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
