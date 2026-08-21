# MoeInitRoutingV4

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：MoE的routing计算，根据MoeGatingTopKSoftmax算子的计算结果做routing处理，支持非量化、静态量化和动态量化模式。本算子针对MoeInitRoutingV3算子做了如下功能变更，请根据实际情况选择合适的算子：

    1.active_num从属性改为可选输入，支持图模式下动态传入active_num值。

    2.新增可选输入topkWeight和可选输出expandedTopkWeightOut，支持对topkWeight按排序后的索引进行重排，使得expandedTopkWeightOut与expandedXOut一一对应。topkWeight与expandedTopkWeightOut必须同时传入或同时不传入。

    3.仅支持Ascend 950架构。

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
    - 静态quant：
        $$
        quantResult = round(x * scaleOptional + offsetOptional)
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

  5.对quantResult取前min(activeNum, NUM_ROWS*K)个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]/K]
    $$

  6.如果输入topkWeight，对topkWeight按sortedRowIdx做位置映射，得出expandedTopkWeightOut：

    $$
    expandedTopkWeightOut[i]=topkWeight[sortedRowIdx[i]]
    $$

    其中i的取值范围为[0, min(activeNum, availableIdxNum))。

  7.expandedRowIdxOut的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRangeOptional范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
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
        <td>x</td>
        <td>输入</td>
        <td>MOE的输入，即token特征输入，对应公式中x。</td>
        <td>FLOAT32、FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2。</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdx</td>
        <td>输入</td>
        <td>每一行特征对应的K个处理专家，里面元素专家id不能超过专家数。对应公式中expertIdx。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scaleOptional</td>
        <td>可选输入</td>
        <td>表示用于计算quant结果的参数。如果不输入表示计算时不使用scale，对应公式中scale。<br>• quantMode为1的INT8动态量化场景下，如果输入则要求为2D的Tensor，shape为(expertEnd-expertStart, H)。<br>• quantMode为13的INT4动态量化场景下，如果输入则要求shape为(1, H)，表示按H维广播的smooth scale。<br>• 仅quantMode=-1且x的数据类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2时，scale数据类型支持FLOAT8_E8M0。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>offsetOptional</td>
        <td>可选输入</td>
        <td>表示用于计算quant结果的偏移值。在非量化场景下和动态quant场景下不输入，对应公式中offsetOptional。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>activeNumOptional</td>
        <td>可选输入</td>
        <td>表示总的最大处理row数，输出expandedXOut只有这么多行是有效的。不输入时activeNum默认为NUM_ROWS*K；输入时为标量Tensor（shape为()或(1,)），数据类型为INT64，值大于等于0，0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>topkWeightOptional</td>
        <td>可选输入</td>
        <td>topk专家的路由权重，用于按排序索引重排以与expandedXOut一一对应。输入shape为(NUM_ROWS, K)，数据类型仅支持FLOAT32。topkWeight与expandedTopkWeightOut必须同时传入或同时不传入。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>属性</td>
        <td>表示每个专家能够处理的tokens数。Dropless场景下仅校验其值，不使用该参数；DropPad场景下取值范围为(0, NUM_ROWS]。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>属性</td>
        <td>表示专家数，expertTokensNumType为key_value模式时，取值范围为[1, 5120]，其它模式取值范围为[1, 10240]。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>属性</td>
        <td>表示是否为DropPad场景，取值为0 和1。<br>• 0：表示Dropless场景，该场景下不校验expertCapacity。<br>• 1：表示DropPad场景。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumType</td>
        <td>属性</td>
        <td>取值为0、1和2 。<br>• 0：表示comsum模式。<br>• 1：表示count模式，即输出的值为各个专家处理的token数量的累计值。<br>• 2：表示key\_value模式，即输出的值为专家和对应专家处理token数量的累计值。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumFlag</td>
        <td>属性</td>
        <td>取值为false和true。<br>• false：表示不输出expertTokensCountOrCumsumOut。<br>• true：表示输出expertTokensCountOrCumsumOut。</td>
        <td>Bool</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>属性</td>
        <td>取值为0、1、-1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17。<br>• 0：表示静态quant场景。<br>• 1：表示动态quant场景，expandedXOut量化到INT8。<br>• -1：表示非量化场景。<br>• 2：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E5M2。<br>• 3：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E4M3FN。<br>• 4：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），expandedXOut量化到FLOAT8_E5M2。<br>• 5：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），expandedXOut量化到FLOAT8_E4M3FN。<br>• 6：表示HIF8直转量化场景下，expandedXOut量化到HIFLOAT8。<br>• 7：表示HIF8 PERTENSOR量化场景，expandedXOut量化到HIFLOAT8。<br>• 8：表示HIF8 PERTOKEN量化场景，expandedXOut量化到HIFLOAT8。<br>• 9：表示MXFP4量化场景，expandedXOut量化到FLOAT4_E2M1。<br>• 11：表示FP8 PerBlock量化场景（BlockSize=128），expandedXOut量化到FLOAT8_E5M2，expandedScaleOut为FLOAT32三维布局。<br>• 12：表示FP8 PerBlock量化场景（BlockSize=128），expandedXOut量化到FLOAT8_E4M3FN，expandedScaleOut为FLOAT32三维布局。<br>• 13：表示INT4动态量化场景，expandedXOut量化到INT4。<br>• 14：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），expandedXOut量化到FLOAT8_E5M2。<br>• 15：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），expandedXOut量化到FLOAT8_E4M3FN。<br>• 16：表示MXFP8 RoundScale+Amax量化场景，expandedXOut量化到FLOAT8_E5M2。<br>• 17：表示MXFP8 RoundScale+Amax量化场景，expandedXOut量化到FLOAT8_E4M3FN。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeExpertRangeOptional</td>
        <td>可选属性</td>
        <td>长度为2，数组内的值为[expertStart, expertEnd],表示活跃的expert范围在expertStart和expertEnd之间，左闭右开。要求值大于等于0，并且expertEnd不大于expertNum。</td>
        <td>ListInt</td>
        <td>-</td>
      </tr>
      <tr>
        <td>rowIdxType</td>
        <td>属性</td>
        <td>表示expandedRowIdxOut使用的索引类型，取值为0、1。<br>• 0：表示gather类型的索引。<br>• 1：表示scatter类型的索引。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedXOut</td>
        <td>输出</td>
        <td>根据expertIdx进行扩展过的特征。非量化场景下数据类型同x，量化场景quantMode为0、1时数据类型支持INT8，quantMode为13且x数据类型为FLOAT32或BFLOAT16时数据类型支持INT4，quantMode为2、4、14、16时数据类型支持FLOAT8_E5M2，quantMode为3、5、15、17时数据类型支持FLOAT8_E4M3FN，quantMode为6、7、8时数据类型支持HIFLOAT8，quantMode为9时数据类型支持FLOAT4_E2M1，quantMode为11、12时数据类型分别支持FLOAT8_E5M2、FLOAT8_E4M3FN。
        <br>• Dropless场景shape为[NUM_ROWS * K, H]。<br>• Active场景shape为[min(activeNum, NUM_ROWS * K), H]。<br>• DropPad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。</td>
        <td>FLOAT32、FLOAT16、BFLOAT16、INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdxOut</td>
        <td>输出</td>
        <td>expandedXOut和x的索引映射关系，shape为[NUM_ROWS*K]，前availableIdxNum个元素为有效数据，其余无效数据，当rowIdxType为0时，无效数据由-1填充；当rowIdxType为1时，无效数据未初始化。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOut</td>
        <td>输出</td>
        <td>当expertTokensNumFlag为true时输出有效数据，为false时输出为空tensor。<br>• 在expertTokensNumType为0或1的场景下，表示activeExpertRangeOptional范围内expert对应的处理token的总数，输出shape为[expertEnd-expertStart]。<br>• 在expertTokensNumType为2的场景下，表示activeExpertRangeOptional范围内token总数为非0的expert，以及对应expert处理token的总数，输出shape为[expertNum, 2]。</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedScaleOut</td>
        <td>输出</td>
        <td>输出量化计算过程中scaleOptional的中间值。<br>• 非量化场景下，当scaleOptional输入时，shape为[NUM_ROWS*K, 1]，输出FLOAT32类型；当输入x数据类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2时，如果scaleOptional输入，则shape为[NUM_ROWS*K, CeilDiv(H, 64), 2]，输出FLOAT8_E8M0类型。DropPad场景下shape为[expertNum * expertCapacity]，输出FLOAT32类型。<br>• 动态量化场景下（quantMode为1），当scaleOptional输入时，shape为[NUM_ROWS*K]，输出FLOAT32类型。<br>• 静态量化场景下（quantMode为0）、HIF8直转量化场景下（quantMode为6）、HIF8 PERTENSOR量化场景下（quantMode为7），输出为空tensor。<br>• HIF8 PERTOKEN量化场景下（quantMode为8），shape为[NUM_ROWS*K]，输出FLOAT32类型。<br>• MXFP8量化场景下（quantMode为2、3、16、17），输出FLOAT8_E8M0类型，Shape为[NUM_ROWS*K, M]，其中M=CeilAlign(CeilDiv(H,32),2)。<br>• MXFP4量化场景下（quantMode为9），输出FLOAT8_E8M0类型，Shape为[NUM_ROWS*K, M, 2]，其中M=CeilDiv(H, 64)。<br>• FP8 PerGroup量化场景下（quantMode为4、5、14、15），输出FLOAT32类型，Shape为[NUM_ROWS*K, CeilDiv(H, 128)]。<br>• FP8 PerBlock量化场景下（quantMode为11、12），输出FLOAT32类型，Shape为[NUM_ROWS*K, CeilDiv(H, 256), 2]。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedTopkWeightOut</td>
        <td>可选输出</td>
        <td>按排序索引重排后的路由权重，与expandedXOut一一对应。topkWeight输入时必须同时输出。<br>• Dropless场景下shape为[min(activeNum, NUM_ROWS*K), 1]，前effectiveNum个元素为有效数据，其余未初始化，effectiveNum=min(activeNum, availableIdxNum)。<br>• DropPad场景下shape为[expertNum*expertCapacity, 1]，已分配容量的位置为有效数据，未分配容量的位置填充为0。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- expertCapacity在Dropless场景下仅校验其值，不使用该参数；在DropPad场景下必须校验且取值范围为(0, NUM_ROWS]。
- 支持quantMode为13的INT4动态量化场景，需同时满足：
  - x数据类型为FLOAT32或BFLOAT16，expandedXOut数据类型为INT4。
  - H为偶数，用于沿H维每两个INT4值打包为1个字节。
  - scaleOptional不输入，或输入shape为(1, H)、数据类型为FLOAT32；offsetOptional不输入。
- DropPad模式特殊约束（dropPadMode=1时）：
  - quantMode仅支持-1（非量化），且数据类型仅支持FLOAT16、BFLOAT16、FLOAT32、INT8、HIFLOAT8。
  - rowIdxType仅支持0（gather索引）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_init_routing_v4](examples/test_aclnn_moe_init_routing_v4.cpp) | 通过[aclnnMoeInitRoutingV4](docs/aclnnMoeInitRoutingV4.md)接口方式调用MoeInitRoutingV4算子。 |
| 图模式     | [test_geir_moe_init_routing_v4](examples/test_geir_moe_init_routing_v4.cpp) | 通过[算子IR](op_graph/moe_init_routing_v4_proto.h)构图方式调用MoeInitRoutingV4算子。 |
