# MoeInitRoutingV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持不量化和动态量化模式。本接口针对V2接口[aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)做了如下功能变更，请根据实际情况选择合适的接口：

    1.增加动态量化功能，支持输出expendX的int8动态量化输出。

    2.增加参数activeExpertRangeOptional，支持筛选有效范围内的expertId。

    3.删除属性expertTokensBeforeCapacityFlag、删除输出expertTokensBeforeCapacityOut（使用expertTokensCountOrCumsumOut进行输出）。

- 计算公式：

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.计算quant结果：
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

  5.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRangeOptional范围内的元素的个数，-1也不在该范围内，表示无效专家，不参与路由计算：
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
        <td>每一行特征对应的K个处理专家，里面元素专家id不能超过专家数，-1表示无效专家（该位置不参与路由，会被过滤）。对应公式中expertIdx。</td>
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
        <td>activeNum</td>
        <td>属性</td>
        <td>表示总的最大处理row数，输出expandedXOut只有这么多行是有效的。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>属性</td>
        <td>表示每个专家能够处理的tokens数，取值范围大于等于0。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>属性</td>
        <td>表示专家数，expertTokensNumType为key\_value模式时，取值范围为[0, 5120],其它模式取值范围[0, 10240]。</td>
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
        <td>取值为0、1、-1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17。<br>• 0：表示静态quant场景。<br>• 1：表示动态quant场景，expandedXOut量化到INT8。<br>• -1：表示不量化场景。<br>• 2：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E5M2。<br>• 3：表示MXFP8量化场景，expandedXOut量化到FLOAT8_E4M3FN。<br>• 4：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），expandedXOut量化到FLOAT8_E5M2。<br>• 5：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），expandedXOut量化到FLOAT8_E4M3FN。<br>• 6：表示HIF8直转量化场景下，expandedXOut量化到HIFLOAT8。<br>• 7：表示HIF8 PERTENSOR量化场景，expandedXOut量化到HIFLOAT8。<br>• 8：表示HIF8 PERTOKEN量化场景，expandedXOut量化到HIFLOAT8。<br>• 9：表示MXFP4量化场景，expandedXOut量化到FLOAT4_E2M1。<br>• 11：表示FP8 PerBlock量化场景（BlockSize=128），expandedXOut量化到FLOAT8_E5M2，expandedScaleOut为FLOAT32三维布局。<br>• 12：表示FP8 PerBlock量化场景（BlockSize=128），expandedXOut量化到FLOAT8_E4M3FN，expandedScaleOut为FLOAT32三维布局。<br>• 13：表示INT4动态量化场景，expandedXOut量化到INT4。<br>• 14：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），expandedXOut量化到FLOAT8_E5M2。<br>• 15：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），expandedXOut量化到FLOAT8_E4M3FN。<br>• 16：表示MXFP8 RoundScale+Amax量化场景，expandedXOut量化到FLOAT8_E5M2。<br>• 17：表示MXFP8 RoundScale+Amax量化场景，expandedXOut量化到FLOAT8_E4M3FN。</td>
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
        <td>表示expandedRowIdxOut使用的索引类型，取值为0、1。（性能模板仅支持1）<br>• 0：表示gather类型的索引。<br>• 1：表示scatter类型的索引。</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedXOut</td>
        <td>输出</td>
        <td>根据expertIdx进行扩展过的特征。非量化场景下数据类型同x，量化场景quantMode为0、1时数据类型支持INT8，quantMode为13且x数据类型为FLOAT32或BFLOAT16时数据类型支持INT4，quantMode为2、4、14、16时数据类型支持FLOAT8_E5M2，quantMode为3、5、15、17时数据类型支持FLOAT8_E4M3FN，quantMode为6、7、8时数据类型支持HIFLOAT8，quantMode为9时数据类型支持FLOAT4_E2M1，quantMode为11、12时数据类型分别支持FLOAT8_E5M2、FLOAT8_E4M3FN。
        <br>• Dropless场景shape为[NUM_ROWS * K, H]。<br>• DropPad场景下要求是一个3D的Tensor，shape为[expertNum, expertCapacity, H]。</td>
        <td>FLOAT32、FLOAT16、BFLOAT16、INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdxOut</td>
        <td>输出</td>
        <td>expandedXOut和x的索引映射关系，前availableIdxNum\*H个元素为有效数据，其余无效数据，当rowIdxType为0时，无效数据由-1填充；当rowIdxType为1时，无效数据未初始化。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOut</td>
        <td>输出</td>
        <td>• 在expertTokensNumType为1的场景下，表示activeExpertRangeOptional范围内expert对应的处理token的总数。<br>• 在expertTokensNumType为2的场景下，表示activeExpertRangeOptional范围内token总数为非0的expert，以及对应expert处理token的总数。<br>• expertTokensNumType为0或1时，输出shape为[expertEnd-expertStart]；expertTokensNumType为2时，输出shape为[expertNum, 2]。</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedScaleOut</td>
        <td>输出</td>
        <td>输出量化计算过程中scaleOptional的中间值。<br>• 非量化场景下为可选输入，如果输入则要求为1D的Tensor,类型为FLOAT32。当输入x数据类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2时,如果输入则要求3D的Tensor,类型为FLOAT8_E8M0。当DropPad场景输出是一个1D的Tensor，shape为[expertNum * expertCapacity]，类型为FLOAT32。<br>• quantMode为2、3、9时,数据类型支持FLOAT8_E8M0。<br>• quantMode为16、17时,数据类型支持FLOAT8_E8M0，Shape为[NUM_ROWS*K, M]，其中M=CeilAlign(CeilDiv(H,32),2)。<br>• quantMode为4、5、14、15时，数据类型支持FLOAT32，且要求为2D的Tensor，shape为[NUM_ROWS*K, CeilDiv(H, 128)]。<br>• quantMode为11、12时,数据类型支持FLOAT32,且要求为3D的Tensor。<br>• 其余场景数据类型支持FLOAT32。</td>
        <td>FLOAT32、FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入值域限制：
  - activeNum当前未使用，校验需等于NUM_ROWS*K。
  - expertCapacity在Dropless场景下仅校验其值，不使用该参数；在DropPad场景下必须校验且取值范围为(0, NUM_ROWS]。
  - dropPadMode支持取值为0和1，分别代表Dropless场景和DropPad场景。
  - expertTokensNumType当前只支持0、1 和2，分别代表cumsum模式、count模式和key\_value模式。
  - expertTokensNumFlag只支持true，代表输出expertTokensCountOrCumsumOut。
  - quantMode:
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持1、-1，分别代表动态量化场景和不量化场景。
    - <term>Ascend 950PR/Ascend 950DT</term>：
      - 支持-1、1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17，分别表示不量化、动态量化到INT8、MXFP8量化到FLOAT8_E5M2、MXFP8量化到FLOAT8_E4M3FN、FP8 PerGroup量化到FLOAT8_E5M2、FP8 PerGroup量化到FLOAT8_E4M3FN、按直转方式量化到HIFLOAT8、按PERTENSOR模式量化到HIFLOAT8、按PERTOKEN模式量化到HIFLOAT8，MXFP4量化到FLOAT4_E2M1，FP8 PerBlock量化到FLOAT8_E5M2，FP8 PerBlock量化到FLOAT8_E4M3FN，FP8 PerGroup量化到FLOAT8_E5M2并启用Amax下限，FP8 PerGroup量化到FLOAT8_E4M3FN并启用Amax下限，INT4动态量化，MXFP8 RoundScale+Amax量化到FLOAT8_E5M2，MXFP8 RoundScale+Amax量化到FLOAT8_E4M3FN。
      - 支持quantMode为13的INT4动态量化场景，需同时满足：
        - x数据类型为FLOAT32或BFLOAT16，expandedXOut数据类型为INT4。
        - H为偶数，用于沿H维每两个INT4值打包为1个字节；NUM_ROWS不要求为偶数。
        - scaleOptional不输入，或输入shape为(1, H)、数据类型为FLOAT32，表示对activeExpertRangeOptional范围内的expert按H维广播smooth scale；offsetOptional不输入。
        - expertTokensNumType为0或1时，expertTokensCountOrCumsumOut的shape为[expertEnd-expertStart]；expertTokensNumType为2时，expertTokensCountOrCumsumOut的shape为[expertNum, 2]。

- <term>Ascend 950PR/Ascend 950DT</term> DropPad模式特殊约束（dropPadMode=1时）：
  - rowIdxType仅支持取值为0（gather索引）。
  - activeExpertRangeOptional必须为[0, expertNum]。
  - expertTokensNumType仅支持取值为1（count模式）。
  - quantMode在DropPad模式下仅支持-1（非量化），且数据类型仅支持FLOAT16、BFLOAT16、FLOAT32、INT8、HIFLOAT8。
  - expandedXOut必须是3D Tensor，shape为[expertNum, expertCapacity, H]。

- 其他限制：该算子部分产品支持两种性能模板，进入两种性能模板需要分别额外满足以下条件，不满足条件则进入通用模板。
  - 支持性能模板的产品：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品。</term>
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品。</term>

  - 进入低时延性能模板需要同时满足以下条件：
    - x、expertIdx、scaleOptional输入Shape要求分别为：(1, 7168)、(1, 8)、(256, 7168)。
    - x数据类型要求：BFLOAT16。
    - 属性要求：activeExpertRangeOptional=[0, 256]、 quantMode=1、expertTokensNumType=2、expertNum=256。

  - 进入大batch性能模板需要同时满足以下条件：
    - NUM_ROWS范围为[384, 8192]
    - K=8
    - expertNum=256
    - expertEnd-expertStart<=32
    - quantMode=-1
    - rowIdxType=1
    - expertTokensNumType=1
