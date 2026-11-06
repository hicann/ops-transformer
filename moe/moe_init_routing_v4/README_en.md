# MoeInitRoutingV4

English | [简体中文](./README.md)

## Product Support

| Product                                                        | Supported |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: Performs the routing computation of MoE. This operator performs routing processing based on the computation result of the MoeGatingTopKSoftmax operator, and supports the non-quantization, static quantization, and dynamic quantization modes. This operator provides the following functional changes compared with the MoeInitRoutingV3 operator. Select an appropriate operator based on actual requirements:

    1.active_num is changed from an attribute to an optional input, supporting dynamic passing of the active_num value in graph mode.

    2.Adds the optional input topkWeight and the optional output expandedTopkWeightOut, which support reordering topkWeight by the sorted indices so that expandedTopkWeightOut corresponds one-to-one with expandedXOut. topkWeight and expandedTopkWeightOut must be passed in or omitted together.

    3.Only the Ascend 950 architecture is supported.

- Computation formula:

  1.Sort the input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding indices sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.Map positions by using sortedRowIdx to obtain expandedRowIdxOut:
    - When rowIdxType is 1, output scatter indices

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - When rowIdxType is 0, output gather indices

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.Compute the histogram result for each expert in sortedExpertIdx to obtain expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.If quantMode is not -1, compute the quantization result:
    - Static quantization:
        $$
        quantResult = round(x * scaleOptional + offsetOptional)
        $$

    - Dynamic quantization:
        - If scale is not entered:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - If scale is entered:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - When quantMode is 13, dynamic quantization uses the symmetric quantization range [-8, 7], the denominator in the scale computation is 7, and every two INT4 values along the H dimension are packed into 1 byte in the quantization result.

  5.Take the values at the positions corresponding to the first min(activeNum, NUM_ROWS*K) entries of sortedRowIdx in quantResult to obtain expandedXOut:

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]/K]
    $$

  6.If topkWeight is entered, map the positions of topkWeight by sortedRowIdx to obtain expandedTopkWeightOut:

    $$
    expandedTopkWeightOut[i]=topkWeight[sortedRowIdx[i]]
    $$

    The value range of i is [0, min(activeNum, availableIdxNum)).

  7.The number of valid elements availableIdxNum in expandedRowIdxOut is computed as the number of elements in expertIdx within the activeExpertRangeOptional range:
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Name</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input of MoE, that is, the token feature input, corresponding to x in the formula.</td>
        <td><ul>
          <li>quantMode=-1: supports FLOAT16, BFLOAT16, FLOAT32, INT8, HIFLOAT8, FLOAT4_E2M1, FLOAT8_E4M3FN, and FLOAT8_E5M2;</li>
          <li>quantMode=0 or 1: supports FLOAT16, BFLOAT16, and FLOAT32;</li>
          <li>quantMode=2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, or 17: supports FLOAT16 and BFLOAT16;</li>
          <li>quantMode=13: supports FLOAT32 and BFLOAT16;</li>
          </ul></td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdx</td>
        <td>Input</td>
        <td>The K experts that process each row of features. The expert id of each element in it cannot exceed the number of experts. Corresponds to expertIdx in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scaleOptional</td>
        <td>Optional input</td>
        <td>Parameter used to compute the quantization result. If it is not entered, scale is not used in the computation. Corresponds to scale in the formula.<br>• In the INT8 dynamic quantization scenario where quantMode is 1, if entered, it must be a 2D tensor with shape (expertEnd-expertStart, H).<br>• In the INT4 dynamic quantization scenario where quantMode is 13, if entered, it must have shape (1, H), indicating a smooth scale broadcast along the H dimension.<br>• The FLOAT8_E8M0 data type of scale is supported only when quantMode is -1 and the data type of x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2.</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>offsetOptional</td>
        <td>Optional input</td>
        <td>Offset value used to compute the quantization result. It is not entered in the non-quantization scenario or the dynamic quantization scenario. Corresponds to offsetOptional in the formula.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>activeNumOptional</td>
        <td>Optional input</td>
        <td>Maximum total number of rows to process. Only this number of rows in the output expandedXOut are valid. If it is not entered, activeNum defaults to NUM_ROWS*K. If entered, it is a scalar tensor (shape () or (1,)) of the INT64 data type. The value must be greater than or equal to 0. A value of 0 indicates the Dropless scenario. A value greater than 0 indicates the Active scenario, which limits the total number of tokens processed by all experts.</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>topkWeightOptional</td>
        <td>Optional input</td>
        <td>Routing weights of the topk experts, used for reordering by the sorted indices so that the result corresponds one-to-one with expandedXOut. The input shape is (NUM_ROWS, K), and only the FLOAT32 data type is supported. topkWeight and expandedTopkWeightOut must be passed in or omitted together.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>Attribute</td>
        <td>Number of tokens that each expert can process. In the Dropless scenario, only the value is checked and this parameter is not used. In the DropPad scenario, the value range is (0, NUM_ROWS].</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>Attribute</td>
        <td>Number of experts. When expertTokensNumType is the key_value mode, the value range is [1, 5120]. In other modes, the value range is [1, 10240].</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>Attribute</td>
        <td>Indicates whether the DropPad scenario is used. The value is 0 or 1.<br>• 0: Dropless scenario. In this scenario, expertCapacity is not checked.<br>• 1: DropPad scenario.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumType</td>
        <td>Attribute</td>
        <td>The value is 0, 1, or 2.<br>• 0: cumsum mode.<br>• 1: count mode, that is, the output values are the cumulative number of tokens processed by each expert.<br>• 2: key\_value mode, that is, the output values are the experts and the cumulative number of tokens processed by the corresponding experts.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumFlag</td>
        <td>Attribute</td>
        <td>The value is false or true.<br>• false: expertTokensCountOrCumsumOut is not output.<br>• true: expertTokensCountOrCumsumOut is output.</td>
        <td>Bool</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>Attribute</td>
        <td>The value is 0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, or 17.<br>• 0: static quantization scenario.<br>• 1: dynamic quantization scenario, where expandedXOut is quantized to INT8.<br>• -1: non-quantization scenario.<br>• 2: MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2.<br>• 3: MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN.<br>• 4: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where expandedXOut is quantized to FLOAT8_E5M2.<br>• 5: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where expandedXOut is quantized to FLOAT8_E4M3FN.<br>• 6: HIF8 direct conversion quantization scenario, where expandedXOut is quantized to HIFLOAT8.<br>• 7: HIF8 PERTENSOR quantization scenario, where expandedXOut is quantized to HIFLOAT8.<br>• 8: HIF8 PERTOKEN quantization scenario, where expandedXOut is quantized to HIFLOAT8.<br>• 9: MXFP4 quantization scenario, where expandedXOut is quantized to FLOAT4_E2M1.<br>• 11: FP8 PerBlock quantization scenario (BlockSize=128), where expandedXOut is quantized to FLOAT8_E5M2 and expandedScaleOut uses the FLOAT32 3D layout.<br>• 12: FP8 PerBlock quantization scenario (BlockSize=128), where expandedXOut is quantized to FLOAT8_E4M3FN and expandedScaleOut uses the FLOAT32 3D layout.<br>• 13: INT4 dynamic quantization scenario, where expandedXOut is quantized to INT4.<br>• 14: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where expandedXOut is quantized to FLOAT8_E5M2.<br>• 15: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where expandedXOut is quantized to FLOAT8_E4M3FN.<br>• 16: MXFP8 RoundScale+Amax quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2.<br>• 17: MXFP8 RoundScale+Amax quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeExpertRangeOptional</td>
        <td>Optional attribute</td>
        <td>Length is 2, and the values in the array are [expertStart, expertEnd], indicating that the active expert range is between expertStart and expertEnd, with the left boundary closed and the right boundary open. The values must be greater than or equal to 0, and expertEnd must not be greater than expertNum.</td>
        <td>ListInt</td>
        <td>-</td>
      </tr>
      <tr>
        <td>rowIdxType</td>
        <td>Attribute</td>
        <td>Index type used by expandedRowIdxOut. The value is 0 or 1.<br>• 0: gather-type index.<br>• 1: scatter-type index.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedXOut</td>
        <td>Output</td>
        <td>Features expanded based on expertIdx. In the non-quantization scenario, the data type is the same as that of x. In the quantization scenario, the INT8 data type is supported when quantMode is 0 or 1; the INT4 data type is supported when quantMode is 13 and the data type of x is FLOAT32 or BFLOAT16; the FLOAT8_E5M2 data type is supported when quantMode is 2, 4, 14, or 16; the FLOAT8_E4M3FN data type is supported when quantMode is 3, 5, 15, or 17; the HIFLOAT8 data type is supported when quantMode is 6, 7, or 8; the FLOAT4_E2M1 data type is supported when quantMode is 9; and the FLOAT8_E5M2 and FLOAT8_E4M3FN data types are supported when quantMode is 11 and 12, respectively.
        <br>• In the Dropless scenario, the shape is [NUM_ROWS * K, H].<br>• In the Active scenario, the shape is [min(activeNum, NUM_ROWS * K), H].<br>• In the DropPad scenario, it must be a 3D tensor with shape [expertNum, expertCapacity, H].</td>
        <td>FLOAT32, FLOAT16, BFLOAT16, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdxOut</td>
        <td>Output</td>
        <td>Index mapping between expandedXOut and x. The shape is [NUM_ROWS*K]. The first availableIdxNum elements are valid data, and the rest are invalid data. When rowIdxType is 0, invalid data is filled with -1; when rowIdxType is 1, invalid data is not initialized.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOut</td>
        <td>Output</td>
        <td>Valid data is output when expertTokensNumFlag is true, and an empty tensor is output when expertTokensNumFlag is false.<br>• In the scenario where expertTokensNumType is 0 or 1, indicates the total number of tokens processed by each expert within the activeExpertRangeOptional range; the output shape is [expertEnd-expertStart].<br>• In the scenario where expertTokensNumType is 2, indicates the experts whose total number of tokens is not 0 within the activeExpertRangeOptional range and the total number of tokens processed by the corresponding experts; the output shape is [expertNum, 2].</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedScaleOut</td>
        <td>Output</td>
        <td>Intermediate value of scaleOptional during the quantization computation.<br>• In the non-quantization scenario, when scaleOptional is entered, the shape is [NUM_ROWS*K] and the output type is FLOAT32. When the data type of the input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, if scaleOptional is entered, the shape is [NUM_ROWS*K, CeilDiv(H, 64), 2] and the output type is FLOAT8_E8M0. In the DropPad scenario, the shape is [expertNum * expertCapacity] and the output type is FLOAT32.<br>• In the dynamic quantization scenario (quantMode is 1), when scaleOptional is entered, the shape is [NUM_ROWS*K] and the output type is FLOAT32.<br>• In the static quantization scenario (quantMode is 0), the HIF8 direct conversion quantization scenario (quantMode is 6), and the HIF8 PERTENSOR quantization scenario (quantMode is 7), the output is an empty tensor.<br>• In the HIF8 PERTOKEN quantization scenario (quantMode is 8), the shape is [NUM_ROWS*K] and the output type is FLOAT32.<br>• In the MXFP8 quantization scenario (quantMode is 2, 3, 16, or 17), the output type is FLOAT8_E8M0 and the Shape is [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H,32),2).<br>• In the MXFP4 quantization scenario (quantMode is 9), the output type is FLOAT8_E8M0 and the Shape is [NUM_ROWS*K, M, 2], where M=CeilDiv(H, 64).<br>• In the FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15), the output type is FLOAT32 and the Shape is [NUM_ROWS*K, CeilDiv(H, 128)].<br>• In the FP8 PerBlock quantization scenario (quantMode is 11 or 12), the output type is FLOAT32 and the Shape is [NUM_ROWS*K, CeilDiv(H, 256), 2].</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedTopkWeightOut</td>
        <td>Optional output</td>
        <td>Routing weights reordered by the sorted indices, corresponding one-to-one with expandedXOut. It must be output together when topkWeight is entered.<br>• In the Dropless scenario, the shape is [min(activeNum, NUM_ROWS*K), 1]. The first effectiveNum elements are valid data, and the rest are not initialized, where effectiveNum=min(activeNum, availableIdxNum).<br>• In the DropPad scenario, the shape is [expertNum*expertCapacity, 1]. Positions with allocated capacity contain valid data, and positions without allocated capacity are filled with 0.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- In the Dropless scenario, only the value of expertCapacity is checked and this parameter is not used. In the DropPad scenario, expertCapacity must be checked and its value range is (0, NUM_ROWS].
- The INT4 dynamic quantization scenario where quantMode is 13 is supported only when all the following conditions are met:
  - The data type of x is FLOAT32 or BFLOAT16, and the data type of expandedXOut is INT4.
  - H is an even number, so that every two INT4 values along the H dimension are packed into 1 byte.
  - scaleOptional is not entered, or the entered shape is (1, H) and the data type is FLOAT32; offsetOptional is not entered.
- Special constraints in DropPad mode (when dropPadMode=1):
  - quantMode supports only -1 (non-quantization), and only the FLOAT16, BFLOAT16, FLOAT32, INT8, and HIFLOAT8 data types are supported.
  - rowIdxType supports only 0 (gather index).

## Invocation

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_init_routing_v4](examples/test_aclnn_moe_init_routing_v4.cpp) | Invoke the MoeInitRoutingV4 operator through the [aclnnMoeInitRoutingV4](docs/aclnnMoeInitRoutingV4_en.md) API. |
| aclnn API  | [test_aclnn_moe_init_routing_v5](examples/test_aclnn_moe_init_routing_v5.cpp) | Invoke the MoeInitRoutingV4 operator through the [aclnnMoeInitRoutingV5](docs/aclnnMoeInitRoutingV5.md) API. On top of the V4 parameters, the scaleAlg input parameter is added to select the scale algorithm during quantization. |
| Graph mode     | [test_geir_moe_init_routing_v4](examples/test_geir_moe_init_routing_v4.cpp) | Invoke the MoeInitRoutingV4 operator by building a graph through the [operator IR](op_graph/moe_init_routing_v4_proto.h). |
| PyTorch API | - | Invoke the moe_init_routing operator through the [moe_init_routing](docs/torchapi_moe_init_routing_en.md) API. |
