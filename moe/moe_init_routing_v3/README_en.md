# MoeInitRoutingV3

English | [简体中文](./README.md)

## Product Support

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Functional Description

- Operator function: routing calculation of MoE. It performs routing processing based on the calculation result of [aclnnMoeGatingTopKSoftmaxV2](../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md) and supports the non-quantization, static quantization, and dynamic quantization modes. This interface provides the following function changes compared with the V2 interface [aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md). Select an appropriate interface based on actual requirements:

    1. Adds the dynamic and static quantization functions, supports the int8 quantization mode for output expandX, and adds multiple FP8/FP4/INT4/HIFLOAT8 quantization modes.

    2. Adds the parameter activeExpertRangeOptional, which supports filtering expertId within the valid range.

    3. Deletes the attribute expertTokensBeforeCapacityFlag and the output expertTokensBeforeCapacityOut (expertTokensCountOrCumsumOut is used for output instead).

- Computation formulas:

  1. Sort the input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding sequence numbers sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2. Perform position mapping based on sortedRowIdx to obtain expandedRowIdxOut:
    - When rowIdxType is 1, the scatter index is output

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - When rowIdxType is 0, the gather index is output

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3. Count the histogram result of each expert in sortedExpertIdx to obtain expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4. If quantMode is not -1, calculate the quant result:
    - Static quant

      $$
      quantResult=round((x∗scaleOptional)+offsetOptional)
      $$

    - Dynamic quant:
        - If scale is not input:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
        - If scale is input:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - When quantMode is 13, dynamic quantization uses the symmetric quantization range [-8, 7], the denominator in the scale calculation is 7, and every two INT4 values of the quantization result along the H dimension are packed into 1 byte.

  5. If the active expert range is the full expert range, tokens are moved according to the scatter index; otherwise, tokens are moved according to the gather index. When dropPadMode is 1, the number of tokens that each expert needs to process is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and insufficient tokens are padded with 0. Obtain expandedXOut:
    - Non-quantization scenario
      - Moved according to the scatter index

        $$
        expandedXOut[i]=x[scatterRowIdx[i] // K]
        $$

      - Moved according to the gather index

        $$
        expandedXOut[gatherRowIdx[i]]=x[i // K]
        $$

    - Quantization scenario
      - Moved according to the scatter index

        $$
        expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
        $$

      - Moved according to the gather index

        $$
        expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
        $$

  6. The number of valid elements availableIdxNum of expandedRowIdxOut is calculated as the number of elements in expertIdx within the activeExpertRangeOptional range. -1 is also out of this range, indicating an invalid expert that does not participate in routing calculation:
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input of MoE, that is, the token feature input, corresponding to x in the formula. The shape is (NUM_ROWS, H). In the MXFP4/INT4 dynamic quantization scenarios where quantMode is 9 or 13, and in the non-quantization pass-through scenario where quantMode is -1 and the data type of x is FLOAT4_E2M1, H must be an even number.</td>
        <td><ul>
          <li>quantMode=-1: supports FLOAT16, BFLOAT16, FLOAT32, INT8, HIFLOAT8, FLOAT4_E2M1, FLOAT8_E4M3FN, and FLOAT8_E5M2;</li>
          <li>quantMode=0, 1: supports FLOAT16, BFLOAT16, and FLOAT32;</li>
          <li>quantMode=2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17: supports FLOAT16 and BFLOAT16;</li>
          <li>quantMode=13: supports FLOAT32 and BFLOAT16;</li>
          <li>The preceding types must also meet the product support restrictions described below: A2/A3 supports only quantMode=-1, 0, and 1, and non-quantized input supports only FLOAT16, BFLOAT16, FLOAT32, and INT8; the other listed types and quantization modes are supported only by 950.</li>
          </ul></td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdx</td>
        <td>Input</td>
        <td>K experts that process the tokens in each row. The expert IDs of elements cannot exceed the number of experts, and -1 indicates an invalid expert (this position does not participate in routing and is filtered out). Corresponds to expertIdx in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scaleOptional</td>
        <td>Optional input</td>
        <td>Parameter used to calculate the quant result. If not input, scale is not used during calculation. Corresponds to scale in the formula.<br>• Optional input in the non-quantization scenario. If input, it must be a 1D tensor with shape (NUM_ROWS,) and the FLOAT32 type. When the data type of the input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, if input, it must be a 3D tensor with shape (NUM_ROWS, CeilDiv(H, 64), 2) and the FLOAT8_E8M0 type.<br>• Required in the static quantization scenario. The input must be a 1D tensor with shape [1, ].<br>• Optional input in the INT8 dynamic quantization scenario where quantMode is 1. If input, it must be a 2D tensor with shape (expertEnd-expertStart, H).<br>• Optional input in the INT4 dynamic quantization scenario where quantMode is 13. If input, the shape must be (1, H), indicating the smooth scale broadcast along the H dimension.<br>• Required in the HIF8 PERTENSOR quantization scenario (quantMode is 7). The input must be a 1D tensor with shape [1, ].<br>• Not input in the MXFP8 quantization scenario (quantMode is 2, 3, 16, or 17).<br>• Not input in the HIF8 direct conversion and HIF8 PERTOKEN quantization scenarios (quantMode is 6 or 8).<br>• Not input in the MXFP4 quantization scenario (quantMode is 9).<br>• Not input in the FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15).<br>• Not input in the FP8 PerBlock quantization scenario (quantMode is 11 or 12).</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>offsetOptional</td>
        <td>Optional input</td>
        <td>Offset used to calculate the quant result. Corresponds to offsetOptional in the formula.<br>• Required in the static quantization scenario. The input must be a 1D tensor with shape [1, ].<br>• Not input in the non-quantization scenario, dynamic quantization (quantMode is 1), MXFP8 quantization (quantMode is 2, 3, 16, or 17), HIF8 quantization (quantMode is 6, 7, or 8), MXFP4 quantization (quantMode is 9), FP8 PerGroup quantization (quantMode is 4, 5, 14, or 15), FP8 PerBlock quantization (quantMode is 11 or 12), or INT4 dynamic quantization (quantMode is 13).</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>activeNum</td>
        <td>Attribute</td>
        <td><term>Atlas A2 products</term> and <term>Atlas A3 products</term>: When dropPadMode=0, activeNum supports values greater than or equal to -1. -1 and 0 indicate no limit on the number of processed rows, and a value greater than 0 indicates that at most min(activeNum, NUM_ROWS*K) rows are processed. <term>Ascend 950PR&950DT products</term>: This attribute is not used to limit the number of processed rows and accepts only -1, 0, or NUM_ROWS*K.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>Attribute</td>
        <td>Indicates the number of tokens that each expert can process. This parameter is not used in the Dropless scenario; in the DropPad scenario, it must be checked and the value range is (0, NUM_ROWS].</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>Attribute</td>
        <td>Indicates the number of experts. When expertTokensNumType is the key\_value mode, the value range is [1, 5120]; in other modes, the value range is [1, 10240].</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>Attribute</td>
        <td>Indicates whether the DropPad scenario is used. The value is 0 or 1.<br>• 0: indicates the Dropless scenario.<br>• 1: indicates the DropPad scenario.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertTokensNumType</td>
        <td>Attribute</td>
        <td>The value is 0, 1, or 2.<br>• 0: indicates the cumsum mode.<br>• 1: indicates the count mode, that is, the output value is the cumulative number of tokens processed by each expert.<br>• 2: indicates the key\_value mode, that is, the output value is the expert and the cumulative number of tokens processed by the corresponding expert.</td>
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
        <td>The value is 0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, or 17.<br>• 0: indicates the static quant scenario.<br>• 1: indicates the dynamic quant scenario, where expandedXOut is quantized to INT8.<br>• -1: indicates the non-quantization scenario.<br>• 2: indicates the MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2.<br>• 3: indicates the MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN.<br>• 4: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where expandedXOut is quantized to FLOAT8_E5M2.<br>• 5: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where expandedXOut is quantized to FLOAT8_E4M3FN.<br>• 6: indicates the HIF8 direct conversion quantization scenario, where expandedXOut is quantized to HIFLOAT8.<br>• 7: indicates the HIF8 PERTENSOR quantization scenario, where expandedXOut is quantized to HIFLOAT8.<br>• 8: indicates the HIF8 PERTOKEN quantization scenario, where expandedXOut is quantized to HIFLOAT8.<br>• 9: indicates the MXFP4 quantization scenario, where expandedXOut is quantized to FLOAT4_E2M1.<br>• 11: indicates the FP8 PerBlock quantization scenario (BlockSize=128), where expandedXOut is quantized to FLOAT8_E5M2, and expandedScaleOut uses the FLOAT32 3D layout.<br>• 12: indicates the FP8 PerBlock quantization scenario (BlockSize=128), where expandedXOut is quantized to FLOAT8_E4M3FN, and expandedScaleOut uses the FLOAT32 3D layout.<br>• 13: indicates the INT4 dynamic quantization scenario, where expandedXOut is quantized to INT4.<br>• 14: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where expandedXOut is quantized to FLOAT8_E5M2.<br>• 15: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where expandedXOut is quantized to FLOAT8_E4M3FN.<br>• 16: indicates the MXFP8 RoundScale+Amax quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2.<br>• 17: indicates the MXFP8 RoundScale+Amax quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeExpertRangeOptional</td>
        <td>Optional attribute</td>
        <td>The length is 2, and the values in the array are [expertStart, expertEnd], indicating that the active expert range is between expertStart and expertEnd, with the left bound closed and the right bound open. The values must be greater than or equal to 0, and expertEnd must not be greater than expertNum. In the Drop/Pad scenario, expertStart equals 0 and expertEnd equals expertNum.</td>
        <td>ListInt</td>
        <td>-</td>
      </tr>
      <tr>
        <td>rowIdxType</td>
        <td>Attribute</td>
        <td>Indicates the index type used by expandedRowIdxOut. The value is 0 or 1. (The performance template supports only 1.)<br>• 0: indicates the gather index.<br>• 1: indicates the scatter index.</td>
        <td>INT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expandedXOut</td>
        <td>Output</td>
        <td>Features expanded based on expertIdx. In the non-quantization scenario, the data type is the same as that of x. In the quantization scenario, the data type supports INT8 when quantMode is 0 or 1; supports INT4 when quantMode is 13 and the data type of x is FLOAT32 or BFLOAT16; supports FLOAT8_E5M2 when quantMode is 2, 4, 14, or 16; supports FLOAT8_E4M3FN when quantMode is 3, 5, 15, or 17; supports HIFLOAT8 when quantMode is 6, 7, or 8; supports FLOAT4_E2M1 when quantMode is 9; supports FLOAT8_E5M2 and FLOAT8_E4M3FN respectively when quantMode is 11 or 12.
        <br>• In the Dropless scenario, the shape is [NUM_ROWS * K, H].<br>• In the DropPad scenario, it must be a 3D tensor with shape [expertNum, expertCapacity, H].</td>
        <td>FLOAT32, FLOAT16, BFLOAT16, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdxOut</td>
        <td>Output</td>
        <td>Index mapping between expandedXOut and x. The output shape is (NUM_ROWS*K, ). The first availableIdxNum elements are valid data, and the rest are invalid data. When rowIdxType is 0, invalid data is filled with -1; when rowIdxType is 1, invalid data is not initialized.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertTokensCountOrCumsumOut</td>
        <td>Output</td>
        <td>• In the scenario where expertTokensNumType is 0, indicates the prefix sum of the total number of tokens processed by experts within the activeExpertRangeOptional range after sorting.<br>• In the scenario where expertTokensNumType is 1, indicates the total number of tokens processed by experts within the activeExpertRangeOptional range.<br>• In the scenario where expertTokensNumType is 2, indicates the experts whose total number of tokens within the activeExpertRangeOptional range is non-zero, and the total number of tokens processed by the corresponding experts.<br>• When expertTokensNumType is 0 or 1, the output shape is [expertEnd-expertStart]; when expertTokensNumType is 2, the output shape is [expertNum, 2].</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedScaleOut</td>
        <td>Output</td>
        <td>Outputs intermediate values of scaleOptional in the quantization calculation. The output shape is the product of all dimensions of the shape of expandedXOut except the last dimension.<br>• In the non-quantization scenario, when scaleOptional is input, the shape is [NUM_ROWS*K], the first availableIdxNum elements are valid data, and the output type is FLOAT32. When the data type of the input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, if scaleOptional is input, the shape of expandedScaleOut is [NUM_ROWS*K, CeilDiv(H, 64), 2], and the output type is FLOAT8_E8M0. In the Drop/Pad scenario, the output is a 1D tensor with shape [expertNum * expertCapacity], and the output type is FLOAT32.<br>• In the dynamic quantization scenario, when scaleOptional is input, the first availableIdxNum elements are valid data.<br>• Not output in the static quantization scenario.<br>• In the MXFP8 quantization scenario (quantMode is 2 or 3), the output type is FLOAT8_E8M0, the shape is [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H,32),2), and the first availableIdxNum rows of NUM_ROWS*K are valid data.<br>• In the MXFP8 RoundScale+Amax quantization scenario (quantMode is 16 or 17), the output type is FLOAT8_E8M0, the shape is [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H,32),2), and the first availableIdxNum rows of NUM_ROWS*K are valid data.<br>• In the scenario of quantization to HIFLOAT8 by direct conversion, expandedScaleOut is not output.<br>• In the scenario of quantization to HIFLOAT8 in PERTENSOR mode, expandedScaleOut is not output.<br>• In the scenario of quantization to HIFLOAT8 in PERTOKEN mode, the output type is FLOAT32, and the shape is [NUM_ROWS*K].<br>• In the MXFP4 quantization scenario (quantMode is 9), the output type is FLOAT8_E8M0, the shape is [NUM_ROWS*K, M, 2], where M=CeilDiv(H, 64), and the first availableIdxNum rows of NUM_ROWS*K are valid data.<br>• In the FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15), the output type is FLOAT32, the shape is [NUM_ROWS*K, CeilDiv(H,128)], and the first availableIdxNum rows of NUM_ROWS*K are valid data.<br>• In the FP8 PerBlock quantization scenario (quantMode is 11 or 12), the output type is FLOAT32, the shape is [NUM_ROWS*K, CeilDiv(H,256), 2], and the first availableIdxNum rows of NUM_ROWS*K are valid data.</td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- Input value range restrictions:
  - activeNum: <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: When dropPadMode=0, activeNum supports values greater than or equal to -1. -1 and 0 indicate no limit on the number of processed rows, and a value greater than 0 indicates that at most min(activeNum, NUM_ROWS*K) rows are processed. <term>Ascend 950PR&950DT products</term>: This attribute is not used to limit the number of processed rows and accepts only -1, 0, or NUM_ROWS*K.
  - expertCapacity is not used in the Dropless scenario; in the DropPad scenario, it must be checked and the value range is (0, NUM_ROWS].
  - dropPadMode supports the values 0 and 1, indicating the Dropless scenario and the DropPad scenario respectively.
  - expertTokensNumType currently supports only 0, 1, and 2, indicating the cumsum mode, count mode, and key\_value mode respectively.
  - quantMode:
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Support 1, 0, and -1, indicating the dynamic quantization, static quantization, and non-quantization scenarios respectively. When quantMode=-1, x supports only FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - <term>Ascend 950PR&950DT products</term>:
      - Support -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, and 17, indicating non-quantization, static quantization to INT8, dynamic quantization to INT8, MXFP8 quantization to FLOAT8_E5M2, MXFP8 quantization to FLOAT8_E4M3FN, FP8 PerGroup quantization to FLOAT8_E5M2, FP8 PerGroup quantization to FLOAT8_E4M3FN, quantization to HIFLOAT8 by direct conversion, quantization to HIFLOAT8 in PERTENSOR mode, quantization to HIFLOAT8 in PERTOKEN mode, MXFP4 quantization to FLOAT4_E2M1, FP8 PerBlock quantization to FLOAT8_E5M2, FP8 PerBlock quantization to FLOAT8_E4M3FN, INT4 dynamic quantization, FP8 PerGroup quantization to FLOAT8_E5M2 with the Amax lower limit enabled, FP8 PerGroup quantization to FLOAT8_E4M3FN with the Amax lower limit enabled, MXFP8 RoundScale+Amax quantization to FLOAT8_E5M2, and MXFP8 RoundScale+Amax quantization to FLOAT8_E4M3FN.
      - Support the INT4 dynamic quantization scenario where quantMode is 13. The following conditions must be met:
        - The data type of x is FLOAT32 or BFLOAT16, and the data type of expandedXOut is INT4.
        - H is an even number, used to pack every two INT4 values along the H dimension into 1 byte.
        - scaleOptional is not input, or the input shape is (1, H) with the FLOAT32 data type, indicating the smooth scale broadcast along the H dimension for experts within the activeExpertRangeOptional range; offsetOptional is not input.
        - When expertTokensNumType is 0 or 1, the shape of expertTokensCountOrCumsumOut is [expertEnd-expertStart]; when expertTokensNumType is 2, the shape of expertTokensCountOrCumsumOut is [expertNum, 2].

- Special constraints of <term>Ascend 950PR&950DT products</term> in DropPad mode (when dropPadMode=1):
  - rowIdxType supports only the value 0 (gather index).
  - activeExpertRangeOptional must be [0, expertNum].
  - expertTokensNumType supports only the value 1 (count mode).
  - In DropPad mode, quantMode supports only -1 (non-quantization), and the data types support only FLOAT16, BFLOAT16, FLOAT32, INT8, and HIFLOAT8.
  - expandedXOut must be a 3D tensor with shape [expertNum, expertCapacity, H].

- Other restrictions: Some products support multiple performance templates for this operator. To enter each performance template, the following additional conditions must be met respectively; otherwise, the general template is used.
  - Products that support performance templates:
    - <term>Atlas A2 products.</term>
    - <term>Atlas A3 products.</term>

  - Entering the low-latency performance template requires all the following conditions:
    - The input shapes of x, expertIdx, and scaleOptional must be (1, 7168), (1, 8), and (256, 7168) respectively.
    - The data type of x must be BFLOAT16.
    - Attribute requirements: activeExpertRangeOptional=[0, 256], quantMode=1, expertTokensNumType=2, and expertNum=256.

  - Entering the large-batch performance template requires all the following conditions:
    - The NUM_ROWS range is [384, 8192]
    - K=8
    - expertNum=256
    - expertEnd-expertStart<=32
    - quantMode=-1
    - rowIdxType=1
    - expertTokensNumType=1

  - Products that support the counting sort performance template:
    - <term>Ascend 950PR&950DT products</term>.

  - Entering the counting sort FullLoad performance template requires all the following conditions:
    - The data type of x is BFLOAT16, FLOAT16, FLOAT32, or INT8
    - expertNum<=1024
    - expertEnd-expertStart<=32
    - quantMode=-1
    - dropPadMode=0
    - The UB space meets the full-load requirement after the buffers of this template are superimposed

  - Entering the counting sort CutOrigin performance template requires all the following conditions (this template is attempted only when the FullLoad conditions are not met):
    - expertNum<=1024
    - expertEnd-expertStart<=128
    - quantMode is -1 or 0
    - NUM_ROWS * H * x.dtype >= 1.5 * totalUbSize
    - NUM_ROWS * K >= 8192
    - When dropPadMode=1 (DropPad scenario), NUM_ROWS>=512 is also required
    - The UB space meets the requirement after the buffers of this template are superimposed
