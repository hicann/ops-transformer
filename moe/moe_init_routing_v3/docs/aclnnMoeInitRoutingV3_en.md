# aclnnMoeInitRoutingV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:26:12.203Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs MoE routing calculation and routes the calculation result of [aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md). Supports non-quantization, static quantization, and dynamic quantization modes. This API has the following function changes compared with the V2 API [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md). Select an appropriate API based on actual requirements:<br>
  <ol><li>Adds dynamic and static quantization functions and supports int8 quantization mode output of expandX.</li><li>Removes the output expertTokensBeforeCapacityOut and adds the output expertTokensCountOrCumsumOut.</li><li>Is compatible with the original V2 output mode and adds support for the key_value output format: redefines the original attributes expertTokensBeforeCapacityFlag (bool) and expertTokensCountOrCumsumFlag (int) as expertsTokensNumFlag (bool) and expertTokensNumType (int), respectively. The following table lists the output format mapping:</li></ol>
  <table align="center">
    <tr>
      <th>DropPadMode</th>
      <th>expertsTokensNumFlag</th>
      <th>expertTokensNumType</th>
      <th style="text-align: center;">Output Format Description</th>
    </tr>
    <tr align="center">
      <td>0</td>
      <td>true</td>
      <td>0</td>
      <td align="left">cumsum mode: expertTokensCountOrCumsumOut indicates the prefix sum histogram of the count of tokens processed by each expert after sorting.</td>
    </tr>
    <tr align="center">
      <td>0</td>
      <td>true</td>
      <td>1</td>
      <td align="left">count mode. expertTokensCountOrCumsumOut indicates the individual count histogram of tokens processed by each expert after sorting.</td>
    </tr>
    <tr align="center">
      <td>0</td>
      <td>true</td>
      <td>2</td>
      <td align="left">key_value mode. The output shape is [expert_num, 2], indicating each expert and the cumulative number of non-zero tokens processed by the expert.</td>
    </tr>
    <tr align="center">
      <td>1</td>
      <td>true</td>
      <td>1</td>
      <td align="left">The output mode is count mode.</td>
    </tr>
    <tr align="center">
      <td>Not enabled</td>
      <td>false</td>
      <td>Not enabled</td>
      <td align="left">Does not output expertTokensCountOrCumsumOut.</td>
    </tr>
  </table>

- Formula:

  1.Sort input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding indices sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.Perform position mapping based on sortedRowIdx to obtain expandedRowIdxOut:
    - When rowIdxType equals 1, output the scatter index

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - When rowIdxType equals 0, the gather index is output

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.Compute the histogram result for each expert in sortedExpertIdx to obtain expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.If quantMode does not equal -1, calculate the quantization result:
     - Static quantization

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$

    - Dynamic quantization:
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

        - When quantMode is 13, dynamic quantization uses the symmetric quantization range [-8, 7], the denominator in the scale calculation is 7, and every two INT4 values along the H dimension in the quantization result are packed into 1 byte.

  5.If the active expert range is the full expert range, tokens are moved according to the Scatter index; otherwise, tokens are moved according to the Gather index. When dropPadMode is 1, the number of tokens to be processed by each expert is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and insufficient tokens are padded with 0. expandedXOut is obtained as follows:
    - Non-quantized scenario
      - Move according to the Scatter index

      $$
      expandedXOut[i]=x[scatterRowIdx[i] // K]
      $$

      - Move according to the Gather index

      $$
      expandedXOut[gatherRowIdx[i]]=x[i // K]
      $$

    - Quantization scenario
      - Movement according to the Scatter index

      $$
      expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
      $$

      - Movement according to the Gather index

      $$
      expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
      $$

  6.The number of valid elements in expandedRowIdxOut is availableIdxNum. It is calculated as the number of elements in expertIdx that are within the range specified by activeExpertRangeOptional. -1 is not within the range either, indicating an invalid expert that does not participate in routing calculation:

    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMoeInitRoutingV3GetWorkspaceSize" API to obtain the input parameters, calculate the required workspace size, and obtain the executor that includes the operator calculation process. Then call the "aclnnMoeInitRoutingV3" API to execute the calculation.

```Cpp
aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *expertIdx,
  const aclTensor   *scaleOptional,
  const aclTensor   *offsetOptional,
  int64_t            activeNum,
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
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnMoeInitRoutingV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMoeInitRoutingV3GetWorkspaceSize

- **Parameters**:

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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x (aclTensor)</td>
      <td>Input</td>
      <td>MOE input, that is, the token feature input.</td>
      <td><ul>
        <li>shape is (NUM_ROWS, H);</li>
        <li>In the MXFP4/INT4 dynamic quantization scenario where quantMode is 9 or 13, and in the non-quantized passthrough scenario where quantMode is -1 and the data type of x is FLOAT4_E2M1, H must be an even number.</li>
        </ul></td>
      <td><ul>
        <li>quantMode=-1: supports FLOAT16, BFLOAT16, FLOAT32, INT8, HIFLOAT8, FLOAT4_E2M1, FLOAT8_E4M3FN, and FLOAT8_E5M2;</li>
        <li>quantMode=0 or 1: supports FLOAT16, BFLOAT16, and FLOAT32;</li>
        <li>quantMode=2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, or 17: supports FLOAT16 and BFLOAT16;</li>
        <li>quantMode=13: supports FLOAT32 and BFLOAT16;</li>
        </ul></td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdx (aclTensor)</td>
      <td>Input</td>
      <td>K processing experts corresponding to each row of features. The expert ID of each element cannot exceed the number of experts. -1 indicates an invalid expert (the position does not participate in routing and will be filtered out).</td>
      <td>Shape is (NUM_ROWS, K).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the parameter used to calculate quantization results.</td>
      <td><ul>
        <li>If not input, scale is not used in the calculation;</li>
        <li>Optional input in the non-quantization scenario. If input, it must be a 1D tensor with shape (NUM_ROWS,) and data type FLOAT32. When the data type of input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, if input, it must be a 3D tensor with shape (NUM_ROWS, CeilDiv(H, 64), 2) and data type FLOAT8_E8M0;</li>
        <li>Mandatory input in the static quantization scenario. The input must be a 1D tensor with shape [1, ];</li>
        <li>Optional input in the INT8 dynamic quantization scenario where quantMode is 1. If input, it must be a 2D tensor with shape (expertEnd-expertStart, H). Optional input in the INT4 dynamic quantization scenario where quantMode is 13. If input, the shape must be (1, H), indicating a smooth scale broadcast along the H dimension.</li>
        <li>Not input in the MXFP8 quantization scenario (quantMode is 2 or 3).</li>
        <li>Not input in the HIF8 direct conversion and HIF8 PERTOKEN quantization scenarios (quantMode is 6 or 8).</li>
        <li>Mandatory input in the HIF8 PERTENSOR quantization scenario (quantMode is 7). The input must be a 1D tensor with shape [1, ].</li>
        <li>Not input in the MXFP4 quantization scenario (quantMode is 9).</li>
        <li>Not input in the FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15).</li>
        <li>Not input in the FP8 PerBlock quantization scenario (quantMode is 11 or 12).</li>
        <li>Not input in the MXFP8 RoundScale+Amax quantization scenario (quantMode is 16 or 17).</li>
        </ul></td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the offset value used for calculating the quant result.</td>
      <td><ul>
        <li>Not input in the non-quantized scenario;</li><li>Must be input in the static quantization scenario; input requirements: 1D tensor, shape is [1, ];</li>
        <li>Not input in dynamic quantization (quantMode is 1), MXFP8 quantization (quantMode is 2, 3), HIF8 quantization (quantMode is 6, 7, 8), MXFP4 quantization (quantMode is 9), FP8 PerGroup quantization (quantMode is 4, 5, 14, 15), FP8 PerBlock quantization (quantMode is 11, 12), INT4 dynamic quantization (quantMode is 13), and the MXFP8 RoundScale+Amax quantization scenario (quantMode is 16, 17).</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum total number of rows to process. The specific behavior depends on the product.</td>
      <td><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When dropPadMode=0, activeNum supports values greater than or equal to -1; -1 and 0 indicate no limit on the number of rows to process; if greater than 0, at most min(activeNum, NUM_ROWS*K) rows are processed. <term>Ascend 950PR&950DT products</term>: This attribute does not limit the number of rows to process and only accepts -1, 0, or NUM_ROWS*K.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of tokens that each expert can process.</td>
      <td>This parameter is not used in Dropless scenarios. In DropPad scenarios, the input must be validated and its value range is (0, NUM_ROWS].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of experts.</td>
      <td>When expertTokensNumType is key_value mode, the value range is [1, 5120]; in other modes, the value range is [1, 10240].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates whether it is the DropPad scenario.</td>
      <td>The value is 0 or 1
        <br>0: indicates the Dropless scenario;
        <br>1: indicates the DropPad scenario;
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumType (int64_t)</td>
      <td>Input</td>
      <td>Indicates different histogram modes.</td>
      <td>Value is 0, 1, and 2
        <br>0: indicates the cumsum mode;
        <br>1: indicates the count mode;
        <br>2: indicates the key_value mode;
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumFlag (bool)</td>
      <td>Input</td>
      <td>Indicates whether to output expertTokensCountOrCumsumOut.</td>
      <td>Value is false or true</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates different quantization scenarios.</td>
      <td>Value is 0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, and 17 (the support status varies across products; see the description after the table)
        <br>0: indicates the static quantization scenario;
        <br>1: indicates the dynamic quantization scenario, where expandedXOut is quantized to INT8;
        <br>-1: indicates the non-quantization scenario;
        <br>2: indicates the MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2;
        <br>3: indicates the MXFP8 quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN;
        <br>4: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where expandedXOut is quantized to FLOAT8_E5M2;
        <br>5: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), where expandedXOut is quantized to FLOAT8_E4M3FN;
        <br>6: indicates the HIF8 direct conversion quantization scenario, where expandedXOut is quantized to HIFLOAT8;
        <br>7: indicates the HIF8 PERTENSOR quantization scenario, where expandedXOut is quantized to HIFLOAT8 according to the pertensor mode;
        <br>8: indicates the HIF8 PERTOKEN quantization scenario, where expandedXOut is quantized to HIFLOAT8 according to the pertoken mode;
        <br>9: indicates the MXFP4 quantization scenario, where expandedXOut is quantized to FLOAT4_E2M1;
        <br>11: indicates the FP8 PerBlock quantization scenario (BlockSize=128), where expandedXOut is quantized to FLOAT8_E5M2 and expandedScaleOut uses the FLOAT32 3D layout;
        <br>12: indicates the FP8 PerBlock quantization scenario (BlockSize=128), where expandedXOut is quantized to FLOAT8_E4M3FN and expandedScaleOut uses the FLOAT32 3D layout;
        <br>13: indicates the INT4 dynamic quantization scenario, where expandedXOut is quantized to INT4;
        <br>14: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where expandedXOut is quantized to FLOAT8_E5M2;
        <br>15: indicates the FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), where expandedXOut is quantized to FLOAT8_E4M3FN;
        <br>16: indicates the MXFP8 RoundScale+Amax quantization scenario, where expandedXOut is quantized to FLOAT8_E5M2;
        <br>17: indicates the MXFP8 RoundScale+Amax quantization scenario, where expandedXOut is quantized to FLOAT8_E4M3FN;
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeExpertRangeOptional (aclIntArray)</td>
      <td>Input</td>
      <td>Indicates the range of active experts.</td>
      <td>Optional parameter. If not provided or nullptr (None) is passed in, the default range is [0, expertNum). Length is 2. The values in the array are [expertStart, expertEnd], with a left-closed and right-open interval. Values must be greater than or equal to 0, and expertEnd must not be greater than expertNum; in the Drop/Pad scenario, expertStart equals 0, expertEnd equals expertNum.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIdxType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the index type used by expandedRowIdxOut.</td>
      <td>Value is 0 or 1.
        <br>0: indicates a gather-type index.
        <br>1: indicates a scatter-type index.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut (aclTensor)</td>
      <td>Output</td>
      <td>Features expanded according to expertIdx.</td>
      <td><ul>
        <li>In the Dropless scenario, the shape is [NUM_ROWS * K, H].</li>
        <li>In the Active scenario, the shape is [min(activeNum, NUM_ROWS * K), H].</li>
        <li>In the Drop/Pad scenario, this parameter must be a 3D tensor with the shape [expertNum, expertCapacity, H].</li>
        <li>In the non-quantization scenario, the data type is the same as that of x. In the quantization scenario, INT8 is supported when quantMode is 0 or 1, FLOAT8_E5M2 is supported when quantMode is 2, 4, 14, or 16, FLOAT8_E4M3FN is supported when quantMode is 3, 5, 15, or 17, HIFLOAT8 is supported when quantMode is 6, 7, or 8, FLOAT4_E2M1 is supported when quantMode is 9, FLOAT8_E5M2 and FLOAT8_E4M3FN are supported when quantMode is 11 or 12, respectively, and INT4 is supported when quantMode is 13 and the data type of x is FLOAT32 or BFLOAT16.</li>
      </ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut(aclTensor)</td>
      <td>Output</td>
      <td>Index mapping between expandedXOut and x.</td>
      <td>Output shape is (NUM_ROWS*K, ). The first availableIdxNum elements are valid data, and the remaining invalid data is determined by rowIdxType:
        <ul><li>When rowIdxType is 0, invalid data is filled with -1.</li>
        <li>When rowIdxType is 1, invalid data is not initialized.</li></ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut(aclTensor)</td>
      <td>Output</td>
      <td>Outputs statistics or accumulated values of the number of tokens processed by each expert.</td>
      <td><ul>
        <li>When expertTokensNumType is 0, indicates the prefix sum of the total tokens processed by experts within the range activeExpertRangeOptional after sorting.</li>
        <li>When expertTokensNumType is 1, indicates the total tokens processed by experts within the range activeExpertRangeOptional.</li>
        <li>When expertTokensNumType is 2, indicates experts with a non-zero total number of tokens within the range activeExpertRangeOptional, and the total tokens processed by the corresponding experts.</li>
        <li>When expertTokensNumType is 0 or 1, the output shape is [expertEnd-expertStart]; when expertTokensNumType is 2, the output shape is [expertNum, 2].</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedScaleOut (aclTensor)</td>
      <td>Output</td>
      <td>Outputs intermediate values of scaleOptional in different quantization processes.</td>
      <td> The output shape is the product of all dimensions of the expandedXOut shape except the last dimension.
        <ul style="list-style-type: circle;">
        <li>In the non-quantization scenario, when scaleOptional is input, the shape is [NUM_ROWS*K], the first availableIdxNum elements are valid data, and the output is of the FLOAT32 type. If the data type of input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2 and scaleOptional is input, the shape of expandedScaleOut is [NUM_ROWS*K, CeilDiv(H, 64), 2] and the output is of the FLOAT8_E8M0 type. In the Drop/Pad scenario, the output is a 1D Tensor with the shape of [expertNum * expertCapacity], and the output is of the FLOAT32 type.</li>
        <li>In the dynamic quantization scenario, when scaleOptional is input, the first availableIdxNum elements are valid data.</li>
        <li>Not output in the static quantization scenario.</li>
        <li>In the MXFP8 quantization scenario (quantMode is 2 or 3), the output is of the FLOAT8_E8M0 type, with the shape of [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H,32),2). Among the NUM_ROWS*K rows, the first availableIdxNum rows are valid data.</li>
        <li>In the MXFP8 RoundScale+Amax quantization scenario (quantMode is 16 or 17), the output is of the FLOAT8_E8M0 type, with the shape of [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H,32),2). Among the NUM_ROWS*K rows, the first availableIdxNum rows are valid data.</li>
        <li>In the scenario where data is quantized to HIFLOAT8 by direct conversion, expandedScaleOut is not output.</li>
        <li>In the scenario where data is quantized to HIFLOAT8 in PERTENSOR mode, expandedScaleOut is not output.</li>
        <li>In the scenario where data is quantized to HIFLOAT8 in PERTOKEN mode, the output is of the FLOAT32 type, with the shape of [NUM_ROWS*K].</li>
        <li>In the MXFP4 quantization scenario, the output is of the FLOAT8_E8M0 type, with the shape of [NUM_ROWS*K, M, 2], where M=CeilDiv(H, 64). Among the NUM_ROWS*K rows, the first availableIdxNum rows are valid data.</li>
        <li>In the FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15), the output is of the FLOAT32 type, with the shape of [NUM_ROWS*K, CeilDiv(H,128)]. Among the NUM_ROWS*K rows, the first availableIdxNum rows are valid data.</li>
        <li>In the FP8 PerBlock quantization scenario (quantMode is 11 or 12), the output is of the FLOAT32 type, with the shape of [NUM_ROWS*K, CeilDiv(H,256), 2]. Among the NUM_ROWS*K rows, the first availableIdxNum rows are valid data.</li></ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that users need to allocate on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the calculation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The calculation input and calculation output are null pointers.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161002</td>
      <td>The data types of the input and output are not within the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>      The shapes of the input and output tensors are not within the supported range.<br>
      The input attributes are not within the supported range.<br>
      </td>
    </tr>
  </tbody></table>

- **Differences in product support status**
  - Differences in quantMode support status:

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The supported values are -1, 0, and 1. When quantMode=-1, x supports only FLOAT16, BFLOAT16, FLOAT32, and INT8.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: The supported values are -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, and 17.
    <!-- end id8 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term> support only the values of the following parameters:
    - The activeNum parameter is not used. The supported values are -1, 0, or NUM_ROWS*K.
    - In the Dropless scenario, expertCapacity is not used; in the DropPad scenario, it must be validated, and the value range is (0, NUM_ROWS].
    - dropPadMode supports the values 0 and 1. The DropPad mode (dropPadMode=1) has the following additional constraints: <ul><li>rowIdxType only supports taking the value 0 (gather index).</li><li>activeExpertRangeOptional must be [0, expertNum].</li><li>expertTokensNumType only supports taking the value 1 (count mode).</li><li>In the DropPad mode, quantMode only supports -1 (non-quantized), and only the data types FLOAT16, BFLOAT16, FLOAT32, INT8, and HIFLOAT8 are supported.</li></ul>
    - expertTokensNumType only supports taking the values 0, 1, and 2.
    - expertTokensNumFlag only supports taking the value true.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term> support the INT4 dynamic quantization scenario where quantMode is 13. The following conditions must be met simultaneously:
    - The data type of x is FLOAT32 or BFLOAT16, and the data type of expandedXOut is INT4.
    - H must be even, so that every two INT4 values along the H dimension are packed into 1 byte; NUM_ROWS is not required to be even.
    - activeNum equals NUM_ROWS*K.
    - scaleOptional is not input, or its shape is (1, H) and its data type is FLOAT32, indicating that the smooth scale is broadcast along the H dimension to the experts within the range of activeExpertRangeOptional; offsetOptional is not input.
    - When the value of expertTokensNumType is 0 or 1, the shape of expertTokensCountOrCumsumOut is [expertEnd-expertStart]; when the value of expertTokensNumType is 2, the shape of expertTokensCountOrCumsumOut is [expertNum, 2].

  <!-- end id10 -->

## aclnnMoeInitRoutingV3

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnMoeInitRoutingV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which includes the operator calculation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns aclnnStatus status codes. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeInitRoutingV3 defaults to a deterministic implementation.

- The operator supports multiple performance templates on the following products. Each template requires meeting additional admission conditions; otherwise, the generic template is used:
  - Products supporting performance templates:

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3 products</term>
    <!-- end id12 -->

  - Admission conditions for performance templates:
    <table>
      <tr align="center">
        <th style="text-align: center;">Performance Template Type</th>
        <th style="text-align: center;">Admission Conditions</th>
      </tr>
      <tr>
        <td align="center">Low-latency performance template</td>
        <td>All of the following conditions must be met:<ul><li>Input shape requirements for x, expertIdx, and scaleOptional are (1, 7168), (1, 8), and (256, 7168), respectively.</li><li>Data type of x: BFLOAT16.</li><li>Attribute requirements: activeExpertRangeOptional=[0, 256], quantMode=1, expertTokensNumType=2, expertNum=256</li></ul></td>
      </tr>
      <tr>
        <td align="center">Large-batch performance template</td>
        <td>All of the following conditions must be met:<ul><li>NUM_ROWS is within the range [384, 8192] and K=8.</li><li>Attribute requirements: expertNum=256, expertEnd-expertStart<=32, quantMode=-1, rowIdxType=1, expertTokensNumType=1</li></ul></td>
      </tr>
      <tr>
        <td align="center"><br>Full-load performance template</td>
        <td>In scenarios where the operator input shape is small, the multi-core synchronization time between operations accounts for a high proportion and becomes a performance bottleneck. Therefore, a performance template is added for this specialized scenario. In this template, data moving in, sort, and calculation are all completed in the same kernel. The following conditions must be met: <ul style="list-style-type: circle;"><li>Attribute requirements: dropPadMode=0; for Ascend 950PR&950DT products, dropPadMode=1 is also supported in non-quantization scenarios, in which case quantMode=-1, rowIdxType=0, and activeExpertRange=[0, expertNum]</li></ul></td>
      </tr>
    </table>

- Empty tensor processing:
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When NUM_ROWS=0, the empty tensor processing path is used.
  - <term>Ascend 950PR&950DT products</term>: When NUM_ROWS=0 or K=0, there are no routing elements; the empty tensor processing path is used, and the expert count is 0; the output shape must still meet the constraints of the corresponding mode.
  - <term>Ascend 950PR&950DT products</term>: When NUM_ROWS*K&gt;0 and H=0, the normal routing process is still used, and expandedRowIdxOut and expertTokensCountOrCumsumOut are generated.

- Input values of inf/-inf/nan are not supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v3.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)
#define LOG_PRINT(message, ...) \
    do { \
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
    // Boilerplate for resource initialization
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main()
{
    // 1. Boilerplate: device/stream initialization. For details, see the list of ACL public APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs according to the API definition.
    std::vector<int64_t> xShape = {3, 2};
    std::vector<int64_t> expertIdxShape = {3, 4};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> offsetShape = {1};

    std::vector<int64_t> expandedXOutShape = {12, 2};
    std::vector<int64_t> expandedRowIdxOutShape = {12};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalShape = {4};
    std::vector<int64_t> expandedScaleOutOptionalShape = {12};

    std::vector<int64_t> activeExpertRangeArray = {0, 4};

    void *xDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;

    void *expandedXOutDeviceAddr = nullptr;
    void *expandedRowIdxOutDeviceAddr = nullptr;
    void *expertTokensCountOrCumsumOutOptionalDeviceAddr = nullptr;
    void *expandedScaleOutOptionalDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;

    int64_t activeNum = 12;
    int64_t expertCapacity = 0;
    int64_t expertNum = 256;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    bool expertTokensNumFlag = true;
    int64_t quantMode = -1;
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());
    int64_t rowIdxType = 1;

    aclTensor *expandedXOut = nullptr;
    aclTensor *expandedRowIdxOut = nullptr;
    aclTensor *expertTokensCountOrCumsumOutOptional = nullptr;
    aclTensor *expandedScaleOutOptional = nullptr;

    std::vector<float> xHostData = {0.1, 0.1, 0.2, 0.2, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 3, 0, 2, 1, 3, 0, 1, 3, 2};
    std::vector<float> scaleHostData = {0.3423, 0.1652, 0.2652};
    std::vector<float> offsetHostData = {1.8369};

    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalHostData = {0, 0, 0, 0};
    std::vector<float> expandedScaleOutOptionalHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT,
                          &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, expandedRowIdxOutShape, &expandedRowIdxOutDeviceAddr,
                          aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensCountOrCumsumOutOptionalHostData, expertTokensCountOrCumsumOutOptionalShape,
                          &expertTokensCountOrCumsumOutOptionalDeviceAddr, aclDataType::ACL_INT64,
                          &expertTokensCountOrCumsumOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedScaleOutOptionalHostData, expandedScaleOutOptionalShape,
                          &expandedScaleOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &expandedScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnMoeInitRoutingV3.
    ret = aclnnMoeInitRoutingV3GetWorkspaceSize(
        x, expertIdx, scale, offset, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensNumType,
        expertTokensNumFlag, quantMode, activeExpertRange, rowIdxType, expandedXOut, expandedRowIdxOut,
        expertTokensCountOrCumsumOutOptional, expandedScaleOutOptional, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeInitRoutingV3.
    ret = aclnnMoeInitRoutingV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV3 failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the results from the device memory to the host side. The code needs to be modified according to the specific API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr,
                      expandedXSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(expandedRowIdxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]),
                      expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokensCountOrCumsumOutOptionalShape);
    std::vector<int64_t> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(), expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]),
                      expertTokensCountOrCumsumOutOptionalDeviceAddr, expertTokensBeforeCapacitySize * sizeof(int64_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %ld\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(expandedScaleOutOptionalShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(), dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]),
                      expandedScaleOutOptionalDeviceAddr, dynamicQuantScaleSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dynamicQuantScaleSize; i++) {
        LOG_PRINT("dynamicQuantScaleData[%ld] is: %f\n", i, dynamicQuantScaleData[i]);
    }
    // 6. Release aclTensor and aclScalar. The code needs to be modified according to the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensCountOrCumsumOutOptional);
    aclDestroyTensor(expandedScaleOutOptional);

    // 7. Release device resources. The code needs to be modified according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokensCountOrCumsumOutOptionalDeviceAddr);
    aclrtFree(expandedScaleOutOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
