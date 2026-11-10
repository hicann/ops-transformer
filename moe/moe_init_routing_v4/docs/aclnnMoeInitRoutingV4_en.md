# aclnnMoeInitRoutingV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:29:32.214Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: routing computation of MoE. The API performs routing processing based on the computation result of [aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2_en.md) and supports the non-quantization, static quantization, and dynamic quantization modes. This API has the following function changes compared with V3 API [aclnnMoeInitRoutingV3](../../moe_init_routing_v3/docs/aclnnMoeInitRoutingV3_en.md). Select an appropriate API as required:<br>
  <ol><li>active_num is an optional input. Currently, only the value NUM_ROWS*K is supported; Active truncation is not supported.</li><li>Optional input topkWeight and optional output expandedTopkWeightOut are added. topkWeight can be rearranged based on the sorted index so that expandedTopkWeightOut and expandedXOut are in one-to-one correspondence. topkWeight and expandedTopkWeightOut must be passed in together or not at all.</li></ol>

- Formula:

  1.Sort the input expertIdx to obtain the sorted result sortedExpertIdx and the corresponding indices sortedRowIdx:

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.Map positions based on sortedRowIdx to obtain expandedRowIdxOut:
    - If rowIdxType is 1, output scatter indices

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - If rowIdxType is 0, output gather indices

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.Compute the histogram for each expert in sortedExpertIdx to obtain expertTokensCountOrCumsumOutOptional:

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.If quantMode is not -1, compute the quant result:
     - Static quant

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$

    - Dynamic quant:
        - If scale is not provided:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - If scale is provided:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - When quantMode is 13, dynamic quantization uses the symmetric quantization range [-8, 7], with 7 as the denominator in scale computation. Every two INT4 values along the H dimension in the quantization result are packed into 1 byte.

  5\. If the active expert range is the full expert range, tokens are moved based on Scatter indexes; otherwise, tokens are moved based on Gather indexes. When dropPadMode is 1, the number of tokens to be processed by each expert is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and the rest are filled with 0. expandedXOut is obtained:
    - Non-quantization scenario
      - Move tokens based on Scatter indexes

      $$
      expandedXOut[i]=x[scatterRowIdx[i] // K]
      $$

      - Move tokens based on Gather indexes

      $$
      expandedXOut[gatherRowIdx[i]]=x[i // K]
      $$

    - Quantization scenario
      - Move tokens based on Scatter indexes

      $$
      expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
      $$

      - Move tokens based on Gather indexes

      $$
      expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
      $$

  6.If topkWeight is provided, rearrange topkWeight based on the sorted indices so that expandedTopkWeightOut corresponds one-to-one with expandedXOut:
    - Dropless use case (dropPadMode=0)

      $$
      expandedTopkWeightOut[i] = topkWeight[sortedRowIdx[i]], \quad 0 \le i < effectiveNum
      $$

      where effectiveNum is:

      $$
      effectiveNum = availableIdxNum
      $$

    - DropPad use case (dropPadMode=1)

      $$
      expandedTopkWeightOut[dstIdx] = topkWeight[globalSortIdx], \quad dstIdx \ne -1
      $$

      where globalSortIdx is the position of expertIdx in the flattened layout (i.e., globalSortIdx = rowIdx * K + colIdx, corresponding to topkWeight[rowIdx, colIdx]), dstIdx is the destination index at the corresponding position in expandedRowIdxOut, and unallocated expert capacity positions in expandedTopkWeightOut are filled with 0.

  7.availableIdxNum is the number of valid elements in expandedRowIdxOut, calculated as the number of elements in expertIdx within the activeExpertRangeOptional range

    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeInitRoutingV4GetWorkspaceSize" API first to obtain the input parameters, calculate the required workspace size, and obtain the executor that contains the operator computation process. Then call the "aclnnMoeInitRoutingV4" API to perform computation.

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
      <td>MoE input, that is, the token feature input.</td>
      <td>The shape is (NUM_ROWS, H).</td>
      <td><ul>
        <li>quantMode=-1: FLOAT16, BFLOAT16, FLOAT32, INT8, HIFLOAT8, FLOAT4_E2M1, FLOAT8_E4M3FN, and FLOAT8_E5M2 are supported;</li>
        <li>quantMode=0 and 1: FLOAT16, BFLOAT16, and FLOAT32 are supported;</li>
        <li>quantMode=2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, and 17: FLOAT16 and BFLOAT16 are supported;</li>
        <li>quantMode=13: FLOAT32 and BFLOAT16 are supported;</li>
        </ul></td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertIdx (aclTensor)</td>
      <td>Input</td>
      <td>K experts for processing each row of features. The expert ID of each element cannot exceed the number of experts.</td>
      <td>The shape is (NUM_ROWS, K).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensor)</td>
      <td>Input</td>
      <td>Specifies the parameter used for calculating quantization results.</td>
      <td><ul>
        <li>If not provided, scale is not used for calculation;</li>
        <li>Optional input in non-quantized scenarios. If provided, it must be a 1D tensor with shape (NUM_ROWS,) and type FLOAT32. When the data type of input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, it must be a 3D tensor with shape (NUM_ROWS, CeilDiv(H, 64), 2) and type FLOAT8_E8M0 if provided;</li>
        <li>Required in static quantization scenarios. The input must be a 1D tensor with shape [1, ];</li>
        <li>Optional input in the INT8 dynamic quantization scenario where quantMode is 1. If provided, it must be a 2D tensor with shape (expertEnd-expertStart, H). Optional input in the INT4 dynamic quantization scenario where quantMode is 13. If provided, the shape must be (1, H), indicating a smooth scale broadcast along the H dimension.</li>
        <li>Do not input in MXFP8 quantization scenarios (quantMode is 2 or 3).</li>
        <li>Do not input in the HIF8 direct conversion or HIF8 PERTOKEN quantization scenarios (quantMode is 6 or 8).</li>
        <li>Required in the HIF8 PERTENSOR quantization scenario (quantMode is 7). The input must be a 1D tensor with shape [1, ].</li>
        <li>Do not input in the MXFP4 quantization scenario (quantMode is 9).</li>
        <li>Do not input in FP8 PerGroup quantization scenarios (quantMode is 4, 5, 14, or 15).</li>
        <li>Do not input in FP8 PerBlock quantization scenarios (quantMode is 11 or 12).</li>
        <li>Do not input in MXFP8 RoundScale+Amax quantization scenarios (quantMode is 16 or 17).</li>
      </ul></td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensor)</td>
      <td>Input</td>
      <td>Specifies the offset value used to calculate the quant result.</td>
      <td><ul>
        <li>Not required in non-quantized scenarios.</li><li>Required in static quantization scenarios. Input requirements: a 1D tensor with shape [1, ].</li>
        <li>Not required for dynamic quantization (quantMode is 1), MXFP8 quantization (quantMode is 2 or 3), HIF8 quantization (quantMode is 6, 7, or 8), MXFP4 quantization (quantMode is 9), FP8 PerGroup quantization (quantMode is 4, 5, 14, or 15), FP8 PerBlock quantization (quantMode is 11 or 12), INT4 dynamic quantization (quantMode is 13), or MXFP8 RoundScale+Amax quantization scenarios (quantMode is 16 or 17).</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNumOptional (aclTensor)</td>
      <td>Input</td>
      <td>Maximum total number of rows to process, currently fixed at NUM_ROWS*K. The actual number of valid rows depends on routing rules such as expert range filtering.</td>
      <td><ul>
        <li>Optional input. If not provided, activeNum defaults to NUM_ROWS*K;</li>
        <li>If provided, it must be a scalar Tensor (shape () or (1,)) of data type INT64. Only the value NUM_ROWS*K is supported. Currently, this parameter cannot be used to limit the number of routes processed.</li>
      </ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>0-1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topkWeightOptional (aclTensor)</td>
      <td>Input</td>
      <td>Routing weights of topk experts, rearranged by sorted index for one-to-one correspondence with expandedXOut.</td>
      <td><ul>
        <li>Optional input. If not provided, expandedTopkWeightOut is not output;</li>
        <li>Input shape is (NUM_ROWS, K), and the data type supports only FLOAT32.</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of tokens each expert can process.</td>
      <td>In the Dropless use case, only its value is validated and this parameter is not used; in the DropPad use case, the value range is (0, NUM_ROWS].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of experts.</td>
      <td>If expertTokensNumType is key_value mode, the value range is [1, 5120]; for other modes, the value range is [1, 10240].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates whether it is the DropPad scenario.</td>
      <td>Value options: 0 and 1
        <br>0: Dropless scenario. expertCapacity is not verified in this scenario.
        <br>1: DropPad scenario.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumType (int64_t)</td>
      <td>Input</td>
      <td>Represents different histogram modes.</td>
      <td>The value can be 0, 1, or 2.
        <br>0: cumsum mode;
        <br>1: count mode;
        <br>2: key_value mode;
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensNumFlag (bool)</td>
      <td>Input</td>
      <td>Indicates whether to output expertTokensCountOrCumsumOut. </td>
      <td>Value options: false and true</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates different quantization scenarios.</td>
      <td>The value can be 0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17
        <br>0: static quantization scenario;
        <br>1: dynamic quantization scenario, with expandedXOut quantized to INT8;
        <br>-1: non-quantized scenario;
        <br>2: MXFP8 quantization scenario, with expandedXOut quantized to FLOAT8_E5M2;
        <br>3: MXFP8 quantization scenario, with expandedXOut quantized to FLOAT8_E4M3FN;
        <br>4: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), with expandedXOut quantized to FLOAT8_E5M2;
        <br>5: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale), with expandedXOut quantized to FLOAT8_E4M3FN;
        <br>6: HIF8 direct conversion quantization scenario, with expandedXOut quantized to HIFLOAT8;
        <br>7: HIF8 PERTENSOR quantization scenario, with expandedXOut quantized to HIFLOAT8 in pertensor mode;
        <br>8: HIF8 PERTOKEN quantization scenario, with expandedXOut quantized to HIFLOAT8 in pertoken mode;
        <br>9: MXFP4 quantization scenario, with expandedXOut quantized to FLOAT4_E2M1;
        <br>11: FP8 PerBlock quantization scenario (BlockSize=128), with expandedXOut quantized to FLOAT8_E5M2 and expandedScaleOut in the FLOAT32 3D layout;
        <br>12: FP8 PerBlock quantization scenario (BlockSize=128), with expandedXOut quantized to FLOAT8_E4M3FN and expandedScaleOut in the FLOAT32 3D layout;
        <br>13: INT4 dynamic quantization scenario, with expandedXOut quantized to INT4;
        <br>14: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), with expandedXOut quantized to FLOAT8_E5M2;
        <br>15: FP8 PerGroup quantization scenario (GroupSize=128, RoundScale+Amax), with expandedXOut quantized to FLOAT8_E4M3FN;
        <br>16: MXFP8 RoundScale+Amax quantization scenario, with expandedXOut quantized to FLOAT8_E5M2;
        <br>17: MXFP8 RoundScale+Amax quantization scenario, with expandedXOut quantized to FLOAT8_E4M3FN.
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeExpertRangeOptional(aclIntArray)</td>
      <td>Input</td>
      <td>Represents the range of active experts.</td>
      <td>The length is 2, and the value is [expertStart, expertEnd], which is a left-closed and right-open interval. The values must be greater than or equal to 0, and expertEnd must not be greater than expertNum. In the Drop/Pad scenario, expertStart is 0 and expertEnd is expertNum.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowIdxType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the index type used by expandedRowIdxOut.</td>
      <td>Value: 0 or 1
        <br>0: gather type index
        <br>1: scatter type index</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut (aclTensor)</td>
      <td>Output</td>
      <td>Features expanded based on expertIdx.</td>
      <td><ul>
        <li>In Dropless scenarios, the shape is [NUM_ROWS * K, H].</li>
        <li>In Drop/Pad scenarios, the tensor must be 3D, with the shape [expertNum, expertCapacity, H].</li>
        <li>In non-quantization scenarios, the data type is the same as that of x. In quantization scenarios, the supported data type is INT8 when quantMode is 0 or 1; INT4 when quantMode is 13 and the data type of x is FLOAT32 or BFLOAT16; FLOAT8_E5M2 when quantMode is 2, 4, 14, or 16; FLOAT8_E4M3FN when quantMode is 3, 5, 15, or 17; HIFLOAT8 when quantMode is 6, 7, or 8; FLOAT4_E2M1 when quantMode is 9; and FLOAT8_E5M2 and FLOAT8_E4M3FN when quantMode is 11 and 12, respectively.</li>
      </ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut (aclTensor)</td>
      <td>Output</td>
      <td>Index mapping relationship between expandedXOut and x.</td>
      <td>The output shape is (NUM_ROWS*K, ). The first availableIdxNum elements are valid data, and the rest are invalid data determined by rowIdxType:
        <ul><li>When rowIdxType is 0, invalid data is filled with -1.</li>
        <li>When rowIdxType is 1, invalid data is uninitialized.</li></ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut (aclTensor)</td>
      <td>Output</td>
      <td>Statistics or cumulative value of the number of tokens processed by each expert.</td>
      <td><ul>
        <li>If expertTokensNumFlag is false, the output is an empty tensor.</li>
        <li>If expertTokensNumType is 0, the output is the prefix sum of the total number of tokens processed by each expert in activeExpertRangeOptional after sorting. The output shape is [expertEnd-expertStart].</li>
        <li>If expertTokensNumType is 1, the output is the total number of tokens processed by each expert in activeExpertRangeOptional. The output shape is [expertEnd-expertStart].</li>
        <li>If expertTokensNumType is 2, the output is the experts with a non-zero total number of tokens in activeExpertRangeOptional, and the total number of tokens processed by the corresponding experts. The output shape is [expertNum, 2].</li>
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
      <td> The output shape is the product of all dimensions of expandedXOut except the last one.
        <ul style="list-style-type: circle;">
        <li>In non-quantization scenarios, if scaleOptional is input, the shape is [NUM_ROWS*K], the first availableIdxNum elements are valid data, and the output is of the FLOAT32 type. If the data type of input x is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2 and scaleOptional is input, the shape of expandedScaleOut is [NUM_ROWS*K, CeilDiv(H, 64), 2], and the output is of the FLOAT8_E8M0 type. In the Drop/Pad use case, the output is a 1D tensor with shape [expertNum * expertCapacity] and is of the FLOAT32 type.</li>
        <li>In dynamic quantization scenarios, if scaleOptional is input, the first availableIdxNum elements are valid data.</li>
        <li>In static quantization scenario (quantMode is 0), HIF8 direct conversion quantization scenario (quantMode is 6), and HIF8 PERTENSOR quantization scenario (quantMode is 7), expandedScaleOut is not output.</li>
        <li>In MXFP8 quantization scenario (quantMode is 2, 3, 16, or 17), the output is of the FLOAT8_E8M0 type, with shape [NUM_ROWS*K, M], where M=CeilAlign(CeilDiv(H,32),2). The first availableIdxNum rows of the NUM_ROWS*K rows are valid data.</li>
        <li>When data is quantized to HIFLOAT8 through direct conversion, expandedScaleOut is not output.</li>
        <li>When data is quantized to HIFLOAT8 in PERTENSOR mode, expandedScaleOut is not output.</li>
        <li>When data is quantized to HIFLOAT8 in PERTOKEN mode, the output is of the FLOAT32 type, with shape [NUM_ROWS*K].</li>
        <li>In MXFP4 quantization scenario, the output is of the FLOAT8_E8M0 type, with shape [NUM_ROWS*K, M, 2], where M=CeilDiv(H, 64). The first availableIdxNum rows of the NUM_ROWS*K rows are valid data.</li>
        <li>In FP8 PerGroup quantization scenario (quantMode is 4, 5, 14, or 15), the output is of the FLOAT32 type, with shape [NUM_ROWS*K, CeilDiv(H, 128)]. The first availableIdxNum rows of the NUM_ROWS*K rows are valid data.</li>
        <li>In FP8 PerBlock quantization scenario (quantMode is 11 or 12), the output is of the FLOAT32 type, with shape [NUM_ROWS*K, CeilDiv(H,256), 2]. The first availableIdxNum rows of the NUM_ROWS*K rows are valid data.</li></ul>
      </td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedTopkWeightOut (aclTensor)</td>
      <td>Output</td>
      <td>Routing weights rearranged by sorted index, in one-to-one correspondence with expandedXOut.</td>
      <td><ul>
        <li>Optional output; must be output together with the topkWeight input;</li>
        <li>In the Dropless use case, the shape is [NUM_ROWS*K, 1]; the first effectiveNum elements are valid data and the rest are uninitialized, where effectiveNum=availableIdxNum;</li>
        <li>In the DropPad use case, the shape is [expertNum*expertCapacity, 1]; positions with allocated capacity contain valid data, and positions without allocated capacity are filled with 0;</li>
        <li>Only the FLOAT32 data type is supported.</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
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
      <td>The compute input and compute output are null pointers.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of the input and output are not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>      The shapes of the input and output tensors are not in the supported range.<br>
      The input attributes are not in the supported range.<br>
      </td>
    </tr>
  </tbody></table>

## aclnnMoeInitRoutingV4

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnMoeInitRoutingV4GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor for the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When explicitly provided, `activeNumOptional` only supports the value `NUM_ROWS*K`. Active truncation is currently not supported.

- Deterministic computation:
  - aclnnMoeInitRoutingV4 defaults to a deterministic implementation.

- The performance template is supported on the following products. Additional admission conditions must be met; otherwise, the generic template is used:
  - Products that support the performance template:
    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: Supported
    <!-- end id7 -->
  - Admission conditions of the performance template:
    <table>
      <tr align="center">
        <th style="text-align: center;">Performance Template Type</th>
        <th style="text-align: center;">Admission Conditions</th>
      </tr>
      <tr>
        <td align="center">Full-load performance template</td>
        <td>When the operator input shape is small, the multi-core synchronization time between operations accounts for a high proportion and becomes the performance bottleneck. Therefore, a performance template is added for this specialized use case. In this template, data transfer-in, sorting, and computation are completed in the same kernel. The following conditions must be met: <ul style="list-style-type: circle;"><li>Attribute requirements: dropPadMode=0. The non-quantized scenario also supports dropPadMode=1, in which case quantMode=-1, rowIdxType=0, and activeExpertRange=[0, expertNum]</li></ul></td>
      </tr>
    </table>

- Special constraints for DropPad mode (when dropPadMode=1):
  - quantMode supports only -1 (non-quantization).
  - rowIdxType supports only 0 (gather index).
  - activeExpertRangeOptional must be [0, expertNum].

- For the MXFP4/INT4 dynamic quantization scenario where quantMode is 9 or 13, and the non-quantized passthrough scenario where quantMode is -1 and the data type of x is FLOAT4_E2M1, H must be an even number. The INT4 dynamic quantization scenario where quantMode is 13 must also meet the following requirements: the data type of x is FLOAT32 or BFLOAT16, dropPadMode is 0, and offsetOptional is not input.

- Empty tensor processing:
  - <term>Ascend 950PR&950DT products</term>: When NUM_ROWS=0 or K=0, there are no routing elements. The empty tensor processing path is used, and the expert count is 0. The output shapes must still meet the constraints of the corresponding mode.
  - <term>Ascend 950PR&950DT products</term>: When NUM_ROWS*K&gt;0 and H=0, the normal routing process is still followed to generate expandedRowIdxOut and expertTokensCountOrCumsumOut.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v4.h"

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
    // Boilerplate for resource initialization.
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate the strides of the contiguous tensor.
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
    // 1. Boilerplate: device/stream initialization. For details, see the list of acl external APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {3, 2};
    std::vector<int64_t> expertIdxShape = {3, 4};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> offsetShape = {1};
    std::vector<int64_t> activeNumShape = {};
    std::vector<int64_t> topkWeightShape = {3, 4};

    std::vector<int64_t> expandedXOutShape = {12, 2};
    std::vector<int64_t> expandedRowIdxOutShape = {12};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalShape = {4};
    std::vector<int64_t> expandedScaleOutOptionalShape = {12};
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

    int64_t expertCapacity = 4;
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
    aclTensor *expandedTopkWeightOut = nullptr;

    std::vector<float> xHostData = {0.1, 0.1, 0.2, 0.2, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 3, 0, 2, 1, 3, 0, 1, 3, 2};
    std::vector<float> scaleHostData = {0.3423, 0.1652, 0.2652};
    std::vector<float> offsetHostData = {1.8369};
    std::vector<int64_t> activeNumHostData = {12};
    std::vector<float> topkWeightHostData = {0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65};

    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalHostData = {0, 0, 0, 0};
    std::vector<float> expandedScaleOutOptionalHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> expandedTopkWeightOutHostData(expandedTopkWeightOutShape[0] * expandedTopkWeightOutShape[1],
                                                     0.0f);

    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(activeNumHostData, activeNumShape, &activeNumDeviceAddr, aclDataType::ACL_INT64,
                          &activeNumOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(topkWeightHostData, topkWeightShape, &topkWeightDeviceAddr, aclDataType::ACL_FLOAT,
                          &topkWeightOptional);
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
    ret = CreateAclTensor(expandedTopkWeightOutHostData, expandedTopkWeightOutShape, &expandedTopkWeightOutDeviceAddr,
                          aclDataType::ACL_FLOAT, &expandedTopkWeightOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnMoeInitRoutingV4.
    ret = aclnnMoeInitRoutingV4GetWorkspaceSize(
        x, expertIdx, scale, offset, activeNumOptional, topkWeightOptional, expertCapacity, expertNum, dropPadMode,
        expertTokensNumType, expertTokensNumFlag, quantMode, activeExpertRange, rowIdxType, expandedXOut,
        expandedRowIdxOut, expertTokensCountOrCumsumOutOptional, expandedScaleOutOptional, expandedTopkWeightOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV4GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMoeInitRoutingV4
    ret = aclnnMoeInitRoutingV4(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV4 failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate: synchronously wait for task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from device memory to the host. Modify the code based on the actual API definition
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
    auto expertTokensCountSize = GetShapeSize(expertTokensCountOrCumsumOutOptionalShape);
    std::vector<int64_t> expertTokensCountData(expertTokensCountSize, 0);
    ret = aclrtMemcpy(expertTokensCountData.data(), expertTokensCountData.size() * sizeof(expertTokensCountData[0]),
                      expertTokensCountOrCumsumOutOptionalDeviceAddr, expertTokensCountSize * sizeof(int64_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensCountSize; i++) {
        LOG_PRINT("expertTokensCountData[%ld] is: %ld\n", i, expertTokensCountData[i]);
    }

    auto expandedScaleSize = GetShapeSize(expandedScaleOutOptionalShape);
    std::vector<float> expandedScaleData(expandedScaleSize, 0);
    ret = aclrtMemcpy(expandedScaleData.data(), expandedScaleData.size() * sizeof(expandedScaleData[0]),
                      expandedScaleOutOptionalDeviceAddr, expandedScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedScaleSize; i++) {
        LOG_PRINT("expandedScaleData[%ld] is: %f\n", i, expandedScaleData[i]);
    }

    auto expandedTopkWeightSize = GetShapeSize(expandedTopkWeightOutShape);
    std::vector<float> expandedTopkWeightData(expandedTopkWeightSize, 0);
    ret =
        aclrtMemcpy(expandedTopkWeightData.data(), expandedTopkWeightData.size() * sizeof(expandedTopkWeightData[0]),
                    expandedTopkWeightOutDeviceAddr, expandedTopkWeightSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedTopkWeightSize; i++) {
        LOG_PRINT("expandedTopkWeightData[%ld] is: %f\n", i, expandedTopkWeightData[i]);
    }
    // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition
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
    aclDestroyIntArray(activeExpertRange);

    // 7. Release device resources. Modify the code based on the actual API definition
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
