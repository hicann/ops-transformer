# QuantGroupedMatMulAlltoAllv

English | [简体中文](./README.md)

## Product Support Status

| Product                                                       | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- **Operator function**: Fuses GroupedMatMul, Unpermute, and AlltoAllv for routed experts, and enables parallel fusion with the shared-expert MatMul. **Computation comes before communication**.

- **Computation formula**:
    - Routed experts:
        $$
        gmmY = (gmmX @ gmmWeight) * gmmXScale * gmmWeightScale \\
        unpermuteOut = Unpermute(gmmY) \\
        y = AlltoAllv(unpermuteOut)
        $$

    - Shared expert:

        $$
        mmY = (mmX @  mmWeight) * mmXScaleOptional * mmWeightScaleOptional
        $$

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1200px">
      <colgroup>
          <col style="width: 188px">
          <col style="width: 85px">
          <col style="width: 220px">
          <col style="width: 280px">
          <col style="width: 200px">
          <col style="width: 90px">
          <col style="width: 100px">
          <col style="width: 80px">
      </colgroup>
      <thead>
          <tr>
              <th>Parameter Name</th>
              <th>Input/Output</th>
              <th>Description</th>
              <th>Usage Instructions</th>
              <th>Data Type</th>
              <th>Data Format</th>
              <th>Dimensions(shape)</th>
              <th>Non-contiguous Tensor</th>
          </tr>
      </thead>
      <tbody>
          <!-- gmmX -->
          <tr>
              <td>gmmX</td>
              <td>Input</td>
              <td>Input gmmX in the formula.</td>
              <td>Shape (A, H1).</td>
              <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- gmmWeight -->
          <tr>
              <td>gmmWeight</td>
              <td>Input</td>
              <td>Input gmmWeight in the formula.</td>
              <td>Shape (e, H1, N1). e is the number of experts deployed on each card, H1 is the hidden size, and N1 is the intermediate dimension of the routed-expert FFN.</td>
              <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
              <td>ND</td>
              <td>3</td>
              <td>x</td>
          </tr>
          <!-- gmmXScale -->
          <tr>
              <td>gmmXScale</td>
              <td>Input</td>
              <td>Quantization factor of gmmX.</td>
              <td><li>pertensor quantization: shape (1).</li><li>mx quantization: shape (A, ceil(H1/64), 2)</li></td>
              <td>FLOAT32, FLOAT8_E8M0</td>
              <td>ND</td>
              <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
              <td>x</td>
          </tr>
          <!-- gmmWeightScale -->
          <tr>
              <td>gmmWeightScale</td>
              <td>Input</td>
              <td>Quantization factor of gmmWeight.</td>
              <td><li>pertensor quantization: shape (1).</li><li>mx quantization: shape (e, N1, ceil(H1/64), 2), or (e, ceil(H1/64), N1, 2) when the weight is transposed</li></td>
              <td>FLOAT32, FLOAT8_E8M0</td>
              <td>ND</td>
              <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
              <td>x</td>
          </tr>
          <!-- sendCountsTensorOptional -->
          <tr>
              <td>sendCountsTensorOptional</td>
              <td>Input</td>
              <td>send count used by AlltoAllv.</td>
              <td>Only empty is supported currently. Shape (e * ep, ). e is the number of experts deployed on each card, and ep is the size of the EP domain.</td>
              <td>INT64</td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- recvCountsTensorOptional -->
          <tr>
              <td>recvCountsTensorOptional</td>
              <td>Input</td>
              <td>recv count used by AlltoAllv.</td>
              <td>An empty tensor is passed by default. Shape (e * ep, ). e is the number of experts deployed on each card, and ep is the size of the EP domain.</td>
              <td>INT64</td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmXOptional -->
          <tr>
              <td>mmXOptional</td>
              <td>Input</td>
              <td>Input mmX in the formula.</td>
              <td>Shape (bs, H2). bs is the number of experts deployed on each card, and H2 is the hidden size.</td>
              <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- mmWeightOptional -->
          <tr>
              <td>mmWeightOptional</td>
              <td>Input</td>
              <td>Input mmWeight in the formula.</td>
              <td>Shape (H2, N2). H2 is the hidden size, and N2 is the intermediate dimension of the shared-expert FFN.</td>
              <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- mmXScaleOptional -->
          <tr>
              <td>mmXScaleOptional</td>
              <td>Input</td>
              <td>Quantization factor of mmX.</td>
              <td><li>pertensor quantization: shape (1).</li><li>mx quantization: shape (BS, ceil(H2/64), 2)</li></td>
              <td>FLOAT32, FLOAT8_E8M0</td>
              <td>ND</td>
              <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
              <td>x</td>
          </tr>
          <!-- mmWeightScaleOptional -->
          <tr>
              <td>mmWeightScaleOptional</td>
              <td>Input</td>
              <td>Quantization factor of mmWeight.</td>
              <td><li>pertensor quantization: shape(1).</li><li>mx quantization: shape (N2, ceil(H2/64), 2), or (ceil(H2/64), N2, 2) when the weight is transposed</li></td>
              <td>FLOAT32, FLOAT8_E8M0</td>
              <td>ND</td>
              <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
              <td>x</td>
          </tr>
          <!-- commQuantScaleOptional -->
          <tr>
              <td>commQuantScaleOptional</td>
              <td>Input</td>
              <td>Quantization factor for low-bit communication.</td>
              <td>Reserved parameter. Only empty is supported currently.</td>
              <td>FLOAT32</td>
              <td>ND</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- gmmXQuantMode -->
          <tr>
              <td>gmmXQuantMode</td>
              <td>Input</td>
              <td>Quantization mode of gmmX.</td>
              <td>The quantization mode must be passed. Currently, 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- gmmWeightQuantMode -->
          <tr>
              <td>gmmWeightQuantMode</td>
              <td>Input</td>
              <td>Quantization mode of gmmWeight.</td>
              <td>The quantization mode must be passed. Currently, 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmXQuantMode -->
          <tr>
              <td>mmXQuantMode</td>
              <td>Input</td>
              <td>Quantization mode of mmX.</td>
              <td>If mmX is not empty, the quantization mode must be passed. Currently, 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- mmWeightQuantMode -->
          <tr>
              <td>mmWeightQuantMode</td>
              <td>Input</td>
              <td>Quantization mode of mmWeight.</td>
              <td>If mmWeight is not empty, the quantization mode must be passed. Currently, 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- commQuantMode -->
          <tr>
              <td>commQuantMode</td>
              <td>Input</td>
              <td>Quantization mode for low-bit communication.</td>
              <td>The low-bit feature is reserved currently. 0 must be passed, indicating no quantization.</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- commQuantDtypeOptional -->
          <tr>
              <td>commQuantDtypeOptional</td>
              <td>Input</td>
              <td>Data type for low-bit communication.</td>
              <td>The low-bit feature is reserved currently. -1 must be passed.</td>
              <td>INT64</td>
              <td>-</td>
              <td>1</td>
              <td>x</td>
          </tr>
          <!-- groupSize (new)-->
          <tr>
              <td>groupSize</td>
              <td>Input</td>
              <td>PerGroup quantization group size.</td>
              <td>Group sizes along the three dimensions used for the Matmul computation. This is a reserved parameter, and only 0 can be configured. The value does not take effect. The groupSize input is composed of the three values groupSizeM, groupSizeN, and groupSizeK concatenated. Each value occupies 16 bits, occupying the lower 48 bits of the int64_t type groupSize (the upper 16 bits are invalid). The computation formula is: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
              <td>INT64</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- group -->
          <tr>
              <td>group</td>
              <td>Input</td>
              <td>Communicator identifier.</td>
              <td>The string length must be greater than 0 and less than 128.</td>
              <td>char*</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- epWorldSize -->
          <tr>
              <td>epWorldSize</td>
              <td>Input</td>
              <td>Size of the communicator.</td>
              <td>2/4/8/16/32/64/128/256 are supported.</td>
              <td>INT64</td>
              <td>-</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- sendCounts -->
          <tr>
              <td>sendCounts</td>
              <td>Input</td>
              <td>send count used by AlltoAllv. Indicates the number of tokens sent by other ranks to each expert on the current rank.</td>
              <td>The supported dimension is e * ep. It is flattened into one dimension as <code>sendCounts[fromRank][expertId]</code>. For example, when e=3, the order is <code>e0,e1,e2,e0,e1,e2, ...</code></td>
              <td>aclIntArray* (element type INT64)</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- recvCounts -->
          <tr>
              <td>recvCounts</td>
              <td>Input</td>
              <td>recv count used by AlltoAllv. Indicates the number of tokens that the current card needs to receive after AlltoAllv.</td>
              <td>The supported dimension is e * ep. It is flattened into one dimension as <code>recvCounts[fromRank][expertId]</code>. For example, when e=3, the order is <code>e0,e1,e2,e0,e1,e2, ...</code></td>
              <td>aclIntArray* (element type INT64)</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- transGmmWeight -->
          <tr>
              <td>transGmmWeight</td>
              <td>Input</td>
              <td>Whether the right matrix of GroupedMatMul is transposed.</td>
              <td>Must be passed. There is no default value.</td>
              <td>BOOL</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- transMmWeight -->
          <tr>
              <td>transMmWeight</td>
              <td>Input</td>
              <td>Whether the right matrix of MatMul is transposed.</td>
              <td>Must be passed. There is no default value.</td>
              <td>BOOL</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- y -->
          <tr>
              <td>y</td>
              <td>Output</td>
              <td>Output of the grouped matmul computation.</td>
              <td>Empty tensors are not supported. Shape (BSK, N1).</td>
              <td>FLOAT16, BFLOAT16</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- mmYOptional -->
          <tr>
              <td>mmYOptional</td>
              <td>Output</td>
              <td>Output of the matmul computation.</td>
              <td>Shape (bs, N2).</td>
              <td>FLOAT16, BFLOAT16</td>
              <td>ND</td>
              <td>2</td>
              <td>x</td>
          </tr>
          <!-- workspaceSize -->
          <tr>
              <td>workspaceSize</td>
              <td>Output</td>
              <td>Returns the size of the workspace to be requested on the device.</td>
              <td>-</td>
              <td>UINT64</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
          <!-- executor -->
          <tr>
              <td>executor</td>
              <td>Output</td>
              <td>Returns the op executor, which contains the operator computation process.</td>
              <td>-</td>
              <td>aclOpExecutor*</td>
              <td>ND</td>
              <td>-</td>
              <td>-</td>
          </tr>
      </tbody>
  </table>

  The relationship between the enumerated values of gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, and commQuantMode and the [quantization modes](../../docs/zh/context/quant_mode_introduction_en.md) is as follows:

  * 0: no quantization -- not supported currently
  * 1: pertensor
  * 2: perchannel
  * 3: pertoken
  * 4: pergroup
  * 5: perblock
  * 6: mx quantization
  * 7: pertoken dynamic quantization

## Constraints

- Deterministic computation:
  - aclnnQuantGroupedMatMulAlltoAllv is deterministic by default.
- Communication engine constraints:
  - Ascend 950DT products: CCU communication and AI_CPU communication are supported. CCU supports only intra-node interconnection within the UB domain, and AI_CPU supports inter-node interconnection within the UB domain.
- e * epWorldSize supports a maximum of 256. e indicates the number of experts on a single card and supports a maximum of 32. epWorldSize supports 2/4/8/16/32/64/128/256;
- The shape of gmmX is (A, H1). A is the sum of sendCounts, and the value range of H1 is (0, 65536);
- The shape of gmmWeight is (e, H1, N1), and the value range of N1 is (0, 65536);
- The shape of y is (BSK, N1). In the first dimension, K is in the range [2, 8], and BSK is the sum of recvCounts;
- mmX is the left matrix of the shared expert, with the shape of (BS, H2). The value range of H2 is (0, 12288];
- mmWeight is the right matrix of the shared expert, with the shape of (H2, N2). The value range of N2 is (0, 65536);
- sendCounts is the number of tokens sent to other cards, and the array size is e * epWorldSize;
- recvCounts is the number of tokens received from other cards, and the array size is e * epWorldSize;
- The quantization Scale, Mode, and other parameters of the routed expert and the shared expert are all required;
- The low-bit communication Mode is a required parameter, while DType and Scale are optional. When Mode is non-0, DType and Scale must be provided;
- Variables used in shapes in the parameter description:
  - BSK: number of tokens received by the current card. It is the sum of the recvCounts parameter. Value range: (0, 52428800).
  - H1: hidden size of the routed expert. Value range: (0, 65536).
  - H2: hidden size of the shared expert. Value range: (0, 12288].
  - e: number of experts on a single card. 0<e<=32. e * epWorldSize supports a maximum of 256.
  - N1: intermediate dimension of the routed-expert FFN. Value range: (0, 65536).
  - N2: intermediate dimension of the shared-expert FFN. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: number of TopK experts selected. Range of K: [2, 8].
  - A: number of tokens sent by the current card. It is the sum of the sendCounts parameter.
  - The sum of A over all cards in the EP communicator equals the sum of BSK over all cards.
  - For mx quantization when gmmX and gmmWeight are FLOAT4_E2M1, H1 and H2 must be even numbers and cannot be 2. When transGmmWeight and transMmWeight are false, N1 and N2 must be even numbers.
  - The transpose states of gmmWeight and gmmWeightScale must be consistent: both transposed or both not transposed. The transpose states of mmWeight and mmWeightScale must also be consistent.
  - groupSize:
    - groupSize is valid only when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher; in other scenarios, 0 must be passed.
    - The groupSize value supports formula-based derivation: the passed groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. When one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK used for computation are reset based on the shapes of the gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs. The setting principle: if groupSizeM=0, the quantization grouping value along the m dimension is derived by the API, and the derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the gmmX/mmX shape, and scaleM is the same as m in the gmmXScale/mmXScale shape; if groupSizeK=0, the quantization grouping value along the k dimension is derived by the API, and the derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is the same as k in the gmmX/mmX shape, and scaleK is the same as k in the gmmXScale/mmXScale shape; if groupSizeN=0, the quantization grouping value along the n dimension is derived by the API, and the derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is the same as n in the gmmWeight/mmWeight shape, and scaleN is the same as n in the gmmWeightScale/mmWeightScale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - If the reset condition is met, when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher and their data types are all FLOAT8_E8M0, the value combination [groupSizeM, groupSizeN, groupSizeK] is derived as [1, 1, 32], and the corresponding groupSize value is 4295032864.

- Quantization parameter constraints:
  - The current version supports pertensor quantization and mx quantization.
- Type Constraints
  - pertensor quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | y |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16/BFLOAT16 |

  - mx quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | y |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |

  - The type of mmX is the same as that of gmmX, the type of mmWeight is the same as that of gmmWeight, and the type of mmY is the same as that of y.

## Calling Instructions

| Calling Method   | Sample Code                                  | Description                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn API | [test_aclnn_quant_grouped_mat_mul_allto_allv.cpp](./examples/test_aclnn_quant_grouped_mat_mul_allto_allv.cpp) | Calls the quant_grouped_mat_mul_allto_allv operator in the quantization scenario through the [aclnnQuantGroupedMatMulAlltoAllv](docs/aclnnQuantGroupedMatMulAlltoAllv_en.md) API. |
