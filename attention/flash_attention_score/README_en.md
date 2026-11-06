# FlashAttentionScore

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 training products</term>|      √     |
|<term>Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |

## Function Description

- Operator function: In training scenarios, the FlashAttention algorithm is used to implement self-attention computation:

    - When pseType=1, perform add first and then mul.
    - When pseType≠1, perform mul first and then add.

- Computation formula:

  The forward computation formula of attention is as follows:

    - When pseType=1, the formula is as follows:
      $$
      attention\_out=Dropout(Softmax(Mask(scale*(pse+(query*d\_scale\_q)*(key*d\_scale\_k)^T), atten\_mask)), keep\_prob)*(value*d\_scale\_v)
      $$

    - When pseType≠1, the formula is as follows:
      $$
      attention\_out=Dropout(Softmax(Mask(scale*((query*d\_scale\_q)*(key*d\_scale\_k)^T) + pse),atten\_mask),keep\_prob)*(value*d\_scale\_v)
      $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>query</td>
      <td>Input</td>
      <td>Input query in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>Input key in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Input value in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>realShiftOptional</td>
      <td>Optional input</td>
      <td>pse in the formula, indicating positional encoding.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dropMaskOptional</td>
      <td>Optional input</td>
      <td>Dropout in the formula, indicating the data dropout mask. A value of 1 means the data is kept, and 0 means the data is dropped.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddingMaskOptional</td>
      <td>Optional input</td>
      <td>Reserved parameter, not used yet.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td>atten_mask in the formula, indicating the attention mask. A value of 1 means the bit does not participate in the computation (does not take effect), and 0 means the bit participates in the computation.</td>
      <td>BOOL, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prefixOptional</td>
      <td>Optional input</td>
      <td>N value of each batch in the prefix sparse computation scenario.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>Optional input</td>
      <td>Actual Q sequence length.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqKvlenOptional</td>
      <td>Optional input</td>
      <td>Actual KV sequence length.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qStartIdxOptional</td>
      <td>Optional input</td>
      <td>Start index of Q.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kvStartIdxOptional</td>
      <td>Optional input</td>
      <td>Start index of KV.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dScaleQOptional</td>
      <td>Optional input</td>
      <td>d_scale_q in the formula, the full quantization parameter of query in FP8 scenarios.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dScaleKOptional</td>
      <td>Optional input</td>
      <td>d_scale_k in the formula, the full quantization parameter of key in FP8 scenarios.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dScaleVOptional</td>
      <td>Optional input</td>
      <td>d_scale_v in the formula, the full quantization parameter of value in FP8 scenarios.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>queryRopeOptional</td>
      <td>Optional input</td>
      <td>RoPE rotary positional encoding input of Q.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keyRopeOptional</td>
      <td>Optional input</td>
      <td>RoPE rotary positional encoding input of K.</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinkOptional</td>
      <td>Optional input</td>
      <td>Sinking parameter.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pScaleOptional</td>
      <td>Optional input</td>
      <td>Scale factor of P.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>scale in the formula, indicating the scale factor. It is used as the scalar value of Muls in the computation flow.</li>
          <li>Default value: 1.0.</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepProb</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>keep_prob in the formula, indicating the probability that data is kept.</li>
          <li>Default value: 1.0.</li>
        </ul>
      </td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used for sparse computation. Indicates the left boundary of the sliding window.</li>
          <li>Default value: 2147483647.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used for sparse computation. Indicates the right boundary of the sliding window.</li>
          <li>Default value: 2147483647.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>Required attribute</td>
      <td>Number of heads on a single card, that is, the length of the N axis of the input query.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputLayout</td>
      <td>Required attribute</td>
      <td>
        <ul>
          <li>Data layout format of the input query, key, and value.</li>
          <li>Supports BSH, SBH, BSND, BNSD, and TND.</li>
          <li>TND uses a compact layout. Tokens of all sequences are concatenated along the T axis, and the shapes of query, key, and value are [T, N, D]. T is the total number of actual tokens in all sequences, N is the number of attention heads, and D is the dimension of each head. actualSeqQlenOptional and actualSeqKvlenOptional provide the cumulative lengths of the query and key/value sequences respectively.</li>
        </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Used to improve precision.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Represents the sparse mode. Supported values: 0, 1, 2, 3, 4, 5, and 6.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pseType</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Controls the execution order of add and mul. Supported values: 0, 1, 2, and 3.</li>
          <li>Default value: 1.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seed</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Random seed of Dropout.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Offset of Dropout.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outDtype</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Output precision control.</li>
          <li>Default value: 0.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxOutLayout</td>
      <td>Optional attribute</td>
      <td>
        <ul>
          <li>Data layout format of the Softmax output.</li>
          <li>Default value: empty.</li>
        </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>Output</td>
      <td>Max intermediate result of the Softmax computation, used for backward computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>Output</td>
      <td>Sum intermediate result of the Softmax computation, used for backward computation.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxOut</td>
      <td>Output</td>
      <td>Reserved parameter, not used yet.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attentionOut</td>
      <td>Output</td>
      <td>attention_out in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Atlas A2 training products</term>: The three data types FLOAT8_E5M2, FLOAT8_E4M3FN, and HIFLOAT8 are not supported.
- <term>Atlas A3 training products</term>: The three data types FLOAT8_E5M2, FLOAT8_E4M3FN, and HIFLOAT8 are not supported.

## Constraints

- The data types of the inputs query, key, value, and realShiftOptional must be consistent.
- The inputLayout of the inputs query, key, and value must be consistent.
- For constraints on the data shape, take BSND and BNSD of inputLayout as an example (H=N\*D for BSH and SBH):
    - B: The value range is 1\~2M. When prefixOptional is used, B supports a maximum of 2K.
    - N: The value range is 1\~256.
    - S: The value range is 1\~1M.
    - D: The value range is 1\~768. If the types of query, key, and value are FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8, the value range of D is 1\~128.
- If the types of query, key, and value are FLOAT8_E5M2, FLOAT8_E4M3FN, or HIFLOAT8, optional parameters such as queryRopeOptional, keyRopeOptional, realShiftOptional, attenMaskOptional, dropMaskOptional, keepProb, and pseType are not supported.
- The value range of keepProb is (0, 1].
- In some scenarios, an excessively large computation workload may cause the operator execution to time out (an aicore error is reported, and errorStr is: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload is affected by parameters such as B, S, N, and D. The larger the values, the larger the workload.
- When pseType is 2 or 3, only equal lengths of Sq and Skv are currently supported.

## Calling Description

| Calling Method           | Calling Sample                                                                                    | Description                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn call | [test_aclnn_flash_attention_score](./examples/test_aclnn_flash_attention_score.cpp) | In non-TND scenarios, calls the FlashAttention operator through the [aclnnFlashAttentionScore](docs/aclnnFlashAttentionScoreV2_en.md) API.             |
| Graph mode call | [test_geir_flash_attention_score](./examples/test_geir_flash_attention_score.cpp) | Calls the FlashAttention operator by building a graph through [Ascend IR](./docs/ascendir_FlashAttentionScore_en.md). |

## Reference Resources

- [Operator Design Principles](./docs/FA_design_en.md)
