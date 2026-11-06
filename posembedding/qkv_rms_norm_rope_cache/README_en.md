# QkvRmsNormRopeCache

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: takes the fused qkv tensor as input, splits it into q, k, and v tensors through SplitVD, performs the fused RmsNorm, ApplyRotaryPosEmb, Quant, and Scatter operations, and outputs q_out, k_cache, v_cache, q_out_before_quant (optional), k_out_before_quant (optional), and v_out_before_quant (optional).
- The scenarios currently supported by this operator are as follows:

  |         Scenario Type           |            Overview            |
  |:----------------------|:-----------------------------|
  |<ul><li>cache_mode is PA_NZ</li><li>q is not quantized</li><li>k and v support non-quantization, symmetric quantization, and asymmetric quantization</li><li>q_out_before_quant/k_out_before_quant/v_out_before_quant are not output</li></ul>|The qkv shape is [$B_{qkv}$ * $S_{qkv}$, $N_{qkv}$ * $D_{qkv}$], and q, k, and v have exactly the same D dimension. The main computation process and output mapping: <br><ul><li>qkv goes through SplitVD->q, k, v</li><li>q goes through RmsNorm, RoPE->q_out</li><li>k goes through RmsNorm, RoPE, Quant (optional), Scatter->k_cache</li><li>v goes through Quant (optional), Scatter->v_cache</li></ul>|

- Computation formula:

  (1) SplitVD:

  In the following formula, $N_q$, $N_k$, and $N_v$ denote the numbers of attention heads of the q, k, and v components respectively, which must satisfy:

  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] * D_{qkv}] \\
  k &= qkv[..., [N_q:-N_v] * D_{qkv}] \\
  v &= qkv[..., [-N_v:] * D_{qkv}]
  \end{aligned}
  $$

  (2) RmsNorm:

  Here, x and y denote the input tensor and the output tensor of RmsNorm respectively. The normalization is performed along the last dimension (feature dimension), and this computation rule applies to both the q and k components.

  $$
  squareX = x * x
  $$

  $$
  meanSquareX = squareX.mean(dim = -1, keepdim = True)
  $$

  $$
  rms = \sqrt{meanSquareX + epsilon}
  $$

  $$
  y = (x / rms) * gamma
  $$

  (3) RoPE (Half-and-Half):

  Here, y refers to the output result of the RmsNorm computation.

  $$
  y1 = y[\ldots, :d/2]
  $$

  $$
  y2 = y[\ldots, d/2:]
  $$

  $$
  y\_RoPE = torch.cat((-y2, y1), dim = -1)
  $$

  $$
  y\_embed = (y * cos) + y\_RoPE * sin
  $$

  (4) Quant:

  No quantization:

  $$
  kQuant = kRoPE \\
  vQuant = v
  $$

  Symmetric quantization:

  $$
  kQuant = kRoPE / kScale \\
  vQuant = v / vScale
  $$

  Asymmetric quantization:

  $$
  kQuant = kRoPE / kScale + kOffset \\
  vQuant = v / vScale + vOffset
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
      <td>qkv</td>
      <td>Input</td>
      <td>Input data used to split out q, k, and v, corresponding to qkv in the formula. The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>qkv</sub> * D<sub>qkv</sub>].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_gamma</td>
      <td>Input</td>
      <td>Input data used for the rms_norm computation of q, corresponding to gamma in the formula. The data type is the same as that of the input qkv, and the shape is [D<sub>qkv</sub>].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_gamma</td>
      <td>Input</td>
      <td>Input data used for the rms_norm computation of k, corresponding to gamma in the formula. The data type is the same as that of the input qkv, and the shape is [D<sub>qkv</sub>].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for the rope computation, which applies the cosine transformation to the input tensor, corresponding to cos in the formula. The data type is the same as that of the input qkv, and the shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, 1 * D_rope], where D_rope = D<sub>qkv</sub> and (D<sub>qkv</sub> * the number of bytes occupied by the qkv data type) must be divisible by 32.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Input data used for the rope computation, which applies the sine transformation to the input tensor, corresponding to sin in the formula. The data type and format are the same as those of the input cos.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Specifies the specific index positions for writing to the cache. The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_out</td>
      <td>Input/Output</td>
      <td>Cache allocated in advance, where the input and output reuse the same address. The data type is the same as that of the input qkv, and the shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_cache</td>
      <td>Input/Output</td>
      <td>Cache allocated in advance, where the input and output reuse the same address. The data type is the same as that of the input qkv (k is not quantized) or INT8 (k is quantized). The shape is [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16] (k is not quantized) or [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32] (k is quantized).</td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_cache</td>
      <td>Input/Output</td>
      <td>Cache allocated in advance, where the input and output reuse the same address. The data type is the same as that of the input qkv (v is not quantized) or INT8 (v is quantized). The shape is [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16] (v is not quantized) or [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32] (v is quantized).</td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_scale</td>
      <td>Optional input</td>
      <td>This input parameter is required when the data type of k_cache is INT8, corresponding to kScale in the formula. The shape is [N<sub>k</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_scale</td>
      <td>Optional input</td>
      <td>This input parameter is required when the data type of v_cache is INT8, corresponding to vScale in the formula. The shape is [N<sub>v</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_offset</td>
      <td>Optional input</td>
      <td>This parameter input is required when the data type of k_cache is INT8, the corresponding k_scale input exists, and the quantization scenario is asymmetric quantization, corresponding to kOffset in the formula. The shape is [N<sub>k</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_offset</td>
      <td>Optional input</td>
      <td>This parameter input is required when the data type of v_cache is INT8, the corresponding v_scale input exists, and the quantization scenario is asymmetric quantization, corresponding to vOffset in the formula. The shape is [N<sub>v</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_out_before_quant</td>
      <td>Optional output</td>
      <td>Data to be written to q_out.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_out_before_quant</td>
      <td>Optional output</td>
      <td>Data to be written to k_cache, that is, the intermediate computation result before quantization and Scatter.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_out_before_quant</td>
      <td>Optional output</td>
      <td>Data to be written to v_cache, that is, the intermediate computation result before quantization and Scatter.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qkv_size</td>
      <td>Attribute</td>
      <td>Passed in the order of [B<sub>qkv</sub>, S<sub>qkv</sub>, N<sub>qkv</sub>, D<sub>qkv</sub>] and provides the specific sizes of the B, S, N, and D dimensions of the input parameter qkv matrix.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_nums</td>
      <td>Attribute</td>
      <td>Passed in the order of [N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>] and provides the specific size of the N dimension in the qkv component units of the input parameter qkv matrix.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul>
      <li>Used to prevent division-by-zero errors in the RmsNorm computation, corresponding to epsilon in the formula.</li>
      <li>The default value is 1e-6.</li>
      </ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cache_mode</td>
      <td>Optional attribute</td>
      <td><ul>
      <li>Selection flag of the cache format. Currently only PA_NZ is supported.</li>
      <li>The default value is PA_NZ.</li>
      </ul></td>
      <td>CHAR*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_output_qkv</td>
      <td>Optional attribute</td>
      <td><ul>
      <li>Indicates whether to output the original values of the corresponding content in each cache output before quantization and Scatter.</li>
      <li>The default value is false.</li>
      </ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraints

* Input shape restrictions:
    * B<sub>qkv</sub> is the batch_size of the input qkv, and S<sub>qkv</sub> is the sequence length of the input qkv. The sizes are determined by qkvSize.
    * N<sub>qkv</sub> is the head number of the input qkv, and D<sub>qkv</sub> is the head dim of the input qkv. Currently only 128 is supported. D<sub>q</sub>, D<sub>k</sub>, and D<sub>k</sub> are the head dims of q, k, and v respectively. D<sub>qkv</sub> = D<sub>q</sub> = D<sub>k</sub> = D<sub>v</sub> is required, and D<sub>qkv</sub> must satisfy that (D<sub>qkv</sub> * the number of bytes occupied by the qkv data type) is divisible by 32.
    * According to the rope rules, D<sub>k</sub> and D<sub>q</sub> must be even numbers. In the scenario where cache_mode is PA_NZ, D<sub>k</sub> and D<sub>q</sub> must be aligned to 32B, and BlockSize must be aligned to 32B.
    * For the preceding 32B alignment cases, the alignment value is determined by the data type of the cache. Taking BlockSize as an example, if the data type of the cache is int8, BlockSize % 32 = 0 must be satisfied; if the data type of the cache is float16, BlockSize % 16 = 0 must be satisfied; if the dtypes of the k_cache and v_cache parameters are inconsistent, BlockSize must satisfy both BlockSize % 32 = 0 and BlockSize % 16 = 0.
    * BlockNum is the number of memory blocks written to the cache. The size is determined by the user input scenario, and BlockNum >= Ceil(S<sub>qkv</sub> / BlockSize) * B<sub>qkv</sub> is required.
    * requireMemory denotes the space required to store data and must satisfy: requireMemory >= (B<sub>qkv</sub> * S<sub>qkv</sub> * N<sub>qkv</sub> * D<sub>qkv</sub> + 2 * D<sub>qkv</sub> + 2 * B<sub>qkv</sub> * S<sub>qkv</sub> * D<sub>qkv</sub> + B<sub>qkv</sub> * S<sub>qkv</sub> * N<sub>q</sub> * D<sub>qkv</sub> + BlockNum * BlockSize * N<sub>v</sub> * D<sub>qkv</sub> + BlockNum * BlockSize * N<sub>k</sub> * D<sub>qkv</sub>) * sizeof(FLOAT16) + B<sub>qkv</sub> * S<sub>qkv</sub> * sizeof(INT64) + (2 * N<sub>k</sub> * D<sub>qkv</sub> + 2 * N<sub>v</sub>) * sizeof(FLOAT). If the computed requireMemory size exceeds the total GM space size of the current AI processor, this operator is not supported.
* Other restrictions:
    * For index, the value range of index is [-1, BlockNum * BlockSize). Values cannot be duplicated. When index is -1, the update is skipped.
    * k_scale and v_scale denote the scaling factors of symmetric quantization. Therefore, if they are passed in, their values cannot be 0.

## Calling Instructions

| Calling Method   | Calling Sample           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_qkv_rms_norm_rope_cache](examples/test_aclnn_qkv_rms_norm_rope_cache.cpp) | Calls the QkvRmsNormRopeCache operator through the [aclnnQkvRmsNormRopeCache](docs/aclnnQkvRmsNormRopeCache_en.md) API. |
| Graph mode | [test_geir_qkv_rms_norm_rope_cache](examples/test_geir_qkv_rms_norm_rope_cache.cpp)  | Calls the QkvRmsNormRopeCache operator through [Operator IR](op_graph/qkv_rms_norm_rope_cache_proto.h) graph construction.         |
