# KvRmsNormRopeCache

English | [简体中文](./README.md)

## Product Support Status

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor products</term> | √ |
|  <term>Kirin 9030 processor products</term> | √ |

## Description

- Operator function: fuses the RMSNorm normalization computation (corresponding to $rms\_size$) and RoPE (Rotary Position Embedding) position encoding (corresponding to $rope\_size$) in the MLA (Multi-head Latent Attention) structure, as well as the ScatterUpdate operation that updates the KV cache.

- Supported scenarios:

  | Scenario Type | kv Component Source | Description |
    | :------: | :------: | :------ |
    |V1|rms_size=Dv=512<br>rope_size=Dk=64<br>vOptional=None|kv merged-axis mode: splits the trailing axis of the input tensor kv into a left half for the rms_norm computation and a right half for the rope computation, and then scatters the computation results into two caches respectively.<li>It is strongly related to the DeepSeekV3 network structure and supports only the scenario where N=1.</li><li>The data Dv required by the rms_norm computation and the data Dk required by the rope computation are split from the D of the input kv, and the sizes of Dk and Dv must satisfy Dk+Dv=Dkv.|
    |V2|Dv=128<br>rms_size=Dk=Dkv=192<br>rope_size=64<br>The shape of vOptional is [Bkv, Nkv, Skv, Dv]|kv separated-axis mode: performs the rms_norm computation on the input tensor kv, then performs the rope computation on the first 64 dimensions of the trailing axis and overwrites the corresponding elements, and finally scatters the result into k_cache. It performs intermediate processing on the input tensor vOptional and finally scatters the result into ckv_cache.</li><li>N=1/2/4/8 is supported</li><li>In this scenario, the trailing axes of k and v are separated. kv stores only the trailing axis of the k component, and vOptional stores the trailing axis of the v component.</li>|

    * <term>Ascend 950PR&950DT products</term>: only the V1 scenario is supported.
    * <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: the V1 and V2 scenarios are supported.

- Computation formula:

  The shape of the input tensor kv is defined as $[Bkv, N, Skv, Dkv]$, and the shape of the tensor vOptional is defined as $[Bkv, N, Skv, Dv]$.

  (1) RmsNorm:

  $$
  x=kv[...,:rms\_size]
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  y = \operatorname{RmsNorm}(x)
  $$

  (2) interleaveRope:

  $$
  x=\begin{cases}
  kv[...,Dv:], \quad vOptional = None\\
  {\operatorname{RmsNorm}(\mathbf{x})}[...,:rope\_size],  \quad vOptional != None
  \end{cases}
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y\_rope=x\_part1*cos+x\_part2*sin
  $$

  $$
  rope\_out=\begin{cases}y\_rope, \quad vOptional = None \\
  concat(y\_rope, {\operatorname{RmsNorm}(\mathbf{x})}[...,rope\_size:]),  \quad vOptional != None
  \end{cases}
  $$

  (3) Quantization computation:

  x denotes the raw data to be written to k_cache and ckv_cache and serves as the input of the quantization process.

  $$
  x = x * scale,\ if\ scale\ !=\ None
  $$

  $$
  x = x + offset,\ if\ offset\ !=\ None
  $$

  $$
  y = \begin{cases}x, \quad scale == None \space and \space offset == None \\ round(x).clamp(-128,127), \quad others
  \end{cases}
  $$

  (4) Scatter write-out:

  The input tensor index corresponds to the index mapping table of each element in the input KV cache. Take the index of a specific element in x as $b \in Bkv$ and $s \in Skv$, and $n$ is the attention head index,

  $$
  scatter\_idx = index(b, s) \\
  $$

   Quant denotes the preceding quantization computation process, applied to the in-place update parameters k\_cache and ckv\_cache:

  $$
  k\_cache[scatter\_idx, ...] = Quant(x = rope\_out, scale = k\_scale, offset = k\_offset)[b, n, s]
  $$

  $$
  ckv\_cache[scatter\_idx, ...] = \begin{cases} Quant(x = \operatorname{RmsNorm}(x), scale = v\_scale, offset = v\_offset)[b, n, s], \quad vOptional = None \\
  Quant(x = vOptional, scale = v\_scale, offset = v\_offset)[b, n, s],  \quad vOptional != None
  \end{cases}
  $$

  (5) Raw result write-out:

  When $is\_output\_kv=True$ and it is valid:

  $$
  k\_rope = rope\_out
  $$

  $$
  c\_kv = \begin{cases}\operatorname{RmsNorm}(x), \quad vOptional = None \\
  vOptional,  \quad vOptional != None
  \end{cases}
  $$


## Parameter Description

<table style="undefined;table-layout: fixed; width: 1920px"><colgroup>
  <col style="width: 84px">
  <col style="width: 128px">
  <col style="width: 448px">
  <col style="width: 288px">
  <col style="width: 72px">
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
      <td>kv</td>
      <td>Input</td>
      <td>Input data used to split out the data Dv required by the rms_norm computation and the data Dk required by the rope computation, corresponding to `kv` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Input data used for the rms_norm computation, corresponding to `gamma` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for the rope computation. It performs the cosine transformation on the input tensor, corresponding to `cos` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Input data used for the rope computation. It performs the sine transformation on the input tensor, corresponding to `sin` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Specifies the specific index positions for writing to the cache.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_cache</td>
      <td>Input/Output</td>
      <td>Cache allocated in advance. The input and output reuse the same address.</td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ckv_cache</td>
      <td>Input/Output</td>
      <td>Cache allocated in advance. The input and output reuse the same address.</td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_rope_scale</td>
      <td>Optional attribute</td>
      <td>This input parameter is required when the data type of k_cache is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_kv_scale</td>
      <td>Optional attribute</td>
      <td>This input parameter is required when the data type of ckv_cache is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_rope_offset</td>
      <td>Optional attribute</td>
      <td>This parameter input is required when the data type of k_cache is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, the corresponding k_rope_scale input exists, and the quantization scenario is asymmetric quantization.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_kv_offset</td>
      <td>Optional attribute</td>
      <td>This parameter input is required when the data type of ckv_cache is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, the corresponding c_kv_scale input exists, and the quantization scenario is asymmetric quantization.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_optional</td>
      <td>Optional attribute</td>
      <td>Only in the kv separated-axis scenario (V2), it serves as the input source of the Dv component for immediate scatter.<br>The first three dimensions of the shape must be the same as kv, and the data type must be the same as kv.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Prevents division-by-zero errors in the rms_norm computation, corresponding to eps in the formula.</li><li>The default value is 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cache_mode</td>
      <td>Optional attribute</td>
      <td>Selection flag of the cache format. The types are Norm, PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, and PA_BLK_NZ.</td>
      <td>CHAR*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_output_kv</td>
      <td>Optional attribute</td>
      <td>Control flag for the k_rope and c_kv outputs.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k_rope</td>
      <td>Output</td>
      <td>rope computation result, corresponding to `rope_out` in the interleaveRope computation formula.<br>Controlled by is_output_kv. It must be output when is_output_kv is true.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_kv</td>
      <td>Output</td>
      <td>rms_norm computation result, corresponding to `y` in the rmsNorm computation formula.<br>Controlled by is_output_kv. It must be output when is_output_kv is true.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* Supported data types of the cache:

  * Non-quantization mode: the cache type must be the same as kv.
    * <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: BFLOAT16 and FLOAT16 are supported.
    * <term><term>Kirin X90 processor products</term>, <term>Kirin 9030 processor products</term></term>: only FLOAT16 is supported.

  * Quantization mode:
    * <term>Ascend 950PR&950DT products</term>: INT8, HIFLOAT8, FLOAT8E5M2, and FLOAT8E4M3FN are supported.
    * <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term><term>Kirin X90 processor products</term>, <term>Kirin 9030 processor products</term></term>: only INT8 is supported.

## Constraints

  * This operator is implemented deterministically by default.
  * Input shape limits:

      * kv is a 4-dimensional tensor with the shape [Bkv,N,Skv,D]. Bkv is the batch size of the input kv, and Skv is the sequence length of the input kv. Their sizes are determined by the user input scenario and have no explicit limit.
      * N is the head number of the input kv. The V1 scenario is strongly related to the DeepSeekV3 network structure and supports only the scenario where N=1. The V2 scenario supports N=1/2/4/8.
      * D is the head dim of the input kv. According to the rope rule, Dk is an even number. If cache_mode is an NZ scenario (cache_mode is PA_NZ or PA_BLK_NZ), Dk and Dv must be 32B aligned. This rule applies to all scenarios and computation types.
      * If cache_mode is a PA scenario (cache_mode is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), block_size must be 32B aligned.
      * For the preceding 32B alignment cases, the alignment value is determined by the data type of the cache. Taking block_size as an example, if the data type of the cache is int8, block_size%32=0 is required. If the data type of the cache is float16, block_size%16=0 is required. If the dtypes of the kCacheRef and ckvCacheRef parameters are inconsistent, block_size must satisfy both block_size%32=0 and block_size%16=0.
      * block_num is the number of memory blocks written to the cache. Its size is determined by the user input scenario and has no explicit limit.
      * The rotary position encoding (RoPE) parameter items (cos, sin) must satisfy the shape constraints:
        * The shape can be the 4-dimensional [Bkv,N,Skv,Dk] or [Bkv,N,1,Dk].
        * The shapes of cos and sin must be the same.
      * The quantization parameter items (k_rope_scale, k_rope_offset, c_kv_scale, c_kv_offset) must satisfy the shape constraints:
        * The number of dimensions and the N-axis size (if present) of all quantization parameter items must be the same.
        * The valid shape constraints on quantization parameter items differ across scenarios:

          | Scenario Type | Quantization Parameter Shape |
          | :------: | :------ |
          |V1|<li>The shapes of k_rope_scale and k_rope_offset support [1, Dk], [Dk,], and [1,].</li><li>The shapes of c_kv_scale and c_kv_offset support [1, Dv], [Dv,], and [1,].</li>|
          |V2|<li>The shapes of k_rope_scale and k_rope_offset support [N, Dk].</li><li>The shapes of c_kv_scale and c_kv_offset support [N, Dv].</li>|

          * <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: the V1 scenario does not support the quantization parameter item shape [1,].

      * None of the input tensors support empty tensors.
      * None of the inputs support invalid values, including but not limited to ±inf and nan.

  * Parameter description:

    * The data types of kv, gamma, cos, sin, and vOptional in the input parameters must be exactly the same.
    * k_cache and ckv_cache are <b>in-place update parameters</b>. Their data types depend on the corresponding input components and the corresponding scale and offset. Details are as follows:

      | cache_type | offset==None | offset!=None |
      | :-------------------------: | :-----------: | :----------------------------------------------: |
      | scale==None | Same as kv | Invalid input, intercepted |
      | scale!=None | INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN | INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN |

      * In quantization mode, the dtypes of k_cache and ckv_cache should be the data types supported on the corresponding products.
      * In non-quantization mode, the quantization parameters (k_rope_scale, k_rope_offset, c_kv_scale, c_kv_offset) must be set to None, and the dtypes of k_cache and ckv_cache must be the same as kv.

    * The mapping between the input components and the corresponding quantization factor scale and quantization offset is as follows:

      | kv Component | Component Input Source | Corresponding scale Input | Corresponding offset Input | Corresponding Output cache |
      | :------: | :------: | :------: | :------: | :------: |
      | k component | In the V1 scenario, corresponding to kv[..., Dv:].<br>In the V2 scenario, corresponding to kv[..., :]. | k_rope_scale | k_rope_offset | k_cache |
      | v component | In the V1 scenario, corresponding to kv[..., :Dv].<br>In the V2 scenario, corresponding to vOptional[..., :]. |c_kv_scale | c_kv_offset | ckv_cache |

      * k_cache: the quantization coefficients are k_rope_scale and k_rope_offset.
      * ckv_cache: the corresponding quantization coefficients are c_kv_scale and c_kv_offset.

    * In the output parameters, the types of k_rope and c_kv must be the same as kv.

  * Quantization mode constraints:

    <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
      <col style="width: 400px">
      <col style="width: 160px">
      <col style="width: 448px">
      </colgroup>
      <thead>
        <tr>
          <th>Quantization Mode Description<br>(the scale and offset inputs have an 'AND' relationship)</th>
          <th>Supported cache Modes</th>
          <th>Availability Description</th>
        </tr></thead>
      <tbody>
      <tr>
          <td rowspan="5">Non-quantization mode<li>scale input: k_rope_scale==None && c_kv_scale==None</li><li>offset input: k_rope_offset==None && c_kv_offset==None</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5">Both V1 and V2 are supported.</td>
      </tr>
      <tr>
          <td>PA/PA_BNSD</td>
      </tr>
      <tr>
          <td>PA_NZ</td>
      </tr>
      <tr>
          <td>PA_BLK_BNSD</td>
      </tr>
      <tr>
          <td>PA_BLK_NZ</td>
      </tr>
      <tr>
          <td rowspan="5">Static quantization mode<li>scale input: at least one of k_rope_scale and c_kv_scale is not null.</li><li>offset input: the offset input is valid only when the corresponding scale is not null. If the corresponding offset is null, it is <b>static symmetric quantization</b>. If the corresponding offset is not null, it is <b>static asymmetric quantization</b>.</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5"><li><b>Static symmetric quantization</b> and <b>static asymmetric quantization</b> differ in support.</li><li>K and V can independently select different quantization modes.</li></td>
      </tr>
      <tr>
          <td>PA/PA_BNSD</td>
      </tr>
      <tr>
          <td>PA_NZ</td>
      </tr>
      <tr>
          <td>PA_BLK_BNSD</td>
      </tr>
      <tr>
          <td>PA_BLK_NZ</td>
      </tr>
      </tbody>
    </table>

    * Static quantization mode support details:
      * Ascend 950PR&950DT products: only the V1 scenario is supported, and <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.
      * <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
        - V1 scenario: for cachemode values other than `Norm`, only <b>static symmetric quantization</b> is supported. Even if a valid offset is passed in, the operator does not process it and still treats it as <b>static symmetric quantization</b>. For cachemode `Norm`, no static quantization is supported.
        - V2 scenario: for all cachemode values, <b>static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.

  * Cache and index related constraints:

    | cachemode| k_cache Shape | ckv_cache Shape | index Shape | Description |
    | :------: | :------: | :------: | :------: | :------ |
    |Norm|[Bkv, N, Scache, Dk]|[Bkv, N, Scache, Dv]|[Bkv, Skv]|KV-Cache update mode. index indicates the offset under each batch.<br>The value range of index values is [-1,Scache). Under different Bkv values, the values can be repeated.<br>$Scache \ge Skv$|
    |PA/PA_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|PagedAttention mode. index indicates the offset of each token.<br>The value range of index values is [-1,block_num * block_size). The values cannot be repeated.<br>$block\_size>1,\\ block\_num \ge Ceil(Skv * Bkv / block\_size)$|
    |PA_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|PagedAttention mode where the cache data format is FRACTAL_NZ. index indicates the offset of each token.<br>The value range of index values is [-1,block_num * block_size). The values cannot be repeated.<br>$block\_size>1,\\ block\_num \ge Ceil(Skv * Bkv / block\_size)$|
    |PA_BLK_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|Special PagedAttention mode. index indicates the start offset of each block (not in one-to-one correspondence with tokens).<br>The value range of index values is [-1,block_num * block_size). The value of value/block_size cannot be repeated.<br>$block\_size>1,\\ block\_num \ge Ceil(Skv / block\_size) * Bkv$|
    |PA_BLK_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|Special PagedAttention mode where the cache data format is FRACTAL_NZ. index indicates the start offset of each block.<br>The value range of index values is [-1,block_num * block_size). The value of value/block_size cannot be repeated.<br>$block\_size>1,\\ block\_num \ge Ceil(Skv / block\_size) * Bkv$|

    * This operator supports the following slotmapping modes:

      | slotmapping Mode | Size Relationship between kv and cache | Additional index Constraints |
      | :---: | :--------: | :-------- |
      |Excess mapping|Norm mode: $Bkv * Scache < Bkv * Skv$<br>Non-Norm mode: $block\_num * block\_size < Bkv * Skv$|The number of bs of the input KV exceeds the number of cache slots, used for cache compression or gating.<br>Norm mode: within each Bkv, the number of valid elements does not exceed Scache, and all remaining elements are -1.<br>Non-Norm mode: the total number of valid elements does not exceed $block\_num * block\_size$, and all remaining elements are -1.|
      |Dense mapping|Norm mode: $Bkv * Scache = Bkv * Skv$<br>Non-Norm mode: $block\_num * block\_size = Bkv * Skv$|The number of bs of the input KV exactly corresponds to the number of cache slots.|
      |Sparse mapping|Norm mode: $Bkv * Scache > Bkv * Skv$<br>Non-Norm mode: $block\_num * block\_size > Bkv * Skv$|The number of bs of the input KV is less than the number of cache slots.|

    * When cache_mode is Norm:
      * The shape of index is 2-dimensional [Bkv, Skv]. The value range of index values is [-1, Scache). Under different Bkv values, the values can be repeated.
      * The B-axis size of the input cache must be exactly the same as $Bkv$.
      * Scache is the sequence length of the input cache. Its size is determined by the user input scenario and has no explicit limit.
    * When cache_mode is PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ, the data layout in the cache is as follows:
      * In non-quantization mode: k_cache is [block_num, Dk//16, block_size, 1, 16], and ckv_cache is [block_num, Dv//16, block_size, 1, 16].
      * In static quantization mode: k_cache is [block_num, Dk//32, block_size, 1, 32], and ckv_cache is [block_num, Dv//32, block_size, 1, 32].
    * When cache_mode is PA_BNSD or PA_NZ, the shape is 1-dimensional [Bkv * Skv]. The value range of index values is [-1,block_num * block_size). The values cannot be repeated.
    * When cache_mode is PA_BLK_BNSD or PA_BLK_NZ, the shape is 1-dimensional [Bkv * ceil_div(Skv,block_size)]. The value range of index values is [-1,block_num * block_size). The value of value/block_size cannot be repeated.
  * is_output_kv constraints:
    * It outputs intermediate processing results: enables k_rope and c_kv.
    * It is valid when cache_mode is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ.
    * When cache_mode is Norm, it is valid only when the quantization mode is enabled in the V2 scenario.

  * Input combination constraints:

    * The items in this constraint apply only to <b>Atlas A3 products</b> and <b>Atlas A2 products</b>.
    * Under all cache_mode values, `non-broadcast mode` is always supported: that is, the `rotary position encoding (RoPE) parameters` and `quantization parameters` whose `[B, N, S]` dimensions are strictly consistent with `kv`.
    * Under various cache_mode values, the supported valid input shape patterns are as follows:

      <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
        <col style="width: 160px">
        <col style="width: 280px">
        <col style="width: 280px">
        <col style="width: 320px">
        </colgroup>
        <thead>
          <tr>
            <th>cache Mode</th>
            <th>kv Shape</th>
            <th>Rotary Position Encoding (RoPE) Parameters<br>Shape</th>
            <th>Other Parameters</th>
          </tr>
          </thead>
        <tbody>
        <tr>
            <td>Norm</td>
            <td rowspan="6"><li>V1 scenario: [Bkv, 1, Skv, Dv+Dk] (kv)</li><li>V2 scenario: <ul><li>[Bkv, N, Skv, Dk] (kv)</li><li>[Bkv, N, Skv, Dv] (vOptional)</li></ul></li></td>
            <td rowspan="2"><li>S-axis non-broadcast is supported: Srope=Skv.</li><li>S-axis broadcast is supported: Srope=1.</li></td>
            <td><li>In <b>non-quantization mode</b>: is_output_kv is invalid, and k_rope and c_kv are invalid.</li><li>V1 scenario: <ul><li>Only <b>non-quantization mode</b> is supported, and all quantization parameter items are invalid.</li></ul></li><li>V2 scenario: <ul><li><b>Non-quantization mode</b> and <b>static quantization mode</b> are supported.</li><li>The quantization parameter items corresponding to k and v must be 2-dimensional [N, D].</li></ul></li></td>
        </tr>
        <tr>
            <td>PA</td>
            <td><li>In <b>non-quantization mode</b>, S-axis broadcast and S-axis non-broadcast are supported.</li><li>In <b>static quantization mode</b>, only S-axis non-broadcast is supported; otherwise the input is invalid.</li><li>Quantization parameter items: <ul><li>V1 scenario: the 2-dimensional [1,D] or 1-dimensional [D] shape is supported. Only <b>static symmetric quantization</b> is supported.</li><li>V2 scenario: the 2-dimensional [N,D] shape is supported. <b>Static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.</li></ul></li></td>
        </tr>
        <tr>
            <td>PA_BNSD</td>
            <td rowspan="4"><li>S-axis broadcast is not supported: Srope=Skv must be satisfied.</li></td>
            <td rowspan="4"><li>Non-quantization mode is supported.</li><li>Quantization parameter items:</li><ul><li>V1 scenario: the 2-dimensional [1,D] or 1-dimensional [D] shape is supported. Only <b>static symmetric quantization</b> is supported.</li><li>V2 scenario: the 2-dimensional [N,D] shape is supported. <b>Static symmetric quantization</b> and <b>static asymmetric quantization</b> are supported.</li></ul></td>
        </tr>
        <tr>
            <td>PA_NZ</td>
        </tr>
        <tr>
            <td>PA_BLK_BNSD</td>
        </tr>
        <tr>
            <td>PA_BLK_NZ</td>
        </tr>
        </tbody>
      </table>

    * <b>Input combinations not covered in the preceding table cause undefined behavior of the operator</b>.

  * vOptional:
    * This parameter is limited to the aclnnKvRmsNormRopeCacheV2 API. The aclnnKvRmsNormRopeCache API does not support this parameter!
    * This parameter is limited to <b>Atlas A3 products</b> and <b>Atlas A2 products</b>.
      * This parameter is a required input only in the <b>kv separated-axis scenario (V2)</b>, and is ignored as an invalid parameter in other types.
      * When vOptional exists, its type must be the same as kv, and its `[B, N, S]` dimensions must also be the same as kv.
    * Ascend 950PR&950DT products: this parameter is not intercepted, but the actual function does not support it and the parameter is not processed.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_kv_rms_norm_rope_cache](examples/test_aclnn_kv_rms_norm_rope_cache.cpp) | Calls the KvRmsNormRopeCache operator through the [aclnnKvRmsNormRopeCache](docs/aclnnKvRmsNormRopeCache_en.md) API. |
| Graph mode | [test_geir_kv_rms_norm_rope_cache](examples/test_geir_kv_rms_norm_rope_cache.cpp)  | Calls the KvRmsNormRopeCache operator by building the graph through the [operator IR](op_graph/kv_rms_norm_rope_cache_proto.h).         |
