# MlaPrologV3

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

- **功能更新**：（相对与aclnnMlaPrologV2weightNz的差异）
    - 新增query与key的尺度矫正因子，分别对应qcQrScale（$\alpha_q$）与kcScale（$\alpha_{kv}$）。
    - 新增可选输入与参数，将cache_mode由必选改为可选。具体包括：
        - actualSeqLenOptional：用于BS合轴且CacheMode="PA_BLK_BSND"/"PA_BLK_NZ"时，指定当前batch中实际的序列长度。
        - kNopeClipAlphaOptional：表示对kv_cache做clip操作时的缩放因子。
        - queryNormFlag：表示是否输出query_norm，以及量化场景下的dequant_scale_q_norm。
        - weightQuantMode：表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式。
        - kvCacheQuantMode：表示kv_cache的量化模式。
        - queryQuantMode：表示query的量化模式。
        - ckvkrRepoMode：表示kv_cache和kr_cache的存储模式。
        - quantScaleRepoMode：表示量化scale的存储模式。
        - tileSize：表示pertoken-pergroup量化时每个tile的大小。
        - queryNormOptional：公式中tokenX做rmsNorm后的输出tensor（对应$c^Q$）。
        - dequantScaleQNormOptional：query_norm的输出tensor的量化参数。
    - 调整cacheIndex参数的名称与位置，对应当前的cacheIndexOptional。
- **算子功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分为四路，首先对输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后分为两路，第一路乘以$W^{UQ}$和$W^{UK}$经过两次上采样后得到$q^N$；第二路乘以$W^{QR}$后经过旋转位置编码（ROPE）得到$q^R$；第三路是输入$x$乘以$W^{DKV}$进行下采样和RmsNorm后传入Cache中得到$k^C$；第四路是输入$x$乘以$W^{KR}$后经过旋转位置编码后传入另一个Cache中得到$k^R$。
- **计算公式**：

    RmsNorm公式

    $$
    \mathrm{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\mathrm{RMS}(x)}
    $$

    $$
    \mathrm{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query计算公式，包括下采样，RmsNorm和两次上采样

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$
    其中 $\alpha_q$ 是Query的尺度矫正参数。

    对Query进行ROPE旋转位置编码

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    Key计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    其中 $\alpha_{kv}$ 是Key的尺度矫正参数。

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型 | 数据格式   |
|----------------------------|-----------|-----|----------------------------------------------------------------------|----------------------------|
| token_x                     | 输入      | 公式中计算Query和Key的输入tensor。 | BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | ND         |
| weight_dq                   | 输入      | 公式中计算Query的下采样权重矩阵$W^{DQ}$。 <br> 不转置的情况下各个维度的表示：（k，n）。 | BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | FRACTAL_NZ |
| weight_uq_qr                 | 输入      | 公式中计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。 <br> 不转置的情况下各个维度的表示：（k, n）。| BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | FRACTAL_NZ |
| weight_uk                   | 输入      | 公式中计算Key的上采样权重$W^{UK}$。 | BFLOAT16 | ND         |
| weight_dkv_kr                | 输入      | 公式中计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。 <br> 不转置的情况下各个维度的表示：（k, n）。| BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | FRACTAL_NZ |
| rmsnorm_gamma_cq             | 输入      | 计算$c^Q$的RmsNorm公式中$\gamma$参数。 | BFLOAT16 | ND         |
| rmsnorm_gamma_ckv            | 输入      | 计算$c^{KV}$的RmsNorm公式中$\gamma$参数。 | BFLOAT16 | ND         |
| rope_sin                    | 输入      | 旋转位置编码的正弦参数矩阵。 | BFLOAT16 | ND         |
| rope_cos                    | 输入      | 旋转位置编码的余弦参数矩阵。 | BFLOAT16 | ND         |
| kv_cache                 | 输入/ 输出| cache索引的aclTensor，计算结果原地更新（对应$k^C$）。|  BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | ND         |
| kr_cache                 | 输入/ 输出| key位置编码的cache，计算结果原地更新（对应$k^R$）。 |  BFLOAT16、INT8 | ND         |
| cache_index                 | 输入      | 存储kvCache和krCache的索引。 | INT64| ND         |
| dequant_scale_x      | 输入      | token_x的反量化参数。  | FLOAT8_E8M0、FLOAT | ND         |
| dequant_scale_w_dq    | 输入      | weight_dq的反量化参数。 | FLOAT8_E8M0、FLOAT | ND         |
| dequant_scale_w_uq_qr  | 输入      | MatmulQcQr矩阵乘后反量化的perchannel参数。 | FLOAT8_E8M0、FLOAT | ND         |
| dequant_scale_w_dkv_kr | 输入      | weight_dkv_kr的反量化参数。 | FLOAT8_E8M0、FLOAT | ND         |
| quant_scale_ckv      | 输入      | KVCache输出量化参数。 | FLOAT | ND         |
| quant_scale_ckr      | 输入      | KRCache输出量化参数。 | FLOAT | ND         |
| smooth_scales_cq     | 输入      | RmsNormCq输出动态量化参数。 | FLOAT | ND         |
| actual_seq_len                 | 输入      | 预留参数，当前版本暂未使用，必须传入空指针。 | INT32 | ND         |
| k_nope_clip_alpha    | 输入      | 对kv_cache做clip操作时的缩放因子。  | FLOAT | ND         |    
| rmsnorm_epsilon_cq           | 输入      | 计算$c^Q$的RmsNorm公式中$\epsilon$参数。 | DOUBLE | -          |
| rmsnorm_epsilon_ckv          | 输入      | 计算$c^{KV}$的RmsNorm公式中$\epsilon$参数。 | DOUBLE | -          |
| cache_mode          | 输入      | kvCache模式。 | CHAR* | -          |
| query_norm_flag           | 输入      | 表示是否输出query_norm，Host侧参数。 | BOOL | -          |
| weight_quant_mode          | 输入      | 表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式。 | INT64 | -          |
| kv_cache_quant_mode           | 输入      | 表示kv_cache的量化模式。 | INT64 | -          |
| query_quant_mode          | 输入      | 表示query的量化模式。 | INT64 | -          |
| ckvkr_repo_mode           | 输入      | 表示kv_cache和kr_cache的存储模式。 | INT64 | -          |
| quant_scale_repo_mode           | 输入      | 表示量化scale的存储模式。 | INT64 | -          |
| tile_size          | 输入      | 表示pertoken-pergroup量化时每个tile的大小，需要传入128。 | INT64 | -          |
| qc_qr_scale          | 输入      | Query的尺度矫正参数，对应$\alpha_q$，默认传1.0。 | DOUBLE | -          |
| kc_scale          | 输入      | Key的尺度矫正参数，对应$\alpha_{kv}$，默认传1.0。 | DOUBLE | -          |
| query                   | 输出      | 公式中Query的输出tensor（对应$q^N$）。 | BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | ND         |
| query_rope               | 输出      | 公式中Query位置编码的输出tensor（对应$q^R$）。 | BFLOAT16 | ND |
| dequant_scale_q_nope | 输出     | 表示Query的输出tensor的量化参数。   | FLOAT | ND         |
| query_norm               | 输出      | 公式中tokenX做rmsNorm后的输出tensor（对应$c^Q$）。 | BFLOAT16、FLOAT8_E4M3FN、INT8、HIFLOAT8 | ND |
| dequant_scale_q_norm | 输出     | query_norm的输出tensor的量化参数。   | FLOAT、FLOAT8_E8M0 | ND         |

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - token_x、weight_dq、weight_uq_qr、weight_dkv_kr、kv_cache、query、query_norm不支持FLOAT8_E4M3FN、HIFLOAT8数据类型。
  - dequant_scale_x、dequant_scale_w_dq、dequant_scale_w_uq_qr、dequant_scale_w_dkv_kr、dequant_scale_q_norm不支持FLOAT8_E8M0数据类型。

## 约束说明

- shape约束
    - 若tokenX的维度采用BS合轴，即(T, He)：
        - ropeSin和ropeCos的shape为(T, Dr)。
        - 当CacheMode为PA_BSND或PA_NZ时，cacheIndex的shape为(T)。
        - 当CacheMode为PA_BLK_BSND或PA_BLK_NZ时，cacheIndex的shape为(Sum(Ceil(S_i/BlockSize)))，S_i为每个Batch中的S的长度。
        - 当CacheMode为PA_BLK_BSND或PA_BLK_NZ时，actualSeqLenOptional需要传入，维度为(B)。
        - int8/fp8/hif8全量化场景下，dequantScaleXOptional的shape为(T, 1)；mxfp8全量化场景下，dequantScaleXOptional的shape为(T, He/32)。
        - queryOut的shape为(T, N, Hckv)。
        - queryRopeOut的shape为(T, N, Dr)。
        - int8/mxfp8/fp8/hif8全量化场景下，dequantScaleQNopeOutOptional的shape为(T, N, 1)，其他场景下为nullptr。
    - 若tokenX的维度不采用BS合轴，即(B, S, He)：
        - ropeSin和ropeCos的shape为(B, S, Dr)。
        - 当CacheMode为PA_BSND或PA_NZ时，cacheIndex的shape为(B, S)。
        - 当CacheMode为PA_BLK_BSND或PA_BLK_NZ时，cacheIndex的shape为(B,Ceil(S/BlockSize))。
        - int8/fp8/hif8全量化场景下，dequantScaleXOptional的shape为(B\*S, 1)；mxfp8全量化场景下，dequantScaleXOptional的shape为(B*S, He/32)。
        - queryOut的shape为(B, S, N, Hckv)。
        - queryRopeOut的shape为(B, S, N, Dr)。
        - int8/mxfp8/fp8/hif8全量化场景下，dequantScaleQNopeOutOptional的shape为(B*S, N, 1)，其他场景下为nullptr。
    - B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则queryOut、queryRopeOut输出空Tensor，kvCacheRef、krCacheRef不做更新。
        - 如果Skv取值为0，则queryOut、queryRopeOut、dequantScaleQNopeOutOptional正常计算，kvCacheRef、krCacheRef不做更新，即输出空Tensor。
    - 当CacheMode为BSND时：
        - tokenX应不采用BS合轴，即维度为(B, S, He)。
        - kvCache的维度为(B,S,Nkv,Dr)。
    - 当CacheMode为TND时：
        - tokenX应采用BS合轴，即维度为(T, He)。
        - kvCache的维度为(T,Nkv,Dr)。
    - 当ckvkrRepoMode=1时：
        - krCache的维度应包含0，支持shape为(0)。

- 特殊约束
  - actualSeqLenOptional传入时，actualSeqLenOptional最后一个数需与T保持一致。
  - pertoken-pergroup量化模式下，ckvkrRepoMode和quantScaleRepoMode必须同时为1；其他量化模式以及非量化场景下，ckvkrRepoMode和quantScaleRepoMode必须同时为0。
  - pertoken-pergroup量化模式下，CacheMode只支持PA_BSND, BSND和TND。
  - 当ckvkrRepoMode值为1时，krCache必须为空Tensor（即shape的乘积为0）。
  - kvcache pertensor量化模式下，kvCacheQuantMode和queryQuantMode必须同时为1。
- aclnnMlaPrologV3WeightNz接口支持场景：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前不支持fp8/hif8/mxfp8全量化场景
    - <term>Ascend 950PR/Ascend 950DT</term>：当前支持所有量化场景
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">场景</th>
      <th>含义</th>
    </tr>
    <tr>
      <td colspan="2">非量化</td>
      <td>
          weight_quant_mode=0，kv_cache_quant_mode=0，query_quant_mode=0<br>
          入参：所有入参皆为非量化数据 <br> 
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">部分量化</td>
      <td>kvCache非量化 </td>
      <td>
          weight_quant_mode=1，kv_cache_quant_mode=0，query_quant_mode=0<br>
          入参：weightUqQr传入pertoken量化数据，其余入参皆为非量化数据。dequantScaleWUqQr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kvCache perchannel量化 </td>
      <td>
          weight_quant_mode=1，kv_cache_quant_mode=2，query_quant_mode=0<br>
          入参：weightUqQr传入pertoken量化数据，kvCacheRef、krCacheRef传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleWUqQr、quantScaleCkv、quant_scale_ckr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：kvCacheRef、krCacheRef返回perchannel量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kvCache pertoken-pergroup量化 </td>
      <td>
          weight_quant_mode=1, kv_cache_quant_mode=3, query_quant_mode=0<br>
          入参：weightUqQr传入pertoken量化数据，其余入参皆为非量化数据。dequantScaleWUqQr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：kvCacheRef返回pertoken-pergroup量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8/fp8/hif8全量化</td>
      <td> kvCache非量化</td>
      <td>
          weight_quant_mode=2/4/5，kv_cache_quant_mode=0，query_quant_mode=0<br>
          入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor量化 </td>
      <td>
          weight_quant_mode=2/4/5，kv_cache_quant_mode=1，query_quant_mode=1<br>
          入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，kvCacheRef传入pertensor量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr、quantScaleCkv字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：queryOut返回per-token-head量化数据，kvCacheRef出参返回pertensor量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup量化 </td>
      <td>
          weight_quant_mode=2/4/5，kv_cache_quant_mode=3，query_quant_mode=0<br>
          入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：kvCacheRef出参返回pertoken-pergroup量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8全量化</td>
      <td> kvCache非量化</td>
      <td>
          weight_quant_mode=3，kv_cache_quant_mode=0，query_quant_mode=0<br>
          入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache pertensor量化 </td>
      <td>
          weight_quant_mode=3，kv_cache_quant_mode=1，query_quant_mode=1<br>
          入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，kvCacheRef传入pertensor量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr、quantScaleCkv字段必须传入 <br>
          出参：queryOut返回per-token-head量化数据，kvCacheRef出参返回pertensor量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache pertoken-pergroup量化 </td>
      <td>
          weight_quant_mode=3，kv_cache_quant_mode=3，query_quant_mode=0<br>
          入参：tokenX传入pertoken量化数据，weightDq、weightUqQr、weightDkvKr传入perchannel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入 <br>
          出参：kvCacheRef出参返回pertoken-pergroup量化数据，其余出参返回非量化数据
      </td>
    </tr>
  </table>

## 调用说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v3.cpp">MlaPrologV3接口测试用例代码
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnMlaPrologV3WeightNz.md">aclnnMlaPrologV3WeightNz
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
