# KvRmsNormRopeCache

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算（对应$rms\_size$）与RoPE（Rotary Position Embedding）位置编码（对应$rope\_size$)，以及更新KVCache的ScatterUpdate操作。

- 支持场景：

  | 场景类型 | kv分量来源 | 说明 |
    | :------: | :------: | :------ |
    |V1|rms_size=Dv=512<br>rope_size=Dk=64<br>vOptional=None|kv合轴模式：对输入张量kv的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。<li>与DeepSeekV3网络结构强相关，仅支持N=1的场景。</li><li>rms_norm计算所需数据Dv和rope计算所需数据Dk由输入kv的D切分而来，Dk、Dv大小需满足Dk+Dv=Dkv。|
    |V2|Dv=128<br>rms_size=Dk=Dkv=192<br>rope_size=64<br>vOptional的shape为[Bkv, Nkv, Skv, Dv]|kv分离模式：对输入张量kv进行rms_norm计算，之后对尾轴前64维进行rope计算并覆盖写回对应元素，最终结果scatter写入到k_cache中；对输入张量vOptional进行中间处理，最终结果scatter写入到ckv_cache中。</li><li>支持N=1/2/4/8</li><li>此场景下k与v尾轴分离，kv仅存储k分量尾轴，vOptional则存储v分量尾轴。</li>|

    * <term>Ascend 950PR/Ascend 950DT</term>：仅支持V1场景。
    * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持V1和V2场景。

- 计算公式：

  定义输入张量kv的shape为$[Bkv, N, Skv, Dkv]$以及张量vOptional的shape为$[Bkv, N, Skv, Dv]$。

  (1) RmsNorm：

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

  (3) 量化计算:

  x表示将要写入到k_cache和ckv_cache上的原始数据，作为量化过程的输入。

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

  (4) Scatter写出：

  输入张量index对应输入kv缓存中各元素的索引映射表，取x中具体元素的索引$b \in Bkv$以及$s \in Skv$，$n$为注意力头索引，

  $$
  scatter\_idx = index(b, s) \\
  $$

   Quant表示前述量化计算过程，对原地更新参数k\_cache和ckv\_cache：

  $$
  k\_cache[scatter\_idx, ...] = Quant(x = rope\_out, scale = k\_scale, offset = k\_offset)[b, n, s]
  $$

  $$
  ckv\_cache[scatter\_idx, ...] = \begin{cases} Quant(x = \operatorname{RmsNorm}(x), scale = v\_scale, offset = v\_offset)[b, n, s], \quad vOptional = None \\
  Quant(x = vOptional, scale = v\_scale, offset = v\_offset)[b, n, s],  \quad vOptional != None
  \end{cases}
  $$

  (5) 原始结果写出：

  当$is\_output\_kv=True$且有效时：

  $$
  k\_rope = rope\_out
  $$

  $$
  c\_kv = \begin{cases}\operatorname{RmsNorm}(x), \quad vOptional = None \\
  vOptional,  \quad vOptional != None
  \end{cases}
  $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1920px"><colgroup>
  <col style="width: 84px">
  <col style="width: 128px">
  <col style="width: 448px">
  <col style="width: 288px">
  <col style="width: 72px">
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
      <td>kv</td>
      <td>输入</td>
      <td>用于切分出rms_norm计算所需数据Dv和rope计算所需数据Dk的输入数据，对应公式中的`kv`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>用于rms_norm计算的输入数据，对应公式中的`gamma`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行余弦变换，对应公式中的`cos`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>输入</td>
      <td>用于rope计算的输入数据，对输入张量进行正弦变换，对应公式中的`sin`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输入</td>
      <td>用于指定写入cache的具体索引位置。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_cache</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ckv_cache</td>
      <td>输入/输出</td>
      <td>提前申请的cache，输入输出同地址复用。</td>
      <td>FLOAT16、BFLOAT16、INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_rope_scale</td>
      <td>可选属性</td>
      <td>当k_cache数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN时，需要此输入参数。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_kv_scale</td>
      <td>可选属性</td>
      <td>当ckv_cache数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN时，需要此输入参数。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_rope_offset</td>
      <td>可选属性</td>
      <td>当k_cache数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN，且对应的k_rope_scale输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_kv_offset</td>
      <td>可选属性</td>
      <td>当ckv_cache数据类型为INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN，且对应的c_kv_scale输入存在并量化场景为非对称量化时，需要此参数输入。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_optional</td>
      <td>可选属性</td>
      <td>仅限kv分离场景(V2)中，作为immediate scatter的Dv分量输入来源。<br>shape的前三维度必须与kv保持一致，数据类型必须与kv保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止rms_norm计算除0错误，对应公式中的eps。</li><li>默认值为1e-5。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cache_mode</td>
      <td>可选属性</td>
      <td>cache格式的选择标记。类型有Norm、PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ。</td>
      <td>CHAR*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_output_kv</td>
      <td>可选属性</td>
      <td>k_rope和c_kv输出控制标记。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>k_rope</td>
      <td>输出</td>
      <td>rope计算结果，对应interleaveRope计算公式中的`rope_out`。<br>由is_output_kv控制，当is_output_kv为true时，需输出。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_kv</td>
      <td>输出</td>
      <td>rms_norm计算结果，对应rmsNorm计算公式中的`y`。<br>由is_output_kv控制，当is_output_kv为true时，需输出。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* cache的数据类型支持：

  * 非量化模式：cache类型必须与kv保持一致。
    * <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：可支持BFLOAT16、FLOAT16。
    * <term>Kirin X90/Kirin 9030 处理器系列产品</term>：仅支持FLOAT16。

  * 量化模式：
    * <term>Ascend 950PR/Ascend 950DT</term>：可支持INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN。
    * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Kirin X90/Kirin 9030 处理器系列产品</term>：仅支持INT8。

## 约束说明

  * 本算子默认确定性实现。
  * 输入shape限制：

      * kv为四维张量，shape为[Bkv,N,Skv,D]，Bkv为输入kv的batch size，Skv为输入kv的sequence length，大小由用户输入场景决定，无明确限制。
      * N为输入kv的head number。V1场景与DeepSeekV3网络结构强相关，仅支持N=1的场景。V2场景支持N=1/2/4/8。
      * D为输入kv的head dim。根据rope规则，Dk为偶数。若cache_mode为NZ场景（cache_mode为PA_NZ、PA_BLK_NZ），Dk、Dv需32B对齐。该规则适用于所有场景和计算类型中。
      * 若cache_mode为PA场景（cache_mode为PA、PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ），block_size需32B对齐。
      * 关于上述32B对齐的情形，对齐值由cache的数据类型决定。以block_size为例，若cache的数据类型为int8，则需block_size%32=0；若cache的数据类型为float16，则需block_size%16=0；若kCacheRef与ckvCacheRef参数的dtype不一致，block_size需同时满足block_size%32=0和block_size%16=0。
      * block_num为写入cache的内存块数，大小由用户输入场景决定，无明确限制。
      * 旋转位置编码（RoPE）参数项(cos, sin)需要满足shape约束：
        * shape允许为4维[Bkv,N,Skv,Dk]或[Bkv,N,1,Dk]。
        * cos与sin的shape必须保持一致。
      * 量化参数项(k_rope_scale, k_rope_offset, c_kv_scale, c_kv_offset)需要满足shape约束：
        * 所有量化参数项的维度数量和N轴尺寸（如果存在）必须保持一致。
        * 不同场景下，量化参数项的合法shape约束存在差异：

          | 场景类型 | 量化参数Shape |
          | :------: | :------ |
          |V1|<li>k_rope_scale和k_rope_offset的shape支持：[1, Dk]、[Dk,]、[1,]。</li><li>c_kv_scale和c_kv_offset的shape支持：[1, Dv]、[Dv,]、[1,]。</li>|
          |V2|<li>k_rope_scale和k_rope_offset的shape支持：[N, Dk]。</li><li>c_kv_scale和c_kv_offset的shape支持：[N, Dv]。</li>|

          * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：V1场景不支持量化参数项的shape为[1,]。

      * 输入张量均不支持空Tensor。
      * 所有输入均不支持无效值，包括且不限于：±inf，nan。

  * 参数说明：

    * 输入参数中kv, gamma, cos, sin, vOptional的数据类型必须完全一致。
    * k_cache和ckv_cache是<b>原地更新参数</b>，它们的数据类型取决于相应的输入分量，以及相应的scale和offset。详情见下：

      | cache_type | offset==None | offset!=None |
      | :-------------------------: | :-----------: | :----------------------------------------------: |
      | scale==None | 与kv保持一致 | 非法输入，拦截 |
      | scale!=None | INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN | INT8、HIFLOAT8、FLOAT8E5M2、FLOAT8E4M3FN |

      * 量化模式时，k_cache和ckv_cache的dtype应为相应产品上支持的数据类型。
      * 非量化模式时，量化参数(k_rope_scale, k_rope_offset, c_kv_scale, c_kv_offset)必须设为None，且k_cache和ckv_cache的dtype必须与kv保持一致。

    * 输入分量关于量化因子scale与量化偏移scale的对应关系如下：

      | kv分量 | 分量输入来源 | 对应scale输入 | 对应offset输入 | 对应输出cache |
      | :------: | :------: | :------: | :------: | :------: |
      | k分量 | V1场景下，对应kv[..., Dv:]。<br>V2场景下，对应kv[..., :]。 | k_rope_scale | k_rope_offset | k_cache |
      | v分量 | V1场景下，对应kv[..., :Dv]。<br>V2场景下，对应vOptional[..., :]。 |c_kv_scale | c_kv_offset | ckv_cache |

      * k_cache：量化系数为k_rope_scale和k_rope_offset。
      * ckv_cache：对应量化系数为c_kv_scale和c_kv_offset。

    * 输出参数中，k_rope和c_kv的类型必须与kv保持一致。

  * 量化模式约束：

    <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
      <col style="width: 400px">
      <col style="width: 160px">
      <col style="width: 448px">
      </colgroup>
      <thead>
        <tr>
          <th>量化模式说明<br>(scale和offset输入之间为'与'关系)</th>
          <th>支持的cache模式</th>
          <th>可用性说明</th>
        </tr></thead>
      <tbody>
      <tr>
          <td rowspan="5">无量化模式<li>scale输入：k_rope_scale==None && c_kv_scale==None</li><li>offset输入：k_rope_offset==None && c_kv_offset==None</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5">V1和V2都支持。</td>
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
          <td rowspan="5">静态量化模式<li>scale输入：k_rope_scale和c_kv_scale至少一个非空。</li><li>offset输入：仅限对应scale为非空时，offset输入合法。相应offset如果为空，则为<b>静态对称量化</b>；相应offset如果非空，则为<b>静态非对称量化</b>。</li></td>
          <td rowspan="1">Norm</td>
          <td rowspan="5"><li><b>静态对称量化</b>和<b>静态非对称量化</b>，支持存在差异。</li><li>支持K和V独立选择不同量化模式。</li></td>
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

    * 静态量化模式支持细节：
      * Ascend 950PR/Ascend 950DT产品：仅支持V1场景，支持<b>静态对称量化</b>和<b>静态非对称量化</b>。
      * <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
        - V1场景：对除`Norm`以外的cachemode，仅支持<b>静态对称量化</b>。即使传入合法的offset，也不会被算子处理，仍视为<b>静态对称量化</b>。对`Norm`模式的cachemode，不支持任何静态量化。
        - V2场景：对所有cachemode，支持<b>静态对称量化</b>和<b>静态非对称量化</b>。

  * cache与index相关约束：

    | cachemode| k_cache 形状 | ckv_cache 形状 | index 形状 | 说明 |
    | :------: | :------: | :------: | :------: | :------ |
    |Norm|[Bkv, N, Scache, Dk]|[Bkv, N, Scache, Dv]|[Bkv, Skv]|KV-Cache 更新模式，index 表示每个 Batch 下的偏移。<br>要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。<br>$Scache \ge Skv$|
    |PA/PA_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|PagedAttention 模式，index 表示每个 token 的偏移。<br>要求index的value值范围为[-1,block_num * block_size)。value数值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × Skv]|Cache 数据格式为 FRACTAL_NZ 的 PagedAttention 模式，index表示每个 token 的偏移。<br>要求index的value值范围为[-1,block_num * block_size)。value数值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_BLK_BNSD|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|特殊 PagedAttention 模式，index 表示每个 block 的起始偏移（不与 token逐一对应）。<br>要求index的value的数值范围为[-1,block_num * block_size)。value/block_size的值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|
    |PA_BLK_NZ|[block_num, block_size, N, Dk]|[block_num, block_size, N, Dv]|[Bkv × ceil(Skv / block_size)]|Cache 数据格式为 FRACTAL_NZ 的特殊的 PagedAttention 模式，index 表示每个 block 的起始偏移。<br>要求index的value的数值范围为[-1,block_num * block_size)。value/block_size的值不能重复。<br>$block\_size>1,\\ block\_num \ge Floor(Skv / block\_size) * Bkv$|

    * Scache为输入cache的sequence length，大小由用户输入场景决定，无明确限制。
    * 当cache_mode为Norm时，shape为2维[Bkv,Skv]，要求index的value值范围为[-1,Scache)。不同的Bkv下，value数值可以重复。
    * 当cache_mode为PA_BNSD、PA_NZ、PA_BLK_BNSD、PA_BLK_NZ时，cache中的数据排布方式为：
      * 非量化模式下：k_cache 为 [block_num, Dk//16, block_size, 1, 16]；ckv_cache 为 [block_num, Dv//16, block_size, 1, 16]。
      * 静态量化模式下：k_cache 为 [block_num, Dk//32, block_size, 1, 32]；ckv_cache 为 [block_num, Dv//32, block_size, 1, 32]。
    * 当cache_mode为PA_BNSD、PA_NZ时，shape为1维[Bkv * Skv]，要求index的value值范围为[-1,block_num * block_size)。value数值不能重复。
    * 当cache_mode为PA_BLK_BNSD、PA_BLK_NZ时，shape为1维[Bkv * ceil_div(Skv,block_size)]，要求index的value的数值范围为[-1,block_num * block_size)。value/block_size的值不能重复。
  * is_output_kv约束：
    * 作用是输出中间处理结果：开启 k_rope 和 c_kv。
    * 在cache_mode为PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, PA_BLK_NZ模式时有效。
    * 在cache_mode为Norm时，仅在V2场景中开启量化模式时有效。

  * 输入组合约束：

    * 本约束内条目，仅适用于<b>Atlas A3 训练系列产品/Atlas A3 推理系列产品</b>、<b>Atlas A2 训练系列产品/Atlas A2 推理系列产品</b>。
    * 在所有cache_mode下，必定支持`非广播模式`：即`[B, N, S]`三个维度与`kv`严格一致的`旋转位置编码（RoPE）参数`和`量化参数`。
    * 在各种cache_mode下，支持的合法输入shape模式如下表：

      <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
        <col style="width: 160px">
        <col style="width: 280px">
        <col style="width: 280px">
        <col style="width: 320px">
        </colgroup>
        <thead>
          <tr>
            <th>cache模式</th>
            <th>kv Shape</th>
            <th>旋转位置编码（RoPE）参数<br>Shape</th>
            <th>其他参数</th>
          </tr>
          </thead>
        <tbody>
        <tr>
            <td>Norm</td>
            <td rowspan="6"><li>V1场景：[Bkv, 1, Skv, Dv+Dk]（kv）</li><li>V2场景：<ul><li>[Bkv, N, Skv, Dk]（kv）</li><li>[Bkv, N, Skv, Dv]（vOptional）</li></ul></li></td>
            <td rowspan="2"><li>支持S轴无广播：Srope=Skv。</li><li>支持S轴广播：Srope=1。</li></td>
            <td><li>在<b>无量化模式</b>下：is_output_kv无效，k_rope 和 c_kv 无效。</li><li>V1场景：<ul><li>仅支持<b>无量化模式</b>，所有量化参数项皆为非法。</li></ul></li><li>V2场景：<ul><li>支持<b>无量化模式</b>和<b>静态量化模式</b>。</li><li>k和v对应的量化参数项必须为2维[N, D]。</li></ul></li></td>
        </tr>
        <tr>
            <td>PA</td>
            <td><li><b>无量化模式</b>下，支持S轴广播和S轴无广播。</li><li><b>静态量化模式</b>下，仅支持S轴无广播，否则为非法输入。</li><li>量化参数项：<ul><li>V1场景：支持shape为2维[1,D]或1维[D]。仅支持<b>静态对称量化</b>。</li><li>V2场景：支持shape为2维[N,D]。支持<b>静态对称量化</b>和<b>静态非对称量化</b>。</li></ul></li></td>
        </tr>
        <tr>
            <td>PA_BNSD</td>
            <td rowspan="4"><li>不支持S轴广播：必须满足Srope=Skv。</li></td>
            <td rowspan="4"><li>支持无量化模式。</li><li>量化参数项：</li><ul><li>V1场景：支持shape为2维[1,D]或1维[D]。仅支持<b>静态对称量化</b>。</li><li>V2场景：支持shape为2维[N,D]。支持<b>静态对称量化</b>和<b>静态非对称量化</b>。</li></ul></td>
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

    * <b>未在上表覆盖范围内的输入组合，将导致算子的未定义行为</b>。

  * vOptional：
    * 该参数仅限aclnnKvRmsNormRopeCacheV2接口，aclnnKvRmsNormRopeCache接口不支持该参数！
    * 该参数仅限<b>Atlas A3 训练系列产品/Atlas A3 推理系列产品</b>、<b>Atlas A2 训练系列产品/Atlas A2 推理系列产品</b>。
      * 该参数仅在<b>kv分离场景(V2)</b>中作为必须入参，在其他类型中会作为无效参数被忽略。
      * 当vOptional存在时，它的类型必须与kv一致，`[B, N, S]`维度也必须与kv一致。
    * Ascend 950PR/Ascend 950DT：不会拦截该参数，但实际功能不支持，也不会处理该参数。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_kv_rms_norm_rope_cache](examples/test_aclnn_kv_rms_norm_rope_cache.cpp) | 通过[aclnnKvRmsNormRopeCache](docs/aclnnKvRmsNormRopeCache.md)接口方式调用KvRmsNormRopeCache算子。 |
| 图模式 | [test_geir_kv_rms_norm_rope_cache](examples/test_geir_kv_rms_norm_rope_cache.cpp)  | 通过[算子IR](op_graph/kv_rms_norm_rope_cache_proto.h)构图方式调用KvRmsNormRopeCache算子。         |
