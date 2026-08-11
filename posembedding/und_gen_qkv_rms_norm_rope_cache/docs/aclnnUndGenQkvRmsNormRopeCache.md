# aclnnUndGenQkvRmsNormRopeCache

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：面向多模态大模型推理的QKV主干融合前处理算子。接口把理解阶段（undecoded）与生成阶段（generated）两段QKV输入按`catIndicesOptional`间接寻址拼成一条输出序列，逐token沿头维度拆分出Q/K/V，对Q和K逐头执行RMSNorm归一化与MRoPE（多模态旋转位置编码），V不参与归一化与旋转；Q作为独立输出返回，K和V按`slotMapping`写入分页KV Cache（`kCacheRef`/`vCacheRef`原地更新）。RMSNorm权重按源token落在理解段还是生成段在两套权重间选择。计算全程float32，输出与KV Cache均为BF16。

- 计算公式：

  完整计算流程可以分解为以下基本计算单元。

  第一步，根据`catIndicesOptional`的间接寻址关系，从理解阶段或生成阶段的QKV输入中取出每个输出token对应的源数据行：

  $$
  X_t = \begin{cases} X_{\text{und}}[c_t] & \text{若 } c_t < L_{\text{und}} \\ X_{\text{gen}}[c_t - L_{\text{und}}] & \text{若 } c_t \geq L_{\text{und}} \end{cases}
  $$

  说明：`catIndicesOptional`给出第t个输出token的源token下标c<sub>t</sub>，据此从理解阶段输入或生成阶段输入中取出对应的QKV数据行。该行沿头维度依次拆分为Q（H<sub>q</sub>个头）、K（H<sub>k</sub>个头）和V（H<sub>v</sub>个头）三段。后续RMSNorm权重也按同一条件同步选择und段（`undWeightsQ`/`undWeightsK`）或gen段（`genWeightsQOptional`/`genWeightsKOptional`）。

  第二步，对每个Q头或K头的D维向量执行RMSNorm归一化：

  $$
  \hat{x}_h = \frac{x_h}{\sqrt{\frac{1}{D} \sum_{i=0}^{D-1} x_{h,i}^2 + \epsilon}} \odot \gamma_h
  $$

  说明：对单个头的D维向量计算均方根值，将原向量除以该值后逐元素乘以对应的RMSNorm权重γ。同一token的所有Q头共用一份Q权重，所有K头共用一份K权重。V不参与此步骤。

  第三步，对归一化后的Q和K执行MRoPE旋转位置编码，将D维向量按前后各半分为低半和高半两部分进行旋转。对低半索引l（0 ≤ l < D/2），取p<sub>l</sub> = p[α(l), t]，cos<sub>l</sub> = C[p<sub>l</sub>, l]，sin<sub>l</sub> = C[p<sub>l</sub>, l + D/2]：

  $$
  \tilde{x}_{\text{lo},l} = \hat{x}_{\text{lo},l} \cdot \cos_l - \hat{x}_{\text{hi},l} \cdot \sin_l
  $$

  $$
  \tilde{x}_{\text{hi},l} = \hat{x}_{\text{hi},l} \cdot \cos_l + \hat{x}_{\text{lo},l} \cdot \sin_l
  $$

  说明：`cosSinCache`的前D/2列为cos、后D/2列为sin。每个低半位置l由轴映射函数α(l)决定从`positions`的哪一轴取位置值，再以该位置值作为行号查`cosSinCache`。`mropeSection`为空时所有位置映射到时间轴，退化为标准RoPE。同一token的所有Q头和K头共用同一份cos/sin。V不参与此步骤。

  第四步，将旋转后的Q和K转换为BF16后写出，V直接透传写入KV Cache。令s<sub>t</sub> = s[t]：

  $$
  Q_{\text{out}}[t] = \mathrm{Cast}_{\text{bf16}}(\tilde{q}_t), \quad K_{\text{cache}}\left[\left\lfloor s_t / B_s \right\rfloor,\ s_t \bmod B_s\right] = \mathrm{Cast}_{\text{bf16}}(\tilde{k}_t), \quad V_{\text{cache}}\left[\left\lfloor s_t / B_s \right\rfloor,\ s_t \bmod B_s\right] = v_t
  $$

  说明：Q按输出token顺序写入独立输出张量；K按`slotMapping`指定的槽位写入分页KV Cache；V不做归一化与旋转，直接按`slotMapping`写入分页KV Cache。槽位号s<sub>t</sub>由调用方预计算，拆成块号与块内行号后定位到cache的对应行。未被`slotMapping`命中的cache位置保持调用方传入的原值。

  其中，X<sub>und</sub>表示参数`undQkv`，X<sub>gen</sub>表示参数`genQkvOptional`，γ<sub>h</sub>表示参数`undWeightsQ`、`undWeightsK`、`genWeightsQOptional`或`genWeightsKOptional`（按源token来自理解阶段还是生成阶段、以及当前是Q头还是K头选择对应权重），C表示参数`cosSinCache`，K<sub>cache</sub>表示参数`kCacheRef`，V<sub>cache</sub>表示参数`vCacheRef`，s表示参数`slotMapping`，p表示参数`positions`，c表示参数`catIndicesOptional`，H<sub>q</sub>表示参数`numHeadsQ`，H<sub>k</sub>表示参数`numHeadsK`，H<sub>v</sub>表示参数`numHeadsV`，ε表示参数`normEps`，m表示参数`mropeSection`，Q<sub>out</sub>表示参数`qOut`，B<sub>s</sub>表示`kCacheRef`的第二维（blockSize），D表示头维度（固定为128），T表示总token数（等于`undQkv`第一维与`genQkvOptional`第一维之和）。

  - L<sub>und</sub>：理解阶段的token数量，即`undQkv`的第一维，非接口参数。
  - α(l)：MRoPE轴映射函数，把每个低半位置l（0 ≤ l < D/2）映射到时间（0）、高度（1）、宽度（2）三轴之一，规则为：

    $$
    \alpha(l) = \begin{cases} 1 & l \bmod 3 = 1 \ \text{且}\ l < 3 m_1 \\ 2 & l \bmod 3 = 2 \ \text{且}\ l < 3 m_2 \\ 0 & \text{其他} \end{cases}
    $$

    该规则只读取`mropeSection`的第2、3个分量（m<sub>1</sub>、m<sub>2</sub>），第1个分量m<sub>0</sub>不参与计算，时间轴是"其余全归它"的兜底轴。因此`mropeSection=[16,16,16]`实际得到时间/高度/宽度 = 32/16/16，而`[64,16,16]`与`[0,16,16]`的轴映射逐位相同。`mropeSection`为空时α(l) ≡ 0，退化为标准RoPE。非接口参数。

  - cos<sub>l</sub>、sin<sub>l</sub>：按α(l)选轴、以`positions`中对应位置值为行号从`cosSinCache`查得的旋转角分量，非接口参数。
  - x<sub>h</sub>：单个头的D维输入向量，从X<sub>t</sub>中按头拆分得到，非接口参数。
  - x̂<sub>h</sub>：RMSNorm归一化后的中间结果，非接口参数。
  - x̂<sub>lo</sub>、x̂<sub>hi</sub>：归一化后向量的低半（前D/2个元素）和高半（后D/2个元素），非接口参数。
  - x̃<sub>lo</sub>、x̃<sub>hi</sub>：MRoPE旋转后的低半和高半结果，非接口参数。
  - q̃<sub>t</sub>、k̃<sub>t</sub>：第t个token的Q、K旋转结果，非接口参数。
  - v<sub>t</sub>：第t个token的V数据，直接透传不做任何转换，非接口参数。
  - Cast<sub>bf16</sub>：将float32中间结果转换为BF16的数据类型转换操作，非接口参数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUndGenQkvRmsNormRopeCache”接口执行计算。

```cpp
aclnnStatus aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor   *undQkv,
  const aclTensor   *undWeightsQ,
  const aclTensor   *undWeightsK,
  const aclTensor   *cosSinCache,
  aclTensor         *kCacheRef,
  aclTensor         *vCacheRef,
  const aclTensor   *slotMapping,
  const aclTensor   *positions,
  const aclTensor   *genQkvOptional,
  const aclTensor   *genWeightsQOptional,
  const aclTensor   *genWeightsKOptional,
  const aclTensor   *catIndicesOptional,
  int64_t            numHeadsQ,
  int64_t            numHeadsK,
  int64_t            numHeadsV,
  double             normEps,
  const aclIntArray *mropeSection,
  aclTensor         *qOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnUndGenQkvRmsNormRopeCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize

- 参数说明

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1500px;"><colgroup>
      <col style="width: 180px;">
      <col style="width: 120px;">
      <col style="width: 300px;">
      <col style="width: 350px;">
      <col style="width: 250px;">
      <col style="width: 100px;">
      <col style="width: 100px;">
      <col style="width: 100px;">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>undQkv（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的X<sub>und</sub>，表示理解阶段QKV输入张量。</td>
        <td><ul><li>不支持空指针或空Tensor，<code>und_len</code>必须为正数。</li><li>第二维N必须等于<code>numHeadsQ + numHeadsK + numHeadsV</code>，沿该维依次拆分为Q/K/V。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[und_len,N,D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>undWeightsQ（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的γ<sub>h</sub>，表示理解阶段Q的RMSNorm权重。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>源token落在理解段时，该token的所有Q头共用此权重。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>undWeightsK（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的γ<sub>h</sub>，表示理解阶段K的RMSNorm权重。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>源token落在理解段时，该token的所有K头共用此权重。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>cosSinCache（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的C，表示旋转位置编码的cos/sin缓存表。</td>
        <td><ul><li>不支持空指针或空Tensor，<code>max_pos</code>必须为正数。</li><li>前<code>D/2</code>列为cos，后<code>D/2</code>列为sin。</li><li>第一维需覆盖<code>positions</code>中出现的最大位置值。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[max_pos,D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>kCacheRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>公式中的K<sub>cache</sub>，表示K的分页KV Cache缓冲区。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>由调用方预分配，接口基于传入Tensor原地更新。</li><li>必须为内存连续Tensor；非连续时接口直接返回失败。</li><li>容量需满足<code>Bn * Bs >= T</code>。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[Bn,Bs,Hk,D]</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>vCacheRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>公式中的V<sub>cache</sub>，表示V的分页KV Cache缓冲区。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>由调用方预分配，接口基于传入Tensor原地更新。</li><li>必须为内存连续Tensor；非连续时接口直接返回失败。</li><li><code>Bn</code>、<code>Bs</code>必须与<code>kCacheRef</code>一致。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[Bn,Bs,Hv,D]</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>slotMapping（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的s，表示每个输出token写入KV Cache的槽位号。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>槽位号由调用方预计算，<code>slot = blockIdx * Bs + rowIdx</code>。</li><li>取值范围应为<code>[0, Bn*Bs-1]</code>且互不重复；重复槽位在多核下写入顺序与结果未定义。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>[T]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>positions（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的p，表示MRoPE三轴位置表，三行依次为时间轴、高度轴、宽度轴。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>第一维固定为3。</li><li>取值应落在<code>[0, max_pos-1]</code>。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>[3,T]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>genQkvOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>公式中的X<sub>gen</sub>，表示生成阶段QKV输入张量。</td>
        <td><ul><li>当前版本必须传入且<code>gen_len</code>为正数，传入nullptr（纯prefill）暂不支持。</li><li>N和D维必须与<code>undQkv</code>一致。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[gen_len,N,D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>genWeightsQOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>公式中的γ<sub>h</sub>，表示生成阶段Q的RMSNorm权重。</td>
        <td><ul><li>当前版本必须传入，传入nullptr暂不支持。</li><li>传入<code>genQkvOptional</code>时必须与<code>genWeightsKOptional</code>成对传入。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>genWeightsKOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>公式中的γ<sub>h</sub>，表示生成阶段K的RMSNorm权重。</td>
        <td><ul><li>当前版本必须传入，传入nullptr暂不支持。</li><li>传入<code>genQkvOptional</code>时必须与<code>genWeightsQOptional</code>成对传入。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>catIndicesOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>公式中的c，表示输出token到源token的映射表。</td>
        <td><ul><li>当前版本必须传入，传入nullptr（单序列恒等映射）暂不支持。</li><li>取值应落在<code>[0, T-1]</code>；小于<code>und_len</code>时取理解段，否则取生成段。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>[T]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>numHeadsQ（int64_t）</td>
        <td>输入</td>
        <td>公式中的H<sub>q</sub>，表示Q的头数。</td>
        <td><ul><li>必须为正数。</li><li><code>(numHeadsQ, numHeadsK, numHeadsV)</code>当前仅支持<code>(8,1,1)</code>和<code>(16,2,2)</code>两种组合。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeadsK（int64_t）</td>
        <td>输入</td>
        <td>公式中的H<sub>k</sub>，表示K的头数。</td>
        <td><ul><li>必须为正数，且等于<code>kCacheRef</code>的第三维。</li><li><code>(numHeadsQ, numHeadsK, numHeadsV)</code>当前仅支持<code>(8,1,1)</code>和<code>(16,2,2)</code>两种组合。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeadsV（int64_t）</td>
        <td>输入</td>
        <td>公式中的H<sub>v</sub>，表示V的头数。</td>
        <td><ul><li>必须为正数，且等于<code>vCacheRef</code>的第三维。</li><li><code>(numHeadsQ, numHeadsK, numHeadsV)</code>当前仅支持<code>(8,1,1)</code>和<code>(16,2,2)</code>两种组合。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps（double）</td>
        <td>输入</td>
        <td>公式中的ε，表示RMSNorm防除零参数。</td>
        <td><ul><li>必须为正数，推荐取<code>1e-6</code>。</li><li>接口内部按FLOAT参与计算。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mropeSection（const aclIntArray*）</td>
        <td>输入</td>
        <td>公式中的m，表示MRoPE三轴分段，决定轴映射函数α(l)。</td>
        <td><ul><li>支持传入空指针或空数组，此时按<code>[D/2,0,0]</code>处理，退化为标准RoPE。</li><li>非空时长度必须为3，每个元素非负，三元素之和不超过<code>D/2</code>。</li><li>只有第2、3个元素参与轴映射，第1个元素不参与计算，详见“功能说明”中α(l)的定义。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>长度为3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qOut（aclTensor*）</td>
        <td>输出</td>
        <td>公式中的Q<sub>out</sub>，表示归一化与旋转后的Q输出张量。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>由调用方预分配，shape必须为<code>[T, numHeadsQ, D]</code>。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[T,Hq,D]</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>不支持空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor**）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>不支持空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
  </div>

- 返回值

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="table-layout: fixed; width: 1000px;"><colgroup>
    <col style="width: 300px;">
    <col style="width: 150px;">
    <col style="width: 550px;">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="white-space: nowrap">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>传入的<code>undQkv</code>、<code>undWeightsQ</code>、<code>undWeightsK</code>、<code>cosSinCache</code>、<code>kCacheRef</code>、<code>vCacheRef</code>、<code>slotMapping</code>、<code>positions</code>、<code>qOut</code>、<code>workspaceSize</code>或<code>executor</code>是空指针。</td>
    </tr>
    <tr>
      <td>提供了<code>genQkvOptional</code>，但<code>genWeightsQOptional</code>或<code>genWeightsKOptional</code>是空指针。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><code>kCacheRef</code>或<code>vCacheRef</code>是非连续Tensor。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>创建<code>aclOpExecutor</code>失败，或输入的Contiguous处理失败。</td>
    </tr>
  </tbody>
  </table>

  > 数据类型、shape与属性取值的完整校验统一收敛在tiling一层实现（aclnn单算子路径与图模式最终都会走tiling，避免同一条约束在多处重复实现导致漂移）。代价是这类非法入参不在第一段接口报错，而是在第二段接口`aclnnUndGenQkvRmsNormRopeCache`执行阶段由tiling返回失败，具体原因见host侧日志。

## aclnnUndGenQkvRmsNormRopeCache

- 参数说明

  <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。当第一段接口返回的<code>workspaceSize</code>为0时，可传入<code>nullptr</code>。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- 返回值

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnUndGenQkvRmsNormRopeCache默认确定性实现。

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

  - **公共约束**
    <a id="公共约束"></a>
    - 计算全程使用float32，KV Cache以BF16存储，不支持FP8/INT8量化。
    - headDim固定为128。
    - `(numHeadsQ, numHeadsK, numHeadsV)`仅支持`(8,1,1)`和`(16,2,2)`两种组合。
    - numHeadsQ + numHeadsK + numHeadsV必须等于`undQkv`的第二维N。
    - und_len必须为正数；T = und_len + gen_len必须为正数。
    - T不设人为上限，真实上限由KV Cache容量决定，需满足blockNum × blockSize ≥ T。上板已验证到T = 64K。
    - blockSize不设限：cache强制为连续BBND，`[Bn, Bs, N, D]`展平后等价`[Bn*Bs, N, D]`，槽位号直接作为扁平行号使用，Bs不参与地址计算，也不参与多核与UB切分。上板已验证Bs ∈ {16, 64, 100, 128, 256, 512}，含非2的幂。
    - `vCacheRef`的Bn、Bs必须与`kCacheRef`一致。
    - `kCacheRef`第三维必须等于numHeadsK，第四维必须等于headDim（128）。
    - `vCacheRef`第三维必须等于numHeadsV，第四维必须等于headDim（128）。
    - `kCacheRef`、`vCacheRef`必须为内存连续Tensor，不支持非连续（含BNBD物理布局）。
    - `genQkvOptional`的N和D维必须与`undQkv`一致。
    - `undWeightsQ`、`undWeightsK`、`genWeightsQOptional`、`genWeightsKOptional`均必须为1维`[D]`，D = headDim = 128。
    - `cosSinCache`必须为2维`[max_pos, D]`，max_pos为正数，D = headDim = 128。
    - `slotMapping`必须为1维`[T]`；`catIndicesOptional`必须为1维`[T]`；`positions`必须为2维`[3, T]`，第一维固定为3。
    - `qOut`必须为3维`[T, numHeadsQ, D]`。
    - `normEps`必须为正数。
    - `mropeSection`为空指针或空数组时按`[D/2, 0, 0]`处理；非空时长度必须为3、每个元素非负、三元素之和不超过D/2。
    - **当前版本要求`genQkvOptional`、`genWeightsQOptional`、`genWeightsKOptional`、`catIndicesOptional`四个参数全部传入，且gen_len为正数**。传入nullptr对应的退化路径（纯prefill、单序列恒等映射）暂不支持，调用第二段接口时会返回失败。后续版本放开。
  - **输入值域约束**（属运行期数据，Host侧无法校验，由调用方保证）
    - `slotMapping`取值必须落在`[0, Bn*Bs-1]`且互不重复。出现重复槽位时多个核会写同一cache行，写入顺序与最终结果均不确定。
    - `positions`取值必须落在`[0, max_pos-1]`，否则会越界访问`cosSinCache`。
    - `catIndicesOptional`取值必须落在`[0, T-1]`。

<!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
/*!
 * \file test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief aclnnUndGenQkvRmsNormRopeCache 两段式接口调用样例（含 slot_mapping 预计算）
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_und_gen_qkv_rms_norm_rope_cache.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t NUM_HEADS_Q = 8;
constexpr int64_t NUM_HEADS_K = 1;
constexpr int64_t NUM_HEADS_V = 1;
constexpr int64_t UND_LEN = 5;
constexpr int64_t GEN_LEN = 3;
constexpr int64_t MAX_POS = 32;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;

// bf16 用 uint16_t 承载：取 float32 的高 16 位，就近舍入
uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(rounded >> 16);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, aclFormat format = ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}
} // namespace

int main()
{
    // 1. 初始化设备与流
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t total = UND_LEN + GEN_LEN;
    const int64_t numHead = NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V;
    const int64_t blockNum = (total + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    // 2. 构造输入 shape
    std::vector<int64_t> undQkvShape = {UND_LEN, numHead, HEAD_DIM};
    std::vector<int64_t> genQkvShape = {GEN_LEN, numHead, HEAD_DIM};
    std::vector<int64_t> weightShape = {HEAD_DIM};
    std::vector<int64_t> cosSinShape = {MAX_POS, HEAD_DIM};
    std::vector<int64_t> kCacheShape = {blockNum, BLOCK_SIZE, NUM_HEADS_K, HEAD_DIM};
    std::vector<int64_t> vCacheShape = {blockNum, BLOCK_SIZE, NUM_HEADS_V, HEAD_DIM};
    std::vector<int64_t> slotMappingShape = {total};
    std::vector<int64_t> positionsShape = {MROPE_AXIS_NUM, total};
    std::vector<int64_t> catIndicesShape = {total};
    std::vector<int64_t> qShape = {total, NUM_HEADS_Q, HEAD_DIM};

    // 3. 造 host 数据（bf16 用 uint16_t 承载）
    // QKV 填一段随位置缓慢变化的非零数据，避免全零输入让样例退化成恒等于 0 的空跑
    std::vector<uint16_t> undQkvHost(GetShapeSize(undQkvShape));
    for (size_t i = 0; i < undQkvHost.size(); ++i) {
        undQkvHost[i] = FloatToBf16(0.05f * static_cast<float>(i % 17) - 0.4f);
    }
    std::vector<uint16_t> genQkvHost(GetShapeSize(genQkvShape));
    for (size_t i = 0; i < genQkvHost.size(); ++i) {
        genQkvHost[i] = FloatToBf16(0.05f * static_cast<float>(i % 13) - 0.3f);
    }
    // RMSNorm 权重：und 段取 1.0，gen 段取 0.5，便于区分两段权重是否按 catIndicesOptional 正确选中
    std::vector<uint16_t> undWeightsQHost(HEAD_DIM, FloatToBf16(1.0f));
    std::vector<uint16_t> undWeightsKHost(HEAD_DIM, FloatToBf16(1.0f));
    std::vector<uint16_t> genWeightsQHost(HEAD_DIM, FloatToBf16(0.5f));
    std::vector<uint16_t> genWeightsKHost(HEAD_DIM, FloatToBf16(0.5f));
    // cos/sin 缓存表：前 D/2 列为 cos、后 D/2 列为 sin，按标准 RoPE 的 theta 生成
    std::vector<float> cosSinHost(GetShapeSize(cosSinShape));
    for (int64_t pos = 0; pos < MAX_POS; ++pos) {
        for (int64_t i = 0; i < HEAD_DIM / 2; ++i) {
            float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(HEAD_DIM));
            float angle = static_cast<float>(pos) * freq;
            cosSinHost[pos * HEAD_DIM + i] = std::cos(angle);
            cosSinHost[pos * HEAD_DIM + HEAD_DIM / 2 + i] = std::sin(angle);
        }
    }
    // KV Cache 由调用方预分配并原地更新，未被 slot_mapping 命中的行保持传入值
    std::vector<uint16_t> kCacheHost(GetShapeSize(kCacheShape), 0);
    std::vector<uint16_t> vCacheHost(GetShapeSize(vCacheShape), 0);
    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0);

    // slot_mapping 由调用方预计算：slot = blockIdx * blockSize + rowIdx
    std::vector<int64_t> slotMappingHost(total);
    for (int64_t t = 0; t < total; ++t) {
        int64_t blockIdx = t / BLOCK_SIZE;
        int64_t rowIdx = t % BLOCK_SIZE;
        slotMappingHost[t] = blockIdx * BLOCK_SIZE + rowIdx;
    }
    // positions：三轴位置
    std::vector<int64_t> positionsHost(MROPE_AXIS_NUM * total);
    for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
        for (int64_t t = 0; t < total; ++t) {
            positionsHost[axis * total + t] = t % MAX_POS;
        }
    }
    // cat_indices：und/gen 交错，out_t -> src_t
    std::vector<int64_t> catIndicesHost = {0, 5, 1, 6, 2, 7, 3, 4};

    // 4. 创建 aclTensor
    void *undQkvDev = nullptr, *genQkvDev = nullptr, *undWqDev = nullptr, *undWkDev = nullptr;
    void *genWqDev = nullptr, *genWkDev = nullptr, *cosSinDev = nullptr, *kCacheDev = nullptr;
    void *vCacheDev = nullptr, *slotMappingDev = nullptr, *positionsDev = nullptr, *catIndicesDev = nullptr;
    void* qDev = nullptr;
    aclTensor *undQkv = nullptr, *genQkvOptional = nullptr, *undWq = nullptr, *undWk = nullptr, *genWq = nullptr;
    aclTensor *genWk = nullptr, *cosSin = nullptr, *kCacheRef = nullptr, *vCacheRef = nullptr, *slotMapping = nullptr;
    aclTensor *positions = nullptr, *catIndicesOptional = nullptr, *qOut = nullptr;

    ret = CreateAclTensor(undQkvHost, undQkvShape, &undQkvDev, ACL_BF16, &undQkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsQHost, weightShape, &undWqDev, ACL_BF16, &undWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsKHost, weightShape, &undWkDev, ACL_BF16, &undWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosSinHost, cosSinShape, &cosSinDev, ACL_FLOAT, &cosSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHost, kCacheShape, &kCacheDev, ACL_BF16, &kCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vCacheHost, vCacheShape, &vCacheDev, ACL_BF16, &vCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(slotMappingHost, slotMappingShape, &slotMappingDev, ACL_INT64, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(positionsHost, positionsShape, &positionsDev, ACL_INT64, &positions);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genQkvHost, genQkvShape, &genQkvDev, ACL_BF16, &genQkvOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genWeightsQHost, weightShape, &genWqDev, ACL_BF16, &genWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genWeightsKHost, weightShape, &genWkDev, ACL_BF16, &genWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(catIndicesHost, catIndicesShape, &catIndicesDev, ACL_INT64, &catIndicesOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qHost, qShape, &qDev, ACL_BF16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // mrope_section：三轴分段
    std::vector<int64_t> mropeSectionData = {16, 16, 16};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionData.data(), mropeSectionData.size());

    // 5. 第一段接口：计算 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
        undQkv, undWq, undWk, cosSin, kCacheRef, vCacheRef, slotMapping, positions, genQkvOptional, genWq, genWk, catIndicesOptional,
        NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V, NORM_EPS, mropeSection, qOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 6. 第二段接口：执行计算
    ret = aclnnUndGenQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUndGenQkvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 拷回结果：qOut 直接读取；k_cache/v_cache 按 slot_mapping 定位对应行
    auto qSize = GetShapeSize(qShape);
    std::vector<uint16_t> qResult(qSize, 0);
    ret = aclrtMemcpy(qResult.data(), qResult.size() * sizeof(qResult[0]), qDev, qSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy qOut result failed. ERROR: %d\n", ret); return ret);

    auto kCacheSize = GetShapeSize(kCacheShape);
    std::vector<uint16_t> kCacheResult(kCacheSize, 0);
    ret = aclrtMemcpy(kCacheResult.data(), kCacheResult.size() * sizeof(kCacheResult[0]), kCacheDev,
                      kCacheSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy k_cache result failed. ERROR: %d\n", ret); return ret);

    // 打印前若干个结果，k_cache 按 slot_mapping[0] 定位到被写入的那一行
    constexpr int64_t PRINT_NUM = 8;
    for (int64_t i = 0; i < PRINT_NUM; ++i) {
        LOG_PRINT("qOut[%ld] = %f\n", i, Bf16ToFloat(qResult[i]));
    }
    int64_t kCacheRowOffset = slotMappingHost[0] * NUM_HEADS_K * HEAD_DIM;
    for (int64_t i = 0; i < PRINT_NUM; ++i) {
        LOG_PRINT("kCache[slot %ld][%ld] = %f\n", slotMappingHost[0], i,
                  Bf16ToFloat(kCacheResult[kCacheRowOffset + i]));
    }
    LOG_PRINT("run aclnnUndGenQkvRmsNormRopeCache success, qOut size = %ld, k_cache size = %ld\n", qSize, kCacheSize);

    // 8. 释放资源
    aclDestroyTensor(undQkv);
    aclDestroyTensor(undWq);
    aclDestroyTensor(undWk);
    aclDestroyTensor(cosSin);
    aclDestroyTensor(kCacheRef);
    aclDestroyTensor(vCacheRef);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(positions);
    aclDestroyTensor(genQkvOptional);
    aclDestroyTensor(genWq);
    aclDestroyTensor(genWk);
    aclDestroyTensor(catIndicesOptional);
    aclDestroyTensor(qOut);
    aclDestroyIntArray(mropeSection);

    aclrtFree(undQkvDev);
    aclrtFree(undWqDev);
    aclrtFree(undWkDev);
    aclrtFree(cosSinDev);
    aclrtFree(kCacheDev);
    aclrtFree(vCacheDev);
    aclrtFree(slotMappingDev);
    aclrtFree(positionsDev);
    aclrtFree(genQkvDev);
    aclrtFree(genWqDev);
    aclrtFree(genWkDev);
    aclrtFree(catIndicesDev);
    aclrtFree(qDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
