# aclnnMsaIndexScoreV2

[📄 查看源码](https://atomgit.com/cann/ops-transformer/tree/master/attention/msa_index_score)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **算子功能**：计算 MSA（MiniMax Sparse Attention）Index Branch 的 KV 块分数。通过 `returnChoice` 选择输出完整分数、TopK 块号，或 TopK 块号及对应分数。Prefill 和 Decode 使用同一接口。

- **计算公式**：

    $$
    S=\operatorname{MaxPool}_{\mathrm{block}}
    \left(\widehat Q_{idx}\widehat K_{idx}^{T}+\mathrm{atten\_mask}\right)
    $$

    $S$ 是每个 query head 的块内最大分数。`sparseMode=0` 时不加因果掩码，取 3 时按右下对齐的因果掩码将不可见位置置为 $-\infty$。同型浮点输入直接做点积，MXFP4 的 query 和 key 用各自的 E8M0 系数逐 32 元素组缩放。

    - **块长 128**：按 head 输出 $S$。由 `initBlocks` 标出的前缀块写入 $10^{30}$，由 `startLoc` 和 `localBlocks` 标出的局部块写入 $10^{29}$，两者重叠时取局部块的值。
    - **块长 16**：在块内 max 后再取 head 间 max，$score=D^{-1/2}\log_2(e)\cdot\max_h S_h$，输出 head 维为 1，dtype 为 FLOAT16。前缀块和局部块保留原分数。
    - **压缩 key**：每个逻辑块仍覆盖 `poolBlockSize` 个原始 token，最多对 `poolBlockSize / compressFactor` 行 key 取 max。例如块长 16、压缩倍数 4 时，每块最多归约 4 行。有效 KV 长度仍按原始 token 计；在因果模式下，以每行覆盖的第一个 token 判断该行是否可见。
    - **融合 TopK 和输出顺序**：`returnChoice=0` 输出完整 score，取 1 时输出 `sparseIndices`，取 2 时一并输出 `sparseValues`。TopK 按可见块的分数选取，同分优先取较小块号。

      块长 128 每个 head 最多选 `topk` 个块，按块号升序写出。块长 16 在跨 head 归约后最多选 128 个块。可见块不超过 128 时按块号升序全部写出，否则优先保留可见前缀和局部块，并将最大入选块号移至最后一个有效槽。

      TopK 不足的槽位填块号 `-1`。对应分数与 score 的对齐补位使用相同的填充值，块长 128 为 FLOAT 最小有限值，块长 16 为 $-\infty$。

- **参数介绍**：

    > - B（Batch Size）表示输入样本批量大小
    > - S（Sequence Length）表示序列长度，$S1$为query侧、$S2$为key侧
    > - T表示所有Batch序列长度累加和，$T1$为query侧、$T2$为key侧
    > - N（Head Num）表示头数，$N1$为query侧、$N2$为key侧
    > - D（Head Dim）表示单个注意力头维度
    > - PageAttention 中 $block\_num$ 为物理页数，$block\_size$ 为每个物理页所存储的 key 行数，$MB$ 为每条请求的页表宽度。压缩时 key 的 $S2$、$T2$ 和物理页长按存储行计，传入的 KV 有效长度按原始 token 计。

## 函数原型

每个算子分为[两段式接口](https://atomgit.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md)，必须先调用`aclnnMsaIndexScoreV2GetWorkspaceSize`接口获取入参并计算所需workspace大小，再调用`aclnnMsaIndexScoreV2`接口执行计算。

ACLNN 为 C 接口，调用时须为原型中的每个形参提供实参。下文允许省略的张量须显式传入 `nullptr`，布局和数值属性须显式传入支持的取值。

```cpp
aclnnStatus aclnnMsaIndexScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *startLoc,
    const aclTensor *blockTableOptional,
    const aclTensor *queryScaleOptional,
    const aclTensor *keyScaleOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKlenOptional,
    char            *layoutKeyOptional,
    int64_t          sparseMode,
    int64_t          initBlocks,
    int64_t          localBlocks,
    char            *layoutQueryOptional,
    int64_t          poolBlockSize,
    int64_t          returnChoice,
    int64_t          topk,
    int64_t          compressFactor,
    const aclTensor *score,
    const aclTensor *sparseIndices,
    const aclTensor *sparseValues,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
aclnnStatus aclnnMsaIndexScoreV2(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMsaIndexScoreV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1700px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 420px">
  <col style="width: 280px">
  <col style="width: 100px">
  <col style="width: 220px">
  <col style="width: 100px">
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的Q_idx。</td>
      <td>必选，禁止传入nullptr。布局由layoutQueryOptional指定。</td>
      <td>Atlas A2/A3 支持 FLOAT16、BFLOAT16 和 A8C8（INT8）。Ascend 950PR&amp;950DT 支持 FLOAT16、BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN 和 MXFP4（FLOAT4_E2M1，使用 E8M0 缩放系数）。</td>
      <td>ND</td>
      <td>3（[T1, N1, D]）或4（[B, S1, N1, D]、[B, N1, S1, D]）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的K_idx。</td>
      <td>必选，禁止传入nullptr。布局由layoutKeyOptional指定。非分页须连续，分页仅允许物理页轴非连续。</td>
      <td>与 query 同型</td>
      <td>ND</td>
      <td>3（[T2, N2, D]）或4（[B, S2, N2, D]、[B, N2, S2, D]、[block_num, block_size, N2, D]、[block_num, N2, block_size, D]）</td>
      <td>仅支持首轴非连续</td>
    </tr>
    <tr>
      <td>startLoc</td>
      <td>输入</td>
      <td>当前query所在逻辑打分块的索引。例如token位置129、块长128时，传入块索引1。</td>
      <td>必选，禁止传入nullptr。块长 128 时用于定位局部窗口，块长 16 的局部窗口由各 token 的可见长度确定。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（[B]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示当前传入的使用PageAttention存储的block映射表。</td>
      <td>BBND / BNBD 必选，禁止传入nullptr。第二维须覆盖实际可见的物理页数。TND / BSND / BNSD 传nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2（[B, MB]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>queryScaleOptional</td>
      <td>输入</td>
      <td>Q 的 A8C8 逐 token、逐 head 反量化系数或 MXFP4 逐组缩放系数。</td>
      <td>A8C8 和 MXFP4×MXFP4 时必选，禁止传入nullptr。其他场景传nullptr。</td>
      <td>A8C8 为 FLOAT，MXFP4 为 FLOAT8_E8M0</td>
      <td>ND</td>
      <td>A8C8 为 [Tq,Hq]；MXFP4 与 query 同布局，末维为 D/32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keyScaleOptional</td>
      <td>输入</td>
      <td>K 的 A8C8 逐 token 反量化系数或 MXFP4 逐组缩放系数。</td>
      <td>A8C8 和 MXFP4 key 时必选，禁止传入nullptr。其他场景传nullptr。</td>
      <td>A8C8 为 FLOAT，MXFP4 为 FLOAT8_E8M0</td>
      <td>ND</td>
      <td>A8C8 分页为 [NP,1,128]，TND 为 [Tk,1] 或 [Tk]；MXFP4 与 key 同布局，末维为 D/32。</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>控制因果可见的mask掩码。</td>
      <td>sparseMode=3 时必选，禁止传入nullptr。模板中 1 表示屏蔽、0 表示可见。sparseMode=0 时传nullptr。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2（[2048, 2048]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>必选，禁止传入nullptr。传入从 0 开始、单调不减的前缀和。BSND / BNSD 时每个请求的长度不能超过 S1。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（[B+1]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKlenOptional</td>
      <td>输入</td>
      <td>各请求压缩前的有效 KV token 数。</td>
      <td>必选，禁止传入nullptr。TND / BSND / BNSD 传从 0 开始、单调不减的前缀和。分页传各请求的有效长度。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（[B]或[B+1]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>输入</td>
      <td>key的数据排布。</td>
      <td>只支持"TND"、"BSND"、"BNSD"、"BBND"、"BNBD"。不传或空串时默认"BBND"。取值须与 key 实际排布一致，布局组合见约束说明。</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>只支持 0 或 3。0 表示不做因果截断，3 表示右下对齐的因果掩码。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>initBlocks</td>
      <td>输入</td>
      <td>local_mask强制选中的头部block数。</td>
      <td>支持 0 到逻辑块上限的整数，0 表示不强制保留前缀块。块长 128 时用于写入高分，块长 16 时用于 TopK 保留。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>localBlocks</td>
      <td>输入</td>
      <td>local_mask强制选中的局部窗口长度。</td>
      <td>支持 0 到逻辑块上限的整数，0 表示不强制保留局部块。块长 128 时窗口右端为startLoc，块长 16 时为当前 token 的最后一个可见块。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional</td>
      <td>输入</td>
      <td>query的数据排布。</td>
      <td>只支持"TND"、"BSND"、"BNSD"。不传或空串时默认"TND"。取值须与 query 实际排布一致，布局组合见约束说明。</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>poolBlockSize</td>
      <td>输入</td>
      <td>打分池化块长。</td>
      <td>支持 16 或 128，非正数按 128 处理，单位为原始 token。分页时，每物理页的 key 行数须是 poolBlockSize / compressFactor 的正整数倍。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnChoice</td>
      <td>输入</td>
      <td>输出模式。</td>
      <td>各平台支持的输入类型均支持 0、1、2。取 0 时输出完整 score，取 1 时输出 TopK 块号，取 2 时输出 TopK 块号及对应分数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>每行的 TopK 输出槽数。</td>
      <td>returnChoice=1 或 2 时，块长 128 支持 1 到 128 的整数，块长 16 只支持 128。returnChoice=0 时不参与计算，仍须传入实参。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressFactor</td>
      <td>输入</td>
      <td>key的压缩倍数。</td>
      <td>Ascend 950 且poolBlockSize=16时只支持 1、2、4、8、16，其余场景只支持 1。1 表示未压缩。有效 KV 长度仍按原始 token 计，物理页长按 key 行计。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>score</td>
      <td>输出</td>
      <td>公式中的score。</td>
      <td>returnChoice=0 时必选，禁止传入nullptr。returnChoice=1 或 2 时传nullptr。末维按 16 对齐。</td>
      <td>块长 128 为 FLOAT，块长 16 为 FLOAT16</td>
      <td>ND</td>
      <td>3（块长16为[1, T1, RoundUp(maxBlockNumPerSeq, 16)]，块长128为[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>输出</td>
      <td>TopK 的逻辑块号。</td>
      <td>returnChoice=1 或 2 时必选，禁止传入nullptr。returnChoice=0 时传nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>3（块长 128 为 [N1,T1,topk]，块长 16 为 [1,T1,128]）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseValues</td>
      <td>输出</td>
      <td>与sparseIndices同序的块分数。</td>
      <td>returnChoice=2 时必选，禁止传入nullptr。returnChoice=0 或 1 时传nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>与sparseIndices相同</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。必选，禁止传入nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回 op 执行器，包含算子计算流程。必选，禁止传入nullptr。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_SUCCESS</td>
      <td>0</td>
      <td>执行成功。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选的指针入参或出参传入了nullptr。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>数据类型、数据格式、维度或取值不满足约束。</td>
    </tr>
  </tbody>
  </table>

## aclnnMsaIndexScoreV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在 Device 侧申请的 workspace 内存地址。workspaceSize&gt;0时禁止传入nullptr。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMsaIndexScoreV2GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>第一段接口返回的 op 执行器。必选，禁止传入nullptr。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。必选，禁止传入nullptr。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_SUCCESS</td>
      <td>0</td>
      <td>执行成功。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>参数不合法。</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- `poolBlockSize` 只支持 16 或 128。块长 16 要求 $N1$ 整除 128。
- TND key 只支持 TND query。BSND / BNSD key 在块长 128 时只支持 TND query，在块长 16 时只支持同布局的 query。BBND / BNBD key 支持平台允许的三种 query 布局。
- BSND / BNSD 布局下，每条请求的 query 长度和按 $\lceil n_k/c\rceil$ 换算的 key 行数须分别在对应张量的容量内。允许单条请求或整批的有效长度为 0。没有可见 KV 时按计算公式填充 score 或 TopK 输出。
- 同型浮点输入的两个缩放系数均传 `nullptr`。A8C8 通过本 V2 接口调用，Qscale 和 Kscale 均为 FP32，Kscale 连续存储。FLOAT16 query 配 INT8 key 为预留组合。MXFP4×MXFP4 仅支持 $N2=1$、$D=64$ 或 128，两个 E8M0 缩放系数均必选，禁止传入 `nullptr`。
- 分页 key 的物理页轴允许非连续，其余长度大于 1 的轴须为紧凑步长。分页 key 的缩放系数须按物理页紧凑存储。非分页 key 须连续。
- 分页时，`blockTableOptional` 的第二维 $MB$ 可以大于实际使用的页数。每个物理页包含 $block\_size/(L/c)$ 个逻辑块，按页表容量分配时，score 末维为 $\mathrm{RoundUp}(MB\times block\_size/(L/c),16)$。非分页按各请求逻辑块数的最大值分配，并向上对齐到 16。整批无 KV 时，仍分配至少 16 列。
- `initBlocks` 和 `localBlocks` 须为非负整数，且各自不超过逻辑块上限。块长 16 的 TopK 模式下，如果任何 token 可能看到超过 128 个逻辑块，两者之和不得超过 128。

<!-- npu="A3, 910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - query 和 key 支持 BFLOAT16、FLOAT16，以及同型 INT8 的 A8C8。
    - `layoutQueryOptional`：仅支持`"TND"`。
    - `poolBlockSize` 只支持 128，BBND / BNBD 的物理页长只支持 128 行。`compressFactor` 只支持 1，`returnChoice` 支持 0、1、2。
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - 扩展支持同型 FLOAT8_E5M2、FLOAT8_E4M3FN，以及配有 E8M0 缩放系数的 MXFP4×MXFP4（FLOAT4_E2M1）。
    - 块长 16 只支持 FLOAT16、BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN 和 MXFP4×MXFP4。BBND / BNBD 支持 16 行或 128 行物理页，且压缩时另支持 $128/c$ 行。
    - 块长 128 的 TopK 模式支持 FLOAT16、BFLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN 和 MXFP4×MXFP4。
<!-- end id9 -->

## 调用示例

以下示例使用 TND query、BBND key、块长 128，输出完整 score。其他布局的 shape 见示例末尾。含 TND、BNBD 的验证代码见 [test_aclnn_msa_index_score_v2.cpp](../examples/test_aclnn_msa_index_score_v2.cpp)。
```cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_msa_index_score_v2.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)         \
  do {                                  \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int64_t RoundUp(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
  void* queryDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* blockTableDeviceAddr = nullptr;
  void* attenMaskDeviceAddr = nullptr;
  void* actualSeqQlenDeviceAddr = nullptr;
  void* actualSeqKlenDeviceAddr = nullptr;
  void* startLocDeviceAddr = nullptr;
  void* scoreDeviceAddr = nullptr;

  aclTensor* queryTensor = nullptr;
  aclTensor* keyTensor = nullptr;
  aclTensor* blockTableTensor = nullptr;
  aclTensor* attenMaskTensor = nullptr;
  aclTensor* actualSeqQlenTensor = nullptr;
  aclTensor* actualSeqKlenTensor = nullptr;
  aclTensor* startLocTensor = nullptr;
  aclTensor* scoreTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
  // TND query + PA_BBND key，sparseMode=3
  constexpr int64_t B = 1;
  constexpr int64_t T1 = 2;
  constexpr int64_t N1 = 2;
  constexpr int64_t N2 = 1;
  constexpr int64_t D = 128;
  constexpr int64_t S2 = 256;
  constexpr int64_t blockSize = 128;
  constexpr int64_t poolBlockSize = 128;
  constexpr int64_t blockNum = 2;
  constexpr int64_t maxBlockNumPerSeq = S2 / poolBlockSize;  // 2
  const int64_t scoreStride = RoundUp(maxBlockNumPerSeq, 16);

  std::vector<int64_t> queryShape = {T1, N1, D};
  std::vector<int64_t> keyShape = {blockNum, blockSize, N2, D};            // BBND
  std::vector<int64_t> blockTableShape = {B, maxBlockNumPerSeq};
  std::vector<int64_t> attenMaskShape = {2048, 2048};
  std::vector<int64_t> actualSeqQlenShape = {B + 1};
  std::vector<int64_t> actualSeqKlenShape = {B};
  std::vector<int64_t> startLocShape = {B};
  std::vector<int64_t> scoreShape = {N1, T1, scoreStride};

  std::vector<uint16_t> queryHostData(GetShapeSize(queryShape), 0x3C00);  // fp16 1.0
  std::vector<uint16_t> keyHostData(GetShapeSize(keyShape), 0x3C00);
  std::vector<int32_t> blockTableHostData = {0, 1};
  std::vector<int8_t> attenMaskHostData(GetShapeSize(attenMaskShape), 0);  // 0: 参与计算
  std::vector<int32_t> actualSeqQlenHostData = {0, static_cast<int32_t>(T1)};
  std::vector<int32_t> actualSeqKlenHostData = {static_cast<int32_t>(S2)};
  std::vector<int32_t> startLocHostData = {1};  // 当前 query 所在逻辑 block 索引
  std::vector<float> scoreHostData(GetShapeSize(scoreShape), 0.0f);

  int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                            aclDataType::ACL_FLOAT16, &resources.queryTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                        aclDataType::ACL_FLOAT16, &resources.keyTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr,
                        aclDataType::ACL_INT32, &resources.blockTableTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &resources.attenMaskDeviceAddr,
                        aclDataType::ACL_INT8, &resources.attenMaskTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(actualSeqQlenHostData, actualSeqQlenShape,
                        &resources.actualSeqQlenDeviceAddr, aclDataType::ACL_INT32,
                        &resources.actualSeqQlenTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(actualSeqKlenHostData, actualSeqKlenShape,
                        &resources.actualSeqKlenDeviceAddr, aclDataType::ACL_INT32,
                        &resources.actualSeqKlenTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(startLocHostData, startLocShape, &resources.startLocDeviceAddr,
                        aclDataType::ACL_INT32, &resources.startLocTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(scoreHostData, scoreShape, &resources.scoreDeviceAddr,
                        aclDataType::ACL_FLOAT, &resources.scoreTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  return ACL_SUCCESS;
}

int ExecuteMsaIndexScore(TensorResources& resources, aclrtStream stream,
                         void** workspaceAddr, uint64_t* workspaceSize) {
  int64_t sparseMode = 3;
  int64_t initBlocks = 0;
  int64_t localBlocks = 1;
  char layoutKey[] = "BBND";
  char layoutQuery[] = "TND";
  int64_t poolBlockSize = 128;  // 打分池化块长
  aclOpExecutor* executor = nullptr;

  // 非量化：keyScaleOptional、queryScaleOptional 传 nullptr；compressFactor 传 1
  int ret = aclnnMsaIndexScoreV2GetWorkspaceSize(
      resources.queryTensor,
      resources.keyTensor,
      resources.startLocTensor,
      resources.blockTableTensor,
      nullptr,  // queryScaleOptional
      nullptr,  // keyScaleOptional
      resources.attenMaskTensor,
      resources.actualSeqQlenTensor,
      resources.actualSeqKlenTensor,
      layoutKey,
      sparseMode,
      initBlocks,
      localBlocks,
      layoutQuery,
      poolBlockSize,
      0,  // returnChoice: score
      16, // topk ignored in score mode
      1,  // compressFactor
      resources.scoreTensor,
      nullptr, // sparseIndices
      nullptr, // sparseValues
      workspaceSize,
      &executor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnMsaIndexScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret;
  }

  if (*workspaceSize > 0ULL) {
    ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return ret;
    }
  }

  ret = aclnnMsaIndexScoreV2(*workspaceAddr, *workspaceSize, executor, stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnMsaIndexScoreV2 failed. ERROR: %d\n", ret);
    return ret;
  }
  return ACL_SUCCESS;
}

int PrintScoreOutResult(const std::vector<int64_t>& shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0.0f);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("score result[%ld] is: %f\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                      aclrtStream stream, int32_t deviceId) {
  if (resources.queryTensor) {
    aclDestroyTensor(resources.queryTensor);
  }
  if (resources.keyTensor) {
    aclDestroyTensor(resources.keyTensor);
  }
  if (resources.blockTableTensor) {
    aclDestroyTensor(resources.blockTableTensor);
  }
  if (resources.attenMaskTensor) {
    aclDestroyTensor(resources.attenMaskTensor);
  }
  if (resources.actualSeqQlenTensor) {
    aclDestroyTensor(resources.actualSeqQlenTensor);
  }
  if (resources.actualSeqKlenTensor) {
    aclDestroyTensor(resources.actualSeqKlenTensor);
  }
  if (resources.startLocTensor) {
    aclDestroyTensor(resources.startLocTensor);
  }
  if (resources.scoreTensor) {
    aclDestroyTensor(resources.scoreTensor);
  }

  if (resources.queryDeviceAddr) {
    aclrtFree(resources.queryDeviceAddr);
  }
  if (resources.keyDeviceAddr) {
    aclrtFree(resources.keyDeviceAddr);
  }
  if (resources.blockTableDeviceAddr) {
    aclrtFree(resources.blockTableDeviceAddr);
  }
  if (resources.attenMaskDeviceAddr) {
    aclrtFree(resources.attenMaskDeviceAddr);
  }
  if (resources.actualSeqQlenDeviceAddr) {
    aclrtFree(resources.actualSeqQlenDeviceAddr);
  }
  if (resources.actualSeqKlenDeviceAddr) {
    aclrtFree(resources.actualSeqKlenDeviceAddr);
  }
  if (resources.startLocDeviceAddr) {
    aclrtFree(resources.startLocDeviceAddr);
  }
  if (resources.scoreDeviceAddr) {
    aclrtFree(resources.scoreDeviceAddr);
  }

  if (workspaceAddr) {
    aclrtFree(workspaceAddr);
  }
  if (stream) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
}

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  TensorResources resources = {};
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  constexpr int64_t T1 = 2;
  constexpr int64_t N1 = 2;
  constexpr int64_t maxBlockNumPerSeq = 2;
  std::vector<int64_t> scoreShape = {N1, T1, RoundUp(maxBlockNumPerSeq, 16)};
  int ret = ACL_SUCCESS;

  ret = Init(deviceId, &stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = InitializeTensors(resources);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  ret = ExecuteMsaIndexScore(resources, stream, &workspaceAddr, &workspaceSize);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("ExecuteMsaIndexScore failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  ret = aclrtSynchronizeStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  PrintScoreOutResult(scoreShape, &resources.scoreDeviceAddr);

  CleanupResources(resources, workspaceAddr, stream, deviceId);
  return 0;
}
```

TND/BNBD与上面BBND示例的差异如下：
```cpp
// BNBD PageAttention：layoutKey="BNBD"，key 为 [block_num, N2, page_size, D]
char layoutKeyBnbd[] = "BNBD";
std::vector<int64_t> keyShapeBnbd = {blockNum, N2, blockSize, D};

// TND packed key：layoutKey="TND"，blockTableOptional 传 nullptr，actualSeqKlenOptional 为 [B+1] 前缀和
char layoutKeyTnd[] = "TND";
constexpr int64_t T2 = 256;
std::vector<int64_t> keyShapeTnd = {T2, N2, D};
std::vector<int64_t> actualSeqKlenShapeTnd = {B + 1};  // 例如 {0, T2}
aclTensor* blockTableTnd = nullptr;

// BBND 页长 16 示例（仅 Ascend 950）：key [block_num, 16, N2, D]，blockTableOptional 禁止传入 nullptr。
// 先块内 max，再 head 间 max。score 为 [1, T1, RoundUp(页表宽度 × 每页打分块数, 16)]。query head 数须整除 128。
char layoutKeyBlk16[] = "BBND";
std::vector<int64_t> keyShapeBlk16 = {blockNum, 16, N2, D};

// query BSND + key BBND（仅 Ascend 950）：layoutQuery="BSND"，query [B, S1, N1, D]，key 仍是分页。
char layoutQueryBsnd[] = "BSND";
std::vector<int64_t> queryShapeBsnd = {B, S1, N1, D};

// query 与 key 同为 BNSD（仅 Ascend 950，逻辑块长 16）：blockTableOptional 传 nullptr。
char layoutQueryBnsd[] = "BNSD";
char layoutKeyBnsd[] = "BNSD";
std::vector<int64_t> queryShapeBnsd = {B, N1, S1, D};
std::vector<int64_t> keyShapeBnsd = {B, N2, S2, D};
```

调用融合 TopK 时，显式传入 `returnChoice=1` 或 2 及所需的 `topk`（块长 16 时传 128），并将 `score` 设为 `nullptr`。按参数表创建 `sparseIndices`，在 `returnChoice=2` 时同时创建 `sparseValues`。压缩 key 时还须显式传入对应的 `compressFactor`，例如 4。
