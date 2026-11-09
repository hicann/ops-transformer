# aclnnSparseLightningIndexer

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- 接口功能：aclnnSparseLightningIndexer 实现 candidate consumer 语义：在 LightningIndexerV2 加权 ReLU 打分框架上消费上游 source 算子（aclnnLightningIndexerV3 开启 candidateTopkBlocks 时输出的 candidateBlockIndices）的候选块索引，对每个 query 行做无 leak 候选内 TopK：可达且命中候选块的位置以真实分数参与排序，其余位置（不可达 / 可达但不在候选内）一律无效不入选；非候选块索引一律不进入 topk 输出，候选内有效数不足 topk 时剩余槽位填 -1，输出每个 query 行的 top-k 索引与（returnValue为1时的）TopK 分数。算子仅注册 ascend910b / ascend910_93 平台（arch22），需与上游 aclnnLightningIndexerV3 的 candidate source 模式配套使用。

- 计算公式（每 batch b、query 行 i，N2=1）：

  $$
  \begin{aligned}
  score(b,i,:) &= \mathrm{DoReduce}(W \odot ReLU(Q_{index} @ K_{index}^{T})) \in \R^{S_2} \\
  score'(b,i,p) &= \begin{cases}
      score(b,i,p) & p < vl(b,i) \text{ 且 } \lfloor p / blockSize \rfloor \in candSet(b,i) \\
      -\infty      & \text{其它（} p \geq vl(b,i) \text{，或可达但不在候选内）}
  \end{cases} \\
  nvalid(b,i) &= \left|\{\, p : score'(b,i,p) > -\infty \,\}\right| \\
  sparseIndices(b,i,k) &= \begin{cases}
      \mathrm{topK}(score'(b,i,:))_k & k < \min(topk,\ nvalid(b,i)) \\
      -1 & \min(topk,\ nvalid(b,i)) \le k < topk
  \end{cases} \\
  sparseValues(b,i,k) &= \begin{cases}
      score(b,i,\ sparseIndices(b,i,k)) & k < \min(topk,\ nvalid(b,i)),\ returnValue=1 \\
      \text{恒空 tensor } (0,) & returnValue=0
  \end{cases}
  \end{aligned}
  $$

  符号说明如下：

  - $vl(b,i)$：行可达前缀长度，由因果mask、sequsedKOptional/cuSeqlensKOptional与cmpRatio（含cmpResidualKOptional余数）推算。
  - $blockSize$：属性candidateBlockSize。
  - $candSet(b,i)$：候选块号集合（来自candidateBlockIndices，重复/越界/-1按"非候选"处理）。
  - $nvalid(b,i)$：候选内有效位置数。
  - $\mathrm{DoReduce}$：沿g（多头）方向的加权求和。
  - $W$：多头权重投影矩阵；$\odot$：逐元素乘法；@：矩阵乘法。

- 候选块语义：候选块 j 表示该行可达前缀在压缩后 K 位置空间上的区间 [j×candidateBlockSize, (j+1)×candidateBlockSize)；块号按 batch 内相对编号消费，每行的块总数 numBlocks = ceil(actS2 / candidateBlockSize)，actS2 为由 sequsedKOptional/cuSeqlensKOptional/k shape 与 cmpRatio（含 cmpResidualKOptional 余数）推算的该 batch 压缩后 K 有效长度。

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSparseLightningIndexerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSparseLightningIndexer”接口执行计算。

```cpp
aclnnStatus aclnnSparseLightningIndexerGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    const aclTensor *candidateBlockIndices,
    const aclTensor *candidateBlockLengthOptional,
    int64_t           topk,
    int64_t           maxSeqlenQ,
    char             *layoutQOptional,
    char             *layoutKOptional,
    int64_t           maskMode,
    int64_t           cmpRatio,
    int64_t           returnValue,
    int64_t           candidateBlockSize,
    const aclTensor  *sparseIndicesOut,
    const aclTensor  *sparseValuesOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSparseLightningIndexer(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnSparseLightningIndexerGetWorkspaceSize

- **参数说明：**

  > [!NOTE]
  >
  > - q、k、w参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、N（Head Num）表示多头数、D（Head Dim）表示每个头的维度、T表示所有Batch输入样本序列长度的累加和。
  > - S1表示q shape中的S，S2表示k shape中的S，T表示q/k shape中的T，N1表示q shape中的N，N2表示k shape中的N，N2仅支持1。
  > - block_num为PageAttention时block总数，block_size为一个block的token数，取值[16, 1024]且为16的倍数。
  > - k为按cmpRatio压缩后的Key序列，每个batch压缩后K有效长度由sequsedKOptional/cuSeqlensKOptional/k shape与cmpRatio（含cmpResidualKOptional余数）推算。

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 250px">
  <col style="width: 380px">
  <col style="width: 180px">
  <col style="width: 110px">
  <col style="width: 160px">
  <col style="width: 110px">
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
    </tr></thead>
  <tbody>
    <tr>
    <td>q（const aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入Q（与上游source算子一致）。</td>
    <td><ul><li>不支持空tensor。</li><li>数据类型与k保持一致。</li><li>layout_q为BSND时，shape为(B, S1, N1, D)。</li><li>layout_q为TND时，shape为(T, N1, D)。</li></ul></td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>k（const aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入K（与上游source算子一致）。</td>
    <td><ul><li>不支持空tensor。</li><li>N2必须为1。</li><li>block_num为PageAttention时block总数，block_size为一个block的token数，取值[16, 1024]且为16的倍数。</li><li>layout_k为BSND时，shape为(B, S2, N2, D)。</li><li>layout_k为TND时，shape为(T, N2, D)。</li><li>layout_k为PA_BBND时，shape为(block_num, block_size, N2, D)。</li></ul></td>
    <td>BFLOAT16、FLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√（仅PA_BBND布局支持0轴非连续）</td>
    </tr>
    <tr>
    <td>w（const aclTensor*）</td>
    <td>输入</td>
    <td>公式中的输入W（与上游source算子一致）。</td>
    <td><ul><li>不支持空tensor。</li><li>layout_q为BSND时，shape为(B, S1, N1)。</li><li>layout_q为TND时，shape为(T, N1)。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensQOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中q的有效token数的累加和（前缀和）。</td>
    <td><ul><li>layout_q为TND时必传，第一个元素固定为0。</li><li>layout_q为BSND时不建议传入。</li><li>shape为(B+1,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cuSeqlensKOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>当前Batch及前序Batch中k的有效token数的累加和（前缀和）。</td>
    <td><ul><li>layout_k为TND时必传。</li><li>layout_k为BSND时传入报错。</li><li>layout_k为PA_BBND时不传。</li><li>shape为(B+1,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sequsedQOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>q每batch实际长度（接口对齐保留位）。</td>
    <td><ul><li>本算子（arch22）不支持传入，非空即被host侧校验拒绝。</li><li>传nullptr即可，q长度以shape S1或cuSeqlensQOptional为准。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sequsedKOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>k每batch实际（压缩后）长度。</td>
    <td><ul><li>PA_BBND时必传。</li><li>BSND/TND时可选传入，传入则参与k有效长度计算（actS2Orig = sequsedK×cmpRatio + cmpResidualK）。</li><li>shape为(B,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>cmpResidualKOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>k压缩前token数量除以cmpRatio的余数。</td>
    <td><ul><li>host侧不校验shape/dtype（跨算子约束）。</li><li>与sequsedKOptional/cuSeqlensKOptional组合参与压缩长度计算。</li><li>需与上游source算子传入的cmpResidualKOptional保持一致。</li><li>shape为(B,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>x</td>
    </tr>
    <tr>
    <td>blockTableOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>PageAttention中KV存储使用的block映射表。</td>
    <td><ul><li>PA_BBND时必传，dim0必须等于B。</li><li>非PA时传入报错。</li><li>第二维长度不能小于maxBlockNumPerSeq（每个batch中最大有效序列长度对应的block数量）。</li><li>shape为(B, maxBlockNumPerSeq)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>x</td>
    </tr>
    <tr>
    <td>outputIdxOffsetOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>输出索引偏移（接口对齐保留位）。</td>
    <td><ul><li>本算子（arch22）不支持传入，非空即被host侧校验拒绝。</li><li>传nullptr即可，输出索引不加偏移。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>metadataOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>元数据（接口对齐保留位）。</td>
    <td><ul><li>arch22不消费且非空即被host侧校验拒绝（输入位仅为IR兼容保留），传nullptr。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>candidateBlockIndices（const aclTensor*）</td>
    <td>输入</td>
    <td>候选块索引（必选，source算子candidateBlockIndices输出直连）。</td>
    <td><ul><li>必传，不支持空tensor。</li><li>candBlocks = shape末维，取值(0, 2048]且为64的倍数。</li><li>layout_q为BSND时，shape为(B, S1, N2, candBlocks)。</li><li>layout_q为TND时，shape为(T, N2, candBlocks)。</li><li>值域[0, numBlocks)或-1，槽位无序。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>candidateBlockLengthOptional（const aclTensor*）</td>
    <td>输入</td>
    <td>预留接口。</td>
    <td><ul><li>仅接受nullptr或元素个数为0的空tensor（任意shape，numel==0），非空报错。</li><li>shape为(0,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>topk（int64_t）</td>
    <td>输入</td>
    <td>topK阶段需要保留的Key token索引数量。</td>
    <td>取值(0, 2048]，恒校验（本算子无超2048回退路径）。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maxSeqlenQ（int64_t）</td>
    <td>输入</td>
    <td>q的最大序列长度（接口对齐保留位）。</td>
    <td>仅接受-1，非-1即被host侧校验拒绝（arch22 kernel不消费）。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutQOptional（char*）</td>
    <td>输入</td>
    <td>用于标识输入q的数据排布格式。</td>
    <td><ul><li>支持BSND、TND。</li><li>layout_k非PA_BBND时，layout_q与layout_k必须一致。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>layoutKOptional（char*）</td>
    <td>输入</td>
    <td>用于标识输入k的数据排布格式。</td>
    <td><ul><li>支持BSND、TND、PA_BBND。</li><li>layout_k非PA_BBND时，layout_q与layout_k必须一致。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskMode（int64_t）</td>
    <td>输入</td>
    <td>表示mask的模式。</td>
    <td><ul><li>0表示无mask，3表示rightDownCausal（因果）mask。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>cmpRatio（int64_t）</td>
    <td>输入</td>
    <td>用于稀疏计算，表示k的压缩倍数。</td>
    <td><ul><li>取值(0, 128]且必须为2的幂，非幂值拒绝。</li><li>本算子host强校验2的幂（上游source算子host仅校验(0, 128]），配置时须与source侧保持一致。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>returnValue（int64_t）</td>
    <td>输入</td>
    <td>是否输出TopK分数。</td>
    <td><ul><li>取值∈{0,1}，其它值报错拒绝。</li><li>0：sparseValuesOut恒空(0,)，kernel不写入。</li><li>1：sparseValuesOut输出入选位置真实TopK分数（与sparseIndicesOut同shape，dtype FLOAT32，无效槽value=0）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>candidateBlockSize（int64_t）</td>
    <td>输入</td>
    <td>候选块粒度，即压缩后K空间中每个候选块包含的位置数。</td>
    <td><ul><li>支持[2, 64]内的2的幂（2、4、8、16、32、64）。</li><li>必须与上游source算子的candidateBlockSize保持一致（调用方约束，框架层无法跨算子校验）。</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sparseIndicesOut（const aclTensor*）</td>
    <td>输出</td>
    <td>公式中的Indices输出（无 leak TopK 索引）。</td>
    <td><ul><li>不支持空tensor。</li><li>有效槽为候选内可达位置按分数降序排列（同分次序不承诺）。</li><li>候选内有效数nvalid &lt; topk时，其后槽位全部填-1（含候选外可达位置在内的所有无效槽一律-1，不输出任何候选外索引）；全无效行输出全-1。</li><li>layout_q为BSND时，shape为(B, S1, N2, topk)。</li><li>layout_q为TND时，shape为(T, N2, topk)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>3-4</td>
    <td>x</td>
    </tr>
    <tr>
    <td>sparseValuesOut（const aclTensor*）</td>
    <td>输出</td>
    <td>Values输出（TopK分数，双分支）。</td>
    <td><ul><li>aclnn层仍做非空校验，须传入非空tensor。</li><li>returnValue为0：实际输出shape为(0,)，kernel不写入。</li><li>returnValue为1：shape与sparseIndicesOut完全相同，有效槽为入选位置真实TopK分数（fp32打分域直出），-1槽对应位value=0。</li><li>layout_q为BSND时shape为(B, S1, N2, topk)，layout_q为TND时shape为(T, N2, topk)。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1、3-4</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
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
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><ul><li>q、k、w、candidate_block_indices、sparse_indices、sparse_values的数据类型和数据格式不在支持的范围内。</li><li>shape非法，如：q/k/candidate_block_indices/sparse_indices维度或各维取值不在支持的范围内；candidate_block_indices末维不在(0, 2048]内或不是64的倍数。</li><li>属性取值非法，如：topk不在(0, 2048]内；cmp_ratio不在(0, 128]内或不是2的幂；mask_mode不为0或3；return_value不为0或1；max_seqlen_q不为-1；candidate_block_size不为[2, 64]内的2的幂；candidate_block_length非空tensor；seqused_q/output_idx_offset/metadata非空。</li></ul></td>
      </tr>
      </tbody>
    </table>

## aclnnSparseLightningIndexer

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
    </colgroup>
    <thead>
      <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseLightningIndexerGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnSparseLightningIndexer默认确定性实现。
- 输入shape限制：
  - q的N1支持[1, 64]，k的N2仅支持1，headDim（D）仅支持128。
  - layout_k非PA_BBND时，layout_q与layout_k必须一致；PA_BBND时block_table必传且dim0等于B，seqused_k必传，k仅允许0轴非连续。
  - layout_k为TND时cuSeqlensKOptional必传；layout_k为BSND时cuSeqlensKOptional传入报错。
- candidate约束：
  - candidateBlockIndices 必传（REQUIRED，INT32）；candBlocks（shape 末维）∈ (0, 2048] 且为 64 的倍数；本算子无 candidate_topk_blocks 属性（宽度由输入 shape 推导）。
  - candidateBlockSize ∈ [2, 64] 内的2的幂；与上游source算子的candidateBlockSize保持一致（调用方约束，框架层无法跨算子校验）。
  - candidateBlockLength 预留：仅接受 nullptr/空 tensor（numel==0），非空报错
    "candidate_block_length is reserved and only empty tensor is supported yet"。
- 属性限制：
  - topk ∈ (0, 2048]，恒校验（>2048 报错）。
  - returnValue ∈ {0,1}，其它值报错拒绝（"Return_value only supports 0 or 1"）。
  - maskMode 仅支持0或3。
  - cmpRatio 由本算子 host 强校验 (0, 128] 且 2 的幂；上游 source 算子（aclnnLightningIndexerV3）host 仅校验 (0, 128]，两侧配置需保持一致。
  - maxSeqlenQ 仅支持-1；seqused_q / output_idx_offset / metadata 由 arch22 host 侧校验拒绝（接口对齐保留位，传 nullptr 即可）。
- 其余约束（layout 组合、PA block_table、N2=1、head_dim=128 等）与上游算子 aclnnLightningIndexerV3 一致。

## 跨算子约束

- candidateBlockIndices 直接对接 aclnnLightningIndexerV3（开启 candidateTopkBlocks 时）的 candidateBlockIndices 输出，块号为同一 forward 内 batch 相对压缩块号，槽位无序（消费侧按集合消费）；
- 候选块号不含 output_idx_offset 偏移；
- candidateBlockSize、cmpRatio 必须与 source 侧配置一致；

- 候选宽度 candBlocks 由输入 tensor shape 末维推导。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer.h"
#include "aclnn/opdev/platform.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

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
                    aclDataType dataType, aclTensor** tensor)
{
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 空 tensor（candidateBlockLengthOptional 预留接口的空传法）
int CreateEmptyAclTensor(aclTensor** tensor)
{
    std::vector<int64_t> shape = {0};
    std::vector<int64_t> strides = {1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* candidateBlockIndicesDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* candidateBlockIndicesTensor = nullptr;
    aclTensor* candidateBlockLengthTensor = nullptr; // 空 tensor 占位
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources)
{
    // BSND/BSND, FP16: B=2, S1=8, N1=8, D=128, S2=64, topk=32, maskMode=0, cmpRatio=1
    // candidateBlockIndices: [B, S1, N2=1, candBlocks=64]（S2=64 / candidateBlockSize=8 → 8 个块，
    // 0..7 为全候选，其余槽位 -1）
    int64_t B = 2;
    int64_t S1 = 8;
    int64_t N1 = 8;
    int64_t D = 128;
    int64_t N2 = 1;
    int64_t S2 = 64;
    int64_t topk = 32;
    int64_t candBlocks = 64;
    int64_t blockSize = 8;
    int64_t numBlocks = S2 / blockSize;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> candidateBlockIndicesShape = {B, S1, N2, candBlocks};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t candidateShapeSize = GetShapeSize(candidateBlockIndicesShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<float> weightsHostData(weightsShapeSize, 1.0f);
    std::vector<int32_t> candidateHostData(candidateShapeSize, -1);
    for (int64_t i = 0; i < candidateShapeSize; i++) {
        int64_t slot = i % candBlocks;
        if (slot < numBlocks) {
            candidateHostData[i] = static_cast<int32_t>(slot); // 全候选（覆盖全部块号）
        }
    }
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 0.0f);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, aclDataType::ACL_FLOAT16,
                              &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, aclDataType::ACL_FLOAT16,
                          &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr, aclDataType::ACL_FLOAT,
                          &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(candidateHostData, candidateBlockIndicesShape, &resources.candidateBlockIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.candidateBlockIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // candidateBlockLengthOptional：预留接口空传法（shape (0,)）
    ret = CreateEmptyAclTensor(&resources.candidateBlockLengthTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    return ACL_SUCCESS;
}

// return_value=0：sparseValuesOut 恒空 (0,)，kernel 不写入（aclnn 层仍要求传入非空 tensor，
// 本示例按 (B, S1, N2, topk) 全尺寸分配）
int ExecuteSparseLightningIndexer(TensorResources& resources, aclrtStream stream, void** workspaceAddr,
                                  uint64_t* workspaceSize)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 0;
    int64_t candidateBlockSize = 8; // [2, 64] 内的2的幂
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    // sequsedQ/outputIdxOffset/metadata 为接口对齐保留位，arch22 传 nullptr
    int ret = aclnnSparseLightningIndexerGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, resources.candidateBlockIndicesTensor, resources.candidateBlockLengthTensor,
        topk, maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio, returnValue, candidateBlockSize,
        resources.sparseIndicesTensor, resources.sparseValuesTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseLightningIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

// return_value=1 调用样例：sparseValuesOut 输出真实 TopK 分数（与 sparseIndicesOut 同 shape，
// 有效槽为入选位置真实分数 fp32，-1 槽对应位 value=0；indices 输出与 return_value=0 时一致）
int ExecuteSparseLightningIndexerRv1(TensorResources& resources, aclrtStream stream)
{
    int64_t topk = 32;
    int64_t maxSeqlenQ = -1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 1;
    int64_t candidateBlockSize = 8;
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }

    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    aclOpExecutor* executor = nullptr;

    // sparseValuesTensor 须传与 sparseIndicesOut 同 shape 的非空 tensor（(B, S1, N2, topk)）
    int ret = aclnnSparseLightningIndexerGetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, resources.candidateBlockIndicesTensor, resources.candidateBlockLengthTensor,
        topk, maxSeqlenQ, layoutQuery, layoutKey, maskMode, cmpRatio, returnValue, candidateBlockSize,
        resources.sparseIndicesTensor, resources.sparseValuesTensor, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseLightningIndexerGetWorkspaceSize(rv=1) failed. ERROR: %d\n", ret);
        return ret;
    }

    if (workspaceSize > 0ULL) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace(rv=1) failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseLightningIndexer(workspaceAddr, workspaceSize, executor, stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseLightningIndexer(rv=1) failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("sparse_indices result[%ld] is: %d\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

// return_value=1 样例的 values 打印：有效槽为真实 TopK 分数，-1 槽对应位为 0
int PrintValuesResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0.0f);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy values result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("sparse_values result[%ld] is: %f\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, aclrtStream stream, int32_t deviceId)
{
    if (resources.queryTensor) {
        aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
        aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
        aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.candidateBlockIndicesTensor) {
        aclDestroyTensor(resources.candidateBlockIndicesTensor);
    }
    if (resources.candidateBlockLengthTensor) {
        aclDestroyTensor(resources.candidateBlockLengthTensor);
    }
    if (resources.sparseIndicesTensor) {
        aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
        aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
        aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
        aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
        aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.candidateBlockIndicesDeviceAddr) {
        aclrtFree(resources.candidateBlockIndicesDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
        aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
        aclrtFree(resources.sparseValuesDeviceAddr);
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

} // namespace

int main()
{
    // 本算子仅注册 ascend910b/ascend910_93（框架级不支持其他 arch）
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201) {
        return 0;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 8;
    int64_t N2 = 1;
    int64_t topk = 32;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
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

    ret = ExecuteSparseLightningIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteSparseLightningIndexer failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);

    // return_value=1 追加样例：同一输入下 sparse_values 输出真实 TopK 分数
    // （有效槽真实分 fp32、-1 槽对应位 0；indices 输出与 rv=0 一致，被本次调用覆盖写回）
    ret = ExecuteSparseLightningIndexerRv1(resources, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteSparseLightningIndexerRv1 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream(rv=1) failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintValuesResult(sparseIndicesShape, &resources.sparseValuesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
