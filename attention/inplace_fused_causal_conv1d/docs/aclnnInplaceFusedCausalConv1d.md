# aclnnInplaceFusedCausalConv1d

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/inplace_fused_causal_conv1d)

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

- 接口功能：对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新等特性。相较于标准causal_conv1d算子，本算子新增APC缓存复用、PD混部、残差连接可选等功能。<br>

- 支持以下场景：
  - 场景一（prefill场景）：

    ```Cpp
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch],开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]（无作用）
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    pad_slot_id: 默认值 -1
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    ```

    其中cu_seq_len为batch内所有变长序列拼接后的总长度。

  - 场景二（prefill和decode混合场景）：

    ```Cpp
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1+m, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch],开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    pad_slot_id: 默认值 -1
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    ```

    其中cu_seq_len为batch内所有变长序列拼接后的总长度。

  - 场景三（decode场景 - 变长序列）：

    ```Cpp
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1+m, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch],开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    pad_slot_id: 默认值 -1
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    ```

    其中state_len必须大于所有batch中最大的token个数加1。

  - 场景四（decode场景 - 固定batch）：

    ```Cpp
    x: [batch, m+1, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1+m, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch],开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    pad_slot_id: 默认值 -1
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    ```

- 计算公式：

  K是卷积核宽度（固定为3），L是原始序列长度，dim是特征维度。
  1. 缓存读取

      缓存行索引：

      $$
      readCacheLine = \begin{cases}
      cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC模式} \\
      cacheIndices[batchId], & \text{非APC且cacheIndices存在} \\
      batchId, & \text{其他}
      \end{cases}
      $$

      Case 1：首次计算（numComputedTokens[batchId] == 0）

      $$
      cachedState[i, dim] = 0, \quad 0 \leq i < K-1
      $$

      $$
      offset = 0
      $$

      Case 2：投机解码模式（numAcceptedTokens存在）

      $$
      offset = numAcceptedTokens[batchId] - 1
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i <   offset + K - 1
      $$

      Case 3：默认模式

      $$
      offset = C - (K - 1)
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
      $$

  2. 缓存拼接

      $$
      paddedInput[i, dim] =
      \begin{cases}
      cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
      x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
      \end{cases}
      $$

  3. 缓存更新

      $$
      Len = offset + K - 1 + L
      $$

      $$
      M = \min(C, \; Len)
      $$

      $$
      writeCacheLine = \begin{cases}
      cacheIndices[batchId, \; idxLast], & \text{APC模式} \\
      cacheIndices[batchId], & \text{非APC且cacheIndices存在} \\
      batchId, & \text{其他}
      \end{cases}
      $$

      $$
      convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
      $$

  4. Offset裁剪

      $$
      x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
      $$

  5. APC缓存填充（可选，APC模式下）

      $$
      seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
      $$

      $$
      seqCompletedOffset = B - seqCompletedOffsetToken
      $$

      $$
      seqEndOffset = (L - seqCompletedOffset) \mod B
      $$

      $$
      lastFullBlockTokenIndex = \begin{cases}
      L - seqEndOffset - B, & seqEndOffset = 0 \\
      L - seqEndOffset, & \text{otherwise}
      \end{cases}
      $$

      $$
      nBlockToFill = idxLast - idxFirst
      $$

      对每个chunk = 0, 1, ..., nBlockToFill - 1：

      $$
      boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
      $$

      $$
      convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
      $$

  6. 因果1维卷积

      $$
      y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
      $$

  7. 零填充重置（可选，当convMode == 1并且numComputedTokens不为空时）

      $$
      resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
      $$

      $$
      y[i, dim] = 0, \quad 0 \leq i < resetIdx
      $$

  8. 残差连接（可选）

      $$
      y[i, dim] = x[i, dim] + y[i, dim]
      $$

  9. 原地更新

      $$
      x[i, dim] = y[i, dim]
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnInplaceFusedCausalConv1dGetWorkspaceSize`接口获取入参并计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnInplaceFusedCausalConv1d`接口执行计算。

```Cpp
aclnnStatus aclnnInplaceFusedCausalConv1dGetWorkspaceSize(
  aclTensor       *x,
  const aclTensor *weight,
  aclTensor       *convStates,
  const aclTensor *queryStartLoc,
  const aclTensor *cacheIndices,
  const aclTensor *initialStateMode,
  const aclTensor *bias,
  const aclTensor *numAcceptedTokens,
  const aclTensor *numComputedTokens,
  const aclTensor *blockIdxFirstScheduledToken,
  const aclTensor *blockIdxLastScheduledToken,
  const aclTensor *initialStateIdx,
  int64_t          activationMode,
  int64_t          padSlotId,
  int64_t          runMode,
  int64_t          maxQueryLen,
  int64_t          residualConnection,
  int64_t          blockSize,
  int64_t          convMode,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceFusedCausalConv1d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInplaceFusedCausalConv1dGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 333px">
  <col style="width: 400px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
  <col style="width: 145px">
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
      <td>x（aclTensor*）</td>
      <td>输入/输出</td>
      <td>计算公式中的x，代表输入序列。卷积结果将原地更新至x。</td>
      <td><ul><li>不支持空tensor。</li><li>prefill场景：shape为[cu_seq_len, dim]。</li><li>decode场景：shape为[cu_seq_len, dim]或[batch, seq_len, dim]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的w，代表因果1维卷积核。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[K, dim]。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>convStates（aclTensor*）</td>
      <td>输入/输出</td>
      <td>计算公式中的cacheState，代表缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[..., stateLen, dim]，第0维的大小不固定。</li><li>stateLen == K-1+m，prefill场景下m=0，decode、PD混部场景下m=[0,7]。</li><li>m表示投机个数，映射numAcceptedTokens参数的值。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryStartLoc（aclTensor*）</td>
      <td>可选输入</td>
      <td>序列起始位置索引，记录各序列在拼接张量x中的起始位置。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[batch+1,]。</li><li>queryStartLoc[i]表示第i个序列的起始偏移。</li><li>queryStartLoc[0]必须为0，queryStartLoc[-1]必须为cu_seq_len，相邻两个数据不相等。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cacheIndices（aclTensor*）</td>
      <td>可选输入</td>
      <td>缓存索引，指定每个序列对应的缓存状态在convStates中的索引。</td>
      <td><ul><li>不支持空tensor，APC开启时不能为None且必须为二维。</li><li>1维shape[batch,]或2维shape[batch, maxNumBlocks], maxNumBlocks范围[1,1024]。</li><li>值需要互不相同（除非等于padSlotId）。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateMode（aclTensor*）</td>
      <td>可选输入</td>
      <td>初始状态标志，表示各序列是否使用缓存数据。</td>
      <td>不支持此字段。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>bias（aclTensor*）</td>
      <td>可选输入</td>
      <td>卷积的偏置。</td>
      <td>不支持此字段。</td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numAcceptedTokens（aclTensor*）</td>
      <td>可选输入</td>
      <td>公式中的numAcceptedTokens，表示每个batch的随机投机数。</td>
      <td><ul><li>1<=numAcceptedTokens中的值<=seqlen，seqlen表示该batch的序列长度。</li><li>shape为[batch,]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numComputedTokens（aclTensor*）</td>
      <td>可选输入</td>
      <td>公式中的numComputedTokens，当前batch已经处理的token总数，用于判断初始状态。</td>
      <td><ul><li>首token时使用零初始化缓存；Pangu V2模式下不能为空。</li><li>shape为[batch,]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxFirstScheduledToken（aclTensor*）</td>
      <td>可选输入</td>
      <td>当前batch的起始位置对应的block索引。</td>
      <td><ul><li>APC开启时不能为空。</li><li>shape为[batch,]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockIdxLastScheduledToken（aclTensor*）</td>
      <td>可选输入</td>
      <td>当前batch的seq_len-1处对应的block索引。</td>
      <td><ul><li>APC开启时不能为空。</li><li>shape为[batch,]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateIdx（aclTensor*）</td>
      <td>可选输入</td>
      <td>初始索引块的索引。</td>
      <td><ul><li>APC开启时不能为空。</li><li>shape为[batch,]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activationMode（int64_t）</td>
      <td>属性</td>
      <td>激活函数类型。</td>
      <td>不支持此字段。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padSlotId（int64_t）</td>
      <td>属性</td>
      <td>用于跳过不需要参与计算的batch。</td>
      <td><ul><li>仅支持不参与计算的变长序列在x的开头或结尾。</li><li>当cacheIndices[i]==padSlotId时跳过该batch。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>runMode（int64_t）</td>
      <td>属性</td>
      <td>用于判断是prefill场景或decode场景。</td>
      <td>历史遗留接口，不支持此字段。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxQueryLen（int64_t）</td>
      <td>属性</td>
      <td>所有batch中的最大seq_len，仅decode场景（固定batch）支持为-1。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualConnection（int64_t）</td>
      <td>属性</td>
      <td>是否做残差连接。</td>
      <td><ul><li>取值为0、1：<br>0：不做残差连接；<br>1：输出y和输入x相加后输出。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>属性</td>
      <td>block块的大小。</td>
      <td><ul><li>取值范围大于等于2，典型值128/256。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>convMode（int64_t）</td>
      <td>属性</td>
      <td>公式中的convMode，支持Qwen3-Next和Pangu V2两种实现。</td>
      <td><ul><li>取值为0、1：<br>0：Qwen3-Next社区版本实现；<br>1：Pangu V2实现。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（int64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x、weight、convStates是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>
      输入和输出的数据类型不在支持的范围内。<br>
      x、weight、convStates、bias的数据类型不一致。<br>
      queryStartLoc、cacheIndices、initialStateMode、numAcceptedTokens、numComputedTokens、blockIdxFirstScheduledToken、blockIdxLastScheduledToken、initialStateIdx的数据类型不一致。<br>
      输入、输出Tensor的shape不在支持的范围内。<br>
      输入的属性不在支持的范围内。<br>
      dim不在指定的取值范围内。<br>
      </td>
    </tr>
  </tbody></table>

## aclnnInplaceFusedCausalConv1d

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceFusedCausalConv1dGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnInplaceFusedCausalConv1d默认确定性实现。

- 输入shape限制：
  - prefill场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024 * 1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64 * batch, 4G], batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len > 8。
  - prefill和decode混合场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024 * 1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64 * batch, 4G], batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len > 8。
  - decode场景（变长序列）：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., k-1+m, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, batch * 8]，每个batch的seq_len范围为[1, 8]。dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len范围[1, 8]。
  - decode场景（固定batch）：
    - x支持3维[batch, m+1, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于batch，同时大于等于cache_indices总维度大小。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - m范围[0, 7]，dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len范围[1, 8]，可为-1。

- 输入值域限制：
  - query_start_loc是累计偏移量，取值范围[0, cu_seq_len]，长度为batch+1，query_start_loc[i]表示第i个序列的起始偏移，query_start_loc[batch+1]表示最后一个序列的结束位置。
  - blockSize 为0或者大于等于2，apc开启时不为0。
  - blockIdxFirstScheduledToken、blockIdxLastScheduledToken、initialStateIdx、num_computed_tokens和cache_indices均存在时表示APC开启，且满足以下条件（i为batch的索引）：
    - cache_indices为2维
    - initialStateIdx[i] <= blockIdxFirstScheduledToken[i]+1
    - initialStateIdx[i] <= blockIdxLastScheduledToken[i]
    - blockIdxFirstScheduledToken[i] <= blockIdxLastScheduledToken[i]
    - blockIdxLastScheduledToken[i] < maxNumBlocks
  - num_accepted_tokens分为None和非None，非None情况下长度为batch，每个元素取值不超过当前batch的seq_len-1且大于0。
  - num_computed_tokens中每个元素取值大于等于0。
  - cache_indices的取值范围为[0, conv_states.dim[0]-1],且元素均不能相等。
  - max_query_len = batch中的最大seq_len。
  - Pangu V2 模式（conv_mode = 1）下，num_computed_tokens不能为 None。
  - 算子入参与中间计算结果，在对应数据类型（float16/bfloat16）下，数值均不会超出该类型值域范围。
  - 算子输入不支持有±inf和nan的情况。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_fused_causal_conv1d.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// IEEE 754 fp16 decode (uint16_t → float)
float Fp16ToFloat(uint16_t raw)
{
    uint32_t sign = (raw >> 15) & 0x1;
    uint32_t exp = (raw >> 10) & 0x1F;
    uint32_t mant = raw & 0x3FF;
    float val;
    if (exp == 0) {
        val = (sign ? -1.0f : 1.0f) * (static_cast<float>(mant) / 1024.0f) * (1.0f / 16384.0f);
    } else if (exp == 31) {
        val = (mant == 0) ? (sign ? -HUGE_VALF : HUGE_VALF) : NAN;
    } else {
        val = (sign ? -1.0f : 1.0f) * (1.0f + static_cast<float>(mant) / 1024.0f) *
              ldexpf(1.0f, static_cast<int>(exp) - 15);
    }
    return val;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
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

    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0, format,
                              shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    return 0;
}

void FreeTensorAndBuffer(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t K = 3;            // kernel width
    int64_t dim = 128;        // feature dimension
    int64_t batch = 4;        // number of sequences
    int64_t numSlots = 8;     // total cache slots (>= batch for non-APC)
    int64_t stateLen = K - 1; // cache state length per slot (= 2)
    // prefill: seq_lens = [5, 3, 7, 4], cuSeqLen = 19
    int64_t cuSeqLen = 19;

    // ---- Tensor shapes ----
    std::vector<int64_t> xShape = {cuSeqLen, dim};
    std::vector<int64_t> weightShape = {K, dim};
    std::vector<int64_t> convStatesShape = {numSlots, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batch + 1};
    std::vector<int64_t> cacheIndicesShape = {batch}; // 1D: non-APC
    std::vector<int64_t> initialStateModeShape = {batch};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> numAcceptedTokensShape = {batch};

    // ---- Host data ----
    // x: fp16 1.0, shape [19, 128]
    std::vector<uint16_t> hostX(cuSeqLen * dim, 0x3C00); // fp16 1.0

    // weight: pass-through at k=0, zero elsewhere
    std::vector<uint16_t> hostWeight(K * dim, 0);
    for (int64_t d = 0; d < dim; d++) {
        hostWeight[0 * dim + d] = 0x3C00; // k=0: 1.0
    }

    // conv_states: zero-initialized, shape [8, 2, 128]
    std::vector<uint16_t> hostConvStates(numSlots * stateLen * dim, 0);

    // query_start_loc: [0, 5, 8, 15, 19]
    std::vector<int32_t> hostQueryStartLoc = {0, 5, 8, 15, 19};

    // cache_indices: each batch uses a distinct cache slot
    std::vector<int32_t> hostCacheIndices = {0, 3, 1, 5};

    // initial_state_mode: 1=from cache, 0=zero-init, 2=from cache (MTP variant)
    std::vector<int32_t> hostInitialStateMode = {1, 0, 2, 1};

    // bias: zero (unused)
    std::vector<uint16_t> hostBias(dim, 0);

    // num_accepted_tokens: 0 for prefill (no speculative decoding)
    std::vector<int32_t> hostNumAcceptedTokens(batch, 0);

    // ---- Device pointers and aclTensors ----
    void *xDev = nullptr, *weightDev = nullptr, *convStatesDev = nullptr;
    void *queryStartLocDev = nullptr, *cacheIndicesDev = nullptr;
    void *initialStateModeDev = nullptr, *biasDev = nullptr;
    void *numAcceptedTokensDev = nullptr;

    aclTensor *xTensor = nullptr, *weightTensor = nullptr, *convStatesTensor = nullptr;
    aclTensor *queryStartLocTensor = nullptr, *cacheIndicesTensor = nullptr;
    aclTensor *initialStateModeTensor = nullptr, *biasTensor = nullptr;
    aclTensor *numAcceptedTokensTensor = nullptr;

    // ---- Create required tensors (x is non-const for inplace) ----
    ret = CreateAclTensor(hostX, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostWeight, weightShape, &weightDev, ACL_FLOAT16, &weightTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(hostConvStates, convStatesShape, &convStatesDev, ACL_FLOAT16, &convStatesTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostQueryStartLoc, queryStartLocShape, &queryStartLocDev, ACL_INT32, &queryStartLocTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostCacheIndices, cacheIndicesShape, &cacheIndicesDev, ACL_INT32, &cacheIndicesTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostInitialStateMode, initialStateModeShape, &initialStateModeDev, ACL_INT32,
                          &initialStateModeTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostBias, biasShape, &biasDev, ACL_FLOAT16, &biasTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(hostNumAcceptedTokens, numAcceptedTokensShape, &numAcceptedTokensDev, ACL_INT32,
                          &numAcceptedTokensTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t activationMode = 0; // 0: None
    int64_t padSlotId = -1;     // -1: no pad-slot skipping
    int64_t runMode = 0;        // 0: prefill
    int64_t maxQueryLen = cuSeqLen;
    int64_t residualConnection = 0; // 0: no residual
    int64_t blockSize = 0;          // 0: non-APC
    int64_t convMode = 0;           // 0: Qwen/Pangu7B

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    ret = aclnnInplaceFusedCausalConv1dGetWorkspaceSize(
        xTensor, weightTensor, convStatesTensor, queryStartLocTensor, cacheIndicesTensor, initialStateModeTensor,
        biasTensor, numAcceptedTokensTensor,
        nullptr, // num_computed_tokens (non-APC: nullptr)
        nullptr, // block_idx_first_scheduled_token
        nullptr, // block_idx_last_scheduled_token
        nullptr, // initial_state_idx
        activationMode, padSlotId, runMode, maxQueryLen, residualConnection, blockSize, convMode, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFusedCausalConv1dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnInplaceFusedCausalConv1d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFusedCausalConv1d failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto xSize = GetShapeSize(xShape);
    std::vector<uint16_t> xResult(xSize, 0);
    ret = aclrtMemcpy(xResult.data(), xSize * sizeof(uint16_t), xDev, xSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy x from device to host failed. ERROR: %d\n", ret); return ret);

    auto csSize = GetShapeSize(convStatesShape);
    std::vector<uint16_t> csResult(csSize, 0);
    ret = aclrtMemcpy(csResult.data(), csSize * sizeof(uint16_t), convStatesDev, csSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy convStates from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("=== x output (inplace, first 10 elements) ===\n");
    for (int64_t i = 0; i < 10 && i < xSize; i++) {
        LOG_PRINT("  x[%lld] = %f\n", (long long)i, (double)Fp16ToFloat(xResult[i]));
    }
    LOG_PRINT("  ...\n");

    LOG_PRINT("=== convStates output (first 10 elements) ===\n");
    for (int64_t i = 0; i < 10 && i < csSize; i++) {
        LOG_PRINT("  convStates[%lld] = %f\n", (long long)i, (double)Fp16ToFloat(csResult[i]));
    }

    FreeTensorAndBuffer(xTensor, xDev);
    FreeTensorAndBuffer(weightTensor, weightDev);
    FreeTensorAndBuffer(convStatesTensor, convStatesDev);
    FreeTensorAndBuffer(queryStartLocTensor, queryStartLocDev);
    FreeTensorAndBuffer(cacheIndicesTensor, cacheIndicesDev);
    FreeTensorAndBuffer(initialStateModeTensor, initialStateModeDev);
    FreeTensorAndBuffer(biasTensor, biasDev);
    FreeTensorAndBuffer(numAcceptedTokensTensor, numAcceptedTokensDev);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
