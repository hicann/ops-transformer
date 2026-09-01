# InplaceFusedCausalConv1d

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列部分数据更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。支持APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新等特性。

- 本算子支持以下场景：

  - 场景一（prefill场景）：

    ```Cpp
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch]或None,开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]（无作用）
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
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
    cache_indices: 不开APC:[batch]或None,开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    max_draft_tokens: 投机个数（multiTokenNum），取值范围[0, 16]
    ```

    其中cu_seq_len为batch内所有变长序列拼接后的总长度。

  - 场景三（decode场景 - 变长序列）：

    ```Cpp
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1+m, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch]或None,开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    max_draft_tokens: 投机个数（multiTokenNum），取值范围[0, 16]
    ```

    其中state_len必须大于所有batch中最大的token个数加1。

  - 场景四（decode场景 - 固定batch）：

    ```Cpp
    x: [batch, m+1, dim]
    weight: [K, dim]，其中K=3
    conv_states: [-1, K-1+m, dim]
    query_start_loc: [batch+1]
    cache_indices: 不开APC:[batch]或None,开APC:[block, maxNumBlocks]
    initial_state_mode: [batch]
    bias: [dim]（无作用）
    num_accepted_tokens: [batch]
    num_computed_tokens: [batch]
    block_idx_first_scheduled_token: 不开APC:None,开APC:[batch]
    block_idx_last_scheduled_token: 不开APC:None,开APC:[batch]
    initial_state_idx: 不开APC:None,开APC:[batch]
    activation_mode:（无作用）
    run_mode:（无作用）
    residual_connection: 不做残差: 0,做残差：1
    block_size: 典型值128/256
    conv_mode：Qwen3-Next模式: 0, Pangu V2: 1
    max_draft_tokens: 投机个数（multiTokenNum），取值范围[0, 16]
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

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px">
  <colgroup>
    <col style="width: 160px">
    <col style="width: 100px">
    <col style="width: 260px">
    <col style="width: 260px">
    <col style="width: 140px">
    <col style="width: 80px">
    <col style="width: 280px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入/输出</td>
      <td>公式中的输入序列x，卷积结果将原地更新至x，无效batch部分保持x原数值不变。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>公式中的因果1维卷积核w。</td>
      <td>同x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>conv_states</td>
      <td>输入/输出</td>
      <td>
        <ul>
          <li>公式中的convStates，缓存状态张量，存储各序列的历史token数据。</li>
          <li>各序列计算完成后原地更新。</li>
        </ul>
      </td>
      <td>同x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_start_loc</td>
      <td>可选输入</td>
      <td>
        <ul>
          <li>x为二维场景下，序列起始位置索引，记录各序列在拼接张量x中的起始位置。</li>
          <li>queryStartLoc[i]表示第i个序列的起始偏移。queryStartLoc[0]必须为0，queryStartLoc[-1]必须为cu_seq_len，相邻两个数据不相等。</li>
        </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cache_indices</td>
      <td>可选输入</td>
      <td>缓存索引，指定每个序列对应的缓存状态在cacheState中的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initial_state_mode</td>
      <td>可选输入</td>
      <td>制定每个序列对应的padding策略。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>卷积的偏置。</td>
      <td>同x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_accepted_tokens</td>
      <td>可选输入</td>
      <td>公式中的numAcceptedTokens。 </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_computed_tokens</td>
      <td>可选输入</td>
      <td>公式中的numComputedTokens，当前batch已经处理的token总数，用于判断初始状态。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_idx_first_scheduled_token</td>
      <td>可选输入</td>
      <td>当前batch的起始位置对应的block索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_idx_last_scheduled_token</td>
      <td>可选输入</td>
      <td>当前batch的seq_len-1处对应的block索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initial_state_idx</td>
      <td>可选输入</td>
      <td>初始索引块的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activation_mode</td>
      <td>可选输入</td>
      <td>激活函数类型。</td>
      <td>STR</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pad_slot_id</td>
      <td>可选输入</td>
      <td>用于跳过不需要参与计算的变长序列。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>run_mode</td>
      <td>可选输入</td>
      <td>表示prefill或者decode场景。历史遗留接口，暂不支持此字段。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_query_len</td>
      <td>可选输入</td>
      <td>所有batch中的最大seq_len，仅decode场景（固定batch）支持为-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residual_connection</td>
      <td>可选输入</td>
      <td>用于残差连接。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>可选输入</td>
      <td>block块的大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>conv_mode</td>
      <td>可选输入</td>
      <td>公式中的convMode，支持Qwen3-Next和Pangu V2两种实现。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_draft_tokens（int64_t）</td>
      <td>属性</td>
      <td>最大投机个数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入shape限制：
  - prefill场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024 * 1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64 * batch, 4G], batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len > max_draft_tokens + 1。
  - prefill和decode混合场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, 1024 * 1024]，dim范围[64, 16384]且是16的倍数，且两者乘积需满足[64 * batch, 4G], batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len > max_draft_tokens + 1。
  - decode场景（变长序列）：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., k-1+m, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - query_start_loc必须存在。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - cu_seq_len范围[batch, batch * (max_draft_tokens + 1)]，每个batch的seq_len范围为[1, max_draft_tokens + 1]。dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len范围[1, max_draft_tokens + 1]。
  - decode场景（固定batch）：
    - x支持3维[batch, m+1, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - conv_states必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于参与计算的batch个数（即cache_indices不等于pad_slot_id的batch个数）。
    - cache_indices为1维[batch, ]或2维[batch, maxNumBlocks]，其中1维表示未开启APC，2维表示开启APC。
    - m范围[0, max_draft_tokens]，dim范围[64, 16384]且是16的倍数，batch范围[1, 256]。
    - maxNumBlocks >= ceiv(max_query_len, block_size)。
    - max_query_len范围[1, max_draft_tokens + 1]，可为-1。

- 输入值域限制：
  - query_start_loc是累计偏移量，取值范围[0, cu_seq_len]，长度为batch+1，query_start_loc[i]表示第i个序列的起始偏移，query_start_loc[batch+1]表示最后一个序列的结束位置。
  - blockSize 为0或者大于等于2，apc开启时不为0。
  - blockIdxFirstScheduledToken、blockIdxLastScheduledToken、initialStateIdx、num_computed_tokens和cache_indices均存在时表示APC开启，且满足以下条件（i为batch的索引）：
    - cache_indices为2维
    - initialStateIdx[i] <= blockIdxFirstScheduledToken[i]+1
    - initialStateIdx[i] <= blockIdxLastScheduledToken[i]
    - blockIdxFirstScheduledToken[i] <= blockIdxLastScheduledToken[i]
    - blockIdxLastScheduledToken[i] < maxNumBlocks
  - num_accepted_tokens分为None和非None，非None情况下长度为batch，prefile对应的元素值为0，decode对应的元素值大于0且小于等于当前batch的seq_len-1。
  - num_computed_tokens中每个元素取值大于等于0。
  - cache_indices的取值范围为[0, conv_states.dim[0]-1],且值均不能相等（除非等于pad_slot_id）。
  - max_query_len = batch中的最大seq_len。
  - max_draft_tokens的取值范围为[0, 16]。
  - Pangu V2 模式（conv_mode = 1）下，num_computed_tokens不能为 None。
  - 算子入参与中间计算结果，在对应数据类型（float16/bfloat16）下，数值均不会超出该类型值域范围。
  - 算子输入不支持有±inf和nan的情况。

  ## 调用说明

  | 调用方式  | 样例代码                                                     | 说明                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn接口 | [test_aclnn_inplace_fused_causal_conv1d](./examples/test_aclnn_inplace_fused_causal_conv1d.cpp) / [test_aclnn_inplace_fused_causal_conv1d_v2](./examples/test_aclnn_inplace_fused_causal_conv1d_v2.cpp) | 通过[aclnnInplaceFusedCausalConv1d](./docs/aclnnInplaceFusedCausalConv1d.md)或[aclnnInplaceFusedCausalConv1dV2](./docs/aclnnInplaceFusedCausalConv1dV2.md)调用InplaceFusedCausalConv1d算子 |
  | 图模式 | - | 通过[算子IR](./op_graph/inplace_fused_causal_conv1d_proto.h)构图方式调用InplaceFusedCausalConv1d算子 |
