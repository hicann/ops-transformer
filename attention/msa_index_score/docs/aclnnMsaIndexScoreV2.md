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

- **算子功能**：计算MSA（MiniMax Sparse Attention）模块Index Branch中的block score。对每个query token与每个KV sparse block，取该block内所有因果可见token的$Q_{idx}$和$K_{idx}$（可选int8反量化）的"matmul+maxpool"运算获得逐block的重要性分数score，用作Index Branch中后续的TopK输入。Prefill与Decode由同一接口承载。

- **计算公式**：

    - 非量化场景：

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - int8量化场景：

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T}  ]
    $$

    完整公式（含因果mask与local_mask）：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    $local\_mask$由`startLoc`、`initBlocks`、`localBlocks`生成：逻辑block $[0, initBlocks)$写入$1\mathrm{e}30$；窗口$[max(0, startLoc+1-localBlocks), startLoc]$写入$1\mathrm{e}29$（覆盖同位置的init）。两者均为0时关闭$local\_mask$。

- **参数介绍**：

    > - B（Batch Size）表示输入样本批量大小
    > - S（Sequence Length）表示序列长度，$S1$为query侧、$S2$为key侧
    > - T表示所有Batch序列长度累加和，$T1$为query侧、$T2$为key侧
    > - N（Head Num）表示头数，$N1$为query侧、$N2$为key侧
    > - D（Head Dim）表示单个注意力头维度；
    > - PageAttention场景下$block\_num$为物理页总数，$block\_size$为每个存储页的token数。$pool\_block\_size$为每个score归约的token数。打分块总数$M_b=\lceil S2/pool\_block\_size\rceil$；页表宽度按$block\_size$计

## 函数原型

每个算子分为[两段式接口](https://atomgit.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md)，必须先调用`aclnnMsaIndexScoreV2GetWorkspaceSize`接口获取入参并计算所需workspace大小，再调用`aclnnMsaIndexScoreV2`接口执行计算。

```cpp
aclnnStatus aclnnMsaIndexScoreV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *blockTableOptional,
    const aclTensor *scaleOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKlenOptional,
    const aclTensor *startLoc,
    char            *layoutKeyOptional,
    int64_t          sparseMode,
    int64_t          initBlocks,
    int64_t          localBlocks,
    char            *layoutQueryOptional,
    int64_t          poolBlockSize,
    const aclTensor *score,
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
      <th>非连续tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的$Q_{idx}$。</td>
      <td>支持TND、BSND、BNSD。由<code>layoutQueryOptional</code>指定，不传时为TND。BSND、BNSD仅Ascend 950，且仅FLOAT16 / BFLOAT16。</td>
      <td>BFLOAT16、FLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3（$[T1, N1, D]$）或4（$[B, S1, N1, D]$、$[B, N1, S1, D]$）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的$K_{idx}$。</td>
      <td>支持TND、BSND、BNSD、BBND、BNBD。非分页（TND / BSND / BNSD）必须与query同布局，且不允许非连续。分页BBND / BNBD可配任意query布局；A2/A3与Ascend 950上允许dim0（物理page）非连续，size>1的非首轴须连续。TND的逻辑块长固定为128，每个head只做一次块内max。BSND / BNSD仅Ascend 950，逻辑块长固定16，且仅FLOAT16 / BFLOAT16。BBND/BNBD的存储页长为128，或仅Ascend 950上的16（FLOAT16 / BFLOAT16）。</td>
      <td>BFLOAT16、FLOAT16、INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3（$[T2, N2, D]$）或4（$[B, S2, N2, D]$、$[B, N2, S2, D]$、$[block\_num, N2, block\_size, D]$、$[block\_num, block\_size, N2, D]$）</td>
      <td>仅支持首轴非连续</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示当前传入的使用PageAttention存储的block映射表。</td>
      <td>仅BBND / BNBD传入，需为2维；第二维长度不能小于$maxBlockNumPerSeq$。TND / BSND / BNSD传<code>nullptr</code>。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2（$[B, S2/block\_size]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>公式中的$scale$，反量化系数。</td>
      <td>非量化场景传入<code>nullptr</code>；量化场景时必选。PA为$[block\_num, N2, block\_size]$；TND为$[T2, N2]$（N2=1时可$[T2]$）。BSND / BNSD不支持int8。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3（$[block\_num, N2, block\_size]$、$[block\_num, block\_size, N2]$）或2（$[T2, N2]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>控制因果可见的mask掩码。</td>
      <td>仅在sparseMode=3时使用，作为base mask控制causal可见；取值为1代表该位不参与计算，为0代表该位参与计算。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2（$[2048, 2048]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>三种query布局都必须传入，单调不减（前缀和）。BSND / BNSD时每个请求的长度不能超过shape里的$S1$。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B+1]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSeqKlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>key为TND / BSND / BNSD时必须传入，单调不减（前缀和，$[B+1]$）；PageAttention场景下为各请求可见$S2$（$[B]$）。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B]$或$[B+1]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>startLoc</td>
      <td>输入</td>
      <td>当前query所在逻辑block索引（非token前缀）。</td>
      <td>与<code>initBlocks</code> / <code>localBlocks</code>一起生成$local\_mask$。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>输入</td>
      <td>key的数据排布。</td>
      <td>取值<code>"TND"</code> / <code>"BSND"</code> / <code>"BNSD"</code> / <code>"BBND"</code> / <code>"BNBD"</code>。不传或空串时默认<code>"BBND"</code>。必须与<code>key</code>实际shape一致，不可仅凭维度推断。非分页取值必须与<code>layoutQueryOptional</code>相同。</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>为0时，代表defaultMask模式；为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>initBlocks</td>
      <td>输入</td>
      <td>$local\_mask$强制选中的头部block数。</td>
      <td>对逻辑block $[0, initBlocks)$写入高分$1\mathrm{e}30$。可选，默认$0$；须$\ge 0$且$\le maxBlockNumPerSeq$。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>localBlocks</td>
      <td>输入</td>
      <td>$local\_mask$强制选中的局部窗口长度。</td>
      <td>窗口为$[max(0, startLoc+1-localBlocks), startLoc]$，写入高分$1\mathrm{e}29$（覆盖同位置init）。可选，默认$1$（对齐MiniMax HF）；与Triton raw score对齐时置$0$。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQueryOptional</td>
      <td>输入</td>
      <td>query的数据排布。</td>
      <td>取值<code>"TND"</code> / <code>"BSND"</code> / <code>"BNSD"</code>。不传或空串时默认<code>"TND"</code>。BSND、BNSD仅Ascend 950。key为TND / BSND / BNSD时必须与<code>layoutKeyOptional</code>相同；key为BBND / BNBD时三种都可以。</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>poolBlockSize</td>
      <td>输入</td>
      <td>打分池化块长。</td>
      <td>每个score归约的token数。只允许16 / 128，非正数按128处理。与BBND / BNBD的存储页长<code>page_size</code>不是同一个量；分页时不能超过页长，且页长必须是它的正整数倍。128为默认值，走原单次块内maxpool。16仅Ascend 950、FLOAT16/BFLOAT16，走两级maxpool。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>score</td>
      <td>输出</td>
      <td>公式中的$score$。</td>
      <td>逐block的重要性分数；末维为对齐后的逻辑block数。逻辑块长为16时，每个block再取head间max，head维为1。块长128时每个head只做一次块内max，head维为$N1$。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3（块长16为$[1, T1, RoundUp(maxBlockNumPerSeq, 16)]$，块长128为$[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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
      <td>必选入参或出参为空指针。</td>
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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMsaIndexScoreV2GetWorkspaceSize获取。</td>
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

- 打分池化块长由`poolBlockSize`决定，缺省128：TND取128时每个head只在该块的128个token上做一次max；取16时走两级maxpool（仅Ascend 950、FLOAT16/BFLOAT16）。BSND/BNSD固定为16，仅Ascend 950，且仅FLOAT16/BFLOAT16。BBND/BNBD的存储页长支持128；Ascend 950上另支持16。`poolBlockSize=16`仅FLOAT16/BFLOAT16。
- `layoutQueryOptional`指定query：`"TND"`为$[T1, N1, D]$，`"BSND"`为$[B, S1, N1, D]$，`"BNSD"`为$[B, N1, S1, D]$。不传时默认`"TND"`。BSND、BNSD仅Ascend 950。
- `layoutKeyOptional`指定key：`"BBND"`（$[block\_num, block\_size, N2, D]$）、`"BNBD"`（$[block\_num, N2, block\_size, D]$）、`"TND"`（$[T2, N2, D]$）、`"BSND"`（$[B, S2, N2, D]$）、`"BNSD"`（$[B, N2, S2, D]$）。不传时默认`"BBND"`。key为TND/ BSND/BNSD时必须与query布局相同；key为BBND/BNBD时query可以是三种中的任意一种。BSND、BNSD仅Ascend 950。
- 逻辑块长为16时（BSND/BNSD，或BBND/BNBD且$pool\_block\_size=16$）：8个长度为16的逻辑块拼成一次128列GEMM。先在每个head内对16个token取max，再对每个block取head间max，即$score[0][t][i] = \max_{h'} \max_{j \in block_i} (Q[t][h'] \cdot K[j])$。head维为1。query的head数必须整除128，因此支持$N1=16$。BBND/BNBD必须传入`blockTableOptional`；BSND/BNSD不得传入。TND经`aclnnMsaIndexScoreV2`显式传`poolBlockSize=16`时也走这条路径。
- PageAttention（`layoutKey`为`"BBND"`/`"BNBD"`）场景下，`blockTableOptional`必须传入；TND/BSND/BNSD不得传入。非分页的`actualSeqKlenOptional`为$[B+1]$前缀和，PA为$[B]$可见长度。BSND/BNSD每个请求的q、kv长度不能超过对应shape里的$S$。
- 非量化场景下，`key` dtype与`query`相同（BFLOAT16/FLOAT16；Ascend 950另支持HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN），`scaleOptional`必须为`nullptr`。量化场景仅支持INT8（fp16 query），且key布局为TND或BBND/BNBD；`scaleOptional`必选，dtype为FLOAT：PA为$[block\_num, N2, block\_size]$，TND为$[T2, N2]$。BSND/BNSD不支持int8与FP8。三种FP8仅950，query与key必须同型。
- `sparseMode`当前仅支持0、3：
    - 为0时，代表defaultMask模式，`attenMaskOptional`传入`nullptr`；
    - 为3时，代表rightDownCausal模式，`attenMaskOptional`必须传入，shape为$[2048, 2048]$，取值为1代表该位不参与计算，为0代表该位参与计算。
- `initBlocks`、`localBlocks`必须$\ge 0$且不超过逻辑block数（PA为`blockTableOptional`第二维；非分页为score末维对齐宽度）。两者均为0时跳过$local\_mask$。
- `q_len`/`kv_len`允许为0（含整batch）。对应请求跳过QK；空KV的score填$-inf$；整batch $T1=0$时`SetBlockDim(1)`（对齐FIA PR 9246）。
- A2/A3与Ascend 950：按估计M-task数启动MIX（单batch为$\mathrm{CeilDiv}(T1\cdot N1,128)$；多batch取packed+$B$上界并截到AIC数），短decode不打满空核。Ascend 950在M-task填不满AIC且可见KV stile $>1$时沿S切`kvChunks`；宽`blockTable`只按可见page估核，避免短KV空转。
- PageAttention`key`（BBND/BNBD）允许首轴非连续（`key | gap | key | ...`），tiling通过`GetInputStride`读取dim0元素stride写入`strideKvBlock`。非首轴（size>1的dim1至末维）必须等于紧凑stride，否则返回不支持/非法输入；TND / BSND / BNSD `key`不允许非连续。`scale`仍按逻辑page紧凑布局。
- PageAttention`blockTable`第二维可以大于实际KV逻辑block数；score末维为$\mathrm{RoundUp}(width, 16)$。Ascend 950 C2UB路径对超过256列的末维按256列滑窗flush，并补写后续$-inf$。

## 调用示例

示例代码如下（query为TND，key为BBND PageAttention，经`aclnnMsaIndexScoreV2`调用；TND key：`layoutKeyOptional="TND"`，`key`为$[T2, N2, D]$，`blockTableOptional`传`nullptr`，`actualSeqKlenOptional`为$[B+1]$前缀和。BNBD：`layoutKeyOptional="BNBD"`，`key`为$[block\_num, N2, block\_size, D]$。query改成BSND / BNSD时把`layoutQueryOptional`设成对应名字；非分页key必须用同一个名字。含TND/BNBD的精度自验证见 [test_aclnn_msa_index_score_v2.cpp](../examples/test_aclnn_msa_index_score_v2.cpp)。
```Cpp
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
  int64_t poolBlockSize = 128;  // 打分池化块长；16 走两级 maxpool（仅 950 fp16/bf16）
  aclOpExecutor* executor = nullptr;

  // 非量化：scaleOptional 传 nullptr
  int ret = aclnnMsaIndexScoreV2GetWorkspaceSize(
      resources.queryTensor,
      resources.keyTensor,
      resources.blockTableTensor,
      nullptr,  // scaleOptional
      resources.attenMaskTensor,
      resources.actualSeqQlenTensor,
      resources.actualSeqKlenTensor,
      resources.startLocTensor,
      layoutKey,
      sparseMode,
      initBlocks,
      localBlocks,
      layoutQuery,
      poolBlockSize,
      resources.scoreTensor,
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

TND/BNBD与上面BBND示例的差异如下
```Cpp
// BNBD PageAttention：layoutKey="BNBD"，key 为 [block_num, N2, page_size, D]
char layoutKeyBnbd[] = "BNBD";
std::vector<int64_t> keyShapeBnbd = {blockNum, N2, blockSize, D};

// TND packed key：layoutKey="TND"，不传 blockTableOptional，actualSeqKlenOptional 为 [B+1] 前缀和
char layoutKeyTnd[] = "TND";
constexpr int64_t T2 = 256;
std::vector<int64_t> keyShapeTnd = {T2, N2, D};
std::vector<int64_t> actualSeqKlenShapeTnd = {B + 1};  // 例如 {0, T2}
aclTensor* blockTableTnd = nullptr;

// BBND 页长 16（仅 Ascend 950，fp16/bf16）：key [block_num, 16, N2, D]，必须传 block_table。
// 先块内 max，再 head 间 max。score 为 [1, T1, RoundUp(block_table 第二维, 16)]。query head 数须整除 128。
char layoutKeyBlk16[] = "BBND";
std::vector<int64_t> keyShapeBlk16 = {blockNum, 16, N2, D};

// query BSND + key BBND（仅 Ascend 950）：layoutQuery="BSND"，query [B, S1, N1, D]，key 仍是分页。
char layoutQueryBsnd[] = "BSND";
std::vector<int64_t> queryShapeBsnd = {B, S1, N1, D};

// query 与 key 同为 BNSD（仅 Ascend 950，fp16/bf16，逻辑块长 16，两级 max）：不传 block_table。
char layoutQueryBnsd[] = "BNSD";
char layoutKeyBnsd[] = "BNSD";
std::vector<int64_t> queryShapeBnsd = {B, N1, S1, D};
std::vector<int64_t> keyShapeBnsd = {B, N2, S2, D};
```
