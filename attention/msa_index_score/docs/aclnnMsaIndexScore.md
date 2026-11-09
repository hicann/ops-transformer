# aclnnMsaIndexScore

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

- **接口功能**：计算MSA（MiniMax Sparse Attention）模块Index Branch中的block score。对每个query token与每个KV sparse block，取该block内所有因果可见token的$Q_{idx}$和$K_{idx}$的"matmul+maxpool"运算获得逐block的重要性分数score，用作Index Branch中后续的TopK输入。Prefill与Decode由同一接口承载。

- **计算公式**：

    - 完整公式（含因果mask与local_mask）：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    非量化时没有$scale$。`sparse_mode=3`时$atten\_mask$为右下因果；`sparse_mode=0`时不叠加。$local\_mask$由`start_loc`、`init_blocks`、`local_blocks`生成，对强制保留的block写入高分。

- **主要计算过程**：

    1. 将每个query token的$Q_{idx}$与KV sparse block内因果可见的$K_{idx}$做矩阵乘，得到token级相关性；
    2. 按`sparse_mode`叠加$atten\_mask$：`sparse_mode=0`不做因果截断，`sparse_mode=3`按rightDownCausal屏蔽不可见位置。
    3. 在每个长度为$block\_size$的sparse block上做Maxpool，得到逐block的重要性分数。
    4. 用`start_loc`、`init_blocks`、`local_blocks`生成$local\_mask$，对强制保留的block写入高分，得到输出`score`。本算子不包含TopK。

- **参数介绍**：

    > - B为batch，S为序列长度（$S1$ query，$S2$ key），T为各batch长度累加（$T1$ / $T2$）。
    > - N为头数（$N1$ query，$N2$ key），D为head dim。
    > - PageAttention下$block\_num$为物理page数，$block\_size$为每页token数。

## 函数原型

每个算子分为[两段式接口](https://atomgit.com/cann/ops-transformer/blob/master/docs/zh/context/two_phase_api.md)，必须先调用`aclnnMsaIndexScoreGetWorkspaceSize`接口获取入参并计算所需workspace大小，再调用`aclnnMsaIndexScore`接口执行计算。

```cpp
aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
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
    const aclTensor *score,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
aclnnStatus aclnnMsaIndexScore(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMsaIndexScoreGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 420px">
  <col style="width: 280px">
  <col style="width: 100px">
  <col style="width: 220px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的$Q_{idx}$。</td>
      <td>支持的shape为：TND。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3（$[T1, N1, D]$）</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的$K_{idx}$。</td>
      <td>支持TND $[T2, N2, D]$、BSND $[B, S2, N2, D]$、BNSD $[B, N2, S2, D]$、BBND、BNBD。query仍为TND。BSND / BNSD不传<code>blockTableOptional</code>。PA key允许dim0（物理page）非连续；size&gt;1的非首轴须连续，否则tiling拒绝。TND / BSND / BNSD不允许非连续。</td>
      <td>与query同型</td>
      <td>ND</td>
      <td>3（$[T2, N2, D]$）或4（$[B, S2, N2, D]$、$[B, N2, S2, D]$、$[block\_num, N2, block\_size, D]$、$[block\_num, block\_size, N2, D]$）</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>表示当前传入的使用PageAttention存储的block映射表。</td>
      <td>PageAttention场景下，blockTableOptional需为2维；第二维长度不能小于$maxBlockNumPerSeq$。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2（$[B, S2/block\_size]$）</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>公式中的$scale$，反量化系数。</td>
      <td>非量化与FP8传入<code>nullptr</code>。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3（$[block\_num, N2, block\_size]$、$[block\_num, block\_size, N2]$）或2（$[T2, N2]$）</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>输入</td>
      <td>控制因果可见的 mask。</td>
      <td>仅在sparseMode=3时使用，作为base mask控制causal可见；取值为1代表该位不参与计算，为0代表该位参与计算。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2（$[2048, 2048]$）</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>当传入TND时，该入参必须传入，单调不减（前缀和）。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B+1]$）</td>
    </tr>
    <tr>
      <td>actualSeqKlenOptional</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>TND / BSND / BNSD为单调不减的前缀和$[B+1]$；其中BSND / BNSD每个请求的长度差不超过$S2$。PageAttention为各请求可见$S2$（$[B]$）。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B]$或$[B+1]$）</td>
    </tr>
    <tr>
      <td>startLoc</td>
      <td>输入</td>
      <td>当前query所在逻辑block索引（非token前缀）。</td>
      <td>与<code>initBlocks</code> / <code>localBlocks</code>一起生成$local\_mask$。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1（$[B]$）</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>输入</td>
      <td>key的数据排布。</td>
      <td>取值<code>"TND"</code> / <code>"BSND"</code> / <code>"BNSD"</code> / <code>"BBND"</code> / <code>"BNBD"</code>。不传或空串时默认<code>"BBND"</code>。必须与<code>key</code>实际shape一致。</td>
      <td>CHAR*</td>
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
    </tr>
    <tr>
      <td>initBlocks</td>
      <td>输入</td>
      <td>$local\_mask$强制选中的头部block数。</td>
      <td>对逻辑block $[0, initBlocks)$写入高分$1\mathrm{e}30$。可选，默认$0$；须$\ge 0$且$\le maxBlockNumPerSeq$。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>localBlocks</td>
      <td>输入</td>
      <td>$local\_mask$强制选中的局部窗口长度。</td>
      <td>窗口为$[max(0, startLoc+1-localBlocks), startLoc]$，写入高分$1\mathrm{e}29$（覆盖同位置init）。可选，默认$1$。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>score</td>
      <td>输出</td>
      <td>公式中的$score$。</td>
      <td>逐block的重要性分数；末维为对齐后的逻辑block数。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3（$[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$）</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
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
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
      - query、key：数据类型支持BFLOAT16、FLOAT16。
  <!-- end id8 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：
      - query、key：数据类型除BFLOAT16、FLOAT16外，还支持FLOAT8_E5M2、FLOAT8_E4M3FN。
  <!-- end id7 -->

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

## aclnnMsaIndexScore

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMsaIndexScoreGetWorkspaceSize获取。</td>
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

- 本接口query为TND，逻辑块长128。
- `layoutKeyOptional`为`"BBND"`、`"BNBD"`、`"TND"`、`"BSND"`（$[B, S2, N2, D]$）或`"BNSD"`（$[B, N2, S2, D]$）。不传时默认`"BBND"`。
- PageAttention必须传入`blockTableOptional`。TND / BSND / BNSD不传`blockTableOptional`，`actualSeqKlenOptional`统一为$[B+1]$前缀和。
- `sparseMode`当前仅支持0、3：
    - 为0时，代表defaultMask模式，`attenMaskOptional`传入`nullptr`；
    - 为3时，代表rightDownCausal模式，`attenMaskOptional`必须传入，shape为$[2048, 2048]$，取值为1代表该位不参与计算，为0代表该位参与计算。
- `initBlocks`、`localBlocks`必须$\ge 0$且不超过逻辑block数（PA为`blockTableOptional`第二维；TND为score末维对齐宽度）。两者均为0时跳过$local\_mask$。
- `q_len` / `kv_len`允许为0。空KV的score填$-inf$。
- 本接口不包含TopK。
- 确定性计算：相同输入下输出确定。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <fenv.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_msa_index_score.h"
#include "securec.h"

namespace {

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t NUM_KV_HEADS = 1; // MSA index cache 为单头共享，P0 仅支持 1
constexpr int64_t SCORE_STRIDE_ALIGN = 16;
constexpr float kNegInf = -3.4028234663852886e+38F;
constexpr float kAtol = 1e-3F;
constexpr float kRtol = 1e-3F;
constexpr float kAtolFp8 = 2e-2F;
constexpr float kRtolFp8 = 2e-2F;
constexpr int64_t kSparseModeRightDown = 3;
constexpr uint32_t kBf16Shift = 16;
constexpr uint32_t kPrngMulA = 1664525U;
constexpr uint32_t kPrngAddA = 1013904223U;
constexpr uint32_t kPrngXorShiftA = 16;
constexpr uint32_t kPrngMulB = 2246822519U;
constexpr uint32_t kPrngXorShiftB = 13;
constexpr uint32_t kPrngMod = 20000U;
constexpr float kPrngDiv = 10000.0F;
constexpr float kQueryAmp = 0.5F;
constexpr float kFillCmpScale = 0.5F;
constexpr float kBoostThr = 1.0e28F;
constexpr float kInt8Scale = 64.0F;
constexpr float kDeqScaleBase = 0.01F;
constexpr float kDeqScaleSpan = 0.02F;
constexpr float kKeyStridePoisonFp = 99.0F;
constexpr int8_t kKeyStridePoisonI8 = 90;
constexpr uint32_t kKeySeedOff = 999983U;
constexpr uint32_t kDeqSeedOff = 424242U;
constexpr int64_t kBlockTableMulB = 7;
constexpr int64_t kBlockTableMulK = 3;
constexpr int64_t kTraceExtraTok = 2;
constexpr int64_t kPrevDimOffset = 2; // 从末维向前推 stride 时的偏移
constexpr size_t kLayoutKeyBufSize = 8;
constexpr int kFp8E4M3ExpBits = 4;
constexpr int kFp8E4M3MantissaBits = 3;
constexpr int kFp8E4M3ExpBias = (1 << (kFp8E4M3ExpBits - 1)) - 1;
constexpr float kFp8E4M3MantissaScale = static_cast<float>(1 << kFp8E4M3MantissaBits);
constexpr int kFp8E5M2ExpBits = 5;
constexpr int kFp8E5M2MantissaBits = 2;
constexpr int kFp8E5M2ExpBias = (1 << (kFp8E5M2ExpBits - 1)) - 1;
constexpr float kFp8E5M2MantissaScale = static_cast<float>(1 << kFp8E5M2MantissaBits);
constexpr int kFp8MinNormalExp = 1; // IEEE 次正规数 ldexp 指数为 1-bias
constexpr int kFp8CodeCount = static_cast<int>(std::numeric_limits<uint8_t>::max()) + 1;

enum class KeyLayout {
    BBND = 0,
    BNBD = 1,
    TND = 2,
    BSND = 3,
    BNSD = 4,
};

std::string LayoutKeyName(KeyLayout layout)
{
    switch (layout) {
        case KeyLayout::TND:
            return "TND";
        case KeyLayout::BSND:
            return "BSND";
        case KeyLayout::BNSD:
            return "BNSD";
        case KeyLayout::BNBD:
            return "BNBD";
        case KeyLayout::BBND:
        default:
            return "BBND";
    }
}

bool IsBatchKeyLayout(KeyLayout layout)
{
    return layout == KeyLayout::BSND || layout == KeyLayout::BNSD;
}

struct TestCase {
    std::string name;
    int64_t numQHeads;
    int64_t headDim;
    int64_t numPages;
    std::vector<int32_t> qLen;     // 每个请求的 query 长度
    std::vector<int32_t> kvLen;    // 每个请求可见的 kv 长度
    std::vector<int32_t> startLoc; // 当前 query 所在逻辑 block 索引（local_mask）
    bool useBf16;
    bool useInt8Key; // true: key=int8，scale=[NP,N_kv,P] 或 TND [T2,N2]
    int64_t sparseMode = kSparseModeRightDown;
    KeyLayout keyLayout = KeyLayout::BBND;
    // 0=无；1=float8_e4m3fn；2=float8_e5m2。仅 950，query/key 同型。
    int fp8Kind = 0;
    // PA key dim0 间隔：1=紧凑；>1 时 storage 为 key|gap|key|...（A2/A3 与 950）。
    int64_t keyDim0Gap = 1;
    // PA block_table 第二维；0 表示按 kv_len 取 maxBlocks。>256 覆盖 950 C2UB 滑窗 flush。
    int64_t tableWidth = 0;
};

constexpr float kLocalScoreInit = 1.0e30F;
constexpr float kLocalScoreLocal = 1.0e29F;
constexpr int64_t kInitBlocks = 0;
constexpr int64_t kLocalBlocks = 1;
constexpr int64_t kAttenMaskSize = 2048;

int64_t CeilDivI64(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t RoundUpI64(int64_t a, int64_t b)
{
    return CeilDivI64(a, b) * b;
}

// 简单可复现的伪随机源，取值落在 [-1, 1)。
float PseudoRandom(uint32_t seed)
{
    seed = seed * kPrngMulA + kPrngAddA;
    seed ^= seed >> kPrngXorShiftA;
    seed = seed * kPrngMulB;
    seed ^= seed >> kPrngXorShiftB;
    return static_cast<float>(seed % kPrngMod) / kPrngDiv - 1.0F;
}

// bf16 <-> fp32：截断低 16 位尾数（round-to-nearest-even 对本用例不必要）。
uint16_t FloatToBf16(float v)
{
    uint32_t bits = 0;
    const errno_t ret = memcpy_s(&bits, sizeof(bits), &v, sizeof(v));
    if (ret != 0) {
        return 0;
    }
    return static_cast<uint16_t>(bits >> kBf16Shift);
}

float Bf16ToFloat(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << kBf16Shift;
    float out = 0.0F;
    const errno_t ret = memcpy_s(&out, sizeof(out), &bits, sizeof(bits));
    if (ret != 0) {
        return 0.0F;
    }
    return out;
}

// OCP E4M3FN：exp=15 且 mant=7 为 NaN，其余有限；bias=7。
float Fp8E4M3fnToFloat(uint8_t x)
{
    const uint32_t s = (static_cast<uint32_t>(x) >> 7) & 1U;
    const uint32_t e = (static_cast<uint32_t>(x) >> 3) & 0xFU;
    const uint32_t m = static_cast<uint32_t>(x) & 7U;
    if (e == 0xFU && m == 7U) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float sign = (s != 0U) ? -1.0F : 1.0F;
    if (e == 0U) {
        return sign * std::ldexp(static_cast<float>(m) / kFp8E4M3MantissaScale, kFp8MinNormalExp - kFp8E4M3ExpBias);
    }
    return sign *
           std::ldexp(1.0F + static_cast<float>(m) / kFp8E4M3MantissaScale, static_cast<int>(e) - kFp8E4M3ExpBias);
}

// IEEE-like E5M2：bias=15，exp=31 为 Inf/NaN。
float Fp8E5M2ToFloat(uint8_t x)
{
    const uint32_t s = (static_cast<uint32_t>(x) >> 7) & 1U;
    const uint32_t e = (static_cast<uint32_t>(x) >> 2) & 0x1FU;
    const uint32_t m = static_cast<uint32_t>(x) & 3U;
    const float sign = (s != 0U) ? -1.0F : 1.0F;
    if (e == 0x1FU) {
        return (m == 0U) ? (sign * std::numeric_limits<float>::infinity()) : std::numeric_limits<float>::quiet_NaN();
    }
    if (e == 0U) {
        return sign * std::ldexp(static_cast<float>(m) / kFp8E5M2MantissaScale, kFp8MinNormalExp - kFp8E5M2ExpBias);
    }
    return sign *
           std::ldexp(1.0F + static_cast<float>(m) / kFp8E5M2MantissaScale, static_cast<int>(e) - kFp8E5M2ExpBias);
}

uint8_t FloatToFp8Nearest(float v, bool e5m2)
{
    uint8_t best = 0;
    float bestDiff = std::numeric_limits<float>::infinity();
    for (int i = 0; i < kFp8CodeCount; ++i) {
        const uint8_t b = static_cast<uint8_t>(i);
        const float d = e5m2 ? Fp8E5M2ToFloat(b) : Fp8E4M3fnToFloat(b);
        if (!std::isfinite(d)) {
            continue;
        }
        const float diff = std::fabs(d - v);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = b;
        }
    }
    return best;
}

template <typename T>
std::vector<T> ScatterPaKeyDim0(const std::vector<T>& packed, int64_t numPages, int64_t pageElems, int64_t gap,
                                T poison)
{
    std::vector<T> wide(static_cast<size_t>(numPages * gap * pageElems), poison);
    for (int64_t p = 0; p < numPages; ++p) {
        const auto src = packed.begin() + static_cast<size_t>(p * pageElems);
        const auto dst = wide.begin() + static_cast<size_t>(p * gap * pageElems);
        std::copy(src, src + static_cast<size_t>(pageElems), dst);
    }
    return wide;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - kPrevDimOffset; i >= 0; i--) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

class DeviceBuffer {
public:
    ~DeviceBuffer()
    {
        for (auto* t : tensors_) {
            if (t != nullptr) {
                (void)aclDestroyTensor(t);
            }
        }
        for (auto* p : addrs_) {
            if (p != nullptr) {
                (void)aclrtFree(p);
            }
        }
    }

    template <typename T>
    aclTensor* Create(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype,
                      void** addrOut = nullptr, const std::vector<int64_t>* storageShape = nullptr,
                      const std::vector<int64_t>* stridesIn = nullptr)
    {
        void* devAddr = nullptr;
        const size_t bytes = host.size() * sizeof(T);
        // 空张量（T1=0 / T2=0）仍需要合法 device 指针；aclrtMalloc(0) 会失败。
        const size_t allocBytes = (bytes == 0) ? static_cast<size_t>(32) : bytes;
        if (aclrtMalloc(&devAddr, allocBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        addrs_.push_back(devAddr);
        if (bytes > 0 && aclrtMemcpy(devAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return nullptr;
        }
        const std::vector<int64_t>& stShape = (storageShape != nullptr) ? *storageShape : shape;
        std::vector<int64_t> strides;
        if (stridesIn != nullptr) {
            strides = *stridesIn;
        } else {
            strides.assign(shape.size(), 1);
            for (int64_t i = static_cast<int64_t>(shape.size()) - kPrevDimOffset; i >= 0; i--) {
                strides[static_cast<size_t>(i)] =
                    shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
            }
        }
        aclTensor* t = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                       stShape.data(), stShape.size(), devAddr);
        tensors_.push_back(t);
        if (addrOut != nullptr) {
            *addrOut = devAddr;
        }
        return t;
    }

private:
    std::vector<void*> addrs_;
    std::vector<aclTensor*> tensors_;
};

/// CPU 参考：score = Maxpool[(scale·)Q@Kᵀ + atten_mask] + local_mask
void ComputeGolden(const TestCase& tc, const std::vector<int32_t>& actualSeqQlen,
                   const std::vector<int32_t>& actualSeqKlen, const std::vector<int32_t>& blockTable, int64_t maxBlocks,
                   int64_t scoreStride, int64_t totalQ, const std::vector<float>& queryF,
                   const std::vector<float>& keyF, const std::vector<float>& deqScale, std::vector<float>& golden)
{
    const int64_t batch = static_cast<int64_t>(tc.qLen.size());
    golden.assign(static_cast<size_t>(tc.numQHeads * totalQ * scoreStride), kNegInf);

    auto keyAt = [&](int64_t pageOrTok, int64_t n, int64_t d) -> float {
        // TND 与 N2=1 的 BSND/BNSD：pageOrTok 是块首 token，块内 token 连续。
        if (tc.keyLayout == KeyLayout::TND || IsBatchKeyLayout(tc.keyLayout)) {
            return keyF[static_cast<size_t>(((pageOrTok + n) * NUM_KV_HEADS) * tc.headDim + d)];
        }
        // BBND [NP,P,1,D] 与 BNBD [NP,1,P,D] 在 N2=1 时同址
        return keyF[static_cast<size_t>(((pageOrTok * BLOCK_SIZE) + n) * tc.headDim + d)];
    };
    auto scaleAt = [&](int64_t pageOrTok, int64_t n) -> float {
        if (!tc.useInt8Key) {
            return 1.0F;
        }
        if (tc.keyLayout == KeyLayout::TND) {
            return deqScale[static_cast<size_t>(pageOrTok + n)];
        }
        return deqScale[static_cast<size_t>(pageOrTok * BLOCK_SIZE + n)];
    };

    for (int64_t b = 0; b < batch; ++b) {
        const int32_t qBegin = actualSeqQlen[b];
        const int32_t qEnd = actualSeqQlen[b + 1];
        const int32_t qLen = qEnd - qBegin;
        const int32_t kvLen = tc.kvLen[b];
        const int64_t numBlocks = CeilDivI64(kvLen, BLOCK_SIZE);
        const int32_t qBlock = tc.startLoc[b];
        const int32_t localStart =
            (qBlock + 1 > static_cast<int32_t>(kLocalBlocks)) ? (qBlock + 1 - static_cast<int32_t>(kLocalBlocks)) : 0;
        int64_t batchS2 = 0;
        if (IsBatchKeyLayout(tc.keyLayout)) {
            for (int32_t kv : tc.kvLen) {
                batchS2 = std::max<int64_t>(batchS2, kv);
            }
        }
        const int32_t cuK = (tc.keyLayout == KeyLayout::TND) ? actualSeqKlen[b] : 0;
        const int32_t batchTokBase = IsBatchKeyLayout(tc.keyLayout) ? static_cast<int32_t>(b * batchS2) : cuK;

        for (int32_t t = qBegin; t < qEnd; ++t) {
            const int32_t tOff = t - qBegin;
            int32_t visibleKeyEnd = kvLen;
            if (tc.sparseMode == kSparseModeRightDown) {
                visibleKeyEnd = kvLen - qLen + tOff + 1;
                if (visibleKeyEnd < 0) {
                    visibleKeyEnd = 0;
                }
                if (visibleKeyEnd > kvLen) {
                    visibleKeyEnd = kvLen;
                }
            }
            for (int64_t h = 0; h < tc.numQHeads; ++h) {
                for (int64_t blk = 0; blk < numBlocks; ++blk) {
                    const int32_t pageOrTok = (tc.keyLayout == KeyLayout::TND || IsBatchKeyLayout(tc.keyLayout)) ?
                                                  (batchTokBase + static_cast<int32_t>(blk * BLOCK_SIZE)) :
                                                  blockTable[b * maxBlocks + blk];
                    float best = 0.0F;
                    bool any = false;
                    for (int64_t n = 0; n < BLOCK_SIZE; ++n) {
                        if (blk * BLOCK_SIZE + n >= visibleKeyEnd) {
                            break;
                        }
                        float acc = 0.0F;
                        const float s = scaleAt(pageOrTok, n);
                        for (int64_t d = 0; d < tc.headDim; ++d) {
                            acc += queryF[((t * tc.numQHeads) + h) * tc.headDim + d] * (keyAt(pageOrTok, n, d) * s);
                        }
                        if (!any || acc > best) {
                            best = acc;
                            any = true;
                        }
                    }
                    if (!any) {
                        continue;
                    }
                    golden[(h * totalQ + t) * scoreStride + blk] = best;
                }
                for (int64_t blk = 0; blk < numBlocks; ++blk) {
                    float boost = 0.0F;
                    if (blk < kInitBlocks) {
                        boost = kLocalScoreInit;
                    }
                    if (blk >= localStart && blk <= qBlock) {
                        boost = kLocalScoreLocal;
                    }
                    if (boost != 0.0F) {
                        golden[(h * totalQ + t) * scoreStride + blk] = boost;
                    }
                }
            }
        }
    }
}

void PrintVec(const std::string& tag, const float* data, int64_t n)
{
    (void)printf("  %s[", tag.c_str());
    for (int64_t i = 0; i < n; ++i) {
        (void)printf("%s%.6g", i == 0 ? "" : ", ", static_cast<double>(data[i]));
    }
    (void)printf("]\n");
}

/// 小尺寸用例的 host 侧逐步拆解：Q/K -> DOT/DEQUANT -> MASK -> MAX。
void PrintTracePipeline(const TestCase& tc, const std::vector<int32_t>& actualSeqQlen,
                        const std::vector<int32_t>& blockTable, int64_t maxBlocks, int64_t scoreStride, int64_t totalQ,
                        const std::vector<float>& queryF, const std::vector<float>& keyF,
                        const std::vector<float>& deqScale, const std::vector<float>& actual)
{
    (void)printf("\n======== HOST TRACE: %s (int8=%d) ========\n", tc.name.c_str(), tc.useInt8Key ? 1 : 0);
    (void)printf("shape: Hq=%ld D=%ld qLen=%d kvLen=%d startLoc=%d blockSize=%ld maxBlocks=%ld scoreStride=%ld\n",
                 tc.numQHeads, tc.headDim, tc.qLen[0], tc.kvLen[0], tc.startLoc[0], BLOCK_SIZE, maxBlocks, scoreStride);
    const int32_t page = blockTable[0];
    for (int64_t t = 0; t < tc.qLen[0]; ++t) {
        for (int64_t h = 0; h < tc.numQHeads; ++h) {
            const int64_t flat = t * tc.numQHeads + h;
            const float* q = &queryF[static_cast<size_t>(flat * tc.headDim)];
            (void)printf("\n-- row flat=%ld (token=%ld head=%ld) --\n", flat, t, h);
            PrintVec("Q", q, tc.headDim);
            const int32_t tOff = static_cast<int32_t>(t);
            int32_t visibleKeyEnd = tc.kvLen[0];
            if (tc.sparseMode == kSparseModeRightDown) {
                visibleKeyEnd = tc.kvLen[0] - tc.qLen[0] + tOff + 1;
                if (visibleKeyEnd < 0) {
                    visibleKeyEnd = 0;
                }
                if (visibleKeyEnd > tc.kvLen[0]) {
                    visibleKeyEnd = tc.kvLen[0];
                }
            }
            float best = kNegInf;
            bool any = false;
            for (int64_t n = 0; n < visibleKeyEnd + kTraceExtraTok && n < BLOCK_SIZE; ++n) {
                const float* k = &keyF[static_cast<size_t>((page * BLOCK_SIZE + n) * tc.headDim)];
                const float ds = tc.useInt8Key ? deqScale[static_cast<size_t>(page) * BLOCK_SIZE + n] : 1.0F;
                float acc = 0.0F;
                for (int64_t d = 0; d < tc.headDim; ++d) {
                    acc += q[d] * (k[d] * ds);
                }
                const bool visible = (n < visibleKeyEnd);
                (void)printf("    S[%ld]=%.6g deqScale=%.6g %s\n", n, static_cast<double>(acc), static_cast<double>(ds),
                             visible ? "KEEP" : "MASK");
                if (visible && (!any || acc > best)) {
                    best = acc;
                    any = true;
                }
            }
            const float deviceOut = actual[static_cast<size_t>((h * totalQ + (actualSeqQlen[0] + t)) * scoreStride)];
            (void)printf("  [MAX]=%.6g [OUT]=%.6g\n", static_cast<double>(any ? best : kNegInf),
                         static_cast<double>(deviceOut));
        }
    }
    (void)printf("======== END HOST TRACE ========\n\n");
    (void)maxBlocks;
}

bool Compare(const std::string& name, const std::vector<float>& actual, const std::vector<float>& golden, float atol,
             float rtol)
{
    size_t badCount = 0;
    size_t infBad = 0;
    size_t total = 0;
    float maxAbsDiff = 0.0F;
    size_t firstBad = 0;
    bool hasBad = false;

    for (size_t i = 0; i < golden.size(); ++i) {
        const float g = golden[i];
        const float a = actual[i];
        if (g <= kNegInf * kFillCmpScale) { // 填充位：要求实测同样是极小值
            if (!(std::isfinite(a) && a <= kNegInf * kFillCmpScale) && !(std::isinf(a) && a < 0)) {
                if (!hasBad) {
                    firstBad = i;
                    hasBad = true;
                }
                ++infBad;
            }
            continue;
        }
        if (g >= kBoostThr) { // local_mask 强制高分
            if (!(a >= kBoostThr)) {
                if (!hasBad) {
                    firstBad = i;
                    hasBad = true;
                }
                ++badCount;
            }
            continue;
        }
        ++total;
        if (!std::isfinite(a) || !std::isfinite(g)) {
            if (!hasBad) {
                firstBad = i;
                hasBad = true;
            }
            ++badCount;
            continue;
        }
        const float diff = std::fabs(a - g);
        maxAbsDiff = diff > maxAbsDiff ? diff : maxAbsDiff;
        if (diff > atol + rtol * std::fabs(g)) {
            if (!hasBad) {
                firstBad = i;
                hasBad = true;
            }
            ++badCount;
        }
    }

    const bool pass = (badCount == 0) && (infBad == 0);
    (void)printf("  [%s] valid=%zu mismatch=%zu fill_mismatch=%zu max_abs_diff=%.6g -> %s\n", name.c_str(), total,
                 badCount, infBad, static_cast<double>(maxAbsDiff), pass ? "PASS" : "FAIL");
    if (!pass) {
        (void)printf("    first mismatch at %zu: actual=%g golden=%g\n", firstBad,
                     static_cast<double>(actual[firstBad]), static_cast<double>(golden[firstBad]));
    }
    return pass;
}

bool RunCase(const TestCase& tc, aclrtStream stream)
{
    const int64_t batch = static_cast<int64_t>(tc.qLen.size());
    std::vector<int32_t> actualSeqQlen(batch + 1, 0);
    for (int64_t b = 0; b < batch; ++b) {
        actualSeqQlen[b + 1] = actualSeqQlen[b] + tc.qLen[b];
    }
    const int64_t totalQ = actualSeqQlen[batch];

    int64_t maxBlocks = 1;
    for (int64_t b = 0; b < batch; ++b) {
        maxBlocks = std::max<int64_t>(maxBlocks, CeilDivI64(tc.kvLen[b], BLOCK_SIZE));
    }
    if (tc.tableWidth > maxBlocks) {
        maxBlocks = tc.tableWidth;
    }
    const int64_t scoreStride = RoundUpI64(maxBlocks, SCORE_STRIDE_ALIGN);

    // block_table 故意打乱，验证 paged 间接寻址。
    std::vector<int32_t> blockTable(static_cast<size_t>(batch * maxBlocks), 0);
    if (tc.numPages > 0) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t k = 0; k < maxBlocks; ++k) {
                blockTable[b * maxBlocks + k] =
                    static_cast<int32_t>((b * kBlockTableMulB + k * kBlockTableMulK + 1) % tc.numPages);
            }
        }
    }

    // fp32 参考数据；再按 dtype 转成低精度输入，golden 用转换后的值以对齐数值路径。
    const bool isTnd = (tc.keyLayout == KeyLayout::TND);
    const bool isBatch = IsBatchKeyLayout(tc.keyLayout);
    int64_t totalK = 0;
    int64_t batchS2 = 0;
    std::vector<int32_t> actualSeqKlenPrefix(static_cast<size_t>(batch + 1), 0);
    if (isTnd || isBatch) {
        for (int64_t b = 0; b < batch; ++b) {
            actualSeqKlenPrefix[b + 1] = actualSeqKlenPrefix[b] + tc.kvLen[b];
        }
    }
    if (isTnd) {
        totalK = actualSeqKlenPrefix[batch];
    } else if (isBatch) {
        for (int32_t kv : tc.kvLen) {
            batchS2 = std::max<int64_t>(batchS2, kv);
        }
        totalK = batch * batchS2;
    }
    const int64_t keyTokens = (isTnd || isBatch) ? totalK : (tc.numPages * BLOCK_SIZE);
    std::vector<float> queryF(static_cast<size_t>(totalQ * tc.numQHeads * tc.headDim));
    std::vector<float> keyF(static_cast<size_t>(keyTokens * tc.headDim));
    std::vector<uint16_t> queryBf(queryF.size());
    std::vector<uint16_t> keyBf(keyF.size());
    std::vector<aclFloat16> queryHf(queryF.size());
    std::vector<aclFloat16> keyHf(keyF.size());
    std::vector<uint8_t> queryFp8(queryF.size());
    std::vector<uint8_t> keyFp8(keyF.size());
    const bool isFp8 = (tc.fp8Kind != 0);
    const bool isE5M2 = (tc.fp8Kind == 2);

    for (size_t i = 0; i < queryF.size(); ++i) {
        const float v = PseudoRandom(static_cast<uint32_t>(i) + 1U) * kQueryAmp;
        if (isFp8) {
            queryFp8[i] = FloatToFp8Nearest(v, isE5M2);
            queryF[i] = isE5M2 ? Fp8E5M2ToFloat(queryFp8[i]) : Fp8E4M3fnToFloat(queryFp8[i]);
        } else if (tc.useBf16) {
            queryBf[i] = FloatToBf16(v);
            queryF[i] = Bf16ToFloat(queryBf[i]);
        } else {
            queryHf[i] = aclFloatToFloat16(v);
            queryF[i] = aclFloat16ToFloat(queryHf[i]);
        }
    }
    for (size_t i = 0; i < keyF.size(); ++i) {
        if (tc.useInt8Key) {
            // int8 量化值：落在 [-64, 63]，golden 用同一整数值的 fp32。
            const int8_t qv = static_cast<int8_t>(
                static_cast<int>(PseudoRandom(static_cast<uint32_t>(i) + kKeySeedOff) * kInt8Scale));
            keyF[i] = static_cast<float>(qv);
        } else {
            const float v = PseudoRandom(static_cast<uint32_t>(i) + kKeySeedOff) * kQueryAmp;
            if (isFp8) {
                keyFp8[i] = FloatToFp8Nearest(v, isE5M2);
                keyF[i] = isE5M2 ? Fp8E5M2ToFloat(keyFp8[i]) : Fp8E4M3fnToFloat(keyFp8[i]);
            } else if (tc.useBf16) {
                keyBf[i] = FloatToBf16(v);
                keyF[i] = Bf16ToFloat(keyBf[i]);
            } else {
                keyHf[i] = aclFloatToFloat16(v);
                keyF[i] = aclFloat16ToFloat(keyHf[i]);
            }
        }
    }
    // 反量化 scale：PA [NP, N_kv=1, P]；TND [T2, N2=1]。
    std::vector<float> deqScale(static_cast<size_t>(keyTokens), 1.0F);
    std::vector<int8_t> keyI8;
    if (tc.useInt8Key) {
        keyI8.resize(keyF.size());
        for (size_t i = 0; i < keyF.size(); ++i) {
            keyI8[i] = static_cast<int8_t>(keyF[i]);
        }
        for (size_t i = 0; i < deqScale.size(); ++i) {
            deqScale[i] = kDeqScaleBase + kDeqScaleSpan * (PseudoRandom(static_cast<uint32_t>(i) + kDeqSeedOff) + 1.0F);
        }
    }

    DeviceBuffer buf;
    const std::vector<int64_t> queryShape = {totalQ, tc.numQHeads, tc.headDim};
    std::vector<int64_t> keyShape;
    if (isTnd) {
        keyShape = {totalK, NUM_KV_HEADS, tc.headDim};
    } else if (tc.keyLayout == KeyLayout::BSND) {
        keyShape = {batch, batchS2, NUM_KV_HEADS, tc.headDim};
    } else if (tc.keyLayout == KeyLayout::BNSD) {
        keyShape = {batch, NUM_KV_HEADS, batchS2, tc.headDim};
    } else if (tc.keyLayout == KeyLayout::BNBD) {
        keyShape = {tc.numPages, NUM_KV_HEADS, BLOCK_SIZE, tc.headDim};
    } else {
        keyShape = {tc.numPages, BLOCK_SIZE, NUM_KV_HEADS, tc.headDim};
    }
    const std::vector<int64_t> scoreShape = {tc.numQHeads, totalQ, scoreStride};

    aclTensor* queryT = nullptr;
    if (isFp8) {
        const aclDataType fp8Dt = isE5M2 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN;
        queryT = buf.Create(queryFp8, queryShape, fp8Dt);
    } else {
        queryT = tc.useBf16 ? buf.Create(queryBf, queryShape, ACL_BF16) : buf.Create(queryHf, queryShape, ACL_FLOAT16);
    }

    const int64_t keyDim0Gap = (isTnd || isBatch || tc.keyDim0Gap < 1) ? 1 : tc.keyDim0Gap;
    std::vector<int64_t> keyStorageShape = keyShape;
    std::vector<int64_t> keyStrides = ContiguousStrides(keyShape);
    const std::vector<int64_t>* keyStoragePtr = nullptr;
    const std::vector<int64_t>* keyStridePtr = nullptr;
    int64_t pageElems = 1;
    if (keyDim0Gap > 1) {
        for (size_t d = 1; d < keyShape.size(); ++d) {
            pageElems *= keyShape[d];
        }
        keyStorageShape[0] = tc.numPages * keyDim0Gap;
        keyStrides[0] = keyDim0Gap * keyStrides[0];
        keyStoragePtr = &keyStorageShape;
        keyStridePtr = &keyStrides;
    }

    aclTensor* keyT = nullptr;
    if (tc.useInt8Key) {
        if (keyDim0Gap > 1) {
            const auto wide = ScatterPaKeyDim0(keyI8, tc.numPages, pageElems, keyDim0Gap, kKeyStridePoisonI8);
            keyT = buf.Create(wide, keyShape, ACL_INT8, nullptr, keyStoragePtr, keyStridePtr);
        } else {
            keyT = buf.Create(keyI8, keyShape, ACL_INT8);
        }
    } else if (isFp8) {
        const aclDataType fp8Dt = isE5M2 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN;
        if (keyDim0Gap > 1) {
            const uint8_t poison = FloatToFp8Nearest(kKeyStridePoisonFp, isE5M2);
            const auto wide = ScatterPaKeyDim0(keyFp8, tc.numPages, pageElems, keyDim0Gap, poison);
            keyT = buf.Create(wide, keyShape, fp8Dt, nullptr, keyStoragePtr, keyStridePtr);
        } else {
            keyT = buf.Create(keyFp8, keyShape, fp8Dt);
        }
    } else if (tc.useBf16) {
        if (keyDim0Gap > 1) {
            const auto wide =
                ScatterPaKeyDim0(keyBf, tc.numPages, pageElems, keyDim0Gap, FloatToBf16(kKeyStridePoisonFp));
            keyT = buf.Create(wide, keyShape, ACL_BF16, nullptr, keyStoragePtr, keyStridePtr);
        } else {
            keyT = buf.Create(keyBf, keyShape, ACL_BF16);
        }
    } else {
        if (keyDim0Gap > 1) {
            const auto wide =
                ScatterPaKeyDim0(keyHf, tc.numPages, pageElems, keyDim0Gap, aclFloatToFloat16(kKeyStridePoisonFp));
            keyT = buf.Create(wide, keyShape, ACL_FLOAT16, nullptr, keyStoragePtr, keyStridePtr);
        } else {
            keyT = buf.Create(keyHf, keyShape, ACL_FLOAT16);
        }
    }
    aclTensor* blockTableT = nullptr;
    if (!isTnd && !isBatch) {
        blockTableT = buf.Create(blockTable, {batch, maxBlocks}, ACL_INT32);
    }
    aclTensor* scaleT = nullptr;
    if (tc.useInt8Key) {
        if (isTnd) {
            scaleT = buf.Create(deqScale, {totalK, NUM_KV_HEADS}, ACL_FLOAT);
        } else {
            scaleT = buf.Create(deqScale, {tc.numPages, NUM_KV_HEADS, BLOCK_SIZE}, ACL_FLOAT);
        }
    }
    aclTensor* actualSeqQlenT = buf.Create(actualSeqQlen, {batch + 1}, ACL_INT32);
    aclTensor* actualSeqKlenT = (isTnd || isBatch) ? buf.Create(actualSeqKlenPrefix, {batch + 1}, ACL_INT32) :
                                                      buf.Create(tc.kvLen, {batch}, ACL_INT32);
    aclTensor* startLocT = buf.Create(tc.startLoc, {batch}, ACL_INT32);
    std::vector<int8_t> attenMaskHost(static_cast<size_t>(kAttenMaskSize * kAttenMaskSize), 0);
    aclTensor* attenMaskT = nullptr;
    if (tc.sparseMode == kSparseModeRightDown) {
        attenMaskT = buf.Create(attenMaskHost, {kAttenMaskSize, kAttenMaskSize}, ACL_INT8);
    }

    std::vector<float> scoreInit(static_cast<size_t>(tc.numQHeads * totalQ * scoreStride), 0.0F);
    void* scoreDev = nullptr;
    aclTensor* scoreT = buf.Create(scoreInit, scoreShape, ACL_FLOAT, &scoreDev);

    if (queryT == nullptr || keyT == nullptr || actualSeqQlenT == nullptr || actualSeqKlenT == nullptr ||
        startLocT == nullptr || scoreT == nullptr || ((!isTnd && !isBatch) && blockTableT == nullptr) ||
        (tc.sparseMode == kSparseModeRightDown && attenMaskT == nullptr)) {
        (void)printf("  [%s] create tensor failed -> FAIL\n", tc.name.c_str());
        return false;
    }

    const std::string layoutKey = LayoutKeyName(tc.keyLayout);
    char layoutKeyBuf[kLayoutKeyBufSize] = {};
    const errno_t cpyRet = memcpy_s(layoutKeyBuf, sizeof(layoutKeyBuf), layoutKey.c_str(), layoutKey.size() + 1);
    if (cpyRet != 0) {
        (void)printf("  [%s] memcpy_s layout_key failed -> FAIL\n", tc.name.c_str());
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    int ret = aclnnMsaIndexScoreGetWorkspaceSize(queryT, keyT, blockTableT, scaleT, attenMaskT, actualSeqQlenT,
                                                 actualSeqKlenT, startLocT, layoutKeyBuf, tc.sparseMode, kInitBlocks,
                                                 kLocalBlocks, scoreT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        (void)printf("  [%s] GetWorkspaceSize failed, ERROR %d -> FAIL\n", tc.name.c_str(), ret);
        return false;
    }
    if (workspaceSize > 0ULL && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        (void)printf("  [%s] malloc workspace failed -> FAIL\n", tc.name.c_str());
        return false;
    }
    ret = aclnnMsaIndexScore(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        (void)printf("  [%s] aclnnMsaIndexScore failed, ERROR %d -> FAIL\n", tc.name.c_str(), ret);
        (void)aclrtFree(workspaceAddr);
        return false;
    }
    (void)printf("  [%s] kernel launched, synchronizing...\n", tc.name.c_str());
    (void)aclrtSynchronizeStream(stream);
    (void)printf("  [%s] synchronized\n", tc.name.c_str());

    std::vector<float> actual(scoreInit.size(), 0.0F);
    (void)aclrtMemcpy(actual.data(), actual.size() * sizeof(float), scoreDev, actual.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }

    std::vector<float> golden;
    ComputeGolden(tc, actualSeqQlen, (isTnd || isBatch) ? actualSeqKlenPrefix : tc.kvLen, blockTable, maxBlocks,
                  scoreStride, totalQ, queryF, keyF, deqScale, golden);
    if (tc.name.find("debug-trace") != std::string::npos) {
        PrintTracePipeline(tc, actualSeqQlen, blockTable, maxBlocks, scoreStride, totalQ, queryF, keyF, deqScale,
                           actual);
    }
    return Compare(tc.name, actual, golden, isFp8 ? kAtolFp8 : kAtol, isFp8 ? kRtolFp8 : kRtol);
}

} // namespace

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(__linux__)
    (void)fedisableexcept(FE_ALL_EXCEPT);
    fenv_t fenvHold;
    (void)feholdexcept(&fenvHold);
#endif
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        (void)printf("[FAIL] init acl failed\n");
        return -1;
    }

    // startLoc 为逻辑 block 索引；因果由 sparseMode=3（rightDownCausal）承担。
    const std::vector<TestCase> cases = {
        // name                       Hq   D    pages  qLen           kvLen         startLoc(block) bf16  int8
        {"L0-debug-trace", 2, 16, 2, {2}, {5}, {16}, false, false},
        {"L0-prefill-aligned", 8, 128, 8, {32}, {256}, {1}, false, false},
        {"L1-prefill-unaligned", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, false, false},
        {"L1-prefill-multi-mtile", 8, 128, 16, {64, 48}, {700, 520}, {4, 3}, false, false},
        {"L1-decode-lq1",
         8,
         128,
         16,
         {1, 1, 1, 1, 1, 1},
         {900, 512, 128, 129, 1, 4096},
         {7, 3, 0, 1, 0, 31},
         false,
         false},
        {"L1-decode-speculative", 8, 128, 16, {4, 2}, {1024, 260}, {7, 2}, false, false},
        {"L1-long-seq-multi-stile", 8, 128, 40, {8}, {4096}, {31}, false, false},
        {"L1-bf16", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, true, false},
        {"L1-head-dim-64", 8, 64, 8, {32, 17}, {300, 130}, {2, 0}, false, false},
        {"L1-heads-16", 16, 128, 8, {13}, {300}, {2}, false, false},
        {"L2-tiny-kv", 8, 128, 4, {1, 1}, {1, 3}, {0, 0}, false, false},
        {"L1-bnbd", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, false, false, kSparseModeRightDown, KeyLayout::BNBD},
        {"L1-tnd-unaligned",
         8,
         128,
         0,
         {32, 17},
         {300, 130},
         {2, 0},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::TND},
        {"L0-tnd-tiny", 2, 16, 0, {2}, {5}, {16}, false, false, kSparseModeRightDown, KeyLayout::TND},
        {"L0-bsnd-tiny", 2, 16, 0, {2}, {5}, {0}, false, false, kSparseModeRightDown, KeyLayout::BSND},
        {"L1-bsnd", 8, 128, 0, {32, 17}, {300, 130}, {2, 0}, false, false, kSparseModeRightDown, KeyLayout::BSND},
        {"L1-bnsd", 8, 128, 0, {32, 17}, {300, 130}, {2, 0}, false, false, kSparseModeRightDown, KeyLayout::BNSD},
        {"L0-bnsd-bf16", 2, 16, 0, {2}, {5}, {0}, true, false, kSparseModeRightDown, KeyLayout::BNSD},
        // FP8：C0=32，headDim 用 128 对齐；D=16 半个 C0 数值不可靠。
        {"L0-fp8-e4m3fn", 2, 128, 2, {2}, {5}, {16}, false, false, kSparseModeRightDown, KeyLayout::BBND, 1},
        {"L0-fp8-e5m2", 2, 128, 2, {2}, {5}, {16}, false, false, kSparseModeRightDown, KeyLayout::BBND, 2},
        {"L1-fp8-e4m3fn-prefill", 8, 128, 8, {32}, {256}, {1}, false, false, kSparseModeRightDown, KeyLayout::BBND, 1},
        // mixed-batch 零长度：部分请求 pad。整 batch 全 0 见文末 L0-all-* / L0-tnd-all-*。
        {"L1-pad-q0", 8, 128, 8, {0, 32}, {256, 256}, {1, 1}, false, false},
        {"L1-pad-kv0", 8, 128, 8, {32, 32}, {0, 256}, {0, 1}, false, false},
        {"L1-pad-q0-kv0", 8, 128, 8, {0, 32}, {0, 256}, {0, 1}, false, false},
        {"L1-pad-mid-q0", 8, 128, 8, {16, 0, 16}, {200, 0, 200}, {1, 0, 1}, false, false},
        {"L1-tnd-pad-q0-kv0", 8, 128, 0, {0, 32}, {0, 256}, {0, 1}, false, false, kSparseModeRightDown, KeyLayout::TND},
        // 整 batch 空序列：host 跳过计算（A2/A3 与 950）。
        {"L0-all-q0", 8, 128, 8, {0}, {256}, {1}, false, false},
        {"L0-all-kv0", 8, 128, 8, {32}, {0}, {0}, false, false},
        {"L0-all-q0-kv0", 8, 128, 8, {0}, {0}, {0}, false, false},
        {"L1-all-q0", 8, 128, 8, {0, 0}, {256, 256}, {1, 1}, false, false},
        {"L0-tnd-all-q0", 8, 128, 0, {0}, {256}, {1}, false, false, kSparseModeRightDown, KeyLayout::TND},
        {"L0-tnd-all-kv0", 8, 128, 0, {32}, {0}, {0}, false, false, kSparseModeRightDown, KeyLayout::TND},
        {"L0-tnd-all-q0-kv0", 8, 128, 0, {0}, {0}, {0}, false, false, kSparseModeRightDown, KeyLayout::TND},
        // PA key dim0 非连续（A2/A3 与 950）：间隔槽下毒，错 stride 会读到 poison。
        {"L0-stride-bbnd", 2, 16, 2, {2}, {5}, {16}, false, false, kSparseModeRightDown, KeyLayout::BBND, 0, 2},
        {"L1-stride-bbnd",
         8,
         128,
         8,
         {32, 17},
         {300, 130},
         {2, 0},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         0,
         2},
        {"L1-stride-bnbd",
         8,
         128,
         8,
         {32, 17},
         {300, 130},
         {2, 0},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BNBD,
         0,
         2},
        // block_table 宽 257：score 末维 RoundUp=272 > UB 单窗 256，950 C2UB 滑窗 flush。
        {"L0-wide-table-257", 2, 16, 2, {2}, {5}, {16}, false, false, kSparseModeRightDown, KeyLayout::BBND, 0, 1, 257},
        {"L0-fp8-wide-table-257",
         2,
         128,
         2,
         {2},
         {5},
         {16},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         1,
         1,
         257},
        {"L1-wide-table-257-bf16",
         8,
         128,
         8,
         {32},
         {256},
         {1},
         true,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         0,
         1,
         257},
        // 短 decode：少量 M-tile + 可选宽 block_table（对齐 vLLM decode 类输入）。
        {"L0-decode-q4-kv4", 4, 128, 4, {4}, {4}, {16}, false, false},
        {"L0-decode-q4-kv4-b2", 4, 128, 8, {4, 4}, {4, 8}, {16, 16}, false, false},
        // vLLM A5 decode 类：T1=4 Hq=4、275 可见 page（跨 C2UB 256 列窗 + 950 S-split）。
        {"L0-decode-q4-kv275", 4, 128, 275, {4}, {35200}, {274}, false, false},
        {"L0-decode-q4-kv4-table275",
         4,
         128,
         4,
         {4},
         {4},
         {16},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         0,
         1,
         275},
        {"L0-fp8-decode-q4-kv4", 4, 128, 4, {4}, {4}, {16}, false, false, kSparseModeRightDown, KeyLayout::BBND, 1},
        {"L0-fp8-decode-q4-kv4-table275",
         4,
         128,
         4,
         {4},
         {4},
         {16},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         1,
         1,
         275},
        {"L0-fp8-decode-q1-kv4-table275",
         4,
         128,
         4,
         {1},
         {4},
         {16},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         1,
         1,
         275},
        {"L0-fp8-decode-q4-kv128-table275",
         4,
         128,
         4,
         {4},
         {128},
         {16},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         1,
         1,
         275},
        {"L0-fp8-decode-q4-kv275",
         4,
         128,
         275,
         {4},
         {35200},
         {274},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         1},
        {"L0-fp8-e5m2-decode-q4-kv275",
         4,
         128,
         275,
         {4},
         {35200},
         {274},
         false,
         false,
         kSparseModeRightDown,
         KeyLayout::BBND,
         2},
    };

    size_t passed = 0;
    size_t ran = 0;
    size_t skipped = 0;
    const char* socName = aclrtGetSocName();
    const std::string soc = (socName == nullptr) ? "" : socName;
    const bool isAscend950 = (soc.find("950") != std::string::npos) || (soc.find("910_95") != std::string::npos);
    (void)printf("running %zu MsaIndexScore cases (soc=%s)\n", cases.size(), soc.empty() ? "?" : soc.c_str());
    for (const auto& tc : cases) {
        if (tc.fp8Kind != 0 && !isAscend950) {
            (void)printf("  [%s] SKIP: FP8 is Ascend 950 only\n", tc.name.c_str());
            ++skipped;
            continue;
        }
        ++ran;
        if (RunCase(tc, stream)) {
            ++passed;
        }
    }
    const std::string extra = (skipped == 0) ? "" : (" (skipped " + std::to_string(skipped) + ")");
    (void)printf("%s: %zu/%zu cases passed%s\n", passed == ran ? "[PASS]" : "[FAIL]", passed, ran, extra.c_str());

    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return passed == ran ? 0 : -1;
}
```
