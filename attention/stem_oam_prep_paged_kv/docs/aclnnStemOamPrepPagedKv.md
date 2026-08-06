# aclnnStemOamPrepPagedKv

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/stem_oam_prep_paged_kv)

## 产品支持情况

<!-- npu="950" id1 -->
<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->

<!-- end id1 -->
<!-- npu="A3" id2 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->

<!-- end id2 -->
<!-- npu="910b" id3 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->

<!-- end id3 -->
<!-- npu="310b" id4 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->

<!-- end id4 -->
<!-- npu="310p" id5 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->

<!-- end id5 -->
<!-- npu="910" id6 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：Stem OAM (Output-Aware Metric) 大模型推理动态稀疏注意力机制的前置评分模块。从 Paged KV Cache 中提取 K/V 数据，经 K Processing (per-token K-scale × group sum + anti-diagonal flip) 和 V Processing (per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average) 计算，输出 kFlat 和 vBias 供 Stem OAM score computation 使用。

- 计算公式：
阶段1：De-page + Cast FP32 + kScaleCache Gather

$$
K\_dense[b], V\_dense[b], kscale\_per\_row[b] = Cast\big(gather(kcache[kv\_indices[b]]),\ FP32\big),\ Cast\big(gather(vcache[kv\_indices[b]]),\ FP32\big),\ gather(kScaleCache[kv\_indices[b]])
$$

$$
k\_padded[b] = \lceil kv\_len[b] / B \rceil \times B, \quad num\_Kb[b] = k\_padded[b] / B, \quad R = B / S
$$

$$
K\_scaled[b, h] = K\_dense[b, h, :] \times kscale\_per\_row[b, h, :] \quad (\text{per-token per-head, broadcast on dim})
$$

阶段2：K Processing(Weighted Group Sum + Anti-diagonal Flip)

$$
K\_group\_sum[b,h,kb,g,:] = \sum_{r=0}^{R-1} K\_blocks[b,h,kb,r,g,:]
$$

$$
K\_group\_rev[b,h,kb,g',:] = K\_group\_sum[b,h,kb,\ S-1-g',:] \quad (\text{anti-diagonal flip})
$$

$$
kFlat[b,h,kb,:] = reshape(K\_group\_rev[b,h,kb,:,:],\ [kflat\_dim = S \times D]) \xrightarrow{Cast} BF16
$$

阶段3：V Processing

**Step 3a: L2 Norm + Max Pool + Log**

$$
norms[b,h,idx,s,:] = \| V\_rows[b,h,idx,s,:] \times vScale[h] \|_2 =  \sqrt{\sum_d (V\_rows[idx,s,d] \times vScale[h])^2}
$$

$$
v\_norm\_down[b,h,idx] = \max_{s} norms[b,h,idx,s,:] \quad (\text{zero beyond } kv\_len)
$$

$$
log\_vals[b,h,idx] = \log(v\_norm\_down[b,h,idx] + \epsilon), \quad \epsilon = 10^{-6}
$$

**Step 3b: Global Normalize (μ/σ over ALL k_down_len values)**

$$
\mu[b,h] = \frac{1}{k\_down\_len} \sum_{idx=0}^{k\_down\_len-1} log\_vals[b,h,idx]
$$

$$
\sigma[b,h] = \sqrt{\frac{\sum(log\_vals - \mu)^2}{k\_down\_len - 1}} \quad (k\_down\_len > 1);\quad \sigma = 0 \quad (k\_down\_len = 1)
$$

**Step 3c: Normalize + ReLU + Block Average**

$$
normalized[b,h,idx] = \frac{log\_vals[b,h,idx] - \mu[b,h]}{\sigma[b,h] + \epsilon}
$$

$$
v\_final[b,h,idx] = \lambda \times ReLU(normalized[b,h,idx])
$$

$$
vBias[b,h,kb] = \frac{1}{R} \sum_{r=0}^{R-1} v\_final[b,h,\ kb \times R + r]
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnStemOamPrepPagedKvGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnStemOamPrepPagedKv"接口执行计算。

```Cpp
aclnnStatus aclnnStemOamPrepPagedKvGetWorkspaceSize(
  const aclTensor  *kCache,
  const aclTensor  *vCache,
  const aclTensor  *kvIndices,
  const aclIntArray  *kvSeqLens,
  const aclTensor  *kScaleCache,
  const aclTensor  *vScale,
  double            lambdaMag,
  const char        *kvLayout,
  int64_t           stemBlockSize,
  int64_t           stemStride,
  const aclTensor  *kFlat,
  const aclTensor  *vBias,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnStemOamPrepPagedKv(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnStemOamPrepPagedKvGetWorkspaceSize

- **参数说明**

  <table class="tg" style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 270px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 115px">
  <col style="width: 135px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">kCache（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Paged K cache。</td>
      <td class="tg-0pky">不支持空Tensor。<br>两种布局, 由kvLayout指定, 当前仅支持"BNBD":<br>"BBND"为[total_blocks, kvBlockSize, H_kv, D=128], <br>"BNBD"为[total_blocks, H_kv, kvBlockSize, D=128]。<br>支持前三维非连续，最后一维必须连续。</td>
      <td class="tg-0pky">FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vCache（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Paged V cache。</td>
      <td class="tg-0pky">不支持空Tensor。<br>布局与kCache保持一致。</td>
      <td class="tg-0pky">与kCache保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">kvIndices（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">每batch KV Block index数组</td>
      <td class="tg-0pky">不支持空Tensor。<br>shape:[batch, max_kv_blocks], max_kv_blocks最大值2048</td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">kvSeqLens（aclIntArray*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">每batch KV序列长度。</td>
      <td class="tg-0pky">不支持空列表。<br>shape:[batch], kv序列长度最大值262144</td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">kScaleCacheOptional（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Per-token per-head K scale。</td>
      <td class="tg-0pky">kCache数据类型为FP8时必填，其他类型可省略（传nullptr）。<br>两种布局, 由kvLayout指定, 当前仅支持"BNBD":<br>"BBND"为[total_blocks, kvBlockSize, H_kv, 1],<br>"BNBD"为[total_blocks, H_kv, kvBlockSize, 1]。<br>支持前三维非连续，最后一维必须连续。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vScaleOptional（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Per-head V scale。</td>
      <td class="tg-0pky">kCache数据类型为FP8时必填，其他类型可省略（传nullptr）。<br>shape:[H_kv]</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">lambdaMag</td>
      <td class="tg-0pky">ATTR</td>
      <td class="tg-0pky">V bias 乘数</td>
      <td class="tg-0pky">取值范围: (0,1]。</td>
      <td class="tg-0pky">double</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">kvLayout</td>
      <td class="tg-0pky">ATTR</td>
      <td class="tg-0pky">KV Cache布局</td>
      <td class="tg-0pky">取值范围: "BBND"、"BNBD"。当前仅支持"BNBD"</td>
      <td class="tg-0pky">char*</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">stemBlockSize</td>
      <td class="tg-0lax">ATTR</td>
      <td class="tg-0lax">Stem block大小</td>
      <td class="tg-0lax">取值范围: %32==0, ≤256, 且stemBlockSize必须是stemStride的整数倍，推荐128。</td>
      <td class="tg-0lax">int64_t</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">stemStride</td>
      <td class="tg-0lax">ATTR</td>
      <td class="tg-0lax">Stride大小</td>
      <td class="tg-0lax">取值范围: %16==0，≤64，≤stemBlockSize，推荐16。</td>
      <td class="tg-0lax">int64_t</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">kFlat（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">K group sum + anti-diag flip 结果。</td>
      <td class="tg-0lax">不支持空Tensor。<br>shape为[batch, H_kv, max_Kb, kflat_dim]，其中kflat_dim=stemStride*D, <br>max_Kb依赖kvSeqLens输入计算获得。</td>
      <td class="tg-0lax">BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">4</td>
      <td class="tg-0lax">×</td>
    </tr>
    <tr>
      <td class="tg-0lax">vBias（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">V block bias结果。</td>
      <td class="tg-0lax">不支持空Tensor。<br>shape为[batch, H_kv, max_Kb], max_Kb依赖kvSeqLens输入计算获得。</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">3</td>
      <td class="tg-0lax">×</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize（uint64_t*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor（aclOpExecutor**）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的kCache、vCache、kFlat、vBias是空指针。<br>kCache数据类型为FP8时kScaleCache、vScale空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>kCache dtype不为FLOAT8_E4M3FN。</td>
    </tr>
    <tr>
      <td>kScaleCache dtype不为FLOAT。</td>
    </tr>
    <tr>
      <td>vScale shape不是1D [H_kv]。</td>
    </tr>
    <tr>
      <td>kvLayout/stemBlockSize/stemStride不满足约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER</td>
      <td>361001</td>
      <td>内部错误，如Tiling计算失败或Kernel查找失败。</td>
    </tr>
  </tbody>
  </table>

## aclnnStemOamPrepPagedKv

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnStemOamPrepPagedKvGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- kvBlockSize ∈ {64, 128}。
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include <cstring>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_stem_oam_prep_paged_kv.h"

  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

  #define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                      void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t batch = 1;
    int64_t totalBlocks = 8;
    int64_t numKvHeads = 4;
    int64_t dimQk = 128;
    int64_t maxKvBlocks = 2;
    int64_t stemBlockSize = 128;
    int64_t stemStride = 16;
    int64_t maxKb = 2;
    int64_t kflatDim = stemStride * dimQk;
    const char *kvLayout = "BNBD";

    std::vector<int64_t> kCacheShape = {totalBlocks, numKvHeads, 64, dimQk};
    std::vector<int64_t> kvIndicesShape = {batch, maxKvBlocks};
    std::vector<int64_t> kScaleCacheShape = {totalBlocks, numKvHeads, 64, 1};
    std::vector<int64_t> vScaleShape = {numKvHeads};
    std::vector<int64_t> kFlatShape = {batch, numKvHeads, maxKb, kflatDim};
    std::vector<int64_t> vBiasShape = {batch, numKvHeads, maxKb};

    void* kCacheDeviceAddr = nullptr;
    void* vCacheDeviceAddr = nullptr;
    void* kvIndicesDeviceAddr = nullptr;
    void* kScaleCacheDeviceAddr = nullptr;
    void* vScaleDeviceAddr = nullptr;
    void* kFlatDeviceAddr = nullptr;
    void* vBiasDeviceAddr = nullptr;

    aclTensor* kCache = nullptr;
    aclTensor* vCache = nullptr;
    aclTensor* kvIndices = nullptr;
    aclIntArray* kvSeqLens = nullptr;
    aclTensor* kScaleCache = nullptr;
    aclTensor* vScale = nullptr;
    aclTensor* kFlat = nullptr;
    aclTensor* vBias = nullptr;

    std::vector<uint8_t> hostKCache(GetShapeSize(kCacheShape), 1);
    std::vector<uint8_t> hostVCache(GetShapeSize(kCacheShape), 1);
    std::vector<int32_t> hostKvIndices({0, 1, 2, 3, 4, 5, 6, 7});
    std::vector<int64_t> hostKvSeqLens({128});
    std::vector<float> hostKScaleCache(GetShapeSize(kScaleCacheShape), 1.0f);
    std::vector<float> hostVScale(numKvHeads, 1.0f);
    std::vector<uint16_t> hostKFlat(GetShapeSize(kFlatShape), 0);
    std::vector<float> hostVBias(GetShapeSize(vBiasShape), 0.0f);

    ret = CreateAclTensor(hostKCache, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &kCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVCache, kCacheShape, &vCacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &vCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKvIndices, kvIndicesShape, &kvIndicesDeviceAddr, aclDataType::ACL_INT32, &kvIndices, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    kvSeqLens = aclCreateIntArray(hostKvSeqLens.data(), hostKvSeqLens.size());
    ret = CreateAclTensor(hostKScaleCache, kScaleCacheShape, &kScaleCacheDeviceAddr, aclDataType::ACL_FLOAT, &kScaleCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVScale, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKFlat, kFlatShape, &kFlatDeviceAddr, aclDataType::ACL_BF16, &kFlat, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostVBias, vBiasShape, &vBiasDeviceAddr, aclDataType::ACL_FLOAT, &vBias, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnStemOamPrepPagedKvGetWorkspaceSize(
        kCache, vCache, kvIndices, kvSeqLens, kScaleCache, vScale,
        0.3, kvLayout, stemBlockSize, stemStride,
        kFlat, vBias, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnStemOamPrepPagedKv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepPagedKv failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(kFlatShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), kFlatDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size && i < 10; i++) {
      float val;
      uint16_t bf16 = resultData[i];
      uint32_t bits = static_cast<uint32_t>(bf16) << 16;
      std::memcpy(&val, &bits, sizeof(val));
      LOG_PRINT("result[%ld] is: %f\n", i, val);
    }

    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kvIndices);
    aclDestroyIntArray(kvSeqLens);
    aclDestroyTensor(kScaleCache);
    aclDestroyTensor(vScale);
    aclDestroyTensor(kFlat);
    aclDestroyTensor(vBias);

    aclrtFree(kCacheDeviceAddr);
    aclrtFree(vCacheDeviceAddr);
    aclrtFree(kvIndicesDeviceAddr);
    aclrtFree(kScaleCacheDeviceAddr);
    aclrtFree(vScaleDeviceAddr);
    aclrtFree(kFlatDeviceAddr);
    aclrtFree(vBiasDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
