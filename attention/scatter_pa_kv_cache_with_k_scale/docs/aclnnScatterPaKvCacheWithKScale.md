# aclnnScatterPaKvCacheWithKScale

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/scatter_pa_kv_cache_with_k_scale)

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

- 接口功能：更新KvCache中指定位置的key和value，同时更新key的scale值。

- 输入输出支持以下场景：
  - 场景一：

    ```python
    key:[batch * seq_len, num_head, head_size]
    value:[batch * seq_len, num_head, head_size]
    key_cache:[num_blocks, num_head, block_size, head_size]
    value_cache:[num_blocks, num_head, block_size, head_size]
    slot_mapping:[batch * seq_len]
    key_scale:[batch * seq_len, num_head]
    key_scale_cache:[num_blocks, num_head, block_size, 1]
    cache_layout:"BNBD"
    ```

    其中key和value的dtype为FLOAT8_E5M2或FLOAT8_E4M3FN，key_scale和key_scale_cache的dtype为FLOAT。

    计算公式：

    对于每个token（i ∈ [0, num_tokens)）和每个头（j ∈ [0, num_head)）：

    ```python
    block_idx = slot_mapping[i] // block_size
    block_offset = slot_mapping[i] % block_size

    key_cache[block_idx][j][block_offset][:] = key[i][j][:]
    value_cache[block_idx][j][block_offset][:] = value[i][j][:]
    key_scale_cache[block_idx][j][block_offset][0] = key_scale[i][j]
    ```

    其中：
    - num_tokens = batch * seq_len
    - block_idx：slot_mapping映射到的block索引
    - block_offset：block内的偏移量

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：仅支持场景一。

<!-- end id7 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnScatterPaKvCacheWithKScale"接口执行计算。

```Cpp
aclnnStatus aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize(
  const aclTensor   *key,
  const aclTensor   *value,
  aclTensor         *keyCacheRef,
  aclTensor         *valueCacheRef,
  const aclTensor   *slotMapping,
  aclTensor         *keyScale,
  aclTensor         *keyScaleCacheRef,
  char              *cacheLayoutOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnScatterPaKvCacheWithKScale(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize

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
      <td class="tg-0pky">key（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">待更新的key值，当前step多个token的key。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li></ul></td>
      <td class="tg-0pky">FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">value（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">待更新的value值，当前step多个token的value。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li></ul></td>
      <td class="tg-0pky">FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyCacheRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">需要更新的key cache，当前layer的key cache。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li></ul></td>
      <td class="tg-0pky">与key保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueCacheRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">需要更新的value cache，当前layer的value cache。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li></ul></td>
      <td class="tg-0pky">与value保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">slotMapping（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">每个token key或value在cache中的存储偏移。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li></ul></td>
      <td class="tg-0pky">INT32、INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyScale（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">待更新的key scale值，当前step多个token的key scale。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li><li>两维tensor，尾轴可以不连续。</li></ul></td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyScaleCacheRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">需要更新的key scale cache，当前layer的key scale cache。</td>
      <td class="tg-0pky"><ul><li>不支持空Tensor。</li><li>四维tensor，最后一维为1。</li></ul></td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">cacheLayoutOptional（char*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">表示keyCacheRef和valueCacheRef的内存排布格式。</td>
      <td class="tg-0lax"><ul><li>当传空指针或"BNBD"时，表示keyCacheRef和valueCacheRef的格式为[num_blocks, num_head, block_size, head_size]。</li></ul></td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
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
      <td>传入的key、value、keyCacheRef、valueCacheRef、slotMapping、keyScale、keyScaleCacheRef是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>参数key、value的数据类型不在支持的范围之内（FLOAT8_E5M2、FLOAT8_E4M3FN）。</td>
    </tr>
    <tr>
      <td>key、keyCacheRef、value、valueCacheRef的数据类型不一致。</td>
    </tr>
    <tr>
      <td>slotMapping的数据类型不在支持的范围之内（INT32、INT64）。</td>
    </tr>
    <tr>
      <td>keyScale、keyScaleCacheRef的数据类型不一致或不在支持的范围之内（FLOAT）。</td>
    </tr>
    <tr>
      <td>key或value的维数不等于3维，slotMapping的维数不等于1维。</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterPaKvCacheWithKScale

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnScatterPaKvCacheWithKScale默认确定性实现。
- key、value、keyCacheRef、valueCacheRef的数据类型必须一致；
- slotMapping的取值范围[0, num_blocks*block_size-1]，且slotMapping内的元素值保证不重复，重复时不保证正确性；
- key和value的前两维shape必须相同；
- keyScale是两维tensor，shape为[batch * seq_len, num_head]，尾轴可以不连续；
- keyScaleCacheRef是四维tensor，shape为[num_blocks, num_head, block_size, 1]，最后一维必须为1，尾轴必须连续。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="950" id8 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_kv_cache_with_k_scale.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
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

    std::vector<int64_t> keyShape = {2, 2, 64};
    std::vector<int64_t> valueShape = {2, 2, 64};
    std::vector<int64_t> keyCacheShape = {4, 2, 16, 64};
    std::vector<int64_t> valueCacheShape = {4, 2, 16, 64};
    std::vector<int64_t> slotMappingShape = {2};
    std::vector<int64_t> keyScaleShape = {2, 2};
    std::vector<int64_t> keyScaleCacheShape = {4, 2, 16, 1};

    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* valueCacheDeviceAddr = nullptr;
    void* keyScaleDeviceAddr = nullptr;
    void* keyScaleCacheDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* valueCache = nullptr;
    aclTensor* keyScale = nullptr;
    aclTensor* keyScaleCache = nullptr;
    char * cacheLayout = "BNBD";

    std::vector<uint8_t> hostKey(256, 1);
    std::vector<uint8_t> hostValue(256, 1);
    std::vector<int32_t> hostSlotMapping(4, 1);
    std::vector<uint8_t> hostKeyCacheRef(8192, 1);
    std::vector<uint8_t> hostValueCacheRef(8192, 1);
    std::vector<float> hostKeyScale(2, 1.0f);
    std::vector<float> hostKeyScaleCacheRef(128, 1.0f);

    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &key, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostValue, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &value, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &keyCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostValueCacheRef, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &valueCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKeyScale, keyScaleShape, &keyScaleDeviceAddr, aclDataType::ACL_FLOAT, &keyScale, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKeyScaleCacheRef, keyScaleCacheShape, &keyScaleCacheDeviceAddr, aclDataType::ACL_FLOAT, &keyScaleCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize(key, value, keyCache, valueCache,
                                                         slotMapping, keyScale, keyScaleCache,
                                                         cacheLayout, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnScatterPaKvCacheWithKScale(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCacheWithKScale failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(keyCacheShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyCacheDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size && i < 10; i++) {
      LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(valueCache);
    aclDestroyTensor(keyScale);
    aclDestroyTensor(keyScaleCache);

    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(valueCacheDeviceAddr);
    aclrtFree(keyScaleDeviceAddr);
    aclrtFree(keyScaleCacheDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```

<!-- end id8 -->
