# aclnnKvCompressEpilog

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/kv_compress_epilog)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：在KV Cache的Epilog阶段，对KV Cache进行原地压缩更新。将`bfloat16`激活值量化压缩后，按`slotMapping`散写到cache，值为 -1的token跳过不处理。`x`尾轴`d`的后64列为rope段、前`d-64`列为nope段。支持三种量化模式：group(bf16 scale)、group(e8m0 scale)、rope hifloat8静态 + nope FLOAT4_E2M1动态。

- 计算公式：

  对`x`的最后一维（d轴）按组计算每组的amax，并量化为目标dtype，记第g组为$x_g$：

  - 场景1（quantMode=0）：group(64)量化为FP8，`scale`存储为`bfloat16`；rope段保留`bfloat16`。

    $$
    scale_g = \frac{\max(|x_g|)}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

  - 场景2（quantMode=1）：同场景1，但`scale`存储为`float8_e8m0`，`roundScale=true`时`scale`向上取到2的幂。

  - 场景3（quantMode=2）：rope段（后64列）做hifloat8**静态**量化，nope段（前d-64列）做per-group **FLOAT4_E2M1 动态**量化（`FP4\_MAX=6.0`），`scale`以`bfloat16`写出，`roundScale`不生效：

    $$
    rope_i = \mathrm{hifloat8}(x_i \cdot xScale), \qquad
    scale_g = \frac{\max(|x_g|)}{FP4\_MAX}, \quad nope_i = \mathrm{FLOAT4\_E2M1}\left(\frac{x_i}{scale_g}\right)
    $$

- 示例：

  ```
  cache shape:        [128, 16, 1, 384]   # 4D [blockNum, blockSize, 1, headDim], num_slots=2048, headDim≥kvCacheCol(=323)
  x shape:            [1024, 256]
  slot_mapping shape: [1024]
  quantGroupSize = 64
  quantMode = 1
  roundScale = true
  xScale = 1.0
  cache out shape:    [128, 16, 1, 384]   (原地更新)
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnKvCompressEpilogGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnKvCompressEpilog”接口执行计算。

```cpp
aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(
  aclTensor       *cacheRef,
  const aclTensor *x,
  const aclTensor *slotMapping,
  int64_t          quantGroupSize,
  int64_t          quantMode,
  bool             roundScale,
  float            xScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnKvCompressEpilog(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnKvCompressEpilogGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 200px">
  <col style="width: 110px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 120px">
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
      <td>cacheRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>表示当前层的KV Cache向量缓存，原地更新，对应计算公式中的量化目标。</td>
      <td><li>不支持空Tensor。</li><li>既是输入也是输出。</li><li>仅支持四维[blockNum, blockSize, 1, headDim]，dim2固定为1。</li></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>[blockNum, blockSize, 1, headDim]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>待量化的激活输入，对应计算公式中x。</td>
      <td><li>不支持空Tensor。</li><li>最后一维d需满足d % 64 == 0且64 < d ≤ 8192。</li></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[bs, d]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>slotMapping（const aclTensor*）</td>
      <td>输入</td>
      <td>token到cache slot的索引映射，值为 -1表示跳过该token。</td>
      <td><li>不支持空Tensor。</li><li>维度需等于x的维度减1。</li><li>元素取值范围为[-1, num_slots - 1]。</li></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>[bs]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>quantGroupSize（int64_t）</td>
      <td>输入</td>
      <td>量化分组大小。</td>
      <td>默认值为64。quantMode=2时仅支持16/32/64，且要求(d-64) % quantGroupSize == 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>枚举值支持0、1、2，其中0表示group量化（scale存储为bfloat16），1表示group量化（scale存储为float8_e8m0），2表示rope段hifloat8静态量化 + nope段FLOAT4_E2M1动态量化（scale存储为bfloat16）。默认值为1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>group模式下是否对每组scale向上取到2的幂。</td>
      <td>默认值为true。quantMode=2下不生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScale（float）</td>
      <td>输入</td>
      <td>quantMode=2时为rope段hifloat8静态量化的缩放系数；quantMode=0/1下预留未使用。</td>
      <td>默认值为1.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
      <td>cacheRef、x、slotMapping存在空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>cacheRef、x或slotMapping为空Tensor。</td>
    </tr>
  </tbody></table>

## aclnnKvCompressEpilog

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnKvCompressEpilogGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnKvCompressEpilog默认确定性实现。
- **cache仅支持四维shape** `[blockNum, blockSize, 1, headDim]`（num_slots = blockNum × blockSize），倒数第二维固定为1，不支持其他维数；仅在blockNum维支持非连续。
- **headDim约束**：cache末维headDim须 ≥ 每行写出字节数kvCacheCol = 对齐(concatCol)。concatCol：quantMode=0/1为(d-64)+128+⌈(d-64)/64⌉×scaleBytes（mode0=2、mode1=1）；quantMode=2为64+(d-64)/2+((d-64)/quantGroupSize)×2。对齐：quantMode=1不补齐，quantMode=0/2按32B对齐。示例：d=256、quantMode=1 → kvCacheCol=323 → headDim ≥ 323。
- slotMapping的维度应等于x的维度减1，即slotMapping为x除最后一维外的所有维度展平。
- x的最后一维（d轴）需满足d % 64 == 0且64 < d ≤ 8192，按每64个元素一组进行逐组量化。
- quantMode=2时，quantGroupSize仅支持16/32/64，且nope段长度(d-64)需能被quantGroupSize整除；x需为bfloat16。
- slotMapping中值为 -1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_compress_epilog.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
    // 固定写法，资源初始化
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
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.(固定写法)device/stream初始化, 参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> cacheShape = {128, 16, 1, 384};  // 4D KV Cache [blockNum,blockSize,1,headDim]，num_slots=2048
    std::vector<int64_t> xShape = {1024, 256};      // 待量化激活
    std::vector<int64_t> slotShape = {1024};        // slot 索引映射
    void* cacheDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* slotDeviceAddr = nullptr;
    aclTensor* cache = nullptr;
    aclTensor* x = nullptr;
    aclTensor* slotMapping = nullptr;
    std::vector<uint8_t> cacheHostData(GetShapeSize(cacheShape), 0);
    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);  // bfloat16 以 uint16 承载
    std::vector<int32_t> slotHostData(GetShapeSize(slotShape), 0);
    // 创建 cache aclTensor（原地输入/输出）
    ret = CreateAclTensor(cacheHostData, cacheShape, &cacheDeviceAddr, aclDataType::ACL_UINT8, &cache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 slotMapping aclTensor
    ret = CreateAclTensor(slotHostData, slotShape, &slotDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 属性
    int64_t quantGroupSize = 64;
    int64_t quantMode = 1;   // group 量化，scale 存储为 float8_e8m0
    bool roundScale = true;
    float xScale = 1.0f;

    // 3.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnKvCompressEpilog第一段接口
    ret = aclnnKvCompressEpilogGetWorkspaceSize(cache, x, slotMapping, quantGroupSize, quantMode, roundScale, xScale,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvCompressEpilogGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnKvCompressEpilog第二段接口
    ret = aclnnKvCompressEpilog(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvCompressEpilog failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.cache 为原地更新，结果已写回 cacheDeviceAddr，可按需拷回 host 查看

    // 6.释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(cache);
    aclDestroyTensor(x);
    aclDestroyTensor(slotMapping);

    // 7.释放device资源
    aclrtFree(cacheDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(slotDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
