# aclnnIndexerQuantCache

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/indexer_quant_cache)

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

- 接口功能：在Indexer注意力机制的Epilog阶段，对KV Cache进行原地压缩更新。将`float16`/`bfloat16`的激活值按逐块动态量化（Per-Block Dynamic Quantization）压缩为FP8（E4M3/E5M2）、INT8（`uint8`）或MX-FP4格式，并按`slotMapping`将量化结果与对应`cacheScale`散写到cache，值为 -1的token跳过不处理。支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。

- 计算公式：

  对`x`的最后一维（d轴）按量化粒度计算每组的amax，并量化为目标dtype，记第g组为$x_g$（MX-FP8/MX-FP4模式每32个元素一组；Normal/HiFloat8模式整行一组）：

  $$
  scale_g = \frac{\max(|x_g|)}{Q\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
  $$

  - 场景1（quantMode=0，MX-FP8）：scale存储为`float8_e8m0`，`roundScale=true`时对scale进行舍入。
  - 场景2（quantMode=1，Normal）：动态逐块量化，scale存储为`float32`。
  - 场景3（quantMode=2，HiFloat8）：输出为$x \times xScale$后的hifloat8。
  - 场景4（quantMode=3，MX-FP4）：按标准MX块（每32元素）量化为FP4（`uint8`每字节打包2个fp4值），scale存储为`float8_e8m0`。

- 示例：

  ```
  # cache/cache_scale 均为 4D [blockNum, blockSize, 1, headDim]; num_slots = blockNum*blockSize = 2048
  cache shape:        [128, 16, 1, 128]   # headDim=128 >= d(128)
  cache_scale shape:  [128, 16, 1, 1]     # mode1 scaleCol=1
  x shape:            [1024, 128]
  slot_mapping shape: [1024]
  quantMode = 1
  roundScale = true
  xScale = 1.0
  cache out shape:        [128, 16, 1, 128]   (原地更新)
  cache_scale out shape:  [128, 16, 1, 1]     (原地更新)
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnIndexerQuantCacheGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIndexerQuantCache”接口执行计算。

```cpp
aclnnStatus aclnnIndexerQuantCacheGetWorkspaceSize(
  aclTensor       *cacheRef,
  aclTensor       *cacheScaleRef,
  const aclTensor *x,
  const aclTensor *slotMapping,
  int64_t          quantMode,
  bool             roundScale,
  float            xScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnIndexerQuantCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnIndexerQuantCacheGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 220px">
  <col style="width: 110px">
  <col style="width: 300px">
  <col style="width: 300px">
  <col style="width: 240px">
  <col style="width: 100px">
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
      <td>cacheRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>表示当前层的分页KV Cache，原地更新，对应计算公式中的量化目标。<b>仅支持四维</b>（shape、headDim长度规则详见约束说明）。MX-FP4模式下cache为FP4（每字节打包2个fp4值）。</td>
      <td><li>不支持空Tensor。</li><li>既是输入也是输出（原地操作）。</li><li>数据类型需与quantMode匹配。</li><li><b>仅支持四维</b>[blockNum, blockSize, 1, headDim]，dim2固定为1，仅在<b>blockNum维支持非连续</b>（分页）。</li></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>[blockNum, blockSize, 1, headDim]（num_slots = blockNum × blockSize）</td>
      <td>√（仅blockNum维）</td>
    </tr>
    <tr>
      <td>cacheScaleRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>表示每块量化的scale因子，原地更新，对应计算公式中scale。<b>仅支持四维</b>，规则同cache，headDim须 ≥ scaleCol（详见约束说明）。</td>
      <td><li>不支持空Tensor。</li><li>既是输入也是输出（原地操作）。</li><li>数据类型需与quantMode匹配。</li><li><b>仅支持四维</b>[blockNum, blockSize, 1, headDim]，dim2固定为1，headDim ≥ scaleCol，仅在<b>blockNum维支持非连续</b>（分页）。</li></td>
      <td>FLOAT、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>[blockNum, blockSize, 1, scaleHeadDim]（scaleHeadDim ≥ scaleCol）</td>
      <td>√（仅blockNum维）</td>
    </tr>
    <tr>
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>待量化的激活输入，对应计算公式中x。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[bs, d]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>slotMapping（const aclTensor*）</td>
      <td>输入</td>
      <td>token到cache slot的索引映射，值为 -1表示跳过该token。</td>
      <td><li>不支持空Tensor。</li><li>维度需等于x的维度减1。</li><li>元素取值范围为[-1, num_slots - 1]。</li></td>
      <td>INT32</td>
      <td>ND</td>
      <td>[bs]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>枚举值支持0、1、2、3，具体参考下文。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>MX-FP8模式下是否对scale进行round操作。</td>
      <td>仅在quantMode=0时生效。建议值为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScale（float）</td>
      <td>输入</td>
      <td>HiFloat8模式下的全局scale乘数。</td>
      <td>仅在quantMode=2时生效。建议值为1.0。</td>
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

  - 数据类型组合需匹配量化模式：quantMode=0，cache为FP8且scale为FLOAT8_E8M0；quantMode=1/2时，cache为FP8/UINT8且scale为FLOAT；quantMode=3，cache为FP4且scale为FLOAT8_E8M0。

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
      <td>cacheRef、cacheScaleRef、x、slotMapping存在空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>cacheRef、x或slotMapping为空Tensor。</td>
    </tr>
  </tbody></table>

## aclnnIndexerQuantCache

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnIndexerQuantCacheGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnIndexerQuantCache默认确定性实现。
- 数据类型组合需与quantMode匹配：Normal/HiFloat8模式cacheScale为FLOAT；MX-FP8/MX-FP4模式cacheScale为FLOAT8_E8M0；MX-FP4模式cache为FP4（uint8打包）。
- **cache 与 cacheScale均仅支持四维shape** `[blockNum, blockSize, 1, headDim]`，倒数第二维固定为1（每token一个量化向量），不支持其他维数；num_slots = blockNum × blockSize。
- cache/cacheScale**仅在blockNum维支持非连续**（分页）：各block可不紧密排布，但block内（blockSize、headDim维）须连续。
- **headDim长度约束**：
  - **cache.headDim ≥ d**（MX-FP4模式以fp4元素计，d个fp4值占 ⌈d/2⌉ 字节）。
  - **cacheScale.headDim ≥ scaleCol**，scaleCol：MX-FP8/MX-FP4（quantMode=0/3）为 ⌈d/32⌉；Normal/HiFloat8（quantMode=1/2）为 1。
  - 示例：d=128、quantMode=0 → scaleCol=4 → cache.headDim ≥ 128 且 cacheScale.headDim ≥ 4。
- x的最后一维（d轴）须能被32整除且 d ≤ 8192。
- slotMapping的维度应等于x的维度减1，即slotMapping为x除最后一维外的所有维度展平。
- slotMapping中值为 -1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_indexer_quant_cache.h"

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

    // 2.构造输入与输出，需要根据API的接口自定义构造（此处以 Normal 模式为例）
    // cache/cacheScale 为 4D [blockNum, blockSize, 1, headDim]; num_slots = 128*16 = 2048
    std::vector<int64_t> cacheShape = {128, 16, 1, 128}; // KV Cache（float8_e4m3fn），headDim=128 >= d
    std::vector<int64_t> scaleShape = {128, 16, 1, 1};   // 每块 scale（float32），mode1 scaleCol=1
    std::vector<int64_t> xShape = {1024, 128};          // 待量化激活
    std::vector<int64_t> slotShape = {1024};            // slot 索引映射
    void* cacheDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* slotDeviceAddr = nullptr;
    aclTensor* cache = nullptr;
    aclTensor* cacheScale = nullptr;
    aclTensor* x = nullptr;
    aclTensor* slotMapping = nullptr;
    std::vector<uint8_t> cacheHostData(GetShapeSize(cacheShape), 0);   // fp8 以 uint8 承载
    std::vector<float> scaleHostData(GetShapeSize(scaleShape), 0);
    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);          // float16 以 uint16 承载
    std::vector<int32_t> slotHostData(GetShapeSize(slotShape), 0);
    // 创建 cache aclTensor（原地输入/输出）
    ret = CreateAclTensor(cacheHostData, cacheShape, &cacheDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &cache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 cacheScale aclTensor（原地输入/输出）
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &cacheScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建 slotMapping aclTensor
    ret = CreateAclTensor(slotHostData, slotShape, &slotDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 属性
    int64_t quantMode = 1;   // Normal 逐块动态量化
    bool roundScale = true;
    float xScale = 1.0f;

    // 3.调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnIndexerQuantCache第一段接口
    ret = aclnnIndexerQuantCacheGetWorkspaceSize(cache, cacheScale, x, slotMapping, quantMode, roundScale, xScale,
                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexerQuantCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnIndexerQuantCache第二段接口
    ret = aclnnIndexerQuantCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexerQuantCache failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.cache 与 cacheScale 为原地更新，结果已写回对应 device 内存，可按需拷回 host 查看

    // 6.释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(cache);
    aclDestroyTensor(cacheScale);
    aclDestroyTensor(x);
    aclDestroyTensor(slotMapping);

    // 7.释放device资源
    aclrtFree(cacheDeviceAddr);
    aclrtFree(scaleDeviceAddr);
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
