# aclnnKvCompressEpilog

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
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

- 接口功能：在KV Cache的Epilog阶段，对KV Cache进行原地压缩更新。将`bfloat16`激活值量化压缩后，按`slotMapping`散写到cache，值为 -1的token跳过不处理，取值越界的token静默跳过。`quantMode=0/1/2`下`x`尾轴`d`的后64列为RoPE段、前`d-64`列为nope段；`quantMode=3/4`对整行量化（无RoPE/nope拆分）。支持五种量化模式：group(bf16 scale)、group(e8m0 scale)、RoPE hifloat8静态 + nope FLOAT4_E2M1动态、MXFP8全行、MXFP4 packed全行。
<!-- npu="A3,910b" id9 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>上`quantMode=1`承载nope INT8对称动态量化（RoPE保留bf16）。
<!-- end id9 -->

- 计算公式：

  对`x`的最后一维（d轴）计算每组的amax，并量化为目标dtype，记第g组为$x_g$：

  - 场景1（quantMode=0）：group(64)量化为FP8，`scale`存储为`bfloat16`；RoPE段保留`bfloat16`。

    $$
    scale_g = \frac{\max(|x_g|)}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

  - 场景2（quantMode=1）：同场景1，但`scale`存储为`float8_e8m0`，`roundScale=true`时`scale`向上取到2的幂。

  - 场景3（quantMode=2）：RoPE段（后64列）做hifloat8**静态**量化，nope段（前d-64列）做per-group **FLOAT4_E2M1动态**量化（`FP4\_MAX=6.0`），`scale`以`bfloat16`写出，`roundScale`不生效：

    $$
    rope_i = \mathrm{hifloat8}(x_i \cdot xScale), \qquad
    scale_g = \frac{\max(|x_g|)}{FP4\_MAX}, \quad nope_i = \mathrm{FLOAT4\_E2M1}\left(\frac{x_i}{scale_g}\right)
    $$

  - 场景4（quantMode=1，<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>上承载INT8）：`d`固定为512，前448列为nope段（每64元素一组共7组）做INT8**对称动态**量化，后64列为RoPE段按`bfloat16`原始位型写出；`scale`以`bfloat16`写出，`roundScale`、`xScale`不生效。每token行定长608B：rope 128B | nope 448B | scale 14B | pad 18B(写0)。取整为数学真值的RNE（四舍六入五取偶）：

    $$
    amax_g = \max(|x_g|), \quad
    scale_g = \begin{cases} 1.0, & amax_g = 0 \\ \frac{amax_g}{127}, & amax_g \neq 0 \end{cases}, \quad
    q_i = \mathrm{clip}\left(\mathrm{RNE}\left(\frac{x_i}{scale_g}\right), -128, 127\right)
    $$

  - 场景5（quantMode=3，MXFP8全行）：整行按`quantGroupSize`（仅支持32）分组做MXFP8动态量化，cache为`float8_e4m3fn/float8_e5m2`（`FP8\_MAX`分别取448/57344）：

    $$
    scale_g = \frac{\max(|x_g|, 10^{-4})}{FP8\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
    $$

    `roundScale=true`时`scale`向上取到2的幂；`scale`以`bfloat16`写出。

  - 场景6（quantMode=4，MXFP4 packed全行）：整行MXFP4量化，每两元素打包一字节E2M1（RNE），共享指数`scale`按位编码为`bfloat16`（含Inf/NaN/0位级修正），`quantGroupSize`支持16/32。

- cache布局：`quantMode=0/1/2`仅支持四维`[blockNum, blockSize, 1, headDim]`；`quantMode=3/4`另支持二维平铺`[N, kvCacheCol]`（slot即行号）与四维分页`[blockNum, blockSize, 1, cacheCol]`（每token交错`[data|scale]`bundle，块尾/步长间隙不写，块步长由cache view stride派生）。行布局（`quantMode=3/4`，平铺）：`[dataCol数据][scaleCol个bf16 scale]`，无pad、行尾不写；dataCol=FP8取d / FP4取d/2，scaleCol=d/quantGroupSize。

- 示例：

  ```python
  cache shape:        [128, 16, 1, 384]   # 4D [blockNum, blockSize, 1, headDim], num_slots=2048, headDim≥kvCacheCol(=323)
  x shape:            [1024, 256]
  slot_mapping shape: [1024]
  quantGroupSize = 64
  quantMode = 1
  roundScale = true
  xScale = 1.0
  cache out shape:    [128, 16, 1, 384]   (原地更新)

  # 场景4（quantMode=1，A2/A3 承载 INT8）
  cache shape:        [128, 16, 1, 608]   # headDim≥608
  x shape:            [1024, 512]
  slot_mapping shape: [1024]
  quantGroupSize = 64
  quantMode = 1
  cache out shape:    [128, 16, 1, 608]   (原地更新，被引用行的前608B被覆写)
  ```

  ```python
  # MXFP4-g16 主场景（quantMode=4，平铺 2D cache）
  cache shape:        [4096, 320]         # 2D [N, concatCol], concatCol = d/2 + 2*(d/16) = 320（无pad）
  x shape:            [1024, 512]
  slot_mapping shape: [1024]
  quantGroupSize = 16
  quantMode = 4
  roundScale = true
  xScale = 1.0
  cache out shape:    [4096, 320]         (原地更新)
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
      <td><li>不支持空Tensor。</li><li>既是输入也是输出。</li><li>quantMode=0/1/2：仅支持四维[blockNum, blockSize, 1, headDim]，dim2固定为1。</li><li>quantMode=3/4：支持二维[N, concatCol]（平铺，无pad）或四维[blockNum, blockSize, 1, cacheCol]（分页）。</li></td>
      <td>UINT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>见使用说明</td>
      <td>quantMode=0/1/2仅blockNum维支持非连续；quantMode=3/4的二维平铺支持行stride非连续，四维分页布局支持块间非连续（blockNum维），间隙保持原值</td>
    </tr>
    <tr>
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>待量化的激活输入，对应计算公式中的x。</td>
      <td><li>不支持空Tensor。</li><li>最后一维d需满足d % 64 == 0且64 < d ≤ 8192。</li><li><term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>上quantMode=1时d必须等于512。</li></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[bs, d]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>slotMapping（const aclTensor*）</td>
      <td>输入</td>
      <td>token到cache slot的索引映射，值为 -1表示跳过该token。</td>
      <td><li>不支持空Tensor。</li><li>维度需等于x的维度减1。</li><li>元素取值范围为[-1, num_slots - 1]。</li></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>[bs]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantGroupSize（int64_t）</td>
      <td>输入</td>
      <td>量化分组大小。</td>
      <td>默认值为64。quantMode=2时仅支持16/32/64，且要求(d-64) % quantGroupSize == 0；<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>上quantMode=1时必须为64。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>枚举值支持0、1、2、3、4，其中0表示group量化（scale存储为bfloat16），1表示group量化（scale存储为float8_e8m0），2表示RoPE段hifloat8静态量化 + nope段FLOAT4_E2M1动态量化（scale存储为bfloat16），3表示MXFP8全行量化（cache为float8_e4m3fn/float8_e5m2，group仅支持32），4表示MXFP4 packed全行量化（cache为uint8，group支持16/32）。默认值为1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>group模式下是否对每组scale向上取到2的幂。</td>
      <td>默认值为true。quantMode=2及<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>的quantMode=1下不生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>xScale（float）</td>
      <td>输入</td>
      <td>quantMode=2时为RoPE段hifloat8静态量化的缩放系数；quantMode=0/1下预留未使用（<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>的quantMode=1亦不生效）。</td>
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

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - cache数据类型：quantMode=0/1/2/4时为UINT8，quantMode=3时为FLOAT8_E4M3FN或FLOAT8_E5M2。
  - quantMode支持0/1/2/3/4，其中quantMode=1表示group量化（scale存储为float8_e8m0）。
<!-- end id10 -->
<!-- npu="A3,910b" id11 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
  - cache数据类型仅支持UINT8。
  - quantMode仅支持1，表示nope段INT8对称动态量化 + RoPE段保留bfloat16（scale存储为bfloat16，要求d=512）；其余取值报错。
<!-- end id11 -->

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口执行下列基础检查；返回成功不表示全部形状、数据类型和属性约束检查已完成，后续tiling检查失败可能由第二段返回：

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
      <td>cacheRef、x或slotMapping为空Tensor；cacheRef布局不符合约束（须连续或仅blockNum维非连续）。</td>
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

  违反下文约束时，后续执行准备或tiling检查失败可能返回以下内部错误。具体错误码取决于失败环节和运行时版本，不保证统一返回161002；请结合运行日志定位，不能仅凭内部错误码判定为参数错误。

  | 返回值 | 错误码 | 说明 |
  | --- | --- | --- |
  | ACLNN_ERR_INNER_NULLPTR | 561103 | API内部nullptr异常；d、quantMode等约束检查失败可能通过此错误返回。 |
  | ACLNN_ERR_INNER | 561000 | API内部异常；不支持的数据类型组合等执行准备失败可能通过此错误返回。 |

## 约束说明

- 确定性计算：aclnnKvCompressEpilog默认确定性实现。
- **cache shape约束**：quantMode=0/1/2仅支持四维`[blockNum, blockSize, 1, headDim]`（num_slots = blockNum × blockSize），倒数第二维固定为1，仅在blockNum维支持非连续；quantMode=3/4支持二维平铺`[N, concatCol]`（slot即行号，尾轴≥concatCol，无32B取整要求）或四维分页`[blockNum, blockSize, 1, cacheCol]`（每token交错[data|scale]bundle，尾轴≥concatCol，块尾/步长间隙不写，块步长由cache view stride派生，连续时为blockSize×cacheCol；32B对齐为建议值，不再强制）。
- **块间非连续（quantMode=3/4）**：四维分页cache各块之间可有间隙；块步长取cache的view stride(0)，须不小于`blockSize × cacheCol`；32B对齐为建议值，不再强制。块内token按`[data|scale]` bundle紧凑排列，块间间隙不写。
- **headDim约束**：cache末维headDim须 ≥ 每行写出字节数。quantMode=3/4：行宽=concatCol（无pad），headDim ≥ concatCol。quantMode=0/1/2：headDim ≥ kvCacheCol = 对齐(concatCol)；concatCol：quantMode=0/1为(d-64)+128+⌈(d-64)/64⌉×scaleBytes（mode0=2、mode1=1）；quantMode=2为64+(d-64)/2+((d-64)/quantGroupSize)×2。对齐：quantMode=1不补齐，quantMode=0/2按32字节对齐。示例：d=256、quantMode=1 → kvCacheCol=323 → headDim ≥ 323。
<!-- npu="A3,910b" id12 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>上quantMode=1的kvCacheCol恒为608，headDim ≥ 608。
<!-- end id12 -->
- slotMapping的维度应等于x的维度减1，即slotMapping为x除最后一维外的所有维度展平。
- x的最后一维（d轴）：quantMode=0/1/2需满足d % 64 == 0且64 < d ≤ 8192；quantMode=3/4需满足0 < d ≤ 8192、d % quantGroupSize == 0且d % 128 == 0（无RoPE/nope拆分）。
- quantMode=2时，quantGroupSize仅支持16/32/64，且nope段长度(d-64)需能被quantGroupSize整除；x需为bfloat16。
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>上quantMode=1为group（e8m0）量化，quantMode支持0/1/2/3/4。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>上，quantMode仅支持1（INT8量化）：d必须等于512（448列nope + 64列RoPE）、quantGroupSize必须为64、headDim须 ≥ 608；roundScale/xScale不生效，其他取值报错。
<!-- end id8 -->
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
