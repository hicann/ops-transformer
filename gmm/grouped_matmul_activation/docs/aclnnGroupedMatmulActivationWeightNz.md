# aclnnGroupedMatmulActivationWeightNz

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

融合分组矩阵乘（Grouped Matrix Multiplication，GMM）、MX反量化和Swish门控线性单元（Swish-Gated Linear Unit，SwiGLU）激活，用于混合专家网络的专家前向计算，直接输出BF16结果，不产生输出scale。

计算公式如下，对`groupList`指定的每个专家执行MX分组矩阵乘：

$$
C_i[m,n] = \sum_{k=0}^{K-1}
\left(X_i[m,k] \times S^X_i[m,\lfloor k/64\rfloor,\lfloor (k\bmod 64)/32\rfloor]\right)
\left(W_i[k,n] \times S^W_i[\lfloor k/64\rfloor,n,\lfloor (k\bmod 64)/32\rfloor]\right)
$$

其中`S^X`、`S^W`为E8M0缩放因子的解码值，须在K方向求和前参与计算。`M`为输入token总数，`K`为输入特征维度，`N`为矩阵乘输出维度，`E`为专家个数，`i`为专家索引。

GMM结果`C`的逻辑shape为`(M,N)`。`swigluMode=2`表示前后分半的SwiGLU变种：沿N轴将前半作为激活分支`C_glu`，后半作为线性分支`C_gate`。激活分支先做上限裁剪，再计算带`gluAlpha`的sigmoid激活；线性分支对称裁剪后加`gluBias`，最后两分支相乘：

$$
u = \min(C_{glu}, clampLimit)
$$

$$
v = \operatorname{clip}(C_{gate},-clampLimit,clampLimit)+gluBias
$$

$$
y = \operatorname{Cast}_{BF16}\left(\frac{u}{1+\exp(-gluAlpha\times u)}\times v\right)
$$

## 函数原型

先调用`aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize`获取计算所需的workspace大小及执行器，再调用`aclnnGroupedMatmulActivationWeightNz`执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor*     x,
    const aclTensor*     xScale,
    const aclTensor*     groupList,
    const aclTensorList* weight,
    const aclTensorList* weightScale,
    const aclTensorList* bias,
    int64_t              dequantMode,
    int64_t              dequantDtype,
    const char*          actType,
    int64_t              groupListType,
    const aclIntArray*   tuningConfig,
    int64_t              swigluMode,
    double               clampLimit,
    double               gluAlpha,
    double               gluBias,
    aclTensor*           y,
    uint64_t*            workspaceSize,
    aclOpExecutor**      executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulActivationWeightNz(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | x（const aclTensor*） | 输入 | 激活矩阵，对应公式中的X。 | 不支持空Tensor。 | FLOAT4_E2M1 | ND | `(M,K)` | √ |
  | xScale（const aclTensor*） | 输入 | 激活的缩放因子，对应公式中的S^X。 | 不支持空Tensor，每64个K元素对应2个缩放因子。 | FLOAT8_E8M0 | ND | `(M,K/64,2)` | √ |
  | groupList（const aclTensor*） | 输入 | 专家分组信息。 | 不支持空Tensor，含义由groupListType指定。 | INT64 | ND | `(E)` | √ |
  | weight（const aclTensorList*） | 输入 | 权重列表，对应公式中的W。 | 长度须为1，包含全部E个专家，不支持空Tensor。转置状态由stride识别，布局见约束说明。 | FLOAT4_E2M1 | FRACTAL_NZ | `(E,K,N)` | √ |
  | weightScale（const aclTensorList*） | 输入 | 权重缩放因子列表，对应公式中的S^W。 | 长度须为1，包含全部E个专家，不支持空Tensor，转置状态须与weight一致。 | FLOAT8_E8M0 | ND | `(E,K/64,N,2)` | √ |
  | bias（const aclTensorList*） | 输入 | 预留的矩阵乘偏置。 | 仅支持nullptr或空列表。 | - | - | - | - |
  | dequantMode（int64_t） | 输入 | 反量化模式。 | 仅支持2，表示MX反量化。 | - | - | - | - |
  | dequantDtype（int64_t） | 输入 | GMM中间结果的数据类型。 | 仅支持0，表示FLOAT32。 | - | - | - | - |
  | actType（const char*） | 输入 | 激活类型。 | 仅支持字符串`"swiglu"`。 | - | - | - | - |
  | groupListType（int64_t） | 输入 | 分组列表解释方式。 | 0表示累计结束位置，1表示各组token数。 | - | - | - | - |
  | tuningConfig（const aclIntArray*） | 输入 | 预留的调优配置。 | 仅支持nullptr或空数组。 | - | - | - | - |
  | swigluMode（int64_t） | 输入 | SwiGLU计算模式。 | 仅支持2，表示前后分半、先裁剪再激活并应用alpha/bias，计算过程见功能说明。 | - | - | - | - |
  | clampLimit（double） | 输入 | 激活和线性分支的裁剪上限。 | 须为不超过FLOAT32最大有限值的正有限数，转换为FLOAT32后须大于0。 | - | - | - | - |
  | gluAlpha（double） | 输入 | sigmoid输入缩放系数。 | 须为有限数且绝对值不超过FLOAT32最大有限值。 | - | - | - | - |
  | gluBias（double） | 输入 | 线性分支偏置。 | 须为有限数且绝对值不超过FLOAT32最大有限值。 | - | - | - | - |
  | y（aclTensor*） | 输出 | SwiGLU计算结果。 | 不支持空Tensor。 | BFLOAT16 | ND | `(M,N/2)` | √ |
  | workspaceSize（uint64_t*） | 输出 | 计算所需的workspace大小，单位为字节。 | 不支持空指针。 | - | - | - | - |
  | executor（aclOpExecutor**） | 输出 | 包含算子计算流程的执行器。 | 不支持空指针，传给第二段接口。 | - | - | - | - |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回值 | 错误码 | 描述 |
  | --- | --- | --- |
  | `ACLNN_ERR_PARAM_NULLPTR` | `161001` | x、xScale、groupList、weight、weightScale、y、workspaceSize或executor为空指针。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | 输入输出的数据类型、数据格式或shape不符合要求，K、N不满足对齐要求，或E超出支持范围。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | weight、weightScale的TensorList长度不为1，或包含空Tensor指针。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | weight、weightScale的转置状态不一致，或storage/stride不符合要求。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | groupList不在Device内存。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | dequantMode不为2、dequantDtype不为0、actType为空或不为`"swiglu"`、groupListType不为0或1，或swigluMode不为2。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | clampLimit、gluAlpha、gluBias含非有限值或绝对值超出FLOAT32最大有限值，或clampLimit转换为FLOAT32后不大于0。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | bias为非空TensorList，或tuningConfig为非空数组。 |
  | `ACLNN_ERR_PARAM_INVALID` | `161002` | 当前设备架构不受支持。 |

## aclnnGroupedMatmulActivationWeightNz

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  | --- | --- | --- |
  | workspace | 输入 | 在Device侧申请的workspace内存地址；workspaceSize为0时可传nullptr。 |
  | workspaceSize | 输入 | 第一段接口获取的workspace大小，单位为字节。 |
  | executor | 输入 | 第一段接口返回的执行器，包含算子计算流程。 |
  | stream | 输入 | 指定执行任务的ACL stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnGroupedMatmulActivationWeightNz默认确定性实现。
- M须大于0。
- K、N须为正数且满足128对齐。
- 输入输出Tensor的数据类型、数据格式及shape如下。weight和weightScale传入的视图shape在转置和非转置布局下相同，转置状态由stride识别。

| Tensor | 数据类型（dtype） | 数据格式 | 调用方传入的视图shape |
| --- | --- | --- | --- |
| x | FLOAT4_E2M1 | ND | `(M,K)` |
| xScale（x_scale） | FLOAT8_E8M0 | ND | `(M,K/64,2)` |
| weight | FLOAT4_E2M1 | FRACTAL_NZ | `(E,K,N)` |
| weightScale | FLOAT8_E8M0 | ND | `(E,K/64,N,2)` |
| groupList | INT64 | ND | `(E,)` |
| y | BFLOAT16 | ND | `(M,N/2)` |

- 常用布局的描述符如下，stride以各Tensor的数据类型元素为单位，权重每字节存储两个元素。

| 布局 | weight的stride | weight的storage shape | weightScale的stride |
| --- | --- | --- | --- |
| 非转置 | `(K*N,N,1)` | `(E,N/64,K/16,16,64)` | `((K/64)*N*2,N*2,2,1)` |
| 转置 | `(K*N,1,K)` | `(E,K/64,N/16,16,64)` | `((K/64)*N*2,2,(K/64)*2,1)` |

- 接口根据stride识别weight和weightScale的转置状态，两者须保持一致。转置weight的stride、weightScale的后三维stride须与上表一致。
- E的取值范围为`[1,1024]`。`groupListType=0`时，groupList须非负、单调非递减，最后一个值不大于M；`groupListType=1`时，各组token数须非负且总和不超过M。允许专家的token数为0，分组数值由调用方保证合法。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  以下示例演示A4W4 WeightNZ输入、BF16输出的两段式接口调用，使用非转置权重。

  ```cpp
  #include <acl/acl.h>
  #include <aclnnop/aclnn_grouped_matmul_activation_weight_nz.h>

  #include <cstdint>
  #include <iostream>
  #include <random>
  #include <vector>

  namespace {
  constexpr int64_t kE = 1;
  constexpr int64_t kM = 32;
  constexpr int64_t kK = 128;
  constexpr int64_t kN = 128;
  constexpr int64_t kScaleK = (kK + 63) / 64;
  constexpr int64_t kNzN0 = 64;
  constexpr int64_t kNzK0 = 16;

  template <typename T>
  uint64_t Numel(const std::vector<int64_t>& shape)
  {
      uint64_t n = 1;
      for (int64_t dim : shape) {
          n *= static_cast<uint64_t>(dim);
      }
      return n;
  }

  template <typename T>
  bool CreateAclTensor(const std::vector<T>& host, const std::vector<int64_t>& logicalShape,
                       const std::vector<int64_t>& storageShape, aclDataType dtype, aclFormat format,
                       aclTensor** tensor, void** device)
  {
      // FLOAT4的storageShape按逻辑元素计数，每字节存放两个元素。
      const uint64_t elements = Numel<T>(storageShape);
      const uint64_t bytes = dtype == ACL_FLOAT4_E2M1 ? (elements + 1) / 2 : elements * sizeof(T);
      if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
          aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
          return false;
      }
      std::vector<int64_t> strides(logicalShape.size(), 1);
      for (int64_t i = static_cast<int64_t>(logicalShape.size()) - 2; i >= 0; --i) {
          strides[i] = logicalShape[i + 1] * strides[i + 1];
      }
      *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dtype, strides.data(), 0, format,
                                storageShape.data(), storageShape.size(), *device);
      return *tensor != nullptr;
  }
  }  // namespace

  int main()
  {
      constexpr int32_t deviceId = 0;
      if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
          return 1;
      }
      aclrtStream stream = nullptr;
      if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
          aclFinalize();
          return 1;
      }

      std::mt19937 rng(0);
      std::uniform_int_distribution<int> byteDist(0, 255);
      std::vector<uint8_t> xData(kM * kK / 2);
      std::vector<uint8_t> weightData(kE * (kN / kNzN0) * (kK / kNzK0) * kNzK0 * kNzN0 / 2);
      for (auto& value : xData) value = static_cast<uint8_t>(byteDist(rng));
      for (auto& value : weightData) value = static_cast<uint8_t>(byteDist(rng));
      std::vector<uint8_t> scaleData(kM * kScaleK * 2, 0x7f);
      std::vector<uint8_t> weightScaleData(kE * kScaleK * kN * 2, 0x7f);
      std::vector<int64_t> groupData = {kM};
      std::vector<uint16_t> yData(kM * (kN / 2), 0);

      aclTensor *x = nullptr, *xScale = nullptr, *groupList = nullptr, *weightTensor = nullptr;
      aclTensor *weightScaleTensor = nullptr, *y = nullptr;
      void *xDev = nullptr, *xScaleDev = nullptr, *groupDev = nullptr, *weightDev = nullptr;
      void *weightScaleDev = nullptr, *yDev = nullptr, *workspace = nullptr;
      aclTensorList *weight = nullptr, *weightScale = nullptr;
      aclOpExecutor* executor = nullptr;
      bool ok = CreateAclTensor(xData, {kM, kK}, {kM, kK}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, &x, &xDev) &&
                CreateAclTensor(scaleData, {kM, kScaleK, 2}, {kM, kScaleK, 2}, ACL_FLOAT8_E8M0,
                                ACL_FORMAT_ND, &xScale, &xScaleDev) &&
                CreateAclTensor(groupData, {kE}, {kE}, ACL_INT64, ACL_FORMAT_ND, &groupList, &groupDev) &&
                CreateAclTensor(weightData, {kE, kK, kN},
                                {kE, kN / kNzN0, kK / kNzK0, kNzK0, kNzN0}, ACL_FLOAT4_E2M1,
                                ACL_FORMAT_FRACTAL_NZ, &weightTensor, &weightDev) &&
                CreateAclTensor(weightScaleData, {kE, kScaleK, kN, 2}, {kE, kScaleK, kN, 2},
                                ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleTensor, &weightScaleDev) &&
                CreateAclTensor(yData, {kM, kN / 2}, {kM, kN / 2}, ACL_BF16, ACL_FORMAT_ND, &y, &yDev);
      if (ok) {
          aclTensor* weightItems[] = {weightTensor};
          aclTensor* weightScaleItems[] = {weightScaleTensor};
          weight = aclCreateTensorList(weightItems, 1);
          weightScale = aclCreateTensorList(weightScaleItems, 1);
          ok = weight != nullptr && weightScale != nullptr;
      }

      constexpr int64_t dequantMode = 2;
      constexpr int64_t dequantDtype = 0;
      constexpr char actType[] = "swiglu";
      constexpr int64_t groupListType = 0;
      constexpr int64_t swigluMode = 2;
      // 设置SwiGLU的裁剪上限、激活系数和线性分支偏置。
      constexpr double clampLimit = 7.0;
      constexpr double gluAlpha = 1.702;
      constexpr double gluBias = 1.0;
      uint64_t workspaceSize = 0;
      aclnnStatus ret = ACL_SUCCESS;
      if (ok) {
          ret = aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
              x, xScale, groupList, weight, weightScale, nullptr, dequantMode, dequantDtype, actType,
              groupListType, nullptr, swigluMode, clampLimit, gluAlpha, gluBias, y, &workspaceSize, &executor);
          if (ret == ACL_SUCCESS && workspaceSize != 0) {
              if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                  ok = false;
              }
          }
          if (ok && ret == ACL_SUCCESS) {
              ret = aclnnGroupedMatmulActivationWeightNz(workspace, workspaceSize, executor, stream);
          }
          if (ok && ret == ACL_SUCCESS && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
              ok = false;
          }
      }
      std::cout << "aclnnGroupedMatmulActivationWeightNz: " << ret << std::endl;

      // executor is released by the ACLNN runtime after execution.
      if (weight != nullptr) aclDestroyTensorList(weight);
      if (weightScale != nullptr) aclDestroyTensorList(weightScale);
      if (x != nullptr) aclDestroyTensor(x);
      if (xScale != nullptr) aclDestroyTensor(xScale);
      if (groupList != nullptr) aclDestroyTensor(groupList);
      if (y != nullptr) aclDestroyTensor(y);
      if (xDev != nullptr) aclrtFree(xDev);
      if (xScaleDev != nullptr) aclrtFree(xScaleDev);
      if (groupDev != nullptr) aclrtFree(groupDev);
      if (weightDev != nullptr) aclrtFree(weightDev);
      if (weightScaleDev != nullptr) aclrtFree(weightScaleDev);
      if (yDev != nullptr) aclrtFree(yDev);
      if (workspace != nullptr) aclrtFree(workspace);
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return ok && ret == ACL_SUCCESS ? 0 : 1;
  }
  ```
<!-- end id7 -->
