# AI CPU算子开发指南

## 概述

> 说明：
>
> 1. 算子开发过程中涉及的基本概念、AI CPU接口等，详细介绍请参考[《TBE&AI CPU算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevWizard)。
> 2. AI CPU算子是使用C++语言开发，运行在AI CPU硬件单元算子。
> 3. build.sh：算子开发过程中涉及的命令可通过`bash build.sh --help`查看，功能参数介绍参考[build参数说明](../install/build.md)。

开发指南以`AddExample`算子开发为例，介绍新算子开发流程以及涉及的交付件，完整样例代码请访问项目`examples`目录。

1. [工程创建](#工程创建)：开发算子前，需完成环境部署并创建算子目录，方便后续算子编译和部署。
2. [算子定义](#算子定义)：确定算子功能与原型定义。
3. [Kernel实现](#kernel实现)：实现Device侧算子核函数。
4. [aclnn适配](#aclnn适配)：自定义算子推荐aclnn接口调用，需提前完成二进制发布。**如采用图模式调用算子**，请参考[图模式适配指南](./graph_develop_guide.md)。
5. [编译部署](#编译部署)：通过工程编译脚本完成自定义算子的编译和安装。
6. [算子验证](#算子验证)：通过常见算子调用方式，验证自定义算子功能。

## 工程创建

**1. 环境部署**

开发算子前，请先参考[环境部署](../install/quick_install.md)完成基础环境搭建。

**2. 目录创建**

目录创建是算子开发的重要步骤，为后续代码编写、编译构建和验证提供统一的目录结构和文件组织方式。

可通过`build.sh`快速创建算子目录，进入项目根目录，执行如下命令：

```bash
# 创建指定算子目录，如bash build.sh --genop_aicpu=examples/add_example
# ${op_class}表示算子类型，如transformer类。
# ${op_name}表示算子名的小写下划线形式，如`AddExample`算子对应为add_example，新增算子不允许与已有算子重名。
bash build.sh --genop_aicpu=${op_class}/${op_name}
```

命令执行成功，会看到如下提示信息：

```bash
Create the AI CPU initial directory for ${op_name} under ${op_class} success
```

创建完成后，目录结构如下所示：

```text
${op_name}                              # 替换为实际算子名的小写下划线形式
├── examples                            # 算子调用示例
│   └── test_aclnn_${op_name}.cpp       # 算子aclnn调用示例
├── op_host                             # Host侧实现
│   └── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
├── op_kernel_aicpu                     # Device侧Kernel实现
│   ├── ${op_name}_aicpu.cpp            # Kernel入口文件，包含主函数和调度逻辑
│   ├── ${op_name}_aicpu.h              # Kernel头文件，包含函数声明、结构定义、逻辑实现
│   └── ${op_name}.json                 # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
├── tests                               # UT实现
│   └── ut                              # kernel/aclnn UT实现
└── CMakeLists.txt                      # 算子cmakelist入口
```

若```${op_class}```为全新算子分类需额外在`CMakeLists`中添加```add_subdirectory(${op_class})```，否则无法正常编译。

 ```text
 if(ENABLE_EXPERIMENTAL)
   # genop新增experimental算子分类
   # add_subdirectory(${op_class})
   add_subdirectory(experimental/transformer)
 else()
   # genop新增非experimental算子分类
   # add_subdirectory(${op_class})
   add_subdirectory(transformer)
 endif()
 ```

## 算子定义

算子定义需要完成两个交付件：`README.md` ```${op_name}.json```

**交付件1：README.md**

开发算子前需要先确定目标算子的功能和计算逻辑。

以自定义`AddExample`算子说明为例，请参考[AddExample算子说明](../../../examples/add_example/README.md)。

**交付件2：${op_name}.json**

算子信息库。

以自定义`AddExample`算子说明为例，请参考[AddExample算子信息库](../../../examples/add_example/op_kernel_aicpu/add_example.json)。

## Kernel实现

### Kernel简介

Kernel是算子在NPU执行的核心部分，Kernel实现包括如下步骤：

```mermaid
graph LR
  H([算子类声明]) -->A([Compute函数实现])
  A -->B([注册算子])
```

### 代码实现

Kernel一共需要两个交付件：```${op_name}_aicpu.cpp``` ```${op_name}_aicpu.h```

**交付件1：${op_name}_aicpu.h**

算子类声明

Kernel实现的第一步，需在头文件```op_kernel_aicpu/${op_name}_aicpu.h```进行算子类的声明，算子类需继承CpuKernel基类。
如需查看详细实现，请参考[add_example_aicpu.h](../../../examples/add_example/op_kernel_aicpu/add_example_aicpu.h)。

```CPP
// 1、算子类声明
// 包含AI CPU基础库头文件
#include "cpu_kernel.h"
// 定义命名空间aicpu(固定不允许修改)，并定义算子Compute实现函数
namespace aicpu {
// 算子类继承CpuKernel基类
class AddExampleCpuKernel : public CpuKernel {
 public:
  ~AddExampleCpuKernel() = default;
  // 声明函数Compute（需要重写），形参CpuKernelContext为CPUKernel的上下文，包括算子输入、输出和属性信息
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
```

**交付件2：${op_name}_aicpu.cpp**

Compute函数实现与AI CPU算子注册

获取输入/输出Tensor信息并进行合法性校验，然后实现核心计算逻辑（如加法操作），并将计算结果设置到输出Tensor中。

如需查看详细实现，请参考[add_example_aicpu.cpp](../../../examples/add_example/op_kernel_aicpu/add_example_aicpu.cpp)。

```C++
// 2、Compute函数实现
#include "add_example_aicpu.h"

namespace {
// 算子名
const char* const kAddExample = "AddExample";
const uint32_t kParamInvalid = 1;
}  // namespace

// 定义命名空间aicpu
namespace aicpu {
// 实现自定义算子类的Compute函数
uint32_t AddExampleCpuKernel::Compute(CpuKernelContext& ctx) {
  // 从CpuKernelContext中获取input tensor
  Tensor* input0 = ctx.Input(0);
  Tensor* input1 = ctx.Input(1);
  // 从CpuKernelContext中获取output tensor
  Tensor* output = ctx.Output(0);

  // 对tensor进行基本校验，判断是否为空指针
  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    return kParamInvalid;
  }

  // 获取input tensor的数据类型
  auto data_type = static_cast<DataType>(input0->GetDataType());
  // 获取input tensor的数据地址，例如输入的数据类型是int32
  auto input0_data = reinterpret_cast<int32_t*>(input0->GetData());
  // 获取tensor的shape
  auto input0_shape = input0->GetTensorShape();

  // 获取output tensor的数据地址，例如输出的数据类型是int32
  auto y = reinterpret_cast<int32_t*>(output->GetData());

  // AddCompute函数根据输入类型执行相应计算。
  // 由于C++自身不支持半精度浮点类型，可借助第三方库Eigen（建议使用3.3.9版本）表示。
  switch (data_type) {
    case DT_FLOAT:
      return AddCompute<float>(...);
    case DT_INT32:
      return AddCompute<int32_t>(...);
      ....
    default : return kParamInvalid;
  }
}

// 3、注册算子Kernel实现，用于框架获取算子Kernel的Compute函数。
REGISTER_CPU_KERNEL(kAddExample, AddExampleCpuKernel);
}  // namespace aicpu
```

## aclnn适配

通常算子开发和编译完成后，会自动生成aclnn接口（一套基于C的API），无需做其他配置，可直接在应用程序中调用aclnn接口实现算子调用。

## 编译部署

算子开发完成后，需对算子工程进行编译，生成自定义算子安装包 `*.run`，具体操作如下：

1. **准备工作。**

   参考[工程创建](#工程创建)完成基础环境搭建，同时检查算子开发交付件是否完备，是否在对应算子分类目录下。
2. **编译自定义算子包。**

   以`AddExample`算子为例，假设开发交付件在`examples`目录，完整代码参见[add_example](../../../examples/add_example/op_kernel_aicpu)目录。

   ```bash
   # 编译指定算子，如bash build.sh --pkg --ops=add_example --opkernel_aicpu -j16
   bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list} --opkernel_aicpu [--experimental] [-j${n}]
   ```

   - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
   - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
   - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"--ops=add_example"。
   - --opkernel_aicpu：开启ENABLE_AICPU_KERNEL，编译kernel_aicpu
   - --experimental（可选）：若编译的算子为贡献算子，需配置--experimental。
   - -j（可选）：指定编译线程数，加快编译速度。

   若提示如下信息，说明编译成功：

   ```bash
   Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
   ```

3. **安装自定义算子包。**

   ```bash
   # 安装run包
   ./build_out/cann-ops-transformer-${vendor_name}_linux-${arch}.run
   ```

   自定义算子包安装在```${ASCEND_HOME_PATH}/opp/vendors```路径中，```${ASCEND_HOME_PATH}```表示CANN软件安装目录，可提前在环境变量中配置。
4. **（可选）卸载自定义算子包。**

   自定义算子包安装后在```${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/scripts```目录会生成`uninstall.sh`，通过该脚本可卸载自定义算子包，命令如下：

   ```bash
   bash ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_transformer/scripts/uninstall.sh
   ```

## 算子验证

验证算子前需确保已配置了环境变量，命令如下：

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_transformer/op_api/lib:${LD_LIBRARY_PATH}
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_transformer
```

- **UT验证**

  算子开发过程中，可通过UT验证（如Kernel）方式进行快速验证，如需查看详细实现，请参考[Kernel UT](../../../examples/add_example/tests/ut/op_kernel_aicpu/test_add_example.cpp)。
- **aclnn调用验证**

  开发好的算子完成编译部署后，可通过aclnn方式验证功能，方法请参考[算子调用方式](../invocation/quick_op_invocation.md)。
