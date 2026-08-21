# AI Core算子开发指南

## 概述

### 使用说明

算子根据运行的硬件单元不同，可分为AI Core算子和AI CPU算子（少数）。AI Core算子使用Ascend C语言开发，运行在AI Core硬件单元；AI CPU算子使用C++语言开发，运行在AI CPU硬件单元。

本文旨在介绍如何基于标准工程开发AI Core算子。

算子开发前，请先了解如下信息：

- 基础知识：请先学习Ascend C编程语言，了解基本语法和原理，参考[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)熟悉硬件架构、算子Tiling/Kernel等概念。
- 开发API：算子开发过程中使用到的接口请参考[《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)、[《基础数据结构和接口》](https://hiascend.com/document/redirect/CannCommunitybasicopapi)。
- 算子工程迁移：针对[Ascend/samples](https://gitee.com/ascend/samples/tree/master)仓已贡献的算子，请参考[附录 > 算子工程迁移](#算子工程迁移)完成存量算子往本项目工程迁移。
- 算子跨平台迁移：针对涉及多平台算子实现迁移的场景，例如将Atlas A2上已实现的算子功能迁移到Ascend 950平台，请参考[附录 > 算子跨平台迁移](#算子跨平台迁移)完成适配。

### 开发流程

以`AddExample`算子为例，介绍标准AI Core算子开发的全流程和交付件，完整样例请访问项目`examples`目录。

1. [工程创建](#工程创建)：开发算子前，需完成环境部署并创建算子目录，方便后续算子编译和部署。

2. [算子定义](#算子定义)：确定算子功能与原型定义。

3. [Tiling实现](#tiling实现)：实现Host侧算子Tiling函数。

4. [Kernel实现](#kernel实现)：实现Device侧算子核函数。

5. [aclnn适配](#aclnn适配)：自定义算子推荐aclnn接口调用，需提前完成二进制发布。**如采用图模式调用算子**，请参考[图模式适配指南](./graph_develop_guide.md)。

6. [编译部署](#编译部署)：通过工程编译脚本完成自定义算子的编译和安装。

7. [算子验证](#算子验证)：通过常见算子调用方式，验证自定义算子功能。

## 工程创建

目录创建是算子开发的重要步骤，为后续代码编写、编译构建和调试提供统一的目录结构和文件组织方式。

可通过`build.sh`快速创建算子目录，进入项目根目录，执行如下命令：

```bash
# 创建指定算子目录，如bash build.sh --genop=examples/add_example
# ${op_class}表示算子类型，如attention类。
# ${op_name}表示算子名的小写下划线形式，如`AddExample`算子对应为add_example，新增算子不允许与已有算子重名。
bash build.sh --genop=${op_class}/${op_name}
```

命令执行成功，会看到如下提示信息：

```bash
Create the initial directory for ${op_name} under ${op_class} success
```

创建完成后，目录结构如下所示：

```text
${op_name}                              # 替换为实际算子名的小写下划线形式
├── examples                            # 算子调用示例
│   ├── test_aclnn_${op_name}.cpp       # 算子aclnn调用示例
├── op_host                             # Host侧实现
│   ├── ${op_name}_def.cpp              # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   ├── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
│   ├── ${op_name}_tiling.cpp           # Tiling实现，将张量划分为多个小块，区分数据类型进行并行计算
│   └── CMakeLists.txt                  # Host侧cmakelist文件
└── op_kernel                           # Device侧Kernel实现
│   ├── ${op_name}_tiling_key.h         # Tilingkey文件，定义Tiling策略的Key，标识不同的划分方式
│   ├── ${op_name}_tiling_data.h        # Tilingdata文件，存储Tiling策略相关的配置数据，如块大小、并行度
│   ├── ${op_name}.cpp                  # Kernel入口文件，包含主函数和调度逻辑
│   └── ${op_name}.h                    # Kernel实现文件，定义Kernel头文件，包含函数声明、结构定义、逻辑实现
└── CMakeLists.txt                      # 算子cmakelist入口
```

若```${op_class}```为全新算子分类需额外在`cmake/custom_build.cmake`中添加```add_subdirectory(${op_class})```，否则无法正常编译，具体修改如下。

```bash
# 编译examples目录下算子
foreach(EXAMPLES_OP_NAME ${ASCEND_OP_NAME})
    set(EXAMPLES_DIR "${OPS_TRANSFORMER_DIR}/examples/${EXAMPLES_OP_NAME}")
    set(EXAMPLES_MC2_DIR "${OPS_TRANSFORMER_DIR}/examples/mc2/${EXAMPLES_OP_NAME}")
    # 在examples目录下新增算子分类时，参考mc2目录增加命令语句如下：
    # set(EXAMPLES_${op_class}_DIR "${OPS_TRANSFORMER_DIR}/examples/${op_class}/${EXAMPLES_OP_NAME}")
    if(IS_DIRECTORY ${EXAMPLES_DIR})
        add_subdirectory(examples/${EXAMPLES_OP_NAME})
        list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/examples/${EXAMPLES_OP_NAME})
    elseif(IS_DIRECTORY ${EXAMPLES_MC2_DIR})
        add_subdirectory(examples/mc2/${EXAMPLES_OP_NAME})
        list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/examples/mc2/${EXAMPLES_OP_NAME})
    # 在examples目录下新增算子分类时，参考mc2目录增加命令语句如下：
    # elseif(IS_DIRECTORY ${EXAMPLES_${op_class}_DIR})
    #     add_subdirectory(examples/${op_class}/${EXAMPLES_OP_NAME})
    #     list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/examples/${op_class}/${EXAMPLES_OP_NAME})
    endif()
endforeach()
```

```bash
# 编译experimental目录下算子
if(ENABLE_EXPERIMENTAL)
    # genop新增experimental算子分类
    # add_subdirectory(${op_class})
    add_subdirectory(experimental/attention)
else()
    # genop新增非experimental算子分类
    # add_subdirectory(${op_class})
    add_subdirectory(attention)
endif()
```

## 算子定义

算子定义需要完成两个交付件：`README.md` ```${op_name}_def.cpp```

**交付件1：README.md**

开发算子前需要先确定目标算子的功能和计算逻辑。

以自定义`AddExample`算子说明为例，请参考[AddExample算子说明](../../../examples/add_example/README.md)。

**交付件2：${op_name}_def.cpp**

算子信息库。

以自定义`AddExample`算子说明为例，请参考[AddExample算子信息库](../../../examples/add_example/op_host/add_example_def.cpp)。

## Tiling实现

### Tiling简介

因NPU中AI Core内部存储空间有限，无法一次性将整个张量数据加载到计算单元中处理，因此需要将输入张量切分为多个小块（Tile），逐块进行计算，这一过程称为Tiling。

用于指导数据切分的算法称为Tiling策略或Tiling算法，其决定了如何将输入数据切分为多个计算块，并指导Kernel如何分配内存、调度计算任务。Tiling与Kernel之间通过`TilingData`结构体进行信息传递。

### 代码实现

Tiling一共需要三个交付件：```${op_name}_tiling.cpp``` ```${op_name}_tiling_key.h``` ```${op_name}_tiling_data.h```

> 说明：
>
> 1. `${op_name}_tiling.cpp`放在`${op_name}/op_host`目录下；
> 2. `${op_name}_tiling_key.h`和`${op_name}_tiling_data.h`放在`${op_name}/op_kernel`目录下；
> 3. 如果`${op_name}_tiling.cpp`中需要引用`${op_name}_tiling_data.h`，请使用相对路径的方式，例如：`#include "../op_kernel/${op_name}_tiling_data.h"`。

**交付件1：${op_name}_tiling.cpp**

Tiling主要切分逻辑。

如需查看详细实现，请参考[add_example_tiling.cpp](../../../examples/add_example/op_host/add_example_tiling.cpp)。

> **样例中函数空实现说明：**
>
> 1. **TilingParse**：图模式标准交付件，保留函数定义以满足框架调用规范，无实际逻辑时可置空。
> 2. **CompileInfo**：图模式标准交付件，保留函数定义以满足框架调用规范，无实际逻辑时可置空。

```CPP
// ${op_name}_tiling.cpp
// 1.Tiling需要获取运行环境信息，包括可用核数、UB(Unified Buffer)大小，并将获取到的信息传递给CompileInfo, 自动生成aclnn不调用该函数，直接返回ge::GRAPH_SUCCESS即可。
static ge::graphStatus TilingParse(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
    // 若手写aclnn接口，可以按照下面步骤完善parse函数
    // // 1.1获取环境信息
    // auto compileInfo = context->GetCompiledInfo<CompileInfo>();
    // OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // auto platformInfo = context->GetPlatformInfo();
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // // 1.2获取可用核数
    // compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // // 1.3获取UB大小
    // uint64_t ubSizePlatForm;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    // compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    // ...
    // return ge::GRAPH_SUCCESS;
}

// 2.Tiling计算主入口
static ge::graphStatus TilingFunc(gert::TilingContext* context){
    // 2.1获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2.2获取输入信息
    // 获取输入张量shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);

    // 如果输入shape是标量，转换为{1}，否则保持原shape不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    // 获取输入张量的描述信息
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 获取数据类型
    dataType = inputDesc->GetDataType();

    // 2.3计算Tiling参数（根据算子功能不同自行设计）
    ...

    // 2.4设置TilingData信息
    ${op_name}TilingData* tiling = context->GetTilingData<${op_name}TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(${op_name}TilingData), 0, sizeof(${op_name}TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = totalIdx;
    tiling->tileNum = TILE_NUM;

    // 2.5设置WorkspaceSize（可选）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
}

// 3.Tiling注册入口
IMPL_OP_OPTILING(${op_name}).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
```

**交付件2：${op_name}_tiling_key.h**

TilingKey是一个算子内为了区分不同的实现而将kernel代码进行区分的方法，kernel侧可以通过TilingKey来选择不同的算法逻辑。

如需查看详细实现，请参考[add_example_tiling_key.h](../../../examples/add_example/op_kernel/add_example_tiling_key.h)。

> **说明：** 如需实现复杂参数组合完成分支选择（涉及多TilingKey场景），请参考[《Ascend C算子开发接口》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“Utils API > Tiling模版编程 > 模版参数含义”。

```CPP
// ${op_name}_tiling_key.h
ASCENDC_TPL_ARGS_DECL(
    ${op_name},
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1)));
```

**交付件3：${op_name}_tiling_data.h**

切分算法相关的参数，比如总数据量大小、每个核数据切块数量，通过结构体存储。

如需查看详细实现，请参考[add_example_tiling_data.h](../../../examples/add_example/op_kernel/add_example_tiling_data.h)。

```CPP
// ${op_name}_tiling_data.h
struct ${op_name}TilingData {
    int64_t totalLength;
    int64_t tileNum;
};
```

## Kernel实现

### Kernel简介

Kernel是算子在NPU执行的核心部分，负责张量数据的加载、计算和存储，是算子功能实现的最终载体。Kernel的实现需要与Tiling策略紧密配合，根据Tiling提供的`TilingData`、`TilingKey`信息进行内存分配和计算调度。

Kernel实现包括如下步骤，整个流程通过`Process`函数串联，实现完整的算子流程。

```mermaid
graph LR
    H([核函数定义]) -->A([定义Kernel类])
    A -->B([初始化函数<br>Init])
    B -->C([主处理函数<br>Process])
    subgraph C [主处理函数Process]
        D([数据搬入<br>CopyIn]) -->E([计算<br>Compute]) -->F([数据搬出<br>CopyOut])
    end
    F -->G([Kernel执行完成])
```

### 代码实现

Kernel一共需要两个交付件：```${op_name}.cpp``` ```${op_name}.h```

> 说明：
>
> 1. `${op_name}.cpp`为kernel的入口函数只能放在`${op_name}/op_kernel`目录下；
> 2. `${op_name}.h`文件可以按照不同SoC或模板放在对应目录下，例如：`${op_name}/op_kernel/arch22`、`${op_name}/op_kernel/arch35`或`${op_name}/op_kernel/impl`等目录下；

**交付件1：${op_name}.cpp**

Kernel入口文件，包含主函数和调度逻辑。

如需查看详细实现，请参考[add_example.cpp](../../../examples/add_example/op_kernel/add_example.cpp)。

```cpp
// 1、核函数定义
// schMode是一个模板参数，用于支持不同数据类型（如float和int32）的计算路径
// __global__ __aicore__表示该函数是个全局函数，可以在AI Core上执行
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling){
    ....
    // Tiling注册入口
    REGISTER_TILING_DEFAULT(AddExampleTilingData);

    // 宏方式获取TilingData
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);

    // 根据TilingKey实例化Kernel对象并完成计算
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_FLOAT)) { // float数据类型走该分支
        NsAddExample::AddExample<float> op;     // 算子Kernel实例获取
        op.Init(x, y, z, &tilingData);          // 算子Kernel实例初始化
        op.Process();                           // 算子Kernel实例执行
    }
    ....
}
```

**交付件2：${op_name}.h**

定义Kernel头文件，包含函数声明、结构定义、逻辑实现等。

如需查看详细实现，请参考[add_example.h](../../../examples/add_example/op_kernel/add_example.h)。

```cpp
// 2、定义Kernel类
template <typename T>
class AddExample
{
public:
    // 默认构造函数，__aicore__表示该函数在AI Core上运行
    __aicore__ inline AddExample(){};
    // 初始化函数，用于设置输入输出地址和Tiling切分信息计算
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData);
    // 主处理函数，执行数据拷贝和计算
    __aicore__ inline void Process();

private:
    // 数据从GM拷贝到LM的函数
    __aicore__ inline void CopyIn(int32_t progress);
    // 数据从LM拷贝到GM的函数
    __aicore__ inline void CopyOut(int32_t progress);
    // 执行计算的函数，datalength表示当前处理的数据长度
    __aicore__ inline void Compute(const int32_t dataLength);

private:
    // 管道对象，用于管理数据流（拷贝和计算的流水线）
    TPipe pipe_;
    // 输入队列X，从GM拷贝到LM，BUFFER_NUM表示buffer数量，开启double buff达到流水并行，为2
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX_;
    // 输入队列Y，从GM拷贝到LM，BUFFER_NUM表示buffer数量，开启double buff达到流水并行，为2
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY_;
    // 输出队列Z，从LM拷贝到GM，BUFFER_NUM表示buffer数量，这里开启double buff达到流水并行，为2
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ_;

    // 输入X的GM地址
    GlobalTensor<T> inputGMX_;
    // 输入Y的GM地址
    GlobalTensor<T> inputGMY_;
    // 输入Z的GM地址
    GlobalTensor<T> outputGMZ_;

    // 总数据长度
    int64_t blockLength_ = 0;
    // 每个block被划分多少块
    int64_t tileNum_ = 0;
    // 每个tile处理数据长度
    int64_t tileLength_ = 0;
    ...
};

// 3、初始化函数Init
template <typename T>
__aicore__ inline void AddExample<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData)
{
    // 3.1 初始化成员变量
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    ...
    // 3.2 初始化GM地址
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    ...
    // 3.3 初始化队列长度
    pipe.InitBuffer(inputQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
    ...
}

// 4、主处理函数Process
template <typename T>
__aicore__ inline void AddExample<T>::Process()
{
    // 计算当前核处理数据循环次数
    int32_t loopCount = tileNum_ * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);              // 数据搬入
        Compute(i);             // 计算
        CopyOut(i);             // 数据搬出
    }
}
...
```

## aclnn适配

通常算子开发和编译完成后，会自动生成aclnn接口（一套基于C的API），可直接在应用程序中调用aclnn接口实现调用算子。

为实现该调用方式，需提前生成算子对应的二进制包，本项目无需手动配置，通过${op_name}_def.cpp已自动生成算子二进制包，支持开发者直接使用。

## 编译部署

算子开发完成后，需对算子工程进行编译，生成自定义算子安装包\*\.run，具体操作如下：

1. **准备工作。**

    参考[工程创建](#工程创建)完成基础环境搭建，同时检查算子开发交付件是否完备，是否在对应算子分类目录下。

2. **配置环境变量。**

    根据实际场景，选择合适的命令。

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/cann/set_env.sh
    # 指定路径安装
    # source ${install_path}/cann/set_env.sh
    ```

3. **编译自定义算子包。**

    以`AddExample`算子为例，假设开发交付件在`examples`目录，完整代码参见[add_example](../../../examples/add_example)目录。

    > 说明：编译过程依赖第三方开源软件，联网场景会自动下载，离线编译场景需要自行安装，具体参考[未联网编译](../install/compile.md#未联网编译)。

    ```bash
    # 编译指定算子，如bash build.sh --pkg --ops=add_example
    bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list} [--experimental]
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"--ops=add_example"。
    - --experimental（可选）：若编译的算子为贡献算子，需配置--experimental。

    若提示如下信息，说明编译成功：

    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```

4. **安装自定义算子包。**

    ```bash
    # 安装run包
    ./build_out/cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```

    自定义算子包安装在```${ASCEND_HOME_PATH}/opp/vendors```路径中，```${ASCEND_HOME_PATH}```表示CANN软件安装目录，可提前在环境变量中配置。

5. **（可选）删除自定义算子包。**

    注意自定义算子包不支持卸载，如需卸载，请删除vendors\/\$\{vendor\_name}目录，并删除vendors/config.ini中load_priority对应\$\{vendor\_name\}的配置项。

## 算子验证

算子开发过程中，可通过如下方式进行验证：

1. [UT验证](#ut验证): 验证交付件代码能否正常运行。UT验证无需NPU环境。

2. [aclnn调用验证](#aclnn调用验证): 验证算子在NPU环境上的功能。aclnn调用验证需要NPU环境。

### UT验证

在主要交付件代码开发过程中，可通过UT验证方式进行快速验证，无需编译部署算子包。

UT目录结构如下，需用户手动创建：

```bash
${op_name}
...                                                     # 其他交付件
└── tests                                               # 测试交付件
    └── ut                                              # UT实现
        ├── op_host
        │   └── test_${op_name}_tiling.cpp              # Tiling UT实现
        │   └── test_${op_name}_infershape.cpp          # Infershape UT实现
        └── op_kernel
            └── test_${op_name}.cpp                     # Kernel UT实现
```

执行UT验证的命令，请参考[算子调用](../invocation/quick_op_invocation.md)。下面将依次介绍各UT交付件的编写。

#### Infershape UT

Infershape UT用于验证host侧Infershape逻辑是否正确，在给定算子的输入后，Infershape能否正确执行、输出是否符合预期，推荐在算子开发阶段同步补齐。

UT编写指导如下，如需查看详细实现，请参考样例UT实现[test_add_example_infershape.cpp](../../../examples/add_example/tests/ut/op_host/test_add_example_infershape.cpp)。

**1. 组织结构与命名建议**

- **头文件**：统一包含`iostream`, `gtest/gtest.h`、`infer_shape_context_faker.h`、`infer_shape_case_executor.h`。
- **测试类**：继承`testing::Test`，实现`SetUpTestCase/TearDownTestCase`统一做数据准备与清理。
- **命名**：测试类建议`${OpName}InfershapeTest`，用例名建议`test_case_xxx`，可读性更高。

测试类示例：

```CPP
class ${OpName}InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "${OpName}InfershapeTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "${OpName}InfershapeTest TearDown" << std::endl;
    }
};
```

**2. 用例基本流程**

1) 调用接口构造用例上下文。需要的参数主要为输入和输出的shape/format/dtype。
    - shape/format/dtype可参考`${op_name}_def.cpp`算子信息库
    - 若某输入在信息库中标记为`ValueDepend`，UT中需同时准备该输入的**真实数据值**。
2) 设定预期结果。
3) 调用接口执行用例。

简化示例：

```CPP
TEST_F(${OpName}InfershapeTest, test_case_xxx)
{
    // 1. 构造用例上下文
    gert::InfershapeContextPara infershapeContextPara(
        "${OpName}",
        {
            {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // input tensor1
            {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // input tensor2
            // 若输入为ValueDepend，需额外传入true和constValue这两个参数
            // 其中constValue为自己定义的变量，如int constValue[2] = {2, 2}
            // {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, constValue}
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // output tensor
        }
    );
    // 2. 设定预期结果
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, -1, -1, 64},
    };
    // 3. 调用接口执行用例
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

#### Tiling UT

Tiling UT用于验证host侧Tiling逻辑是否正确，在给定算子的输入后，Tiling能否正确执行、输出是否符合预期，推荐在算子开发阶段同步补齐。

UT编写指导如下，如需查看详细实现，请参考样例UT实现[test_add_example_tiling.cpp](../../../examples/add_example/tests/ut/op_host/test_add_example_tiling.cpp)。

**1. 组织结构与命名建议**

- **头文件**：统一包含`iostream`, `gtest/gtest.h`、`tiling_context_faker.h`、`tiling_case_executor.h`。
  - 若tiling头文件中已经定义CompileInfo结构体，则也需引入。
- **测试类**：继承`testing::Test`，实现`SetUpTestCase/TearDownTestCase`统一做数据准备与清理。
- **命名**：测试类建议`${OpName}TilingTest`，用例名建议`test_case_xxx`，可读性更高。

测试类示例：

```CPP
class ${OpName}TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "${OpName}TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "${OpName}TilingTest TearDown" << std::endl;
    }
};
```

**2. 用例基本流程**

1) 调用接口构造用例上下文。需要的参数主要为输入和输出的shape/format/dtype、属性以及compileInfo，可参考`${op_name}_def.cpp`算子信息库。
    - shape/format/dtype和属性可参考`${op_name}_def.cpp`算子信息库。
    - 若某输入在信息库中标记为`ValueDepend`，UT中需同时准备该输入的**真实数据值**。
    - compileInfo优先使用tiling头文件中声明的结构体，若tiling头文件没有声明，则在用例中声明。
2) 设定预期结果。
3) 调用接口执行用例。

简化示例：

```CPP
TEST_F(${OpName}TilingTest, test_case_xxx)
{
    // 声明结构体并初始化一个结构体变量
    struct ${OpName}CompileInfo {
    } compileInfo;
    // 1. 构造用例上下文
    gert::TilingContextPara tilingContextPara(
        "${OpName}",
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor1
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // input tensor2
            // 若输入为ValueDepend，需额外传入true和constValue这两个参数
            // 其中constValue为自己定义的变量，如int constValue[2] = {2, 2}
            // {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND, true, constValue}
        },
        {
            {{{32, 4, 4, 4}, {32, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output tensor
        },
        {
            // 属性
            gert::TilingContextPara::OpAttr("${attr_name}", AnyValue::CreateFrom<std::string>("${attr_value}"))
        },
        &compileInfo,
        64,     // tiling阶段获取的核数
        262144, // tiling阶段获取的ub大小，但实际获取的值比指定值少256字节
        4096    // 指定tiling阶段中tiling data的最大值
    );
    // 2. 设定预期结果
    uint64_t expectTilingKey = 0;
    string expectTilingData = "2048 32 10912 ";
    std::vector<size_t> expectWorkspaces = {0};
    // 3. 调用接口执行用例
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
```

#### Kernel UT

Kernel UT用于验证Device侧Kernel逻辑是否正确，在给定输入/Tiling参数后，Kernel能否正确执行、输出是否符合预期，推荐在算子开发阶段同步补齐。

UT编写指导如下，如需查看详细实现，请参考样例UT实现[test_add_example.cpp](../../../examples/add_example/tests/ut/op_kernel/test_add_example.cpp)。

**1. 组织结构与命名建议**

- **头文件**：建议统一包含`gtest/gtest.h`、`tikicpulib.h`、`data_utils.h`与Tiling头文件。
  - 直接引用`op_host/${op_name}_tiling.h`
  - 或在UT目录提供轻量适配头（如`examples/add_example/tests/ut/op_kernel/add_example_tiling.h`）
  - 若Kernel为模板函数，可在UT中直接`#include "../../../op_kernel/${op_name}.cpp"`触发实例化（参考`AddExample`）
- **测试类**：继承`testing::Test`，实现`SetUpTestCase/TearDownTestCase`统一做数据准备与清理（如拷贝数据目录、chmod、生成bin）。
- **命名**：测试类建议`${OpName}KernelTest`，用例名建议`test_case_xxx`，可读性更高。

测试类示例：

```CPP
class ${OpName}KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "${OpName}KernelTest SetUp" << std::endl;
        // 在此统一准备测试数据
    }
    static void TearDownTestCase()
    {
        std::cout << "${OpName}KernelTest TearDown" << std::endl;
    }
};
```

**2. 用例基本流程**

1) 设定输入shape/format/dtype，初次上手可参考`${op_name}_def.cpp`算子信息库。
    - 若某输入在信息库中标记为`ValueDepend`，UT中需同时准备该输入的**真实数据值**。
2) 准备输入/输出/Workspace/Tiling缓冲区（`AscendC::GmAlloc`）。
3) 准备Tiling数据（手动构造或由Tiling函数生成）。
4) 设置`ICPU_SET_TILING_KEY`与`AscendC::SetKernelMode`。
5) 使用`ICPU_RUN_KF`执行Kernel。
6) 结果校验并释放资源（`AscendC::GmFree`）。

简化示例：

```CPP
extern "C" __global__ __aicore__ void ${op_name}(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                GM_ADDR workspace, GM_ADDR tiling);

TEST_F(${OpName}KernelTest, test_case_basic)
{
    // 1.设定输入shape/format/dtype，必要时准备ValueDepend输入值
    // 2.申请输入/输出/workspace/tiling内存
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(...);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(...);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(...);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(...);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(${op_name}TilingData));

    // 3.准备tiling数据（手动构造或由tiling函数生成）
    auto* tilingData = reinterpret_cast<${op_name}TilingData*>(tiling);
    tilingData->... = ...;

    // 4.设置tiling key并执行kernel
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(${op_name}, blockDim, x, y, z, workspace, tiling);

    // 5.结果校验
    EXPECT_EQ(..., ...);

    // 6.释放资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
```

**3. Tiling数据准备方式**

- **手动构造**：适合字段少、逻辑简单。
- **调用Tiling函数自动生成**：适合字段多、依赖属性/shape复杂。可复用`tests/ut/framework_normal/common/tiling_context_faker.h`与`tiling_case_executor.h`。示例：

```CPP
gert::TilingContextPara para("OpName",
    {{{{2, 2, 2, 1}, {2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
    {{{{2, 1, 2, 2}, {2, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
    {gert::TilingContextPara::OpAttr("attr", AnyValue::CreateFrom<int64_t>(1))},
    &compileInfo);

TilingInfo tilingInfo;
ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
uint32_t blockDim = tilingInfo.blockNum;
```

**4. 数据生成与结果比对**

- 可使用`tests/ut/framework_normal/op_kernel/data_utils.h`的`ReadFile/WriteFile`读写二进制。
- 结合`gen_data.py`/`compare_data.py`脚本生成与比对数据，可参考`add_example`的`add_example_data`目录：
  [gen_data.py](../../../examples/add_example/tests/ut/op_kernel/add_example_data/gen_data.py)、
  [compare_data.py](../../../examples/add_example/tests/ut/op_kernel/add_example_data/compare_data.py)。
- 简单算子可直接在UT中计算期望值并比对。
  - 浮点比较建议使用`EXPECT_NEAR/ASSERT_NEAR`并设置合理容差。

### aclnn调用验证

```bash
# 执行前需要导入环境变量
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

开发好的算子完成编译部署后，可通过aclnn方式验证功能，方法请参考[算子调用方式](../invocation/quick_op_invocation.md)。

## 附录

### 算子工程迁移

由于[Ascend/samples](https://gitee.com/ascend/samples/tree/master)工程与本项目工程（参考[工程创建](#工程创建)）有差异，因此算子实现交付件和数量不同，可参考下表迁移`operator`目录中的算子样例。

<table border="1">
  <tr>
    <th>Ascend/samples</th>
    <th>本项目</th>
    <th>迁移方法</th>
    <th>代码示例</th>
  </tr>
  <tr>
    <td rowspan="4">op_host/{op_name}.cpp</td>
    <td>op_host/{op_name}_def.cpp</td>
    <td>将原有op_host/{op_name}.cpp中算子原型描述部分独立出来</td>
    <td><a href="#op_host/{op_name}_def.cpp">op_host/{op_name}_def.cpp</a>
    </td>
  </tr>
  <tr>
    <td>op_host/{op_name}_infershape.cpp</td>
    <td>（可选）将原有op_host/{op_name}.cpp中shape推导部分独立出来</td>
    <td><a href="#op_host/{op_name}_infershape.cpp">op_host/{op_name}_infershape.cpp</a>
    </td>
  </tr>
  <tr>
    <td>op_host/{op_name}_tiling.cpp</td>
    <td>仅保留原有op_host/{op_name}.cpp中的TilingFunc</td>
    <td><a href="#op_host/{op_name}_tiling.cpp">op_host/{op_name}_tiling.cpp</a></td>
  </tr>
  <tr>
    <td>op_graph/{op_name}_graph_infer.cpp</td>
    <td>（可选）将原有op_host/{op_name}.cpp中类型推导部分独立出来</td>
    <td><a href="#op_graph/{op_name}_graph_infer.cpp">op_graph/{op_name}_graph_infer.cpp</a></td>
  </tr>
  <tr>
    <td>op_host/{op_name}_tiling.h</td>
    <td>op_kernel/{op_name}_tiling_data.h</td>
    <td>将原有op_host目录下的宏定义Tiling结构体定义改成C++标准定义</td>
    <td><a href="#op_kernel/{op_name}_tiling_data.h">op_kernel/{op_name}_tiling_data.h</a></td>
  </tr>
  <tr>
    <td rowspan="2">op_kernel/{op_name}.cpp</td>
    <td>op_kernel/{op_name}.h</td>
    <td>保留原有op_host/{op_name}.cpp中Kernel实现的算子类定义部分</td>
    <td><a href="#op_kernel/{op_name}.h">op_kernel/{op_name}.h</a></td>
  </tr>
  <tr>
    <td>op_kernel/{op_name}.cpp</td>
    <td>将原有op_host/{op_name}.cpp中Kernel实现的核函数迁移至cpp文件，同时:
      <li>新增REGISTER_TILING_DEFAULT调用注册Tiling结构体，使用GET_TILING_DATA_WITH_STRUCT获取TilingData</li>
      <li>添加Tiling模板，支持模板参数的传入，根据模板参数的分支选择不同的Kernel侧实现</li>
    </td>
    <td><a href="#op_kernel/{op_name}.cpp">op_kernel/{op_name}.cpp</a></td>
  </tr>
  <tr>
    <td>op_kernel/tiling_key_{op_name}.h</td>
    <td>op_kernel/{op_name}_tiling_key.h</td>
    <td>保留原有op_kernel/tiling_key_{op_name}.h中算子的模板参数定义，若不存在op_kernel/tiling_key_{op_name}.h，新增定义模板参数和模板参数组合</td>
    <td><a href="#op_kernel/{op_name}_tiling_key.h">op_kernel/{op_name}_tiling_key.h</a></td>
  </tr>
</table>

<div id="op_host/{op_name}_def.cpp">
<p style="font-size:18px;"><b>op_host/{op_name}_def.cpp</b></p>
</div>

将原有${op_name}.cpp中算子信息库内容独立迁移至该文件，需要去掉SetInferShape和SetTiling内容。

```CPP
// 原有${op_name}.cpp中算子信息库内容
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
        ....
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);   // 需要去掉SetInferShape
        this->AICore()
            .SetTiling(optiling::TilingFunc)                                       // 需要去掉SetTiling
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
} // namespace ops

// 迁移至op_host/{op_name}_def.cpp后，代码中无SetInferShape和SetTiling内容
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
        ....
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore()
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
} // namespace ops
```

<div id="op_host/{op_name}_infershape.cpp">
<p style="font-size:18px;"><b>op_host/{op_name}_infershape.cpp</b></p>
</div>

图模式场景需要适配该文件，将原有${op_name}.cpp中shape推导部分独立迁至该文件，调用接口IMPL_OP_INFERSHAPE完成InferShape注册。

```CPP
// 原有${op_name}.cpp中的InferShape
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
} // namespace ge

// 迁移至op_host/{op_name}_infershape.cpp后，调用接口IMPL_OP_INFERSHAPE完成InferShape注册
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AddCustom).InferShape(InferShape);   // 在该文件中完成InferShape注册
} // namespace ge
```

<div id="op_host/{op_name}_tiling.cpp">
<p style="font-size:18px;"><b>op_host/{op_name}_tiling.cpp</b></p>
</div>

将原有${op_name}.cpp中TilingFunc迁移至该文件后，调用接口IMPL_OP_OPTILING完成TilingFunc注册。
宏定义TilingData结构体改成标准C++结构体后，TilingFunc中对结构体成员变量不再使用tiling.set_xxx的方式进行赋值，而是直接对成员变量赋值。
若是新增定义模板参数和模板参数组合，TilingFunc中需要同时配置模板参数tilingKey。
可参考[add_example_tiling.cpp](../../../examples/add_example/op_host/add_example_tiling.cpp)。

```CPP
// 原有${op_name}.cpp中TilingFunc
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t DEFAULT_TILE_NUM = 8;
constexpr int MIN_LENGTH_FOR_SPLIT = 2048;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
    ....
    tiling.set_totalLength(totalLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT); // 模板参数tilingkey配置
    context->SetTilingKey(tilingKey);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

// 迁移至op_host/{op_name}_tiling.cpp后，调用接口IMPL_OP_OPTILING完成TilingFunc注册，直接对结构体成员变量赋值，
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t DEFAULT_TILE_NUM = 8;
constexpr int MIN_LENGTH_FOR_SPLIT = 2048;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // TilingData tiling;
    TilingData* tiling = context->GetTilingData<TilingData>();
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
    ....
    tiling->totalLength = totalLength;   // 直接对结构体成员变量赋值
    // tiling.set_totalLength(totalLength);   // 不再使用tiling.set_xxx的方式进行赋值
    // tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    // context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT); // 模板参数tilingkey配置
    context->SetTilingKey(tilingKey);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(AddCustom).Tiling(TilingFunc);   // 在该文件中完成TilingFunc注册
} // namespace optiling
```

<div id="op_graph/{op_name}_graph_infer.cpp">
<p style="font-size:18px;"><b>op_graph/{op_name}_graph_infer.cpp</b></p>
</div>
图模式场景需要适配该文件，将原有${op_name}.cpp中类型推导独立迁移至该文件后，调用接口IMPL_OP完成InferDataType注册。

```CPP
// 原有${op_name}.cpp中InferDataType
namespace ge {
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

// 迁移至op_graph/{op_name}_graph_infer.cpp后，调用接口IMPL_OP完成InferDataType注册
namespace ge {
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP(AddCustom).InferDataType(InferDataType);   // 在该文件中完成InferDataType函数注册
} // namespace ge
```

<div id="op_kernel/{op_name}_tiling_data.h">
<p style="font-size:18px;"><b>op_kernel/{op_name}_tiling_data.h</b></p>
</div>

```CPP
// 原有op_host/{op_name}_tiling.h中的宏定义TilingData结构体
namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XXX, TilingData)
} // namespace optiling

// 迁移至op_kernel/{op_name}_tiling_data.h后，改成C++标准结构体
struct TilingData {
    uint32_t  totalLength;
};
```

<div id="op_kernel/{op_name}.h">
<p style="font-size:18px;"><b>op_kernel/{op_name}.h</b></p>
</div>

保留原有op_host/{op_name}.cpp中kernel实现的算子类定义部分。

<div id="op_kernel/{op_name}.cpp">
<p style="font-size:18px;"><b>op_kernel/{op_name}.cpp</b></p>
</div>

```CPP
// 原有op_kernel/{op_name}.cpp中的核函数实现
template<int D_T_X, int D_T_Y, int D_T_Z, int TILE_NUM, int IS_SPLIT>
 __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if(D_T_X == ADD_TPL_FP32 && D_T_Y == ADD_TPL_FP32 && D_T_Z == ADD_TPL_FP32){
        KernelAdd<float, float, float> op;
        op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
        op.Process1();
    }else if(D_T_X == ADD_TPL_FP16 && D_T_Y == ADD_TPL_FP16 && D_T_Z == ADD_TPL_FP16){
        KernelAdd<half, half, half> op;
        if(IS_SPLIT == 0){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process1();
        }else if(IS_SPLIT == 1){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process2();
        }
    }
}

// 迁移至op_kernel/{op_name}.cpp后，新增REGISTER_TILING_DEFAULT调用注册Tiling结构体，使用GET_TILING_DATA_WITH_STRUCT获取TilingData
template<int D_T_X, int D_T_Y, int D_T_Z, int TILE_NUM, int IS_SPLIT>
 __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // GET_TILING_DATA(tiling_data, tiling);
    REGISTER_TILING_DEFAULT(TilingData);   // 新增REGISTER_TILING_DEFAULT调用注册TilingData结构体
    GET_TILING_DATA_WITH_STRUCT(TilingData, tiling_data, tiling);   // 宏GET_TILING_DATA_WITH_STRUCT获取TilingData
    if(D_T_X == ADD_TPL_FP32 && D_T_Y == ADD_TPL_FP32 && D_T_Z == ADD_TPL_FP32){
        KernelAdd<float, float, float> op;
        op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
        op.Process1();
    }else if(D_T_X == ADD_TPL_FP16 && D_T_Y == ADD_TPL_FP16 && D_T_Z == ADD_TPL_FP16){
        KernelAdd<half, half, half> op;
        if(IS_SPLIT == 0){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process1();
        }else if(IS_SPLIT == 1){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process2();
        }
    }
}
```

<div id="op_kernel/{op_name}_tiling_key.h">
<p style="font-size:18px;"><b>op_kernel/{op_name}_tiling_key.h</b></p>
</div>

保留原有op\_kernel/tiling\_key\_{op\_name}.h中算子的模板参数定义，若不存在op\_kernel/tiling\_key\_{op\_name}.h，请参考[add_example_tiling_key.h](../../../examples/add_example/op_kernel/add_example_tiling_key.h)新增定义模板参数和模板参数组合。

### 算子跨平台迁移

完成算子代码开发后，如需实现多平台间（如Atlas A2/A3等）的算子代码迁移，需考虑硬件结构差异引发的软件实现变更。

您可以参考[算子跨平台迁移指导](./cross_platform_migration_guide.md)，该文档提供了常见的适配要点与方法论，并提供相关算子的适配样例。
