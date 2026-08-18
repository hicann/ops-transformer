# 快速入门：基于ops-transformer仓

## 使用须知

本指南旨在帮助开发者快速上手CANN和`ops-transformer`算子仓的使用。为方便快速了解算子开发全流程，以**AddExample**算子为实践对象，算子源码位于`ops-transformer/examples/add_example`，操作流程如下：

1. **[前提条件](../README.md)**：参考项目README完成环境准备和源码下载，此处不再赘述。快速入门场景**推荐CANNLab或Docker部署**，操作简单。

   > **说明**：CANNLab或Docker环境默认提供最新版本CANN包；如需体验master分支最新能力，可手动搭建环境。

2. **[编译运行](#一编译运行)**：编译自定义算子包并安装，实现快速调用算子。

3. **[算子开发](#二算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。

4. **[算子调试](#三算子调试)**：掌握算子打印和性能采集方法。

5. **[算子验证](#四算子验证)**：学习如何修改算子example样例，以验证算子在不同输入下的功能正确性。

## 一、编译运行

本阶段目的是**快速体验项目标准流程**，验证环境能否成功进行算子源码编译、打包、安装和运行。

### 1. 进入项目源码

- CANNLab云开发环境：

   默认提供最新版本CANN包配套的项目源码，资源一般在`/mnt/workspace/gitCode`目录下，请进入目标分支源码目录。

- 非CANNLab云开发环境：

  根据[release仓库](https://gitcode.com/cann/release-management)源码与CANN版本配套关系，执行如下命令下载源码，\$\{tag\_version\}替换为目标分支标签，例如9.0.0。

  ```bash
  git clone -b ${tag_version} https://gitcode.com/cann/ops-transformer.git && cd ops-transformer
  ```

> 说明：如需切换源码分支版本，请参考如下指导。
>
> 1. 在源码目录执行`git branch`，查询当前源码版本。
> 2. 在源码目录执行`git checkout ${tag_version}`，切换到目标分支源码，注意满足源码与CANN版本配套关系。若源码已存在，执行`git pull`拉取最新源码。

### 2. 编译AddExample算子

编译指定的算子，通用编译命令格式：`bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>`。

> **说明**：编译前请确保已配置CANN环境变量，否则可能因找不到`ASCEND_HOME_PATH`等导致编译失败。默认路径安装时执行：
>
> ```bash
> source /usr/local/Ascend/cann/set_env.sh
> ```

以AddExample算子为例，编译命令如下：

```bash
bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
```

**SoC 取值映射表**<a id="soc-version-mapping"></a>

产品名对应的${soc_version}取值如下，请按实际场景传参。

| 产品系列 | ${soc_version} 取值 |
| -------- | ------------------- |
| Atlas A2 系列（训练/推理） | ascend910b |
| Atlas A3 系列（训练/推理） | ascend910_93 |
| 950 系列 | ascend950 |

若提示如下信息，说明编译成功。

```bash
Self-extractable archive "cann-ops-transformer-custom_linux.${arch}.run" successfully created.
```

编译成功后，run包存放于项目根目录的build_out目录下。

### 3. 安装AddExample算子包

```bash
./build_out/cann-ops-transformer-*linux*.run
```

`AddExample`安装在```${ASCEND_HOME_PATH}/opp/vendors```路径中，```${ASCEND_HOME_PATH}```表示CANN软件安装目录。

> 安装后执行 `grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini` 校验，预期输出含本 vendor。若无输出，通常是 `${ASCEND_HOME_PATH}/opp/vendors` 属主或权限不对，请排查后重新安装。

### 4. 配置环境变量

将自定义算子包的路径加入环境变量，确保运行时能够找到。

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. 快速验证：运行算子样例

通用的运行命令格式：`bash build.sh --run_example <算子名> <运行模式> <包模式>`。

以AddExample为例，其提供了简单算子样例`add_example/examples/test_aclnn_add_example.cpp`，运行该样例验证算子功能是否正常。

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom
```

预期输出：打印算子`AddExample`的加法计算结果，表明算子已成功部署并正确执行。

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 2.000000
...
```

## 二、算子开发

本阶段目的是对已成功运行的AddExample算子尝试**修改核函数代码**。

### 1. 修改Kernel实现

找到AddExample算子的核心kernel实现文件`ops-transformer/examples/add_example/op_kernel/add_example.h`，尝试将算子中的Add操作改为Mul操作：

```cpp
__aicore__ inline void AddExample<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === 在此处将Add替换为Mul ===
    // AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. 编译与验证

重复[编译运行](#一编译运行)章节中的步骤：

1. **重新编译**：
    先回到项目根目录，编译命令如下：

    ```bash
    bash build.sh --pkg --soc=${soc_version} --ops=add_example -j16
    ```

    产品名对应的${soc_version}取值参考[SoC 取值映射表](#soc-version-mapping)，请按实际场景传参。

2. **重新安装**：

    ```bash
    ./build_out/cann-ops-transformer-*linux*.run
    ```

3. **重新验证**：

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

4. **成功标志**：输出结果变成乘法结果。

    ```bash
    add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 1.000000
    add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 1.000000
    add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 1.000000
    add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 1.000000
    add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 1.000000
    add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 1.000000
    add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 1.000000
    add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 1.000000
    ...
    ```

## 三、算子调试

本阶段以AddExample为例，在算子中添加打印并采集算子性能数据，以便后续问题分析定位。

### 1. 打印

算子如果出现执行失败、精度异常等问题，添加打印进行问题分析和定位。

请在`examples/add_example/op_kernel/add_example.h`中进行代码修改。

- **printf**

  该接口支持打印Scalar类型数据，如整数、字符型、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“Ascend C算子开发接口 > SIMD API > 基础API > 调试接口 > 上板打印 > printf”。

  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // 打印当前核计算Block长度
  AscendC::PRINTF("Tiling blockLength is %ld\n", blockLength_);
  ```

- **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“Ascend C算子开发接口 > SIMD API > 基础API > 调试接口 > 上板打印 > DumpTensor”。

  ```c++
  // 在Compute函数中，Add之后、EnQue之前调用
  DumpTensor(zLocal, 0, 128);
  ```

### 2. 性能采集

当算子功能验证正确后，可通过`msprof op`命令采集算子级性能数据。

- **生成可执行文件**

    调用AddExample算子的example样例，生成可执行文件（test_aclnn_add_example），该文件位于项目`ops-transformer/build`目录。

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

- **采集性能数据**

    进入AddExample算子可执行文件目录`ops-transformer/build/`，执行如下命令：

    ```bash
    msprof op ./test_aclnn_add_example
    ```

    执行后会直接打印算子基础信息（如Op Name、Op Type、Task Duration、Block Dim等）和性能瓶颈提示。

采集结果保存在项目`ops-transformer/build/`目录下的`OPPROF_*`文件夹中，命令执行完后会自动解析并导出性能数据文件。如需进一步解读各项性能指标（如流水占比、带宽利用率等），请参见[《msProf工具使用指南》](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)。

## 四、算子验证

本阶段通过修改AddExample算子example样例中的输入数据，验证该算子在多种场景下的功能正确性。

### 1. 修改测试输入

找到并编辑`AddExample`的`ops-transformer/examples/add_example/examples/test_aclnn_add_example.cpp`，修改输入张量的形状和数值。

**修改输入/输出数据**：修改输入、输出的shape信息，以及初始化数据，构造相应的输入、输出tensor。

```c++
int main() {
    // ... 初始化代码 ...

    // === ① 修改selfX的输入 ===
    // 修改前：shape = {32, 4, 4, 4},数值全为1
    // 修改后：将输入shape改为 {8, 8, 8, 8}，并填充不同的测试数据
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 *8
    // 可使用循环填充更有区分度的数据，例如递增序列
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // 填充0-9的循环值
    }
    // === ② 参考selfX，同理修改selfY（第二个输入）与out（输出）的shape及数据 ===

    // ... 后续执行代码 ...
}
```

> **注意**：修改shape时需同步修改对应host侧数据vector的长度（本例`selfX`、`selfY`、`out`三者的vector长度都要从默认的2048改为4096 = 8×8×8×8），并保证三者shape一致，否则会出现数据越界或结果不匹配。

### 2. 重新编译并验证

1. 由于只修改了example测试代码，无需重新编译算子包。

2. 重新执行验证命令：

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

3. 观察算子输出结果是否符合预期。

## 结语

体验完上述流程，已基本完成算子开发过程，如果想进一步贡献新算子或学习更多高阶开发、调试等技能，请访问本项目README学习[进阶教程](../README.md#学习教程)和[贡献指南](../README.md#相关信息)等。
