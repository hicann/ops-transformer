
# 算子调试调优

## 调试定位（AI Core算子）

算子运行过程中，如果出现算子执行失败、精度异常等问题，可以打印各阶段信息，如Kernel中间结果，进行问题分析和定位。

### 1. Host侧日志获取方式

* **plog获取**

    程序执行结束后，默认可在"$HOME/ascend/log"下查看，host日志文件存储路径如下：

    ```bash
    $HOME/ascend/log/debug/plog/plog-pid_*.log
    ```

   开启环境变量ASCEND_SLOG_PRINT_TO_STDOUT可以将log日志直接打屏显示(1：开启打屏，0：关闭打屏)，配置示例如下：

    ```bash
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    ```

    日志相关介绍参见[《日志参考》](https://hiascend.com/document/redirect/CannCommunitylogref)，环境变量介绍参见[《环境变量参考》](https://hiascend.com/document/redirect/CannCommunityEnvRef)。

* **aclnn异常错误信息获取**

    通过aclGetRecentErrMsg接口（参见[《Runtime运行时API》](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)）获取aclnn接口调用过程中的异常信息，使用方法如下：

   ```cpp
   printf("%s",aclGetRecentErrMsg());
   ```

    打印错误信息样例如下：

   ```text
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
   ```

### 2. Kernel调试

常见调试方法如下：

* **printf**

    该接口支持打印Scalar类型数据，如整数、字符、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。

    ```c++
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    tileNum_ = tilingData->tileNum;
    tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
    // 打印当前核计算Block长度
    AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
    ```

* **DumpTensor**

    该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。

    ```c++
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    // 打印zLocal Tensor信息
    DumpTensor(zLocal, 0, 128);
    AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
    ```

* **msDebug单步调试**

  对于复杂场景的问题定位，比如算子卡死、GM/UB访问越界等场景，可以采取[msDebug](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/user_guide/msdebug_user_guide.md)单步调试的方式，具体操作步骤如下：

  1. **编译O0 -g的Kernel**

     使用msDebug调试前，需要编译不带优化且带调试信息的Kernel，通过`--op_debug_config "ccec_O0,ccec_g"`选项实现：

     ```bash
     bash build.sh --pkg --ops=add_example --soc=ascend910b --op_debug_config "ccec_O0,ccec_g"
     ```

     若上述编译失败，或运行算子本身就存在精度问题，可能是由于该算子tilingKey较多（大算子），需要对目标tilingKey对应的分支代码单独调试。此时需要增加`--tiling_key`选项指定目标tilingKey。

     首先可用msdebug直接运行一次算子，观察Kernel启动信息来确定目标tilingKey：

     ```
     [Launch of Kernel RunAicpuKfcResInit on Device 0]
     [Launch of Kernel RunAicpuKfcResInit on Device 1]
     [Launch of Kernel AddExample_c9e4347749ac13425bf7e31c6a5e306a_7 on Device 1]
     [Launch of Kernel AddExample_c9e4347749ac13425bf7e31c6a5e306a_7 on Device 0]
     ```

     其中`_c9e4347749ac13425bf7e31c6a5e306a`即为tilingKey。确定目标tilingKey后，编译命令如下：

     ```bash
     bash build.sh --pkg --ops=add_example --soc=ascend910b --op_debug_config "ccec_O0,ccec_g" --tiling_key=c9e4347749ac13425bf7e31c6a5e306a
     ```

  2. **启动msDebug调试**

     安装算子包后，进入算子可执行文件所在目录，使用msdebug启动调试：

     ```bash
     msdebug ./test_aclnn_add_example
     ```

     更多msDebug使用方法请参见[msDebug](https://gitcode.com/Ascend/msdebug/blob/master/docs/zh/user_guide/msdebug_user_guide.md)算子调试工具文档。

### 3. Kernel检测

[msSanitizer](https://www.hiascend.com/document/redirect/CannCommunityToolMssanitizer)是Ascend C算子内存检测和竞争检测工具，可用于检测Kernel运行过程中的GM/UB越界访问、内存泄漏、并发竞争等问题。下面以`add_example`算子为例，介绍使用步骤：

1. **编译使能检测的Kernel**

   通过`--op_debug_config "sanitizer"`选项编译带检测的Kernel，将下述命令中的`add_example`替换为实际待检测的算子名：

   ```bash
   bash build.sh --pkg --ops=add_example --soc=ascend910b --op_debug_config "sanitizer"
   ```

   对于Ascend 950，当前需临时修改CANN包中的opc编译脚本，在`_gen_compile_cmd_c310`函数中添加`--cce-enable-sanitizer`、`-g`、`-fno-jump-tables`选项：

   ```bash
   vim ${ASCEND_HOME_PATH}/cann/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_v220.py
   ```

2. **安装编译包**

   编译完成后，进入`build_out`目录安装生成的算子包（文件名随SoC版本不同而变化，以950为例）：

   ```bash
   cd build_out
   bash cann-ops-transformer-custom_linux-aarch64.run
   cd ..
   ```

3. **编译并运行测试用例**

   ```bash
   bash build.sh --run_example add_example eager cust --soc=ascend910b
   ```

4. **执行msSanitizer检测**

   使用mssanitizer同时执行内存检测和竞争检测：

   ```bash
   mssanitizer --tool=memcheck --tool=racecheck \
       --log-level=error \
       --kernel-name=AddExample \
       -- build/test_aclnn_add_example > mssanitizer_result.txt 2>&1
   ```

   检测结果输出到`mssanitizer_result.txt`中，更多msSanitizer使用方法请参见[msSanitizer](https://www.hiascend.com/document/redirect/CannCommunityToolMssanitizer)算子检测工具文档。

## 性能调优

算子运行过程中，如果出现执行精度下降、内存占用异常等问题，可通过[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)性能分析工具分析算子各运行阶段指标数据（如吞吐率、内存占用、耗时等），从而确定问题根源，并针对性地优化。

本章以`AddExample`自定义算子为例，主要介绍算子调优中常用的算子上板性能采集和流水图仿真的方式。

**适用场景区别：**

* **上板性能采集**：适用于在真实NPU硬件上运行算子，快速获取算子整体性能指标（如Kernel耗时、Block数、流水占比等），帮助判断算子是否存在性能问题。
* **流水图仿真**：适用于无NPU硬件开发者，或需要深入分析算子内部指令级流水瓶颈，优化指令排布的场景，提供比上板更详细的指令级流水分析。

### 方式一 上板性能采集

* **前提条件**

    完成算子开发和编译后，假设采用aclnn接口方式调用，生成的算子可执行文件（test_aclnn_add_example）所在目录为本项目`examples/add_example/examples/build/bin/`。

* **采集性能数据**

    当需要采集算子上板运行各项流水指标时可以进入算子可执行文件所在目录，执行如下命令：

    ```bash
    msprof op ./test_aclnn_add_example
    ```

    采集结果在本项目`examples/add_example/examples/build/bin/OPPROF_*`目录，采集完成后打印如下信息：

    ```text
    Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
    Op Type: vector
    Task Duration(us): 97.861954
    Block Dim: 8
    Mix Block Dim:
    Device Id: 0
    Pid: 2776181
    Current Freq: 1800
    Rated Freq: 1800
    ```

    其中Task Duration是当前算子Kernel耗时，Block Dim是当前算子执行核数。

    算子各项流水详细指标可关注`OPPROF_*`下`ArithmeticUtilization`文件，包含了当前各项流水的占比，具体介绍参见[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)中“性能数据文件 > msprof op > ArithmeticUtilization（cube及vector类型指令耗时和占比）”章节。

### 方式二 仿真流水图采集

* **前提条件**

    完成算子开发和编译后，假设采用aclnn接口方式调用，生成的算子可执行文件（test_aclnn_add_example）所在目录为本项目`examples/add_example/examples/build/bin/`。

* **针对Ascend 950PR，可使用[NPU Simulator](./npu_sim.md)仿真工具，执行仿真命令，生成仿真数据**

    执行仿真命令，生成仿真数据

    ```bash
    npusim record ./test_aclnn_add_example -s Ascend950 --gen-report
    ```

    仿真结果在本项目`examples/add_example/examples/build/bin/npusim_*/report/results/kernel_*/core_*`目录，流水相关文件为：

    ```bash
    trace_core0.json
    ```

    在Chrome浏览器中输入“chrome://tracing”地址，并将生成的指令流水图文件（trace_core0.json）拖到空白处打开，具体参数介绍参考NPU Simulator中[“仿真结果解析”](./npu_sim.md#仿真结果解析说明)章节。

* **针对Atlas A2/A3系列产品，可使用[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)工具，执行仿真命令，生成仿真数据**

    msProf工具进行算子仿真调优之前，需执行如下命令配置环境变量。

    ```bash
    export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
    ```

    请根据CANN软件包实际安装路径和AI处理器型号对以上环境变量进行修改。

    之后进入算子可执行文件所在目录，执行如下命令：

    ```bash
    msprof op simulator --output=$PWD/pipeline_auto --kernel-name "AddExample" ./test_aclnn_add_example
    ```

      采集结果在本项目`$PWD/pipeline_auto/OPPROF_**`目录中。
      其中流水相关文件路径为`OPPROF_*/simulator/visualize_data.bin`，可以借助[MindStudio Insight](https://www.hiascend.com/document/redirect/MindStudioInsight)工具中“基础操作 > 导入数据”章节查看如何导入流水数据。
