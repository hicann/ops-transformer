# 编译与运行样例

## 前提说明

- 如需编译执行算子API，请确保基础环境已搭建完成，包括驱动、固件、CANN软件包、ops包等。

- 算子API的调用流程和编译运行操作详情请参见[《应用开发（C&C++）》](https://hiascend.com/document/redirect/CannCommunityCppInferWizard)中“单算子调用>单算子API执行>调用aclnn接口示例代码”。

## 编译前准备

本章以开发和运行环境合设场景为例，即带AI处理器的机器既作为开发环境又作为运行环境。该场景下，代码开发和代码运行在同一台机器上。这里以**FlashAttentionScore算子**为例，其他算子的调用逻辑、流程、编译脚本与FlashAttentionScore算子大致一样，请根据实际情况自行修改API调用脚本(\*.cpp)和编译脚本(CMakeLists)。

- **示例代码**

   已知FlashAttentionScore算子实现self-attention（自注意力）的计算：

  - psetype=1时，需要先add再mul。
  - psetype≠1时，需要先mul再add。

- 计算公式：

  注意力的正向计算公式如下：

  - psetype=1时，公式如下：
      $$
      attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
      $$

  - psetype≠1时，公式如下：
      $$
      attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
      $$

  您可以从[aclnnFlashAttentionScore](../../../attention/flash_attention_score/docs/aclnnFlashAttentionScore.md)文档中获取调用示例代码，并将代码文件命名为“**test\_flash_attention_score.cpp**”。

- **CMakeLists文件**

    CMake文件示例如下，请根据实际情况修改：

    ```bash
    # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.18.4)

    # 设置工程名
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # 设置编译选项
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
    add_executable(opapi_test
                   test_flash_attention_score.cpp)

    # 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
    if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
        ${INCLUDE_BASE_DIR}/aclnnop
        # custom包需要额外增加引用路径，参考如下，其中${vendor_name}需要修改为编译custom包时指定的，默认设置"custom"
        # ${ASCEND_PATH}/opp/vendors/${vendor_name}_transformer/op_api/include
    )

    # 设置链接的库文件路径
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_transformer.so
                          # custom包需要将libopapi_transformer.so替换为libcust_opapi.so，参考路径如下，
                          # 其中${vendor_name}需要修改为编译custom包时指定的，默认设置"custom"
                          # ${ASCEND_PATH}/opp/vendors/${vendor_name}_transformer/op_api/lib/libcust_opapi.so
                          )

    # 可执行文件在CMakeLists文件所在目录的bin目录下
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    本项目编译生成算子样例时，需额外链接libopapi_math.so动态库，因为算子在实现过程中调用了部分L0接口，这些接口封装在libopapi_math.so中，因此在编译链接阶段需显式声明此项依赖。

    对于集合通信和MatMul计算融合、并行的算子，统称为通算融合算子（简称MC2算子），包括AllGatherMatmul、AlltoAllAllGatherBatchMatMul、BatchMatMulReduceScatterAlltoAll、MatmulAllReduce、MatmulAllReduceAddRmsNorm、MatmulReduceScatter等。调用该类算子API时，一般会涉及多线程和HCCL（Huawei Collective Communication Library，集合通信库），因此CMake文件需要额外导入如下内容，否则无法成功编译。

  ```bash
  # 设置链接的库文件路径
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_transformer.so
                        # custom包需要将libopapi_transformer.so替换为libcust_opapi.so，参考路径如下，
                        # 其中${vendor_name}需要修改为编译custom包时指定的，默认设置"custom"
                        # ${ASCEND_PATH}/opp/vendors/${vendor_name}_transformer/op_api/lib/libcust_opapi.so
                        -Wl,--no-as-needed
                        ${ASCEND_PATH}/lib64/libhccl.so      # 集合通信库文件
                        ${ASCEND_PATH}/lib64/libhccl_fwk.so
                        ${CMAKE_THREAD_LIBS_INIT})           # 多线程依赖的库文件
  ```

  其中“find_package(Threads REQUIRED)”是CMake用于查找线程库的命令，可自动链接线程库依赖的头文件或间接依赖的库文件。

## 编译与运行

  1. 提前准备好算子的调用代码（\*.cpp）和编译脚本（CMakeLists.txt）。
  2. 配置环境变量。

     安装CANN软件后，使用CANN运行用户登录环境，执行如下命令生效环境变量。

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     其中${INSTALL_DIR}为CANN软件安装后文件存储路径，请根据实际情况替换。
  3. 编译并运行。
      - 进入CMakeLists.txt所在目录，执行如下命令，新建build目录存放生成的编译文件。

        ```bash
        mkdir -p build
        ```

      - 进入build目录，执行cmake命令编译，再执行make命令生成可执行文件。

        ```bash
        cd build
        cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
        make
        ```

        编译成功后，会在build目录的bin文件夹下生成opapi\_test可执行文件。

      - 进入bin目录，运行可执行文件opapi\_test。

        ```bash
        cd bin
        ./opapi_test
        ```

        以FlashAttentionScore算子的运行结果为例，运行后的结果示例如下：

        ```text
        result[0] is: 256.000000
        result[1] is: 256.000000
        result[2] is: 256.000000
        result[3] is: 256.000000
        result[4] is: 256.000000

        ...
        result[65532] is: 256.000000
        result[65533] is: 256.000000
        result[65534] is: 256.000000
        result[65535] is: 256.000000
        ```

        若执行结果报错，未出现预期结果，可以使用aclGetRecentErrMsg接口获取报错具体信息。
        调用aclnnFlashAttentionScoreGetWorkspaceSize报错获取异常信息示例如下：

        ```text
        // q is nullptr
        ret = aclnnFlashAttentionScoreGetWorkspaceSize(
          q, k, v, pse, dropMask, padding, attenmask, prefix, scaleValue,
          keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
          sparseMode, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        上述构造空指针问题获取报错信息示例如下:

        ```text
        aclnnFlashAttentionScoreGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): The query cannot be nullptr.
        ```
