# AllGatherAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：完成AllGather通信与Add计算融合。
- 计算公式：

    $$
    gatherOut=AllGather(a0, a1)
    $$
    $$
    c[i]=gatherOut[i] + b[i]
    $$

- 算子语义示例：(两张卡参与计算，输入前缀分别为rank0和rank1)

  - 输入:
    rank0_a = [1,2,3];
    rank0_b = [4,5,6,7,8,9];
    rank1_a = [10,11,12];
    rank1_b = [13,14,15,16,17,18];

  - 输出：
    rank0_c = AllGatherAdd(rank0_a, rank1_a, rank0_b)
                = AllGather(rank0_a,rank1_a) + rank0_b
                = [1,2,3,10,11,12] + [4,5,6,7,8,9]
                = [5,7,9,17,19,21]

    rank1_c = AllGatherAdd(rank0_a, rank1_a, rank1_b)
                = AllGather(rank0_a,rank1_a) + rank1_b
                = [1,2,3,10,11,12] + [13,14,15,16,17,18]
                = [14,16,18,26,28,30]

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 380px">
 <col style="width: 120px">
 <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>输入</td>
      <td>算子输入a，通信域中所有卡的输入a会先参与AllGather计算生成gatherOut中间结果。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>算子输入b，通信域每张卡的输入b会分别与gatherOut结果进行Add计算生成输出c。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输出</td>
      <td>算子计算的最终输出。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gather_out</td>
      <td>输出</td>
      <td>算子AllGather通信部分的中间输出。</td>
      <td>FLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td><li>Host侧标识通信域的字符串，通信域名称。</li><li>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</li></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>commTurn</td>
      <td>属性</td>
      <td>通信数据切分数，即总数据量/单次通信量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rank_size</td>
      <td>属性</td>
      <td>通信域里面的卡数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

* 当前该示例算子仅支持固定shape：a(240, 256)，b(240 * 2, 256)，和固定rank_size = 2。
* 所有输入不支持空tensor场景，取值范围在[-5,5]之间。
* 确定性计算：
  - allGatherAdd算子默认确定性实现。

## 调用说明

调用本算子前，请确保已本地下载代码仓，并安装好如下基础依赖、NPU驱动和固件已安装。

本项目源码编译用到的依赖如下，请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md)文档中的**前提条件**和**环境准备**章节，完成环境依赖的下载和CANN包的准备，其中，**环境准备**章节中的**安装最新版CANN ops-math包**部分可以跳过。

**编译执行**

本示例算子使用自定义算子包方式编译执行。

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 执行命令示例：（在910b环境下编译all_gather_add算子
    # bash build.sh --pkg --soc=ascend910b --ops=all_gather_add
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas A2 推理系列产品使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops：填写本示例算子名称all_gather_add。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```

    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**

    ```bash
    # 首先到自定义算子包生成目录下
    cd build_out/
    # 然后安装自定义算子包
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```

    自定义算子包安装路径为```${ASCEND_HOME_PATH}/opp/vendors```，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/latest/opp。注意自定义算子包不支持卸载。

    若安装自定义算子包时有类似“environment variable ASCEND_CUSTOM_OPP_PATH=/xxx: is set and has multiple path in it(colon inside), which will cause the custom op installed incorrectly. Please use the	--install-path option to specify an installation path instead.”的报错信息，说明ASCEND_CUSTOM_OPP_PATH路径在环境中被提前设置，需要unset这个环境变量后再重新安装自定义算子包。

**本地验证**

通过项目根目录build.sh脚本，可快速调用算子和UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../../../docs/zh/install/build.md)。

- **执行算子样例**

    - 本示例算子目前仅支持基于单算子API执行方式调用算子，即[两段式aclnn接口](../../../docs/zh/context/two_phase_api.md)调用，本示例的aclnn接口定义在[op_api](../all_gather_add/op_host/op_api)路径下。

      如需执行该样例算子，需按以下步骤操作：

      1.将自定义算子包安装在默认路径下，参考**安装自定义算子包**章节；

      2.编写算子执行代码，通过调用[aclnn_all_gather_add.h](../all_gather_add/op_host/op_api/aclnn_all_gather_add.h)文件中的两段式接口aclnnAllGatherAddGetWorkspaceSize和aclnnAllGatherAdd来完成算子调用。此处需注意，算子执行代码需放置在算子的[examples](../all_gather_add/examples/)路径下，且文件名称需遵循"test_aclnn_[算子名称].cpp"格式，[测试工程运行脚本](../../../build.sh)会根据算子名称匹配对应的执行文件并运行。

      本示例的算子执行样例代码：[test_aclnn_all_gather_add.cpp](examples/test_aclnn_all_gather_add.cpp)。该代码首先按照固定shape随机生成测试数据作为输入，同时在cpu侧对输入数据进行拼接和相加来模拟AllGather、Add的运算并生成golden数据，然后将生成的测试数据转化为Tensor拷贝到设备侧，最后依次调用[本示例aclnn文件](../all_gather_add/op_host/op_api/aclnn_all_gather_add.h)中的两段式接口完成单算子API调用。

      完成自定义算子包的安装和算子执行代码的编写后，执行如下命令：

        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}]
        # 执行命令示例:
        # bash build.sh --run_example all_gather_add eager cust
        ```

        - \$\{op\}：表示待执行算子，算子名小写下划线形式，如all_gather_add。
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
        - \$\{vendor\_name\}（名称可自定义）：与构建的自定义算子包设置一致，默认名为custom。

        算子执行结束后，测试样例会将算子的执行结果拷贝到主机侧与golden进行精度对比，本示例算子的精度要求为千分之一。
        运行结果示例如下：

        ```
        device_0 aclnnAllGatherAdd execute successfully.
        device_1 aclnnAllGatherAdd execute successfully.
        device_0 aclnnAllGatherAdd golden compare successfully.
        device_1 aclnnAllGatherAdd golden compare successfully.
        ```

- **注意事项**
    - 本示例算子仅接受固定shape的tensor为参数，且仅支持在2个昇腾AI处理器上执行。

      本算子执行代码[test_aclnn_all_gather_add.cpp](examples/test_aclnn_all_gather_add.cpp)中的rankId为昇腾AI处理器的逻辑ID，代码中的rankId被赋值为0，1使得算子在ID为：0、1的昇腾AI处理器上执行算子，实际执行过程中可能存在多个进程竞争处理器0、1的问题，导致如下错误：

        ```bash
        [ERROR] HcclCommInit failed. ret = 4
        ```

      此时需要设置[ASCEND_RT_VISIBLE_DEVICES](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0028.html)环境变量来指定实际执行算子的昇腾AI处理器：

        ```bash
        # 查看各Device运行情况
        npu-smi info
        # 若处理器6，7空闲，则指定仅6，7对当前进程可见：
        export ASCEND_RT_VISIBLE_DEVICES=6,7
        ```

        此时，当前进程仅可见6，7两个处理器，则6，7对于当前进程的逻辑ID为0，1。
        重新执行算子即可。
