# 源码构建

源码构建前，请参考本项目README完成环境准备和源码下载，此处不再赘述。

## 安装第三方依赖

> **说明**：对于CANNLab或Docker环境，默认联网，编译过程中会自动下载第三方依赖，无需手动安装，可跳过本章操作。

本项目编译过程依赖的第三方开源软件列表如下：

| 开源软件 | 版本 | 下载地址 |
|---|---|---|
| googletest | 1.14.0 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| json | 3.11.3 | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| pybind11 | 2.13.6 | [pybind11-2.13.6.tar.gz](https://gitcode.com/cann-src-third-party/pybind11/releases/download/v2.13.6/pybind11-2.13.6.tar.gz) |
| eigen | 5.0.0 | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0/eigen-5.0.0.tar.gz) |
| protobuf | 25.1.0 | [protobuf-25.1.tar.gz](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz) |
| opbase(自CANN 9.0.0及以后版本需要下载) | master | [opbase](https://gitcode.com/cann/opbase) |
| cann-cmake | master-046 | [cmake-master-046.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-046.tar.gz) |
| ops-tensor(自CANN 9.1.0及以后版本需要下载) | master | [ops-tensor](https://gitcode.com/cann/ops-tensor) |

若您的编译环境可以访问网络，请参考[联网编译](#联网编译)，编译脚本会自动联网下载第三方软件。否则，请参考[未联网编译](#未联网编译)手动下载第三方软件。

## 算子包形态介绍

- **自定义算子包**：

  选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。生成的自定义算子包优先级高于原始CANN包。该包支持aclnn和图模式调用AI Core、AI CPU算子。

- **ops-transformer包**：

  选择整个项目编译生成的包称为ops-transformer包，可**完整替换**CANN包对应部分。该包支持aclnn和图模式调用AI Core算子。

- **ops-transformer静态库**：

  > 说明：若您需要**基于本项目进行二次发布**并且对**软件包大小有要求**时，建议采用静态库编译，该库可以链接您的应用开发程序，仅保留业务所需的算子，从而实现软件最小化部署。

  表示整个项目编译为一个静态库文件，包含libcann_transformer_static.a和aclnn接口头文件。该包仅支持aclnn调用AI Core算子。

## 联网编译

若在有互联网的环境下编译，编译过程中会自动安装第三方依赖，无需手动安装。不同场景下源码编译和部署命令如下：

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] [--opkernel_aicpu]
    # 以FlashAttentionScore算子编译为例
    # bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score
    # 编译experimental贡献目录下的算子
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=${experimental_op}
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"apply_rotary_pos_emb,rope_quant_kvcache,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译experimental贡献目录下的算子，${experimental_op}为新贡献算子目录名。
    - --opkernel_aicpu（可选）：开启ENABLE_AICPU_KERNEL，编译kernel_aicpu。

    > **说明**：编译纯AICPU算子（仅含op_kernel_aicpu目录的算子）时，必须添加--opkernel_aicpu选项，否则会报"OpFileNotExistsError: File aic-*-ops-info.ini does not exist"错误，详见下方FAQ章节。

    若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是ops-transformer包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。当提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```

    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**

    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```

    自定义算子包安装路径为```${ASCEND_HOME_PATH}/opp/vendors```，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。

    > 安装后执行 `grep load_priority ${ASCEND_HOME_PATH}/opp/vendors/config.ini` 校验，预期输出含本 vendor。若无输出，通常是 `${ASCEND_HOME_PATH}/opp/vendors` 属主或权限不对，请排查后重新安装。

    如果部署算子包时通过配置--install-path参数指定了算子包的安装目录，则在使用自定义算子前，需要执行```source ${install_path}/vendors/${vendor_name}/bin/set_env.bash```命令，set_env.bash脚本中将自定义算子包的安装路径追加到环境变量ASCEND_CUSTOM_OPP_PATH中，使自定义算子在当前环境中生效。

3. **（可选）删除自定义算子包**

    注意自定义算子包不支持卸载，如需卸载，请删除vendors\/\$\{vendor\_name}目录，并删除vendors/config.ini中load_priority对应\$\{vendor\_name\}的配置项。

### ops-transformer包

1. **编译ops-transformer包**

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译除experimental贡献目录外的所有算子
    bash build.sh --pkg --soc=${soc_version}
    # 指定打包类型（run/rpm/deb/all），默认 run
    # bash build.sh --pkg --pkg-type=deb --soc=${soc_version}
    # 编译experimental贡献目录下的所有算子
    # bash build.sh --pkg --experimental --soc=${soc_version}
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --pkg-type（可选）：指定打包类型，取值 run/rpm/deb/all，默认为 run，仅可与 --pkg 同时使用。
    - --experimental（可选）：表示编译experimental贡献目录下的算子。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run" successfully created.
    ```

    \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-transformer包**

    ```bash
    # 安装命令
    ./cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **（可选）卸载ops-transformer包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/ops_transformer/script/uninstall.sh
    ```

### ops-transformer静态库

> 说明：静态库仅支持Atlas A2、Atlas A3、Atlas A5系列产品。experimental算子暂不支持使用静态库。

1. **编译ops-transformer静态库**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --static --soc=${soc_version}
    ```

    \$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Atlas A5系列产品使用"ascend950"。
    若提示如下信息，说明编译并压缩成功。

    ```bash
    [SUCCESS] Build static lib success!
    Successfully created compressed package: ${repo_path}/build_out/cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz
    ```

    \$\{repo\_path\}表示项目根目录，\$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，压缩包存放于build_out目录下。

2. **解压ops-transformer静态库**

    进入build_out目录执行解压命令：

    ```bash
    tar -zxvf ./cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz -C ${static_lib_path}
    ```

    \$\{static\_lib\_path\}表示静态库解压路径。解压后目录结构如下：

    ```text
    ├── cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}
    │   ├── lib64
    │   │   ├── libcann_transformer_static.a        # 静态库文件
    │   └── include
    |       ├── ...                                 # aclnn接口头文件
    ```

## 未联网编译

若在没有连接互联网的环境下编译，需要提前准备好依赖的第三方软件，再进行源码编译。具体过程如下：

1. **检查基础环境是否完备**

    请确保已按[环境部署](../install/quick_install.md)完成基础环境搭建，包括CANN包安装、源码下载等。

    - 在联网环境中，进入[本项目主页](https://gitcode.com/cann/ops-transformer)，通过`下载ZIP`或`clone`按钮，根据指导完成源码下载。
    - 连接离线环境，上传源码至您指定的目录下。若下载的是源码压缩包，请先进行解压。

2. **下载第三方软件依赖**

    在联网环境中提前下载第三方软件，目前有如下方式，请按需选择：

    - 方式1：根据[第三方软件依赖](#安装第三方依赖)提供的表格手动下载，若从其他地址下载，请确保版本号一致。

    - 方式2：通过[third_lib_download.py](../../../scripts/tools/third_lib_download.py)脚本一键下载，该脚本在本项目`scripts/tools/`目录，下载该脚本并执行如下命令：

        ```bash
        python ${scripts_dir}/third_lib_download.py
        ```

    \$\{scripts\_dir\}表示脚本存放路径，下载的第三方软件包默认存放在当前脚本所在目录。

3. **编译算子包**

   将下载好的第三方软件上传至离线环境，可存放在`third_party`目录或自定义目录下。**推荐前者，其编译命令与联网编译场景下的命令一致。**

    - **third\_party目录**（推荐）

      请在本项目根目录创建`third_party`目录（若有则无需创建），将第三方软件拷贝到该指定目录。此时编译命令与联网编译命令一致，具体参考[联网编译](#联网编译)。

    - **自定义目录**

      在离线环境的任意位置新建`${cann_3rd_lib_path}`目录，将第三方软件拷贝到该目录，请确保该目录有权限访问。

        ```bash
      mkdir -p ${cann_3rd_lib_path}
        ```

      此时编译命令需在联网编译命令基础上额外增加`--cann_3rd_lib_path=${cann_3rd_lib_path}`用于指定第三方软件所在路径。假设存放路径为`/path/cann_3rd_lib_path`，不同编译方式对应的命令如下：

      - 自定义算子包

        ```bash
        bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] --cann_3rd_lib_path=${cann_3rd_lib_path}
        # 以FlashAttentionScore算子编译为例
        # bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score --cann_3rd_lib_path=/path/cann_3rd_lib_path
        ```

      - ops-transformer整包

        ```bash
        bash build.sh --pkg --soc=${soc_version} --cann_3rd_lib_path=${cann_3rd_lib_path}
        # bash build.sh --pkg --soc=ascend910b --cann_3rd_lib_path=/path/cann_3rd_lib_path
        ```

      - ops-transformer静态库

        ```bash
        bash build.sh --pkg --static --soc=${soc_version} --cann_3rd_lib_path=${cann_3rd_lib_path}
        # bash build.sh --pkg --static --soc=ascend910b --cann_3rd_lib_path=/path/cann_3rd_lib_path
        ```

4. **安装/卸载算子包**

    未联网和联网场景下编译得到算子包结果一样，默认存放于项目根目录build_out目录下，并且安装和卸载的操作命令也一样，具体参见[联网编译](#联网编译)。

## 本地验证

源码包部署后，可通过项目根目录build.sh执行UT用例，验证项目功能是否正常。

> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

```bash
# 安装根目录下test相关requirements.txt依赖
pip3 install -r tests/requirements.txt
# 方式1: 编译并执行指定算子和对应功能的UT测试用例（选其一）
bash build.sh -u --[opapi|ophost|opkernel] --ops=abs
# 方式2: 编译并执行所有的UT测试用例
# bash build.sh -u
# 方式3: 编译所有的UT测试用例但不执行
# bash build.sh -u --noexec
# 方式4: 编译并执行对应功能的UT测试用例（选其一）
# bash build.sh -u --[opapi|ophost|opkernel]
# 方式5: 编译对应功能的UT测试用例但不执行（选其一）
# bash build.sh -u --noexec --[opapi|ophost|opkernel]
# 方式6: 执行UT测试用例时可指定soc编译
# bash build.sh -u --[opapi|ophost|opkernel] [--soc=${soc_version}]
```

以验证ophost功能是否正常为例，执行如下命令：

```bash
bash build.sh -u --ophost
```

执行完成后出现如下内容，表示执行成功。

```bash
Global Environment TearDown
[==========] ${n} tests from ${m} test suites ran. (${x} ms total)
[  PASSED  ] ${n} tests.
[100%] Built target math_op_host_ut
```

\$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。
> 说明：add_example算子仅是为了帮助理解如何开发算子，仅设计了ophost的ut用例，并未设计opapi和opkernel的ut用例。开发者可以参考其他算子的ut用例以实现看护。

## FAQ

### 本地编译报错"ERROR: No matching distribution found for setuptools>=40.8.0"

```bash
Could not fetch URL XXX (Caused by SSLError(SSLCertVerificationError))
ERROR: Could not find a version that satisfies the requirement setuptools>=40.8.0 (from versions: none)
ERROR: No matching distribution found for setuptools>=40.8.0
```

1、检查[环境部署](./quick_install.md)中根目录requirements.txt中python三方库依赖是否齐全

2、检查环境~/.pip/pip.conf或~/.config/pip/pip.conf指定的pip源是否可用

### 单独编译aicpu算子报错

单独编译纯AICPU算子（仅含op_kernel_aicpu目录的算子）时，若未添加--opkernel_aicpu选项，会出现如下报错：

```bash
OpFileNotExistsError: File aic-*-ops-info.ini does not exist in directory [...]
make: *** [Makefile:462: prepare_build] Error 2
```

该报错原因为未开启ENABLE_AICPU_KERNEL，导致未生成aic-*-ops-info.ini文件。请在编译命令中添加--opkernel_aicpu选项，例如：

```bash
bash build.sh --pkg --soc=${soc_version} --ops=${op_list} --opkernel_aicpu
```

### 编译失败或中断后恢复

编译失败或强制中断可能导致 build 缓存损坏，再次编译异常时，执行 `bash build.sh --make_clean` 清理后重新编译。
