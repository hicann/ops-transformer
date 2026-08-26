# 环境部署

在学习各类[学习教程](../../../README.md#学习教程)操作之前，请先参考下面步骤完成基础环境搭建和源码下载，确保已安装NPU驱动、固件和CANN软件（`Ascend-cann-toolkit`和`Ascend-cann-ops`）等。

## 环境安装

本项目提供多种搭建昇腾环境的方式，请按需选择。

> **说明**：本文提到的编译态和运行态含义如下，请根据实际情况选择。
>
> - 编译态：针对仅编译本项目不运行的场景，只需安装CANN toolkit包。
> - 运行态：针对运行本项目的场景（编译运行或纯运行），需安装驱动、CANN toolkit包、CANN ops包。

|  安装方式  |  使用说明  |  使用场景  |
| ----- | ------ | ------ |
|  CANNLab  | 一站式开发平台，提供在线直接运行的昇腾环境，无需手动安装。<br>当前可提供单机算力，**默认安装最新版本CANN包**。 | 适用于没有昇腾设备的开发者。|
|  Docker  | Docker镜像是一种高效部署方式，已预集成CANN包和必备依赖。<br>当前适用于Atlas A2、A3系列产品，OS支持ubuntu22.04、openEuler24.03等。**默认安装最新版本CANN包**。 |适用有昇腾设备，需要快速搭建环境的开发者。|
|  手动安装  | 手动安装CANN包和基础依赖，灵活性高。 |适用有昇腾设备，想体验手动安装CANN包或体验最新master分支能力的开发者。|

### 方式1：CANNLab

对于无昇腾设备的开发者，可直接使用CANNLab云开发环境，即“**一站式开发平台**”，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动、软件包和依赖，无需手动安装。

> **说明**：环境默认安装最新版本CANN包，源码下载时注意与软件配套。更多关于开发平台的介绍请参考[CANNLab指导](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击“`CANNLab`”按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="../figures/cloudIDE.png" alt="云平台"  width="750px" height="85px">

2. 根据页面提示创建NPU环境并配置规格，启动云开发环境后，单击“`连接 > WebIDE`”进入一站式开发平台。当前开源项目资源默认在`/mnt/workspace/gitCode`目录。

   <img src="../figures/webIDE.png" alt="云平台"  width="1000px" height="150px">

### 方式2：Docker部署

对于有昇腾设备的开发者，若想快速搭建昇腾环境，可使用Docker镜像部署。

> **说明**：
>
> - 镜像文件比较大，下载需要一定时间，请耐心等待。关于docker命令的选项介绍可通过`docker --help`查询。
> - 环境默认安装最新最新版本CANN包，源码下载时注意与软件配套。

1.**安装驱动（运行态依赖）**

驱动是运行态依赖，若仅编译算子，可以不安装。使用`npu-smi info`检查是否有NPU相关信息，若没有，请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》完成驱动安装：

   - 步骤1：在页面中选择自己的产品系列、CPU架构、操作系统，安装方式选择"在线安装（Yum）"。
   - 步骤2：按照页面指引依次完成"配置用户属组"、"安装依赖&配源"、"安装NPU驱动"三个环节。
   - 步骤3：执行`npu-smi info`，若能正常显示NPU设备信息，则说明驱动安装成功。

2.**下载镜像**

- 步骤1：以root用户登录宿主机。确保宿主机已安装Docker引擎（版本1.11.2及以上），使用`docker --version`检查Docker版本，若没有，请参考[Docker官方安装指南](https://docs.docker.com/engine/install/)。
- 步骤2：从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)拉取已预集成CANN软件包及`ops-transformer`所需依赖的镜像。

  示例如下，请自行替换CANN版本号、芯片系列、操作系统、python版本等信息，各字段支持的取值可在上述昇腾镜像仓库页面查询。

    ```bash
    # 以cann:9.1.0-beta.1版本为例
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel
    ```

    > **说明**：镜像标签格式为`<CANN版本>-<芯片系列>-<操作系统>-<Python版本>-devel`。带`-devel`后缀的镜像为算子开发镜像，内含算子开发编译依赖。

3.**运行Docker**

拉取镜像后，需要以特定参数启动容器，以便容器内能访问宿主的昇腾设备。

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel bash
```

> **说明**：`--name`用于指定容器名称，请替换为便于识别的自定义名称；若该名称已被占用，容器会启动失败。

| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--device /dev/davinci0` | 核心：将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行`npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。|
| `--device /dev/davinci_manager` | 映射NPU设备管理接口。 | - |
| `--device /dev/devmm_svm` | 映射设备内存管理接口。 | - |
| `--device /dev/hisi_hdc` | 映射主机与设备间的通信接口。 | - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载设备容器管理接口（DCMI）相关工具和库。 | - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键挂载：将宿主机的NPU驱动库映射到容器内。 | - |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | 挂载驱动版本信息文件。 | - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载CANN软件安装信息文件。 | - |
| `-it` | `-i`（交互式）和`-t`（分配伪终端）的组合参数。 | - |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel` | 指定要运行的Docker镜像。 | 请确保此镜像名和标签（tag）与您通过`docker pull`拉取的镜像完全一致。 |
| `bash` | 容器启动后立即执行的命令。 | - |

### 方式3：手动安装

对于有昇腾设备的开发者，若想手动搭建昇腾环境，请参考下述步骤。

#### 安装软件

- **场景1：基于CANN weekly版本体验最新master能力或开发**

    请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》，选择`Weekly版本`，并根据产品系列、CPU架构、操作系统等下载对应包，最后参考页面提供的命令完成安装。

- **场景2：基于CANN 稳定版本体验分支能力或开发**

    请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》，选择`稳定版本`（仅支持CANN 8.5.0及后续版本），并根据产品系列、CPU架构、操作系统等下载对应包，最后参考页面提供的命令完成安装。

#### 安装基础依赖

本项目基础依赖如下，注意满足版本号要求。

- python >= 3.7.0（建议版本 <= 3.10）
- gcc >= 7.3.0
- cmake >= 3.18.4
- pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
- dos2unix
- make
- patch
- googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

上述依赖可通过项目脚本一键安装，操作步骤如下：

1. 下载源码。

    下载与CANN版本配套的分支源码，命令如下，`${tag_version}`替换为分支标签名。

    ```bash
    git clone -b ${tag_version} https://gitcode.com/cann/ops-transformer.git
    ```

2. 安装依赖。

    先通过项目根目录`install_deps.sh`一键安装上述依赖，命令如下，若遇到不支持系统，请参考该文件自行适配。

    ```bash
    bash install_deps.sh
    ```

    再通过项目根目录`requirements.txt`继续安装python三方库依赖，命令如下。

    ```bash
    pip3 install -r requirements.txt
    ```

## 环境验证

安装完CANN包后，需验证环境和驱动是否正常。

- **检查NPU设备**

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

- **检查CANN版本**

    ```bash
    # 查看CANN toolkit包版本信息（默认路径安装）
    # docker和手动安装场景：
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
    # CANNLab场景：
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_toolkit_install.info

    # 查看CANN ops包版本信息（默认路径安装）
    # docker和手动安装场景：
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    # CANNLab场景：
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

    其中`${arch}`可通过`uname -m`查询当前架构，如aarch64、x86_64。

## 环境变量配置

按需选择合适的命令使环境变量生效。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```
