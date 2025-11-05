# 商发版本说明

若您安装了**商发版**（**8.3.RC1**）CANN开发套件包`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`，需要对部分算子源码进行修改，可通过修改对应的开源项目源码，具体操作如下：

- \$\{cann\_version\}：表示CANN包版本号。
- \$\{arch\}：表示CPU架构，如aarch64、x86_64。

## 获取并安装软件包

如需修改某项目算子的源码，请确保已完成NPU驱动和固件、CANN开发套件包和`cann-opbase_${cann_version}_linux-${arch}.run`包安装。

1. 前提条件。
    参考《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommercialInstSoftware)》，按要求完成NPU驱动和固件、`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`软件包的获取和安装。

2. 安装`cann-opbase_${cann_version}_linux-${arch}.run`包。

    ```bash
    # 1.opbase项目源码下载，以master分支为例
    git clone https://gitcode.com/cann/opbase.git
    # 2.进入项目根目录编译生成run包，默认在根目录build_out目录下
    bash build.sh
    # 3.安装编译包，${install_path}需与toolkit包指定路径一致
    ./cann-opbase_${cann_version}_linux-${arch}.run --full --install-path=${install_path}/ascend-toolkit
    ```

## 安装依赖

开源项目的源码编译用到的依赖如下，请确保已安装并且满足版本要求。
- python >= 3.7.0
- gcc >= 7.3.0
- cmake >= 3.16.0
- pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
- dos2unix
- Gawk
- googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

上述依赖包可通过项目根目录install\_deps.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
```bash
bash install_deps.sh
```

项目使用的python依赖包，可通过根目录下requirements.txt安装，命令如下：
```bash
pip3 install -r requirements.txt
```
## 下载源码
通过`git`命令下载待修改项目的源码：
```bash
# 源码下载，以master分支为例
git clone https://gitcode.com/cann/${ops_project}.git
```
\$\{ops\_project\}表示待修改项目（如ops-math、ops-nn等）。

## 配置环境变量

根据实际场景，选择合适的命令。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 指定路径安装
source ${install_path}/ascend-toolkit/set_env.sh
```

## 编译执行

基于商发版CANN开发套件包修改算子源码时，需使用**自定义算子包**方式编译和安装。

1. **编译自定义算子包。**

    进入待修改项目的根目录，执行如下编译命令：
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    ```
    - --soc：Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子（参见[算子列表](./op_list.md)）。格式形如"apply_rotary_pos_emb,rope_quant_kvcache,..."，多算子之间用英文逗号","分隔。
    
    说明：若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是built-in包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。
    
    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包。**
    ```bash
    ./cann-ops-transformer-${vendor_name}-linux.${arch}.run
    ```

    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/ascend-toolkit/latest。注意自定义算子包不支持卸载。
