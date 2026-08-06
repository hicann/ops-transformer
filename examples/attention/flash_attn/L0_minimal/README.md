# AscendOps — flash_attn_minimal

## 项目简介
AscendOps 是一个轻量级，高性能的算子开发工程模板，它集成了PyTorch、PyBind11和昇腾CANN工具链，提供了从算子内核编写，编译到Python封装的完整工具链。

## 核心特性
🚀 开箱即用 (Out-of-the-Box): 预置完整的昇腾NPU算子开发环境配置，克隆后即可开始开发。

🧩 极简设计 (Minimalist Design): 代码结构清晰直观，专注于核心算子开发流程。

⚡ 高性能 (High Performance): 基于AscendC编程模型，充分发挥昇腾NPU硬件能力。

📦 一键部署 (One-Click Deployment): 集成setuptools构建系统，支持一键编译和安装。

🔌 PyTorch集成 (PyTorch Integration): 无缝集成PyTorch张量操作，支持自动微分和GPU/NPU统一接口。

## 核心组件
1. `csrc/flash_attn_minimal/torch_interface.cpp` 算子Kernel实现
2. `csrc/flash_attn_minimal/CMakeLists.txt` 算子cmake配置

## 环境要求
*   AI处理器：Ascend 950PR/Ascend 950DT
*   Python: 3.8+
*   PyTorch: 2.6.0+
*   PyTorchAdapter 7.1.0+
*   CMake: 3.18.4+
*   CANN Ascend Toolkit

## 环境准备

1. **安装社区版CANN toolkit包**

    根据实际环境，请下载对应的CANN开发套件包`Ascend-cann-toolkit_{version}_linux-{arch}.run`。

    随后安装CANN开发套件包，详情参考[CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit).

    ```bash
    ./Ascend-cann-toolkit_{version}_linux-{arch}.run --full --force --install-path=${install-path}
    ```

    - {version}: CANN版本包路径。
    - {arch}: 系统架构。
    - {install-path}: 指定安装路径。

2. **配置环境变量**

	根据实际场景，选择合适的命令。

    ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   # 指定路径安装
   source ${install-path}/ascend-toolkit/set_env.sh
    ```

3. **安装torch与torch_npu包**

    根据实际环境，下载对应 torch 包，常见的 wheel 文件名示例如下：

   - x86_64 Linux 版本：`torch-${torch_version}+cpu-${python_version}-linux_x86_64.whl`
   - ARM Linux（旧版本通常无 `+cpu` 后缀）：`torch-${torch_version}-${python_version}-linux_aarch64.whl`

    说明：上面示例中的版本号及 `+cpu` 后缀仅作为示例。不同操作系统和架构（尤其是 ARM Linux）对应的 wheel 名称可能没有 `+cpu` 后缀，请以 PyTorch 官方下载页面或 `pip` 实际可用的包名为准。

    下载链接为：[官网地址](https://download.pytorch.org/whl/torch)

    安装命令如下（将 `<torch_whl>` 替换为实际下载的文件名）：

    ```sh
    pip3 install <torch_whl>
    ```

    根据实际环境，安装torch版本配套的torch-npu包，例如：`torch_npu-${torch_version}-${python_version}-linux_${arch}.whl`。

    可以直接使用 pip 命令下载安装（将 `<torch_npu_whl>` 替换为实际下载的文件名），命令如下：

    ```sh
    pip3 install <torch_npu_whl>
    ```

    - ${torch_version}：表示 torch 包版本号。
    - ${python_version}：表示 python 版本号。
    - ${arch}：表示 CPU 架构，如 aarch64、x86_64。

## 安装步骤

1. 进入目录，安装依赖
    ```sh
    pip3 install -r requirements.txt
    ```

2. 从源码构建.whl包
    ```sh
    python3 -m build --wheel -n
    ```

3. 安装构建好的.whl包
    ```sh
    pip3 install dist/ascend_ops-*.whl
    ```

    重新安装请使用以下命令覆盖已安装过的版本：
    ```sh
    pip3 install dist/ascend_ops-*.whl --force-reinstall --no-deps
    ```

4. (可选)再次构建前建议先执行以下命令清理编译缓存
    ```sh
    python3 setup.py clean
    ```

## 开发模式构建

此命令实现即时生效的开发环境配置，执行后即可使源码修改生效，省略了构建完整whl包和安装的过程，适用于需要多次修改验证算子的场景：
  ```sh
  pip3 install --no-build-isolation -e .
  ```

## 使用示例
1. 上板执行

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以flash_attn_minimal算子为例，您可以在`ascend_ops/csrc/flash_attn_minimal/test`目录下找到并执行这个脚本:

```bash
cd ./ascend_ops/csrc/flash_attn_minimal/test/
python3 test_fa.py
```

最终看到如下输出，即为执行成功：
```bash
compare CPU Result vs NPU Result: True
```

同时支持算子性能采集，使用如下命令可以快速采集:
```bash
msprof --application="python3 test_fa.py"
```


## 进阶参考
下述资料可让您更深入的理解算子的开发与实现，实现更优性能的算子。

- [FA简化算子实践](./ascend_ops/csrc/flash_attn_minimal/README.md)。
