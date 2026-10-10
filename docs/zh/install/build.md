# build参数说明

## 简介

build.sh是本项目的构建脚本，默认在项目根目录下，作用是将源代码自动编译、链接和配置，最终生成可执行文件、库文件或其它可供安装或直接运行的目标文件。具体来说，脚本中通过配置不同参数实现多种功能，包含构建多种目标库（如：libophost_transformer.so）、编译算子包、执行单元测试等。

## 使用方法

1. **配置环境变量**

   参考[环境部署](quick_install.md)完成环境变量配置。

   ```bash
   # 默认路径安装，以root用户为例
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **构建命令格式**

   以编译算子包命令为例，样式如下，其中`--vendor_name`与`--ops`在该场景为可选项。

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   全量参数含义参见下方参数说明章节，请按实际情况选择合适的参数。

## 参数说明

build.sh支持多种功能，可通过如下命令查看所有功能参数。

```bash
bash build.sh --help
```

| 参数名              | 可选/必选  | 参数说明                                                                         |
|------------------|--------|------------------------------------------------------------------------------|
| -j${n}           | 可选     | 指定编译线程数，${n}为具体线程数，默认值为8（如：-j8）；若线程数超过CPU核心数，会自动调整为CPU核心数。                   |
| -v               | 可选     | 查看CMake编译配置信息。                                                               |
| -O${n}           | 可选     | 指定编译优化级别，支持O0/O1/O2/O3（如：-O3），${n}为优化级别标识。                                   |
| -u               | 可选     | 启用单元测试（UT）编译模式，编译所有UT目标。                                                     |
| --help，-h        | 可选     | 打印脚本使用帮助信息。                                                                  |
| --ops            | 可选     | 指定待编译的算子，如：apply_rotary_pos_emb,rope_quant_kvcache，多个算子用英文逗号“,”分隔，不可与--ophost、--opapi、--opgraph同时使用。 |
| --module         | 可选     | 指定待编译的模块，如：--module=mc2,attention，多个模块用英文逗号“,”分隔（支持：mc2,attention,moe,ffn,mhc,posembedding,gmm）。 |
| --soc            | 可选     | 指定NPU型号，每次编译只支持1个NPU型号。                                               |
| --jit            | 可选     | 静态图场景下，编译`cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run`整包时不需要编译算子二进制文件（图的运行态会在线编译），可以配置该选项，以提升编译速度。 |
| --static         | 可选     | 配置后，表示生成静态库文件，包含libcann_transformer_static.a和aclnn接口头文件，搭配--pkg参数，生成静态库压缩包。|
| --vendor_name    | 可选     | 指定自定义算子包的名称，默认值为custom。                                                      |
| --build-type     | 可选     | 指定构建类型，支持Release/Debug（如：--build-type=Debug），默认值为Release。不可与--mssanitizer、--oom、--dump_cce同时使用。 |
| --version        | 可选     | 指定版本号（如：--version=1.0.0）。                                                       |
| --cov            | 可选     | 开启单元测试代码覆盖率统计（需配合-u使用），使用lcov生成覆盖率报告。                                           |
| --noexec         | 可选     | 仅编译单元测试二进制文件，不自动执行编译后的UT可执行文件。                                               |
| --valgrind       | 可选     | 使用valgrind运行单元测试进行内存检测（需配合-u使用），启用后会禁用ASAN和--noexec。                                |
| --opkernel       | 可选     | 编译二进制内核。                                                                     |
| --opkernel_aicpu | 可选     | 编译AICPU算子内核。                                                                   |
| --pkg        | 可选     | 生成安装包，不可与-u（UT模式）或--ophost、--opapi、--opgraph同时使用。                            |
| --pkg-type       | 可选     | 指定打包类型，取值：run/rpm/deb/all，默认为run，仅可与--pkg同时使用。                                  |
| --make_clean     | 可选     | 执行基础清理操作（清理编译产物），执行后脚本退出。                                                    |
| --asan           | 可选     | 启用ASAN（AddressSanitizer）内存检测功能。                                            |
| --ophost         | 可选     | 编译libophost_transformer.so库，不可与--pkg、--ops同时使用。                                     |
| --opapi          | 可选     | 编译libopapi_transformer.so库，不可与--pkg、--ops同时使用。                                      |
| --opgraph        | 可选     | 编译libopgraph_transformer.so库，不可与--pkg、--ops同时使用。                                    |
| --onnxplugin     | 可选     | 编译liboponnx_plugin_transformer.so库。                                                 |
| --ophost_test   | 可选     | 编译ophost相关单元测试，与-u --ophost组合等效。                                          |
| --opapi_test    | 可选     | 编译opapi相关单元测试，与-u --opapi组合等效。                                            |
| --opgraph_test  | 可选     | 编译opgraph相关单元测试，与-u --opgraph组合等效。                                            |
| --opkernel_test | 可选     | 编译opkernel相关单元测试，与-u --opkernel组合等效。                                      |
| --run_example    | 可选     | 编译指定算子及模式的样例并执行编译后的可执行文件。                                                    |
| --simulator      | 可选     | 启用仿真器模式执行--run_example任务。仿真模式下，会根据soc_version链接对应的仿真库，目前只支持Ascend950。          |
| --genop          | 可选     | 创建AI Core自定义算子初始目录。                                                       |
| --genop_aicpu    | 可选     | 创建AI CPU自定义算子初始目录。                                                       |
| --experimental  | 可选     | 编译experimental目录下的用户算子。                                                   |
| --oom            | 可选     | 开启kernel侧oom内存检测功能。                                                   |
| --mssanitizer    | 可选     | 开启kernel侧mssanitizer内存检测功能，不可与--oom、--dump_cce同时使用。                              |
| --dump_cce       | 可选     | 开启kernel侧dump预编译文件功能，不可与--mssanitizer同时使用。                                         |
| --bisheng_flags  | 可选     | 指定一个毕昇编译器编译参数，不可与--mssanitizer、--oom、--dump_cce同时使用。     |
| --kernel_template_input     | 可选     | 指定编译kernel时的tilingKey模板，仅支持指定一个模板，模板参数用英文逗号“,”分隔，与--ops同时使用且只能指定一个算子，不会编译该算子所依赖的其他算子二进制文件。     |
| --cann_3rd_lib_path           | 可选     | 离线编译场景下第三方库存放的目录。                                                   |
