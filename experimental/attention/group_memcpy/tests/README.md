# GroupMemcpy 测试

已于 2026-10-28 在 Linux / CANN 9.1 / Ascend 910B3 上完成编译运行：aclnn C++ 141 组正常调用及非法参数检查通过，Torch pytest 150 项全部通过。下列命令为通用验收流程，本机环境、独立打包入口和日志见 [验证记录](../docs/validation.md)。

## 测试矩阵

| 测试层 | 覆盖内容 | 判定 |
| --- | --- | --- |
| CPU 源码契约 | JSON/OpDef 的 dtype 顺序、binary 配置、源文件、schema/实现/独立扩展目录、Python 语法、文档链接 | 静态一致性，不代表可编译 |
| CPU C++ Tiling | 实际生产 TileBytes 函数，1..599999 字节、不同 UB 预算、容量/对齐/尾块、跨核任务覆盖 | 不越界、不重复、不遗漏 |
| aclnn C++ | 9 个 dtype × 15 个元素数 + 6 个任务数，共 141 组正常调用；空指针、长度/shape/dtype/连续性错误 | 全 storage 字节比较；同时检查源未修改、目标前后哨兵未修改 |
| Torch pytest | 同 dtype/边界组合，任务数 1/39/40/41/81/257、重复源、offset 切片、不同 shape 同元素数、原地写回 | 字节完全相等、地址不变 |
| Torch 运行约束 | 当前非默认 stream、异步 Tensor 生命周期、allocator 压力、重叠、非连续/错误 dtype/设备/shape、requires_grad、原地 version/已保存 Tensor 修改检测、inference_mode、Meta | 正确依赖、明确拒绝；双设备用例仅在至少两张卡时执行 |
| 原直调二进制回放 | 原工程导出的 src_i.bin 和 dst_i.bin | 新输出与原输出逐字节完全一致 |

元素数边界为 `0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 94207, 94208, 94209, 188417`。UINT8 情况直接覆盖 184 KiB UB 双队列的 94208 字节 tile 边界；其他 dtype 同时覆盖更大搬运及多 tile。浮点使用原始位模式比较，不用 `allclose` 放宽。

## CPU 检查

在仓库根目录：

~~~bash
python3 experimental/attention/group_memcpy/tests/host/test_source_contract.py
cmake -S experimental/attention/group_memcpy/tests/host -B /tmp/group_memcpy_host_build
cmake --build /tmp/group_memcpy_host_build
ctest --test-dir /tmp/group_memcpy_host_build --output-on-failure
~~~

第一条不依赖第三方包；后续需 CMake 和 C++14 编译器，直接调用生产头文件中的策略。

## 先构建并安装算子包

在 Linux 上按实际路径载入 CANN 环境；仓库的工具链与 torch_npu 版本需互相匹配，并提供 `OpCommand::RunOpApiV2`、`NPUCachingAllocator::recordStream` 等 API。

~~~bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --experimental --soc=ascend910b --ops=group_memcpy --vendor_name=group_memcpy
find build_out -type f -name '*.run'
~~~

执行显示的实际安装包，并载入其安装日志指定的环境脚本。根据实际安装位置设置以下变量，示例路径为默认 CANN vendor 目录：

~~~bash
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann/opp/vendors/group_memcpy
export GROUP_MEMCPY_OPAPI_ROOT="$ASCEND_CUSTOM_OPP_PATH/op_api"
export LD_LIBRARY_PATH="$GROUP_MEMCPY_OPAPI_ROOT/lib:$GROUP_MEMCPY_OPAPI_ROOT/lib64:$LD_LIBRARY_PATH"
~~~

如果还有其他自定义 vendor，追加已有的 `ASCEND_CUSTOM_OPP_PATH`，不要丢失它们的搜索路径。检查安装目录包含本次 `GroupMemcpy` 注册/Kernel 以及导出的两个 aclnn 符号。

## aclnn C++ 运行测试

~~~bash
cmake -S experimental/attention/group_memcpy/tests/aclnn -B /tmp/group_memcpy_aclnn_build \
    -DASCEND_ROOT="$ASCEND_HOME_PATH" \
    -DGROUP_MEMCPY_OPAPI_ROOT="$GROUP_MEMCPY_OPAPI_ROOT"
cmake --build /tmp/group_memcpy_aclnn_build -j
ctest --test-dir /tmp/group_memcpy_aclnn_build --output-on-failure
~~~

[main.cpp](aclnn/main.cpp) 独立完成 Device/stream 创建、TensorList 组装、两段式调用、同步、字节校验和释放，失败返回非零。默认使用设备 0，不能把“程序未报错”代替其中的结果比较。

若安装布局不同，显式向 CMake 传入 `-DOPAPI_LIBRARY=/真实路径/libcust_opapi.so` 及 `-DOPAPI_INCLUDE_DIR=/真实路径/include`；不要误链接未包含本算子的旧库。

## Torch 接口测试

~~~bash
# 在仓库根目录执行，生成包含 experimental 算子的统一扩展包。
bash build.sh --torch_extension --experimental
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_transformer-*.whl
python3 -m pytest experimental/attention/group_memcpy/tests/test_group_memcpy.py -v
~~~

扩展源码由仓库根目录 `torch_extension/setup.py` 自动收集，首次 NPU 调用时通过 `OpBuilder` 编译。需要 pytest、torch、torch_npu 和已安装的本次算子包。没有 torch/torch_npu/NPU 时测试跳过；有 NPU 但缺少扩展或算子注册时应失败，不以 skip 掩盖。

## 原直调二进制回放

原目录没有提供基线 bin，当前未完成这一项。应先在可运行原工程的环境导出每个有效源片段和直调后的目标片段，保留原始结果：

~~~text
group_memcpy_verify/example/
├── input/src_0.bin ... src_N-1.bin
├── output/dst_0.bin ... dst_N-1.bin       # 原直调真实输出，只读基线
└── aclnn_output/                        # 新建的输出目录
~~~

每个文件长度必须恰好为同一个 `W` 字节。回放以一维 UINT8 视图描述相同字节，不做数值转换；因此也适用于原 BFLOAT16 搬运。

~~~bash
# 在独立验证目录 example 下，opapi_test 指向已编译的测试程序。
/tmp/group_memcpy_aclnn_build/opapi_test --replay input output aclnn_output 41 256
~~~

`41` 和 `256` 是命令示例，必须替换成导出基线的真实任务数和每组字节数。程序先核对输入文件大小，再运行 aclnn，检查完整 storage/哨兵，写入新输出并与原始 golden 逐字节比较；目标文件已存在时拒绝覆盖。复测请使用新的输出目录。

独立验证目录提供单独的 CMake 入口和说明；它保留验证材料，不作为伪造的 msopgen 产物。迁移代码仍在仓库算子目录中。
