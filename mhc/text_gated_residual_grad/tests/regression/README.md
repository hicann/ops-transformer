# TextGatedResidualGrad 回归测试

运行工作区大小计算的边界测试（不分配设备内存）：

```bash
c++ -std=c++17 test_workspace_layout.cpp -o /tmp/tgrg-workspace
/tmp/tgrg-workspace
```

运行主机侧信用同步协议测试，需在本目录执行并使用支持 C++17 的编译器：

```bash
c++ -std=c++17 -DHC_GRAD_NPU_ARCH=HC_GRAD_ARCH_A5 -Istubs test_credit_sync.cpp -o /tmp/tgrg-credit-a5
/tmp/tgrg-credit-a5
c++ -std=c++17 -DHC_GRAD_NPU_ARCH=HC_GRAD_ARCH_A2 -Istubs test_credit_sync.cpp -o /tmp/tgrg-credit-a2
/tmp/tgrg-credit-a2
```

该测试执行算子中的调度器和信用同步类，检查两个 AIV 子块，拒绝没有先行生产者的等待，检查保留的标志 ID，并要求退出时所有信用计数器归零。测试模拟事件计数，不模拟硬件流水时序。A2 构建检查默认直接读取调度下共享信用同步辅助类的 mode-2 语义，不验证设备执行。

运行主机侧 FP32 消减误差测试（提取算子中的方法，需要 Python 3 和 C++17）：

```bash
python3 test_dgamma_host.py
# Compile the extracted production methods without running them:
python3 test_dgamma_host.py --compile-only
```

构建算子包及其 ACLNN 示例，加载该算子包的环境后，在支持的 Ascend 设备上运行 BF16 精确消减误差回归测试（需要 numpy）：

```bash
python3 test_dgamma_cancellation.py --runner /path/to/test_aclnn_text_gated_residual_grad --device 4
```

该用例要求每个通道至少有两条 AIV 执行通道。输入将 +2**24 和 +1 放在第 0 条执行通道上两个独立的 1024 行任务中，将 -2**24 放在另一条执行通道上。原有 Kahan 合并逻辑的 dgamma 结果为 -1，修正后的结果精确为 +1。运行程序由算子已有的 `examples/test_aclnn_text_gated_residual_grad.cpp` 构建得到。

输出缓冲区生命周期回归测试使用全 1/全 0 的精确输入。在 56 AIV 的 A5 分块配置下，可暴露 V6 在 MTE3 完成前清除 V1 输出的问题：

```bash
python3 test_pipeline_output_reuse.py --runner /path/to/test_aclnn_text_gated_residual_grad --device 4
```

预期 `dw_gate` 精确为 768。旧竞争问题能否复现取决于分块配置，预期输出则适用于任意核数。这两个设备侧回归测试用于定向诊断，验收应使用项目的独立标杆。

构建算子包后，运行 ACLNN 可选参数配对和结果指针检查。先加载 CANN 环境，并将 `CUSTOM_OPP_VENDOR` 设置为算子包的 `packages/vendors/custom_transformer` 目录：

```bash
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_OPP_VENDOR"
export LD_LIBRARY_PATH="$CUSTOM_OPP_VENDOR/op_api/lib:$LD_LIBRARY_PATH"
c++ -std=c++17 test_optional_outputs.cpp \
  -I "$CUSTOM_OPP_VENDOR/op_api/include" -I "$ASCEND_HOME_PATH/include" \
  -L "$CUSTOM_OPP_VENDOR/op_api/lib" -L "$ASCEND_HOME_PATH/lib64" \
  -lcust_opapi -lascendcl -lnnopbase -o /tmp/tgrg-optional
/tmp/tgrg-optional 0
```

这 20 项仅查询检查覆盖 FP16 和 BF16 下所有有效的可选参数组合、wGate/dwGate 和 gamma/dgamma 配对的两个方向，以及 workspaceSize/executor 为空指针的情况，不启动内核。

`rms_cancellation_cases.json` 保留了精度失败的 ATK 随机种子 2026110550（BF16，N=1944，C=2，H=1280，R=160，提供 gamma，不提供 gate）。在 sigmoid 饱和输入下，旧 RMS 计算在 dx 上产生 6 个小值域误差点，独立标杆为 2 个，未通过保持不变的计数比阈值 2。文件还包含种子 2026111050（BF16，N=4308，C=4，H=2560，R=320，提供 gamma 和 gate），用于暴露 RMS 投影归一化后仍存在的归约误差：算子有 16 个小值域误差点，标杆为 1 个。

使用[独立标杆的 ATK 适配代码](https://gitcode.com/azyuli/text_gated_residual_grad_bench)及其 CPU FP64 参考实现运行这些用例；必须使用 `--disable_id_seed` 保留用例中的随机种子。标杆和 golden 比较均需启用。这些定向用例不能替代更广泛的随机精度测试。
