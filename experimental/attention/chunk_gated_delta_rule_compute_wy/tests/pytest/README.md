# ChunkGatedDeltaRuleComputeWy 精度测试

依赖：已安装包含该算子的 `.run`、统一扩展 wheel、PyTorch、torch_npu 和 pytest；使用空闲的 310P 设备。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 以下为 CANN 9.2.0 默认安装位置，请按实际安装位置调整。
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.2.0/opp/vendors/custom_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
# 从 ops-transformer 仓库根目录执行
cd experimental/attention/chunk_gated_delta_rule_compute_wy/tests/pytest
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=2 python3 -m pytest -m ci -v -ra -s 2>&1 | tee /tmp/chunk_compute_wy_pytest.log
```

通过公开接口 `cann_ops_transformer.ops.chunk_gated_delta_rule_compute_wy` 调用。
覆盖单 chunk、多 chunk、GQA 分组、多 batch 及不同 K/V 维度，共 3 个基础用例。
固定随机种子 42，k 按最后一维归一化，g 非正，beta 在 [0, 1) 内。
CPU FP64 参考实现按算子公式建立单位下三角方程，用前代求解 W/U，避免复用 Kernel 的倍增算法。

逐个检查五个输出的形状、类型和有限值；q/k 要求完全相等，g 使用 rtol=1e-5、atol=1e-6，
W/U 考虑 FP16 中间计算舍入，使用 rtol=5e-3、atol=1e-3，要求所有元素满足误差阈值。
这是基础精度覆盖，尚不包含最大尺寸、非连续输入或非法参数测试。
