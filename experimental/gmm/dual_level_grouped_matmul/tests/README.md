# DualLevelGroupedMatmul 测试

## 精度

`assets/dlgmm_reference.py` 为共享 CPU reference；ACLNN 和 E2E 适配器分别处理 TensorList、NZ allocation 和 Torch 打包数据。Kernel 适配模块供两者复用，不再单独交付重复 Kernel ST。

数值顺序：每 32 K 应用 level1 scale，在 512 K 段内 FP32 累加，再乘 A0×B0 并跨段累加；最终加 bias、转换 FP16/BF16。比较使用 CSV 中的 isclose 阈值，不放宽容差。

ACLNN 和 Torch E2E 各有 11 条 ST，两条通路使用不同的 shape。每条通路独立覆盖权重转置与非转置、FP16/BF16 输出、有无 bias 的全部组合，以及三种分组列表类型和两种 `splitItem` 取值。用例还覆盖单行输入、多基本块、空专家、M/N/K 尾块，以及两级 scale 窗口切换。Torch E2E 同时覆盖 `torch.int8` 和 `torch.uint8` 打包输入。

加载 CANN、已安装算子 vendor 和 Torch 扩展后，在 ops-test-kit 根目录运行：

```bash
OP=/absolute/ops-transformer/experimental/gmm/dual_level_grouped_matmul
python3 -m ttk aclnn -i "$OP/tests/st/arch35/ttk_aclnn_dual_level_grouped_matmul_weight_nz_basic_st.csv" \
  --plugin "$OP/tests/assets" --devices 0 --pc 1 --seed 20261116 -o aclnn_results.csv
python3 "$OP/tests/st/run_e2e.py" e2e -i "$OP/tests/st/arch35/ttk_e2e_dual_level_grouped_matmul_basic_st.csv" \
  --plugin "$OP/tests/assets" --devices 0 --pc 1 --seed 20261116 -o e2e_results.csv
```

vendor Torch 包通过 `CANN_TRANSFORMER_PACKAGE` 指定；标准包默认为 cann_ops_transformer。TTK 须支持 TensorList 的嵌套 stride。CPU reference 依赖 NumPy，BF16 需要 ml_dtypes，E2E 需要 Torch/TorchNPU。

E2E 预处理显式将四维 weight level1 scale 转成实际 ND 存储，再恢复转置 view；不能仅凭 CSV 的 ND 标签或逻辑连续性判断设备格式。

## 核心 UT

Host/API UT 采用 CSV 批量驱动：用例全部定义在测试文件旁的同名 CSV 中（表头键值列，新增用例只需追加行，无需改动测试代码）。tiling 校验 tilingKey 与 tilingData 字段，infershape 按 `case_type=shape/dtype_only` 分别覆盖 shape 推导与 dtype 推导，op_api 按行调用生产 checker 校验 null/offset/stride/storage 及属性非法路径。通过仓库 build.sh 的 `--ophost --opapi -u --experimental --ops=dual_level_grouped_matmul` 运行：

- `ut/op_host/test_dual_level_grouped_matmul_tiling.csv`（tiling 用例）
- `ut/op_host/test_dual_level_grouped_matmul_infershape.csv`（shape/dtype 用例）
- `ut/op_host/op_api/test_aclnn_dual_level_grouped_matmul_weight_nz.csv`（checker 用例）

`ut/generate_ut_csv.py` 可按几何规则重新生成上述三张表（NZ 存储、level0/level1 分组数等均由 M/K/E/N 推导）；手工改 CSV 或改脚本后重跑均可。

Torch 测试覆盖 Meta 输出推导、稀疏分组以及设备侧的 scale shape、存储 dtype、视图偏移、ND/NZ 格式和默认确定性。在安装该扩展的环境运行：

```bash
python3 -m unittest discover -s "$OP/tests/torch" -v
```
