# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Compressor 位级一致性测试支撑库(配合 test_compressor_bitlevel.py 使用).

能力:
1. 确定性构造算子输入: 每类张量独立固定种子, B 变化时数据"前缀一致", 支持跨 B / 跨位置对比;
2. 单次运行 = CPU golden 精度校验(复用 compressor_golden.cpu_compressor/check_result) + NPU 直调;
3. NPU 结果之间的位级(bit-level)对比:
   - pair 用例: B=(b_small, b_large) 两次运行, 共享 batch 的 cmp_kv/state_cache 必须位级一致
     (B 穿过 FULL_LOAD->PERF 模板边界或 PERF 动态D分块悬崖时同样成立);
   - position 用例: 目标样本换到首/中/尾位置, 其 cmp_kv/state_cache 必须位级一致.

环境变量:
   CMP_TEST_DEVICE_ID   指定运行设备号, 默认 0
   CMP_BITLEVEL_STRICT  "1"(默认)位级不一致即失败; "0"降级为告警+阈值校验(便于区分位级尾差与功能错误)
"""

import os
import sysconfig
import importlib

import numpy as np
import torch
import torch_npu

from compressor_golden import check_result, cpu_compressor

DEVICE_ID = int(os.environ.get("CMP_TEST_DEVICE_ID", "0"))
BITLEVEL_STRICT = os.environ.get("CMP_BITLEVEL_STRICT", "1") == "1"

torch.npu.config.allow_internal_format = True


# =====================================================================================================================
# 算子入口加载
# =====================================================================================================================
def load_compressor_op():
    """加载 compressor 算子的 python 入口.

    兼容两种安装方式(优先单算子 wheel, 避免旧整包内 csrc 缺失问题):
      1. 单算子 wheel: pip 安装 cann_ops_transformer_<vendor>(bash build.sh --torch_extension --ops=compressor),
         布局为 <pkg>/ops/attention/compressor;
      2. 整包 wheel: cann_ops_transformer(bash build.sh --torch_extension).
    """
    if getattr(load_compressor_op, "_cached", None) is not None:
        return load_compressor_op._cached
    site_dir = sysconfig.get_paths()["purelib"]
    pkg_names = []
    for name in sorted(os.listdir(site_dir)):
        if (
            name.startswith("cann_ops_transformer_")
            and not name.endswith(".dist-info")
            and os.path.isdir(os.path.join(site_dir, name))
        ):
            pkg_names.append(name)
    pkg_names.append("cann_ops_transformer")  # 整包兜底
    sub_modules = ["ops.attention.compressor", "ops.compressor"]
    errors = []
    for pkg in pkg_names:
        for sub in sub_modules:
            try:
                mod = importlib.import_module("%s.%s" % (pkg, sub))
            except Exception as exc:  # noqa: BLE001
                errors.append("%s.%s: %s" % (pkg, sub, exc))
                continue
            fn = getattr(mod, "compressor", None)
            if callable(fn):
                print("[bitlevel] 使用算子入口: %s.%s" % (pkg, sub))
                load_compressor_op._cached = fn
                return fn
    raise ImportError(
        "未找到可用的 compressor 算子入口, 请先安装 cann_ops_transformer(整包或单算子)wheel.\n%s"
        % "\n".join(errors)
    )


# =====================================================================================================================
# 模板/切分边界推导(arch22: A2/910B系列)
# =====================================================================================================================
def get_aic_core_num(device_id=None):
    """获取设备 Cube(AIC) 核数, 用于推导 PERF 模板动态 D 分块悬崖边界."""
    if device_id is None:
        device_id = DEVICE_ID
    return int(torch.npu.get_device_properties(device_id).cube_core_num)


def calc_perf_dsplit_boundary_b(seq_len, head_dim, coff, aic_core_num=None):
    """推导 PERF(normal)模板"动态D分块悬崖"的 batch 边界 B_dyn(仅 arch22 内核逻辑适用).

    内核逻辑(arch22 compressor_kernel_perf.h SetBaseSize):
      - 初始 dBaseSize=64, dBaseBlockNum=head_dim/64;
      - 所有 batch 等长且 mSize=B*S <= mBaseSize * (AIC // dBaseBlockNum), 且 AIC >= 精细分块数
        (D=512 时为 16, D=128 时为 8), 则 dBaseSize=head_dim/16;
      - 即 B <= B_dyn 时 dBaseSize=32(D=512), B > B_dyn 时 dBaseSize=64.
        D 分块数(dBasicBlockNum 16<->8)变化 -> 同一输出列改由其他核计算,
        hIdxStart=(aiCoreIdx % dBasicBlockNum)*K_L1_BASE 随之改变 -> K轴累加顺序变化.
    典型值: AIC=24 且 D=512、S=4 时 B_dyn=96(即 B=96/97 悬崖对); AIC=20(910B3) 时 B_dyn=64.
    不适用时返回 None.
    """
    if aic_core_num is None:
        aic_core_num = get_aic_core_num()
    if coff == 2:
        m_base = 128
    elif coff == 1:
        m_base = 256
    else:
        return None
    if head_dim == 512:
        fine_block_num = 16
    elif head_dim == 128:
        fine_block_num = 8
    else:
        return None
    if aic_core_num < fine_block_num:
        return None
    groups = aic_core_num // (head_dim // 64)
    if groups < 1:
        return None
    return (m_base * groups) // seq_len


def predict_template(batch_size, cfg):
    """预测 BSH 布局下 batch_size 对应的模板: FULL_LOAD(BSH+S<=4+B*S<=limit) 或 PERF(normal)."""
    seq_len = cfg["seq_len"]
    device_name = torch.npu.get_device_name(DEVICE_ID)
    token_limit = (
        256 if "950" in device_name else 128
    )  # arch35(950)与arch22(A2/910B)阈值不同
    if seq_len <= 4 and batch_size * seq_len <= token_limit:
        return "FULL_LOAD"
    return "PERF(normal)"


def predict_perf_dsplit(batch_size, cfg, aic_core_num=None):
    """预测 PERF 模板下 batch_size 对应的 dBaseSize(典型配置), 仅用于诊断打印."""
    head_dim = cfg["head_dim"]
    coff = cfg["coff"]
    if aic_core_num is None:
        aic_core_num = get_aic_core_num()
    if coff != 2 or head_dim not in (128, 512):
        return None
    fine_block_num = 16 if head_dim == 512 else 8
    if aic_core_num < fine_block_num:
        return 64
    groups = aic_core_num // (head_dim // 64)
    if batch_size * cfg["seq_len"] <= 128 * groups:
        return head_dim // fine_block_num
    return 64


# =====================================================================================================================
# 确定性输入构造
# =====================================================================================================================
def _gen_block_table(start_pos_list, seq_len, cmp_ratio, block_size, nb):
    """复刻 compressor_golden.run_compressor_eager 的 block_table 生成逻辑(纯确定性, 无随机).

    返回 (B, nb) int32 表: 值为 state_cache 第0维物理块 id, 0 表示该列无效(不读不写).
    每行非零列 = 该 batch 需要读取/写入 state 的 block 区间; id 顺序分配, 与 batch 前缀相关.
    """
    batch_num = len(start_pos_list)
    block_num = batch_num * nb
    table = torch.zeros((batch_num, nb), dtype=torch.int32)
    next_block_id = 1
    for i in range(batch_num):
        start_p = int(start_pos_list[i])
        used = seq_len  # BSH 布局且 seqused=None: 每 batch 有效长度恒为 S
        # 需要读取 state 的范围
        cur_start = start_p // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_p // cmp_ratio * cmp_ratio + cmp_ratio
        if start_p % cmp_ratio == 0:
            cur_end = start_p
        cur_end = min(cur_end, start_p + used)
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            if next_block_id < block_num:
                table[i][j] = next_block_id
                next_block_id = next_block_id + 1
        # 需要写入 state 的范围(save_state_seqlens=None)
        end_pos = used
        next_start = (start_p + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
        next_end = (start_p + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
        if (start_p + end_pos) % cmp_ratio == 0:
            next_end = start_p + end_pos
        next_end = min(next_end, start_p + end_pos)
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            if next_block_id < block_num and table[i][j] == 0:
                table[i][j] = next_block_id
                next_block_id = next_block_id + 1
    return table


def build_case_inputs(cfg, batch_size, permutation=None, start_pos_list=None):
    """确定性构造一次运行的完整输入(全部 CPU 张量).

    数据复现规则:
      - 每类张量独立固定种子: kv/score/x 的形状随 B 变化, 但同种子下小 B 的数据是大 B 数据的前缀,
        因此 pair 用例两次运行的共享 batch 输入完全一致; wkv/wgate/ape 形状与 B 无关, 两次运行完全一致;
      - permutation[pos] = 该位置放置的"样本"编号(默认恒等). 样本的 x 行 / 初始 state 块内容 / start_pos
        随样本迁移, 与所在位置无关, 用于 position 用例.
    """
    seq_len = cfg["seq_len"]
    head_dim = cfg["head_dim"]
    hidden_size = cfg["hidden_size"]
    coff = cfg["coff"]
    cmp_ratio = cfg["cmp_ratio"]
    block_size = cfg["block_size"]
    data_type = cfg["data_type"]
    seed = cfg.get("seed", 2026)
    datarange = cfg.get("datarange", [-10, 10])
    if start_pos_list is None:
        start_pos_list = [cfg["default_start_p"]] * batch_size
    assert len(start_pos_list) == batch_size, "start_pos 长度必须等于 B"
    perm = list(permutation) if permutation is not None else list(range(batch_size))
    assert sorted(perm) == list(range(batch_size)), "permutation 必须是 0..B-1 的重排"

    s_max = max(start_pos_list) + seq_len
    nb = (s_max + block_size - 1) // block_size

    lo, hi = datarange[0], datarange[1]
    np.random.seed(seed + 1)
    kv_stream = torch.tensor(
        np.random.uniform(lo, hi, (batch_size * nb, block_size, coff * head_dim))
    ).to(torch.float32)
    np.random.seed(seed + 2)
    score_stream = torch.tensor(
        np.random.uniform(lo, hi, (batch_size * nb, block_size, coff * head_dim))
    ).to(torch.float32)
    np.random.seed(seed + 3)
    x_stream = torch.tensor(
        np.random.uniform(lo, hi, (batch_size, seq_len, hidden_size))
    ).to(data_type)
    np.random.seed(seed + 4)
    wkv = torch.tensor(np.random.uniform(lo, hi, (coff * head_dim, hidden_size))).to(
        data_type
    )
    np.random.seed(seed + 5)
    wgate = torch.tensor(np.random.uniform(lo, hi, (coff * head_dim, hidden_size))).to(
        data_type
    )
    np.random.seed(seed + 6)
    ape = torch.tensor(np.random.uniform(lo, hi, (cmp_ratio, coff * head_dim))).to(
        torch.float32
    )

    # 位置 -> 样本 重排: x 行与 start_pos 按位置取样本
    run_start_pos = [int(start_pos_list[perm[pos]]) for pos in range(batch_size)]
    x_run = x_stream[perm]
    table = _gen_block_table(run_start_pos, seq_len, cmp_ratio, block_size, nb)

    # state 物理块内容按"样本x列"迁移: 表项(pos, j) 指向的物理块放置 样本 perm[pos] 的第 j 列内容,
    # 保证同一样本在任何位置的初始 state 完全一致(与物理块 id 无关)
    kv_state = kv_stream.clone()
    score_state = score_stream.clone()
    for pos in range(batch_size):
        for j in range(nb):
            block_id = int(table[pos][j])
            if block_id != 0:
                kv_state[block_id] = kv_stream[perm[pos] * nb + j]
                score_state[block_id] = score_stream[perm[pos] * nb + j]

    meta = {
        "B": batch_size,
        "S": seq_len,
        "nb": nb,
        "block_num": batch_size * nb,
        "S_max": s_max,
        "perm": perm,
        "start_pos": run_start_pos,
        "seed": seed,
    }
    return {
        "x": x_run,
        "wkv": wkv,
        "wgate": wgate,
        "ape": ape,
        "block_table": table,
        "kv_state": kv_state,
        "score_state": score_state,
        "start_pos": run_start_pos,
        "meta": meta,
    }


# =====================================================================================================================
# CPU golden / NPU 执行与精度校验
# =====================================================================================================================
def run_compressor_golden(inputs, cfg):
    """CPU 侧 golden 计算(复用 compressor_golden.cpu_compressor), 返回输出与 update 掩码."""
    kv_state = inputs["kv_state"].clone()
    score_state = inputs["score_state"].clone()
    update_kv = torch.zeros(kv_state.shape, dtype=torch.bool)
    update_score = torch.zeros(score_state.shape, dtype=torch.bool)
    cpu_out, cmp_kv_mask, *_ = cpu_compressor(
        inputs["x"],
        inputs["wkv"],
        inputs["wgate"],
        kv_state,
        score_state,
        update_kv,
        update_score,
        inputs["ape"],
        block_table=inputs["block_table"],
        cu_seqlens=None,  # BSH 布局
        seqused=None,
        start_pos=list(inputs["start_pos"]),
        cmp_ratio=cfg["cmp_ratio"],
        coff=cfg["coff"],
        cache_mode=cfg["cache_mode"],
    )
    return {
        "cpu_out": cpu_out,
        "cmp_kv_mask": cmp_kv_mask,
        "cpu_kv": kv_state,
        "cpu_score": score_state,
        "update_kv": update_kv,
        "update_score": update_score,
    }


def run_compressor_npu(inputs, cfg):
    """NPU 直调一次, 返回 (cmp_kv输出, state_cache) 的 CPU 拷贝."""
    compressor_op = load_compressor_op()
    torch_npu.npu.set_device(DEVICE_ID)
    device = "npu:%d" % DEVICE_ID
    kv_state = inputs["kv_state"]
    score_state = inputs["score_state"]
    # state_cache 布局: [block_num, block_size, 2*coff*D], 前半 kv, 后半 score, fp32
    state_cache = torch.zeros(
        (kv_state.shape[0], kv_state.shape[1], 2 * kv_state.shape[2]),
        dtype=torch.float32,
    )
    state_cache[:, :, : kv_state.shape[2]] = kv_state
    state_cache[:, :, kv_state.shape[2] :] = score_state
    state_cache = state_cache.to(device)

    x = inputs["x"].to(device)
    wkv = inputs["wkv"].to(device)
    wgate = inputs["wgate"].to(device)
    ape = inputs["ape"].to(device)
    block_table = inputs["block_table"].to(device)
    start_pos = torch.tensor(inputs["start_pos"], dtype=torch.int32).to(device)
    npu_out = compressor_op(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio=cfg["cmp_ratio"],
        state_block_table=block_table,
        cu_seqlens=None,  # BSH 布局
        seqused=None,
        start_pos=start_pos,
        coff=cfg["coff"],
        cache_mode=cfg["cache_mode"],
    )
    return npu_out.cpu(), state_cache.cpu()


def check_golden(inputs, cfg, npu_out, state_cache_cpu, golden_ret):
    """与 CPU golden 做精度校验(与 run_compressor_eager 同口径的 5 项检查)."""
    data_type = str(npu_out.dtype).replace("torch.", "")
    half = golden_ret["cpu_kv"].shape[2]
    results = {}
    print("-" * 30 + "golden check: cmp_kv" + "-" * 30)
    _, results["cmp_kv"] = check_result(
        golden_ret["cpu_out"][golden_ret["cmp_kv_mask"]].to(torch.float32),
        npu_out[golden_ret["cmp_kv_mask"]].to(torch.float32),
        data_type,
    )
    print("-" * 30 + "golden check: kv_state_update" + "-" * 30)
    _, results["kv_state_update"] = check_result(
        golden_ret["cpu_kv"][golden_ret["update_kv"]].to(torch.float32),
        state_cache_cpu[:, :, :half][golden_ret["update_kv"]].to(torch.float32),
        data_type,
    )
    print("-" * 30 + "golden check: score_state_update" + "-" * 30)
    _, results["score_state_update"] = check_result(
        golden_ret["cpu_score"][golden_ret["update_score"]].to(torch.float32),
        state_cache_cpu[:, :, half:][golden_ret["update_score"]].to(torch.float32),
        data_type,
    )
    print("-" * 30 + "golden check: kv_state_origin" + "-" * 30)
    _, results["kv_state_origin"] = check_result(
        golden_ret["cpu_kv"][~golden_ret["update_kv"]].to(torch.float32),
        state_cache_cpu[:, :, :half][~golden_ret["update_kv"]].to(torch.float32),
        data_type,
        0.0,
    )
    print("-" * 30 + "golden check: score_state_origin" + "-" * 30)
    _, results["score_state_origin"] = check_result(
        golden_ret["cpu_score"][~golden_ret["update_score"]].to(torch.float32),
        state_cache_cpu[:, :, half:][~golden_ret["update_score"]].to(torch.float32),
        data_type,
        0.0,
    )
    failed = [key for key, val in results.items() if val != "Pass"]
    assert not failed, "CPU golden 精度校验失败: %s" % failed
    return results


def run_once(cfg, batch_size, permutation=None, start_pos_list=None, tag=""):
    """执行一次: 构造输入 -> CPU golden 校验 -> NPU 直调, 返回全部输入与输出(CPU)."""
    inputs = build_case_inputs(
        cfg, batch_size, permutation=permutation, start_pos_list=start_pos_list
    )
    golden_ret = run_compressor_golden(inputs, cfg)
    npu_out, state_cache_cpu = run_compressor_npu(inputs, cfg)
    check_golden(inputs, cfg, npu_out, state_cache_cpu, golden_ret)
    print(
        "[run_once] tag=%s B=%d start_pos[0:3]=%s 模板=%s"
        % (tag, batch_size, inputs["start_pos"][:3], predict_template(batch_size, cfg))
    )
    return {
        "inputs": inputs,
        "npu_out": npu_out,
        "state_cache": state_cache_cpu,
        "golden": golden_ret,
        "tag": tag,
    }


# =====================================================================================================================
# 位级对比
# =====================================================================================================================
def _first_mismatch(ref, cur, limit=5):
    ref_flat = ref.flatten()
    cur_flat = cur.flatten()
    neq = ref_flat != cur_flat
    idx = torch.nonzero(neq).flatten()
    info = {"total": int(neq.sum()), "numel": int(ref_flat.numel())}
    if info["total"] > 0:
        diff = (ref_flat[idx].to(torch.float32) - cur_flat[idx].to(torch.float32)).abs()
        info["max_abs_diff"] = float(diff.max())
        first = idx[:limit].tolist()
        info["first_idx"] = first
        info["first_pairs"] = [(float(ref_flat[i]), float(cur_flat[i])) for i in first]
    return info


def assert_bit_level(tag, ref, cur):
    """位级一致性断言: 要求 shape/dtype 一致且逐 bit 相等."""
    assert ref.shape == cur.shape and ref.dtype == cur.dtype, (
        "[%s] shape/dtype 不一致: %s%s vs %s%s"
        % (tag, tuple(ref.shape), ref.dtype, tuple(cur.shape), cur.dtype)
    )
    if torch.equal(ref, cur):
        print("[bitlevel][PASS] %s: 位级一致 (numel=%d)" % (tag, ref.numel()))
        return True
    info = _first_mismatch(ref, cur)
    msg = (
        "[bitlevel][FAIL] %s: 位级不一致 %d/%d 个元素, max_abs_diff=%s, "
        "first_idx=%s, first_pairs=%s"
        % (
            tag,
            info["total"],
            info["numel"],
            info.get("max_abs_diff"),
            info.get("first_idx"),
            info.get("first_pairs"),
        )
    )
    if BITLEVEL_STRICT:
        raise AssertionError(
            msg + "\n(设置 CMP_BITLEVEL_STRICT=0 可降级为告警+阈值校验)"
        )
    print(msg)
    # 降级模式: 位级失败时退回阈值校验, 区分"位级尾差"与"功能错误"
    if ref.dtype == torch.bfloat16:
        rtol, atol = 0.0078125, 1e-4
    else:
        rtol, atol = 0.005, 2.5e-5
    close = torch.isclose(
        ref.to(torch.float32), cur.to(torch.float32), rtol=rtol, atol=atol
    )
    ratio = float(close.float().mean())
    print("[bitlevel][WARN] %s: 阈值内占比 %.6f%%" % (tag, ratio * 100.0))
    assert ratio >= 0.995, "%s 连阈值校验都未通过(占比 %.6f%%), 疑似功能错误" % (
        tag,
        ratio * 100.0,
    )
    return False


# =====================================================================================================================
# 用例执行: pair(B变化) / position(位置变化)
# =====================================================================================================================
def run_pair_case(cfg, b_small, b_large):
    """pair 用例: B=(b_small, b_large) 两次同种子运行, 共享 batch(0..b_small-1)结果位级对比.

    覆盖: B 穿过 FULL_LOAD->PERF 模板边界(B=32/33)或 PERF 动态D分块悬崖(dBaseSize 32<->64)时,
    共享 batch 的 cmp_kv 输出与 state_cache 更新必须与 B 无关(位级一致).
    """
    assert b_small < b_large, "pair 用例要求 b_small < b_large"
    aic = get_aic_core_num()
    print("=" * 100)
    print(
        "[pair] B=(%d, %d), 设备=%s, AIC核数=%d, 模板: %s -> %s, 预测PERF dBaseSize: %s -> %s"
        % (
            b_small,
            b_large,
            torch.npu.get_device_name(DEVICE_ID),
            aic,
            predict_template(b_small, cfg),
            predict_template(b_large, cfg),
            predict_perf_dsplit(b_small, cfg, aic),
            predict_perf_dsplit(b_large, cfg, aic),
        )
    )
    run_small = run_once(cfg, b_small, tag="pair_B%d" % b_small)
    run_large = run_once(cfg, b_large, tag="pair_B%d" % b_large)

    n_shared = b_small
    # 前置校验: 两次运行的共享 batch 输入必须一致(block_table 前缀一致)
    table_small = run_small["inputs"]["block_table"]
    table_large = run_large["inputs"]["block_table"]
    assert torch.equal(table_small, table_large[:n_shared]), (
        "两次运行的 block_table 前缀不一致, 请检查数据构造逻辑"
    )

    # 1. cmp_kv 输出对比(共享 batch)
    assert_bit_level(
        "pair_B%d_%d/cmp_kv[0:%d]" % (b_small, b_large, n_shared),
        run_small["npu_out"][:n_shared],
        run_large["npu_out"][:n_shared],
    )

    # 2. state_cache 对比(共享 batch 引用的物理块, 含更新块与只读块)
    shared_ids = sorted({int(v) for v in table_small.flatten().tolist() if int(v) != 0})
    half = run_small["inputs"]["kv_state"].shape[2]
    kv_small = run_small["state_cache"]
    kv_large = run_large["state_cache"]
    assert_bit_level(
        "pair_B%d_%d/kv_state[shared_blocks]" % (b_small, b_large),
        kv_small[shared_ids, :, :half],
        kv_large[shared_ids, :, :half],
    )
    assert_bit_level(
        "pair_B%d_%d/score_state[shared_blocks]" % (b_small, b_large),
        kv_small[shared_ids, :, half:],
        kv_large[shared_ids, :, half:],
    )
    print("[pair] B=(%d, %d) 位级一致性检查完成" % (b_small, b_large))


def _compare_two_samples(cfg, run_a, pos_a, run_b, pos_b, tag):
    """对比同一样本在两次运行(分别位于 pos_a / pos_b)的输出: cmp_kv 行 + state 物理块(按列对齐)."""
    # 1. cmp_kv 输出行
    assert_bit_level(
        "%s/cmp_kv[pos%d_vs_pos%d]" % (tag, pos_a, pos_b),
        run_a["npu_out"][pos_a],
        run_b["npu_out"][pos_b],
    )
    # 2. state 物理块: 同一样本两次运行引用的物理块 id 不同, 但 block_table 行的非零列相同, 按列对齐比较
    row_a = run_a["inputs"]["block_table"][pos_a].tolist()
    row_b = run_b["inputs"]["block_table"][pos_b].tolist()
    assert len(row_a) == len(row_b), "block_table 列数不一致"
    half = run_a["inputs"]["kv_state"].shape[2]
    for j, (id_a, id_b) in enumerate(zip(row_a, row_b)):
        assert (id_a != 0) == (id_b != 0), (
            "[%s] block_table 第%d列有效性不一致: %d vs %d" % (tag, j, id_a, id_b)
        )
        if id_a == 0:
            continue
        assert_bit_level(
            "%s/col%d/kv_state[blk%d_vs_blk%d]" % (tag, j, id_a, id_b),
            run_a["state_cache"][id_a, :, :half],
            run_b["state_cache"][id_b, :, :half],
        )
        assert_bit_level(
            "%s/col%d/score_state[blk%d_vs_blk%d]" % (tag, j, id_a, id_b),
            run_a["state_cache"][id_a, :, half:],
            run_b["state_cache"][id_b, :, half:],
        )


def run_position_case(cfg, batch_size):
    """position 用例: 同 B 多次运行, 目标样本(独立数据+独立start_pos)从位置0(首)换到中/尾.

    交换方式: 目标样本与位置 mid/tail 的样本整体对调(其余样本位置不变).
    覆盖: 目标样本与被交换样本在不同位置(->不同核/不同hIdxStart)下的位级一致性.
    """
    print("=" * 100)
    print(
        "[position] B=%d, 设备=%s, 模板=%s, 目标样本 start_pos=%d, 其余样本 start_pos=%d"
        % (
            batch_size,
            torch.npu.get_device_name(DEVICE_ID),
            predict_template(batch_size, cfg),
            cfg["target_start_p"],
            cfg["default_start_p"],
        )
    )
    # 样本0 = 目标样本(独立数据 + target_start_p), 初始位于位置0(首)
    start_pos_canon = [cfg["default_start_p"]] * batch_size
    start_pos_canon[0] = cfg["target_start_p"]
    reference = run_once(
        cfg,
        batch_size,
        permutation=None,
        start_pos_list=start_pos_canon,
        tag="pos_head",
    )
    mid_pos = batch_size // 2
    placements = {"head": 0, "mid": mid_pos, "tail": batch_size - 1}
    for name, pos in placements.items():
        if pos == 0:
            print("[position:%s] 目标样本已在位置0, 即参考运行" % name)
            continue
        perm = list(range(batch_size))
        perm[0], perm[pos] = perm[pos], perm[0]  # 目标样本(样本0) <-> 位置pos样本 交换
        run = run_once(
            cfg,
            batch_size,
            permutation=perm,
            start_pos_list=start_pos_canon,
            tag="pos_%s" % name,
        )
        # 目标样本: 参考运行位于位置0, 本次运行位于位置pos
        _compare_two_samples(cfg, reference, 0, run, pos, "position_%s/target" % name)
        # 被交换样本: 参考运行位于位置pos, 本次运行位于位置0
        _compare_two_samples(
            cfg, reference, pos, run, 0, "position_%s/displaced" % name
        )
    print("[position] B=%d 位置一致性检查完成" % batch_size)
