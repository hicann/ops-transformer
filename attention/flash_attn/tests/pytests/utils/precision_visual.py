#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""
FlashAttention 精度可视化工具
==============================
将 CPU golden 与 NPU 输出逐元素对比，生成热力图 PNG，直观展示哪些位置精度 fail。

输出文件（保存到 --out_dir）：
  {case}_heatmap_p{N}.png   每页最多 8 个 panel 的 relErr 热力图（绿色=pass，红色=fail）
  {case}_passrate.png        各 panel 精度通过率汇总棒状图

两种使用方式
────────────
① 独立运行（读取 dump_tensors 保存的 txt 文件）：

    python precision_visual.py \\
        --dump_dir ./dump_output --case_id BASE_01 \\
        --out_dir  ./viz_output

    python precision_visual.py \\
        --cpu_txt ./dump_output/BASE_01/cpu_out.txt \\
        --npu_txt ./dump_output/BASE_01/npu_out.txt \\
        --out_dir ./viz_output --case_name BASE_01

② 由 test_flash_attn.py 通过 --visualize 参数自动调用（内存中直接传 tensor）：
    python test_flash_attn.py --case_id BASE_01 --visualize

依赖：matplotlib（pip install matplotlib）
"""

import os
import sys
import argparse
import logging

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 非交互后端，支持无 GUI 服务器环境
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger(__name__)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

RATIO_THRESHOLD = 0.005  # 相对误差阈值（0.5%）

# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_tensor_from_txt(filepath):
    """读取 save_tensor_to_txt 写出的文件，返回 (ndarray[float32], shape_tuple)。"""
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    # 首行格式: "# shape: 1x8x512x128  total: 524288"
    shape_part = lines[0].split("shape:")[-1].split("total:")[0].strip()
    shape = tuple(int(s) for s in shape_part.split("x") if s)
    data = np.array([float(ln) for ln in lines[1:] if ln.strip()], dtype=np.float32)
    return data.reshape(shape), shape


# ─────────────────────────────────────────────────────────────────────────────
# 精度计算
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    cpu: np.ndarray, npu: np.ndarray, threshold: float = RATIO_THRESHOLD
):
    """
    返回:
      rel_err  : 逐元素相对误差（与 cpu/npu 同 shape, float32）
      fail_mask: bool ndarray，True 表示该元素超阈值
      stats    : dict，汇总统计量
    """
    cpu = cpu.astype(np.float32)
    npu = npu.astype(np.float32)
    diff = np.abs(cpu - npu)
    ref_abs = np.abs(cpu)
    rel_err = diff / (ref_abs + 1e-6)

    # 与 check_result 保持一致：threshold = max(ref*0.5%, 0.000025)
    thresh_map = np.maximum(ref_abs * threshold, 0.000025)
    fail_mask = diff > thresh_map

    total = cpu.size
    fail_cnt = int(fail_mask.sum())
    stats = dict(
        shape=cpu.shape,
        total=total,
        fail_cnt=fail_cnt,
        fail_ratio=fail_cnt / total,
        max_rel=float(rel_err.max()),
        mean_rel=float(rel_err.mean()),
        max_abs=float(diff.max()),
    )
    return rel_err, fail_mask, stats


# ─────────────────────────────────────────────────────────────────────────────
# 拆分 panel（与 shape 无关的通用规则）
# ─────────────────────────────────────────────────────────────────────────────


def split_to_panels(arr: np.ndarray):
    """
    将任意维度的 ndarray 拆分为 (num_panels, H, W) 方便逐格绘图。

    规则（基于 ndim）：
      4D (a, b, c, d)  →  a*b panels，每 panel (c, d)
                          label: "a{i}_b{j}"，适合 (B,N,S,D) / (B,N,T,D)
      3D (a, b, c)     →  a   panels，每 panel (b, c)
                          label: "i{k}"，适合 (T,N,D) 以 T 为 panel 或 (N,T,D) 以 N 为 panel
      2D / 1D          →  1   panel
    """
    if arr.ndim == 4:
        a, b, c, d = arr.shape
        panels = arr.reshape(a * b, c, d)
        labels = [f"g{i // b}_h{i % b}" for i in range(a * b)]
    elif arr.ndim == 3:
        a, b, c = arr.shape
        panels = arr  # shape (a, b, c)
        labels = [f"p{i}" for i in range(a)]
    elif arr.ndim == 2:
        panels = arr[np.newaxis]  # (1, H, W)
        labels = ["p0"]
    else:
        panels = arr.reshape(1, 1, -1)
        labels = ["p0"]
    return panels, labels


# ─────────────────────────────────────────────────────────────────────────────
# 热力图绘制
# ─────────────────────────────────────────────────────────────────────────────


def _build_cmap(threshold: float, vmax: float):
    """
    构建双段色图：
      [0,  threshold] → 绿色渐变（pass）
      [threshold, vmax] → 红色渐变（fail）
    """
    n_pass = max(2, int(256 * threshold / max(vmax, 1e-9)))
    n_fail = 256 - n_pass
    colors_pass = plt.cm.Greens_r(np.linspace(0.1, 0.6, n_pass))
    colors_fail = plt.cm.Reds(np.linspace(0.4, 1.0, n_fail))
    all_colors = np.vstack([colors_pass, colors_fail])
    return mcolors.LinearSegmentedColormap.from_list("prec", all_colors, N=512)


def _plot_one_panel(
    ax,
    panel_err: np.ndarray,
    panel_fail: np.ndarray,
    label: str,
    cmap,
    norm,
    threshold: float,
):
    """在 ax 上绘制单个 panel 的热力图 + fail 遮罩。"""
    im = ax.imshow(
        panel_err, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
    )

    # 在 fail 元素上叠加半透明红色遮罩
    rgba = np.zeros((*panel_fail.shape, 4), dtype=np.float32)
    rgba[panel_fail] = [0.85, 0.1, 0.1, 0.50]
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    ax.set_title(label, fontsize=8, pad=3)
    ax.set_xlabel("dim →", fontsize=6, labelpad=1)
    ax.set_ylabel("seq →", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    return im


def plot_precision_heatmaps(
    rel_err: np.ndarray,
    fail_mask: np.ndarray,
    case_name: str,
    out_dir: str,
    stats: dict,
    threshold: float = RATIO_THRESHOLD,
    max_panels: int = 16,
) -> list:
    """
    主绘图函数，生成两类图：
      ① 每页 ≤8 个 panel 的 relErr 热力图
      ② 各 panel 精度通过率汇总棒状图
    返回所有生成的文件路径列表。
    """
    os.makedirs(out_dir, exist_ok=True)
    case_name = case_name.replace("/", "_")
    panels_err, labels = split_to_panels(rel_err)
    panels_fail, _ = split_to_panels(fail_mask)

    n_panels = min(len(panels_err), max_panels)
    vmax = max(float(rel_err.max()) * 1.05, threshold * 3, 1e-6)
    cmap = _build_cmap(threshold, vmax)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    shape_str = "×".join(str(s) for s in stats["shape"])
    fail_ratio = stats["fail_ratio"] * 100
    title_meta = (
        f"shape={shape_str}   failElems={stats['fail_cnt']}/{stats['total']}"
        f" ({fail_ratio:.3f}%)   maxRelErr={stats['max_rel']:.4f}"
    )

    legend_elems = [
        Patch(
            facecolor=[0.85, 0.1, 0.1, 0.5],
            label=f"FAIL (relErr > {threshold * 100:.1f}%)",
        ),
        Patch(facecolor=[0.1, 0.6, 0.2, 0.7], label="PASS"),
    ]

    # ── ① 热力图（每页 8 个 panel）────────────────────────────────────────
    per_page = 8
    n_pages = (n_panels + per_page - 1) // per_page
    saved_files = []

    for pg in range(n_pages):
        start = pg * per_page
        end = min(start + per_page, n_panels)
        n_sub = end - start
        ncols = min(n_sub, 4)
        nrows = (n_sub + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 4.8 + 0.8, nrows * 3.6 + 1.4), squeeze=False
        )
        fig.suptitle(
            f"{case_name}  精度热力图  (page {pg + 1}/{n_pages})\n{title_meta}",
            fontsize=10,
            y=1.01,
        )

        last_im = None
        for idx in range(n_sub):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            err = panels_err[start + idx]
            fal = panels_fail[start + idx].astype(bool)
            last_im = _plot_one_panel(
                ax, err, fal, labels[start + idx], cmap, norm, threshold
            )

        # 隐藏多余子图
        for idx in range(n_sub, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        # 统一 colorbar
        if last_im is not None:
            cbar = fig.colorbar(
                last_im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02, shrink=0.6
            )
            cbar.set_label("relative error", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            # 在 colorbar 上画阈值刻度线
            cbar.ax.axhline(threshold, color="yellow", linewidth=1.5, linestyle="--")
            cbar.ax.text(
                1.8,
                threshold / vmax,
                f"{threshold * 100:.1f}%",
                transform=cbar.ax.transAxes,
                fontsize=6,
                color="goldenrod",
                va="center",
            )

        fig.legend(
            handles=legend_elems, loc="lower left", fontsize=7, framealpha=0.85, ncol=2
        )

        fpath = os.path.join(out_dir, f"{case_name}_heatmap_p{pg + 1}.png")
        plt.tight_layout()
        fig.savefig(fpath, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(fpath)
        logger.info(f"  [viz] heatmap → {fpath}")

    # ── ② 通过率汇总棒状图 ────────────────────────────────────────────────
    pass_rates = [1.0 - float(panels_fail[i].mean()) for i in range(n_panels)]
    fail_rates = [1.0 - r for r in pass_rates]

    fig2, ax2 = plt.subplots(figsize=(max(8, n_panels * 0.6 + 1.5), 4.2))
    bar_colors = ["#e74c3c" if fr > 0 else "#2ecc71" for fr in fail_rates]
    bars = ax2.bar(
        range(n_panels),
        pass_rates,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.7,
    )

    ax2.axhline(1.0, color="#7f8c8d", linestyle="--", linewidth=0.8, label="100% pass")
    ax2.set_ylim(max(0, min(pass_rates) - 0.05), 1.06)
    ax2.set_xticks(range(n_panels))
    ax2.set_xticklabels(labels[:n_panels], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Pass Rate", fontsize=9)
    ax2.set_xlabel("Panel", fontsize=9)
    ax2.set_title(f"{case_name}  各 Panel 精度通过率\n{title_meta}", fontsize=9)

    for bar, rate in zip(bars, pass_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{rate * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=5.5,
            rotation=90 if n_panels > 8 else 0,
        )

    ax2.legend(fontsize=7, loc="lower right")
    plt.tight_layout()

    summary_path = os.path.join(out_dir, f"{case_name}_passrate.png")
    fig2.savefig(summary_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    saved_files.append(summary_path)
    logger.info(f"  [viz] pass rate chart → {summary_path}")

    return saved_files


# ─────────────────────────────────────────────────────────────────────────────
# 对外接口（供 test_flash_attn.py 直接调用）
# ─────────────────────────────────────────────────────────────────────────────


def visualize_from_tensors(
    cpu_tensor,
    npu_tensor,
    case_name: str,
    out_dir: str,
    threshold: float = RATIO_THRESHOLD,
    max_panels: int = 16,
):
    """
    从内存 tensor（torch.Tensor 或 np.ndarray, float32）直接生成可视化图。
    供 test_flash_attn.py 的 --visualize 模式调用。
    """
    to_np = lambda t: (
        t.detach().cpu().numpy()
        if hasattr(t, "detach")
        else np.asarray(t, dtype=np.float32)
    )
    cpu_np = to_np(cpu_tensor).astype(np.float32)
    npu_np = to_np(npu_tensor).astype(np.float32)

    rel_err, fail_mask, stats = compute_metrics(cpu_np, npu_np, threshold)
    logger.info(
        f"\n[viz] {case_name}  shape={stats['shape']}  "
        f"fail={stats['fail_cnt']}/{stats['total']} ({stats['fail_ratio'] * 100:.4f}%)"
    )
    return plot_precision_heatmaps(
        rel_err, fail_mask, case_name, out_dir, stats, threshold, max_panels
    )


def visualize_from_txt(
    cpu_txt: str,
    npu_txt: str,
    case_name: str,
    out_dir: str,
    threshold: float = RATIO_THRESHOLD,
    max_panels: int = 16,
):
    """从 dump_tensors 保存的 txt 文件读取并生成可视化图。"""
    cpu_np, _ = load_tensor_from_txt(cpu_txt)
    npu_np, _ = load_tensor_from_txt(npu_txt)
    return visualize_from_tensors(
        cpu_np, npu_np, case_name, out_dir, threshold, max_panels
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口（独立运行）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlashAttention precision visualization tool",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example:
  # Read from dump directory (used with test_flash_attn.py --dump_tensors)
  python precision_visual.py --dump_dir ./dump_output --case_id BASE_01,TND_01

  # Directly specify txt file
  python precision_visual.py --cpu_txt ./dump_output/BASE_01/cpu_out.txt \\
                              --npu_txt ./dump_output/BASE_01/npu_out.txt
""",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dump_dir",
        type=str,
        metavar="DIR",
        help="dump_tensors root dir, used with --case_id",
    )
    src.add_argument(
        "--cpu_txt", type=str, metavar="FILE", help="directly specify cpu_out.txt path"
    )

    parser.add_argument(
        "--npu_txt",
        type=str,
        metavar="FILE",
        help="directly specify npu_out.txt path (required with --cpu_txt)",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="case name, comma-separated (required with --dump_dir)",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        default=None,
        help="name for chart title (optional with --cpu_txt)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./viz_output",
        help="output directory (default: ./viz_output)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RATIO_THRESHOLD,
        help=f"relative error threshold (default: {RATIO_THRESHOLD})",
    )
    parser.add_argument(
        "--max_panels",
        type=int,
        default=16,
        help="max number of panels to show (default: 16)",
    )
    args = parser.parse_args()

    if args.cpu_txt:
        if not args.npu_txt:
            parser.error("--cpu_txt requires --npu_txt")
        name = args.case_name or os.path.basename(
            os.path.dirname(os.path.abspath(args.cpu_txt))
        )
        visualize_from_txt(
            args.cpu_txt,
            args.npu_txt,
            name,
            args.out_dir,
            args.threshold,
            args.max_panels,
        )
    else:
        if not args.case_id:
            parser.error("--dump_dir requires --case_id")
        for cid in [c.strip() for c in args.case_id.split(",")]:
            cpu_path = os.path.join(args.dump_dir, cid, "cpu_out.txt")
            npu_path = os.path.join(args.dump_dir, cid, "npu_out.txt")
            if not os.path.exists(cpu_path) or not os.path.exists(npu_path):
                logger.info(f"[WARN] {cid}: dump file not found, skipping → {cpu_path}")
                continue
            visualize_from_txt(
                cpu_path, npu_path, cid, args.out_dir, args.threshold, args.max_panels
            )
