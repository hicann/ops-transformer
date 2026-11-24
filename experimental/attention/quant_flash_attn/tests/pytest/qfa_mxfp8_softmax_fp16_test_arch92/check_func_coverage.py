"""func 用例覆盖矩阵检查: 每个特性/交叉特性是否覆盖充分, 并统计总数.

用法: python check_func_coverage.py
"""

import importlib.util
import os
import sys
from collections import Counter

REF = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(REF, "quant_flash_attn_paramset_func_rdv.py")

spec = importlib.util.spec_from_file_location("ps", PATH)
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
CASES = ps.CASES


def s1max(c):
    return max(c["seqused_q"]) if c["seqused_q"] else 0


def s2max(c):
    return max(c["seqused_kv"]) if c["seqused_kv"] else 0


def varlen(c):
    return len(set(c["seqused_q"])) > 1 or len(set(c["seqused_kv"])) > 1


def has_zero(c):
    return any(v == 0 for v in c["seqused_q"]) or any(v == 0 for v in c["seqused_kv"])


def g(c):
    return c["N_q"] // c["N_kv"]


def s1_bucket(c):
    s = s1max(c)
    if s <= 128:
        return "s1<=128"
    if s <= 256:
        return "s1 129..256"
    if s <= 512:
        return "s1 257..512"
    return "s1>512"


def s2_bucket(c):
    s = s2max(c)
    if s < 256:
        return "s2<256"
    if s <= 512:
        return "s2 256..512"
    if s <= 1024:
        return "s2 513..1024"
    return "s2>1024"


def s2_mod512(c):
    r = s2max(c) % 512
    return "s2%512=0" if r == 0 else ("s2%512=256" if r == 256 else "s2%512=other")


SINGLE = {
    "layout": lambda c: c["input_layout"],
    "mask": lambda c: f"mask{c['mask_mode']}",
    "lse": lambda c: f"lse{int(bool(c.get('enable_lse')))}",
    "pa": lambda c: "pa" if c.get("enable_pa") else "nopa",
    "pa_block": lambda c: f"pa_bs{c['block_size']}" if c.get("enable_pa") else "nopa",
    "pa_shuffle": lambda c: (
        f"shuf{int(bool(c.get('pa_shuffle_blocks')))}" if c.get("enable_pa") else "nopa"
    ),
    "s1_bucket": s1_bucket,
    "s2_bucket": s2_bucket,
    "s2_mod512": s2_mod512,
    "s1_mod128": lambda c: "s1%128=0" if s1max(c) % 128 == 0 else "s1%128!=0",
    "s2_mod64": lambda c: "s2%64=0" if s2max(c) % 64 == 0 else "s2%64!=0",
    "g": lambda c: f"g{g(c)}",
    "batch": lambda c: f"b{c['B']}"
    if c["B"] <= 8
    else ("b16..32" if c["B"] <= 32 else "b48+"),
    "varlen": lambda c: f"varlen{int(varlen(c))}",
    "zero": lambda c: f"zero{int(has_zero(c))}",
    "pscale": lambda c: "ps1" if abs(float(c["p_scale"]) - 1.0) < 1e-9 else "ps<1",
    "nkv": lambda c: f"nkv{c['N_kv']}",
}

PAIRS = [
    ("layout", "mask"),
    ("layout", "lse"),
    ("mask", "lse"),
    ("layout", "pa"),
    ("s1_bucket", "s2_bucket"),
    ("g", "layout"),
    ("s1_mod128", "mask"),
    ("s2_mod512", "layout"),
    ("varlen", "layout"),
    ("pscale", "layout"),
    ("batch", "mask"),
    ("pa", "pa_shuffle"),
    ("s1_bucket", "mask"),
    ("s2_bucket", "mask"),
]

print(
    f"总用例 {len(CASES)} | 非PA {sum(1 for c in CASES if not c.get('enable_pa'))} | "
    f"PA {sum(1 for c in CASES if c.get('enable_pa'))}"
)
for name, fn in SINGLE.items():
    print(
        f"  {name:12} {len({fn(c) for c in CASES}):2d} 类: {sorted({fn(c) for c in CASES})}"
    )
for a, b in PAIRS:
    fa, fb = SINGLE[a], SINGLE[b]
    print(f"  交叉 {a} x {b}: {len({(fa(c), fb(c)) for c in CASES})} 种")

gaps = []
for name, fn in SINGLE.items():
    for v, n in Counter(fn(c) for c in CASES).items():
        if n < 2:
            gaps.append(f"单特性覆盖不足: {name}={v} 只有 {n} 条")
for a, b in PAIRS:
    fa, fb = SINGLE[a], SINGLE[b]
    combos = {(fa(c), fb(c)) for c in CASES}
    need = 3 if (a, b) == ("pa", "pa_shuffle") else 4
    if len(combos) < need:
        gaps.append(f"交叉过少: {a} x {b} 仅 {len(combos)} 种")

print("\n结论:", "覆盖充分" if not gaps else "发现缺口")
for item in gaps:
    print("  -", item)
sys.exit(0 if not gaps else 1)
