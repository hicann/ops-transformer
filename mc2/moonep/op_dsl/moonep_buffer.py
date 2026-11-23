# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Buffer：MoonEP 对外接口的统一门面。

它把 MoonEP 通信上下文和 ElasticBuffer 收在一处，对外只露 dispatch /
prefetch_weight / combine / reduce_grad 四个接口：构造时按形状算好
workspace、建好 UbMem 连接、定好窗口槽位与 dtype；前向 dispatch 顺手把
plan 算出来，prefetch_weight / combine / reduce_grad 拿着同一份 plan 干活，
反向 dispatch 把 plan 递回来就能复用，不必重算。

用法：

    buf = Buffer(S, H, Hp, K, E, num_ep_ranks, group=group)

    # 前向
    hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buf.dispatch(
        hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
    )
    gate_pb, up_pb, down_pb = buf.prefetch_weight(
        plan=plan,
        local_gate_weight=w1, local_up_weight=w3, local_down_weight=w2,
    )
    output_sh, gathered_route_weights_sk, _ = buf.combine(
        plan=plan,
        hidden_nvsh=expert_output_nvsh,
        route_weights_nvs=route_weights_nvs,
    )

    # 反向：复用前向 plan
    grad_expert_output_nvsh, _, _, _ = buf.dispatch(grad_output_sh, plan=plan)
    grad_hidden_sh, _, _ = buf.combine(
        plan=plan, hidden_nvsh=grad_hidden_nvsh,
    )
    gate_rb, up_rb, down_rb = buf.reduce_grad(
        plan=plan,
        local_gate_grad=dw1, local_up_grad=dw3, local_down_grad=dw2,
    )
    buf.destroy()

W13 交织形态用 `local_gate_up_weight` + `local_down_weight`；prefetch 返回
`(views, scales)`，views 为 `(gate_up, down)`，`local_*_scales` 成对传入。
"""

from __future__ import annotations

import threading
import warnings

import moonep_plan
import prefetch_weight
import reduce_grad
import torch
import torch.distributed as dist
from moonep_context import (
    HEAP_BASE,
    LAYOUT_TAIL_BYTES,
    MAX_E,
    MAX_R,
    SCALE_ALIGN_BYTES,
    check_heap_capacity,
    create_moonep_context,
    pick_weight_chunk,
    reduce_region_end,
    sym_barrier,
    sync_transport_stream,
)
from moonep_plan import MoonepPlan

# reduce VL=64 尾块约束（moonep_context VL 同源）：W 非 64 倍数时尾块越界 UB
_W_ALIGN = 64
_UE8M0_BLOCK = 32  # mxfp8 block scale 粒度（每 32 权重元素 1 个 ue8m0 字节）
# UB 分块 fail-fast 试算界（fp32 更严口径——bf16 更宽，取严者构造期报错）
_CHUNK_TRIAL_UB = 16384


def _workspace_need(epn, w, slots, gdtype, wdtype):
    """moonep workspace（通信 buffer）实际容量唯一真源。

    ``HEAP_BASE + reduce_region_end(...) + LAYOUT_TAIL_BYTES``：权重+scale
    发布区 + 梯度副本区 [epn, Wg] × 组 + 尾部对齐（副本区无 R 维度）。
    3 组等宽与 W13 交织的 slot 步长不同，按两者较大者预留。
    """
    grad_elem = 2 if gdtype in (torch.bfloat16, torch.float16) else 4
    weight_elem = int(wdtype.itemsize)
    scale_upper = _scale_upper_bytes(w)
    fp8 = prefetch_weight._is_fp8_weight(wdtype)
    # 3 组等宽与 W13 交织的 slot 步长不同（LCM 对齐），按两者较大者预留
    if fp8:
        layouts = [
            ((w, w, w), (scale_upper,) * 3),
            ((2 * w, w), (_scale_upper_bytes(2 * w), scale_upper)),
        ]
    else:
        layouts = [((w, w, w), None), ((2 * w, w), None)]
    need = 0
    for widths, scale_widths in layouts:
        need = max(
            need,
            HEAP_BASE
            + reduce_region_end(
                epn,
                w,
                len(widths),
                int(slots),
                grad_bytes_per_elem=grad_elem,
                widths=widths,
                scale_widths=scale_widths,
                weight_bytes=weight_elem,
            )
            + LAYOUT_TAIL_BYTES,
        )
    return need


def _normalize_grad_dtype(grad_dtype):
    """grad_dtype 归一化："bf16"/"fp16"/"fp32"（或 torch.dtype）→ torch dtype。"""
    if isinstance(grad_dtype, str):
        if grad_dtype == "bf16":
            return torch.bfloat16
        if grad_dtype == "fp16":
            return torch.float16
        if grad_dtype == "fp32":
            return torch.float32
        raise TypeError(
            f"grad_dtype 仅支持 'bf16'/'fp16'/'fp32'"
            f"（或 torch.bfloat16/torch.float16/torch.float32），"
            f"got {grad_dtype!r}"
        )
    if grad_dtype in (torch.bfloat16, torch.float16, torch.float32):
        return grad_dtype
    raise TypeError(
        f"grad_dtype 仅支持 'bf16'/'fp16'/'fp32'"
        f"（或 torch.bfloat16/torch.float16/torch.float32），"
        f"got {grad_dtype!r}"
    )


def _normalize_weight_dtype(weight_dtype):
    """weight_dtype 归一化："bf16"/"fp16"/"e4m3fn"/"e5m2"（或 torch.dtype）→ torch dtype。"""
    if isinstance(weight_dtype, str):
        if weight_dtype == "bf16":
            return torch.bfloat16
        if weight_dtype == "fp16":
            return torch.float16
        if weight_dtype == "e4m3fn":
            return torch.float8_e4m3fn
        if weight_dtype == "e5m2":
            return torch.float8_e5m2
        raise TypeError(
            "weight_dtype 仅支持 'bf16'/'fp16'/'e4m3fn'/'e5m2'（或 torch 对应"
            " dtype；uint8 不支持——mxfp8 用 e4m3fn/e5m2），"
            f"got {weight_dtype!r}"
        )
    if weight_dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return weight_dtype
    raise TypeError(
        "weight_dtype 仅支持 'bf16'/'fp16'/'e4m3fn'/'e5m2'（或 torch 对应"
        f" dtype；uint8 不支持——mxfp8 用 e4m3fn/e5m2），got {weight_dtype!r}"
    )


def _scale_upper_bytes(w_elems):
    """单组 scale 行宽上界（字节）：ceil(W/32) 个 ue8m0 block scale、32B 对齐。"""
    raw = (int(w_elems) + _UE8M0_BLOCK - 1) // _UE8M0_BLOCK
    return (raw + SCALE_ALIGN_BYTES - 1) // SCALE_ALIGN_BYTES * SCALE_ALIGN_BYTES


class Buffer:
    """MoonEP 对外接口统一封装宿主（dispatch / prefetch_weight / combine /
    reduce_grad；plan 在 dispatch 内部完成）。

    构造内化：workspace 精算 → 单通信组解析（moonep ctx 与 ElasticBuffer
    复用同一 ``ProcessGroup``）→ MoonepContext → window_slots/dtype 定型。
    ``num_sms`` / ``token_padding`` 为预留参数，当前分别只接受 None 与缺省值。

    单 group 说明：MoonEP 数据面使用本 rank 私有注册 workspace，EB 自建
    注册内存；两者共用同一 HCCL 通信器是刻意设计。``_ensure_eb`` 的锁只
    保护 EB 懒构造（防并发重复构造），不改变集合操作的流序——调用方须按
    常规 HCCL 约束保证自有 ``dist.*`` 集合操作与 MoonEP 建连/EB 操作不并发
    发起。
    """

    def __init__(
        self,
        S: int,
        H: int,
        Hp: int,
        K: int,
        E: int,
        num_ep_ranks: int,
        group: torch.distributed.ProcessGroup,
        num_sms: int | None = None,
        token_padding: int = 128,
        *,
        weight_dtype: str | torch.dtype = "bf16",
        grad_dtype: str | torch.dtype = "bf16",
        slots: int = 1,
        eb_netlayer: str = "auto",
    ) -> None:
        """构造（集体调用）。

        Args:
            S: 每 rank token 数。
            H: hidden size（EB token 维，同时是专家 FFN 输入维）。
            Hp: 专家 FFN 中间维；权重行宽上界 ``W = H × Hp``。
            K: 每 token top-k。
            E: EP 组专家总数（``E % R == 0``）。
            num_ep_ranks: EP rank 数，取值 2–128，须等于 ``group`` 的 world
                size。
            group: 通信组（必选）。同一个组同时用于 MoonEP ctx 与
                ElasticBuffer。
            num_sms: 预留参数：计划开放的语义是 MoonEP kernel 使用的 AIV
                核数；当前仅支持 None。
            token_padding: 预留参数（当前不生效）。
            weight_dtype: 权重 dtype（bf16/fp16/e4m3fn/e5m2）。
            grad_dtype: 梯度 dtype（bf16/fp16/fp32）。
            slots: 权重窗口槽数（层间流水取 2）。
            eb_netlayer: ElasticBuffer netlayer 透传，``auto`` 或非负数字符串。
        """
        if not isinstance(group, torch.distributed.ProcessGroup):
            raise TypeError(
                f"group 须为 torch.distributed.ProcessGroup，got {type(group).__name__}"
            )
        rank_size = dist.get_world_size(group)
        if rank_size > MAX_R:
            raise ValueError(f"EP 组规模 {rank_size} 超过 MAX_R={MAX_R}")
        if E > MAX_E:
            raise ValueError(f"E={E} 超过 MAX_E={MAX_E}")
        if (
            not isinstance(num_ep_ranks, int)
            or isinstance(num_ep_ranks, bool)
            or not 2 <= num_ep_ranks <= MAX_R
        ):
            raise ValueError(
                f"num_ep_ranks 须为 2–{MAX_R} 的整数，got {num_ep_ranks!r}"
            )
        if num_ep_ranks != rank_size:
            raise ValueError(
                f"num_ep_ranks={num_ep_ranks} 与 group world size={rank_size} 不符"
            )
        if num_sms is not None:
            raise NotImplementedError(
                f"num_sms 为预留参数，当前仅支持 None，got {num_sms!r}"
            )
        if (
            not isinstance(token_padding, int)
            or isinstance(token_padding, bool)
            or token_padding <= 0
        ):
            raise ValueError(f"token_padding 须为正整数，got {token_padding!r}")
        if S < 1:
            raise ValueError(f"S 须 >= 1，got {S}")
        if K < 1:
            raise ValueError(f"K 须 >= 1，got {K}")
        if E < 1:
            raise ValueError(f"E 须 >= 1，got {E}")
        if E % rank_size != 0:
            raise ValueError(
                f"E={E} 必须能被 rank_size={rank_size} 整除（epn = E // rank_size）"
            )
        if slots < 1:
            raise ValueError(f"slots 须 >= 1，got {slots}")
        if H < 1:
            raise ValueError(f"H 须 >= 1，got {H}")
        if Hp < 1:
            raise ValueError(f"Hp 须 >= 1，got {Hp}")
        if not (
            isinstance(eb_netlayer, str)
            and (
                eb_netlayer == "auto"
                or (eb_netlayer.isascii() and eb_netlayer.isdigit())
            )
        ):
            raise ValueError(
                f"eb_netlayer 须为 'auto' 或非负数字符串，got {eb_netlayer!r}"
            )
        w = int(H) * int(Hp)
        gdtype = _normalize_grad_dtype(grad_dtype)
        wdtype = _normalize_weight_dtype(weight_dtype)
        if w % _W_ALIGN != 0:
            raise ValueError(
                f"W={w} 须为 {_W_ALIGN} 的倍数（reduce VL=64 尾块越界防护）"
            )
        pick_weight_chunk(w, _CHUNK_TRIAL_UB)
        epn = E // rank_size
        heap_need = _workspace_need(epn, w, slots, gdtype, wdtype)

        self._group = group
        self._ctx = create_moonep_context(group=self._group, workspace_bytes=heap_need)
        self._ctx.window_slots = slots
        self._ctx.weight_dtype = wdtype
        self._ctx.grad_dtype = gdtype
        check_heap_capacity(self._ctx, heap_need, "Buffer 构造")
        self._eb = None  # 懒构造（首调 dispatch/combine）
        self._eb_lock = threading.Lock()  # 首调并发守卫（EB 构造不可重入）
        self._nmtr = int(S)
        self._eb_netlayer = eb_netlayer
        self._S, self._K, self._E = int(S), int(K), int(E)
        self._H = int(H)
        self._epn = epn
        self._Hp = int(Hp)
        self._slots = int(slots)
        self._weight_dtype = wdtype
        self._grad_dtype = gdtype
        self._heap_need = heap_need
        self._heap_bytes = self._ctx.workspace_bytes
        self._rank = self._ctx.rank
        self._rank_size = self._ctx.rank_size
        self._num_sms = num_sms
        self._token_padding = token_padding
        self._handle_fwd = None
        self._handle_bwd = None
        self._handle_fwd_plan = None
        self._handle_bwd_plan = None
        self._destroyed = False

    # --------------------------------------------------------------------------
    # 内部守卫
    # --------------------------------------------------------------------------

    def _require_alive(self):
        if self._destroyed or self._ctx is None:
            raise RuntimeError("Buffer 已 destroy：视图/plan 张量的使用须先于 destroy")
        return self._ctx

    def _ensure_eb(self):
        if self._eb is not None:
            return
        with self._eb_lock:
            if self._eb is not None:
                return
            try:
                from cann_ops_transformer.ops import ElasticBuffer
            except ImportError as exc:
                raise ImportError(
                    "dispatch/combine 需要 cann_ops_transformer（ElasticBuffer）。"
                    "安装：pip install <cann_ops_transformer whl>"
                ) from exc
            # 仅显式设置时透传——旧版 ElasticBuffer 不接受 netlayer 参数
            eb_kwargs = {}
            if self._eb_netlayer != "auto":
                eb_kwargs["netlayer"] = self._eb_netlayer
            self._eb = ElasticBuffer(
                self._group,
                num_max_tokens_per_rank=self._nmtr,
                hidden=self._H,
                num_topk=self._K,
                **eb_kwargs,
            )

    # --------------------------------------------------------------------------
    # 公开接口
    # --------------------------------------------------------------------------

    def _plan(
        self, tokens_per_expert: torch.Tensor, topk_experts_sk: torch.Tensor
    ) -> MoonepPlan:
        """内部规划入口（dispatch 前向调用）；不对外。"""
        ctx = self._require_alive()
        if tokens_per_expert.shape[0] != self._E:
            raise ValueError(
                f"tokens_per_expert.shape[0]={tokens_per_expert.shape[0]} "
                f"!= 构造期 E={self._E}"
            )
        if tuple(topk_experts_sk.shape) != (self._S, self._K):
            raise ValueError(
                f"topk_experts_sk.shape={tuple(topk_experts_sk.shape)} "
                f"!= 构造期 (S, K)=({self._S}, {self._K})"
            )
        return moonep_plan.moonep_plan(ctx, tokens_per_expert, topk_experts_sk)

    def dispatch(
        self,
        hidden_sh: torch.Tensor,
        route_weights_sk: torch.Tensor | None = None,
        topk_experts_sk: torch.Tensor | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        *,
        plan: MoonepPlan | None = None,
        do_cpu_sync: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, MoonepPlan]:
        """前向（建 plan）或反向（复用 plan）的 token 分发。

        前向：传 ``topk_experts_sk`` + ``tokens_per_expert``（``plan`` 为空），
        内部生成 plan；返回 ``(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)``。
        复用规划：传 ``plan=`` + ``route_weights_sk`` 为同路由前向重分发；
        只传 ``plan=``（无权重）为反向梯度重分发。
        """
        self._require_alive()
        if hidden_sh.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"hidden_sh dtype 须为 bfloat16/float16，got {hidden_sh.dtype}"
            )
        if not hidden_sh.is_contiguous():
            raise ValueError("hidden_sh 须为连续张量")
        if tuple(hidden_sh.shape) != (self._S, self._H):
            raise ValueError(
                f"hidden_sh.shape={tuple(hidden_sh.shape)} 须为 (S, H)="
                f"({self._S}, {self._H})"
            )
        if plan is not None:
            if topk_experts_sk is not None or tokens_per_expert is not None:
                raise ValueError("传 plan（复用规划）时不能再传 topk/tpe")
            if not isinstance(plan, MoonepPlan):
                raise TypeError(f"plan 须为 MoonepPlan，got {type(plan).__name__}")
            # route_weights_sk 随车 = 前向复用 plan（同路由重分发，权重随车）；
            # 不带权重 = 反向梯度重分发（recv_w 为 None）。
            return self._dispatch_and_return(
                hidden_sh,
                plan,
                do_cpu_sync,
                route_weights_sk,
                forward=route_weights_sk is not None,
            )
        if topk_experts_sk is None or tokens_per_expert is None:
            raise ValueError("前向 dispatch 须传 topk_experts_sk 与 tokens_per_expert")
        if tokens_per_expert.shape[0] != self._E:
            raise ValueError(
                f"tokens_per_expert.shape[0]={tokens_per_expert.shape[0]} "
                f"!= 构造期 E={self._E}"
            )
        if tuple(topk_experts_sk.shape) != (self._S, self._K):
            raise ValueError(
                f"topk_experts_sk.shape={tuple(topk_experts_sk.shape)} "
                f"!= 构造期 (S, K)=({self._S}, {self._K})"
            )
        if route_weights_sk is not None and tuple(route_weights_sk.shape) != (
            self._S,
            self._K,
        ):
            raise ValueError(
                f"route_weights_sk.shape={tuple(route_weights_sk.shape)} "
                f"!= 构造期 (S, K)=({self._S}, {self._K})"
            )
        plan = moonep_plan.moonep_plan(self._ctx, tokens_per_expert, topk_experts_sk)
        return self._dispatch_and_return(
            hidden_sh, plan, do_cpu_sync, route_weights_sk, forward=True
        )

    def _dispatch_and_return(self, x, plan, do_cpu_sync, topk_weights, forward):
        self._ensure_eb()
        dispatch_kwargs = {
            "topk_weights": (
                topk_weights.float() if topk_weights is not None else None
            ),
            "do_cpu_sync": do_cpu_sync,
        }
        dispatch_kwargs.update(
            {
                "topk_idx": plan.vtopk.view(self._S, self._K),
                "num_experts": 2 * self._E,
                "num_max_tokens_per_rank": self._nmtr,
            }
        )
        recv_x, _meta, recv_w, handle = self._eb.dispatch(x, **dispatch_kwargs)
        if forward:
            self._handle_fwd = handle
            self._handle_fwd_plan = plan
        else:
            self._handle_bwd = handle
            self._handle_bwd_plan = plan
        return recv_x, recv_w, plan.cu_seqlens, plan

    def _dispatch_cached(
        self,
        hidden_sh: torch.Tensor,
        *,
        plan: MoonepPlan,
        handle,
        route_weights_sk: torch.Tensor | None = None,
        do_cpu_sync: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, MoonepPlan]:
        """Replay a caller-managed ElasticBuffer dispatch handle."""
        self._require_alive()
        self._ensure_eb()
        if hidden_sh.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"hidden_sh dtype 须为 bfloat16/float16，got {hidden_sh.dtype}"
            )
        if not hidden_sh.is_contiguous():
            raise ValueError("hidden_sh 须为连续张量")
        if tuple(hidden_sh.shape) != (self._S, self._H):
            raise ValueError(
                f"hidden_sh.shape={tuple(hidden_sh.shape)} 须为 (S, H)="
                f"({self._S}, {self._H})"
            )
        if not isinstance(plan, MoonepPlan):
            raise TypeError(f"plan 须为 MoonepPlan，got {type(plan).__name__}")
        if route_weights_sk is not None and tuple(route_weights_sk.shape) != (
            self._S,
            self._K,
        ):
            raise ValueError(
                f"route_weights_sk.shape={tuple(route_weights_sk.shape)} "
                f"!= 构造期 (S, K)=({self._S}, {self._K})"
            )
        recv_x, _meta, recv_w, _handle = self._eb.dispatch(
            hidden_sh,
            handle=handle,
            topk_weights=(
                route_weights_sk.float() if route_weights_sk is not None else None
            ),
            do_cpu_sync=do_cpu_sync,
        )
        return recv_x, recv_w, plan.cu_seqlens, plan

    def prefetch_weight(
        self,
        *,
        plan: MoonepPlan,
        local_gate_weight: torch.Tensor | None = None,
        local_up_weight: torch.Tensor | None = None,
        local_down_weight: torch.Tensor | None = None,
        local_gate_up_weight: torch.Tensor | None = None,
        local_gate_scales: torch.Tensor | None = None,
        local_up_scales: torch.Tensor | None = None,
        local_down_scales: torch.Tensor | None = None,
        local_gate_up_scales: torch.Tensor | None = None,
        async_finish: bool = False,
        slot: int = 0,
    ):
        """推送 home 权重到各 rank 预取槽。

        返回 ``(views, scales)``：3 组形态 views 为 ``(gate, up, down)``，
        W13 交织形态为 ``(gate_up, down)``；scales 与权重逐组对应，未量化时
        为空元组。``async_finish=True`` 时返回 ``((views, scales), event)``。
        """
        ctx = self._require_alive()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(
                f"prefetch_weight 需要 MoonepPlan，got {type(plan).__name__}"
            )
        if local_gate_up_weight is None:
            if any(
                x is None
                for x in (local_gate_weight, local_up_weight, local_down_weight)
            ):
                raise ValueError(
                    "prefetch_weight: 3 组形态须同时传 gate/up/down；"
                    "W13 形态传 local_gate_up_weight + local_down_weight"
                )
            if local_gate_up_scales is not None:
                raise ValueError("prefetch_weight: W13 scale 不能用于 3 组形态")
            weights = (local_gate_weight, local_up_weight, local_down_weight)
            scale_args = (local_gate_scales, local_up_scales, local_down_scales)
        else:
            if local_gate_weight is not None or local_up_weight is not None:
                raise ValueError("prefetch_weight: W13 形态不接受 gate/up 独立参数")
            if local_gate_scales is not None or local_up_scales is not None:
                raise ValueError(
                    "prefetch_weight: W13 形态不接受 gate/up 独立 scale，"
                    "请传 local_gate_up_scales"
                )
            if local_down_weight is None:
                raise ValueError("prefetch_weight: W13 形态须同时传 local_down_weight")
            weights = (local_gate_up_weight, local_down_weight)
            scale_args = (local_gate_up_scales, local_down_scales)
        provided = [s is not None for s in scale_args]
        if any(provided) and not all(provided):
            raise ValueError("prefetch_weight: scales 须整组提供或全部省略")
        scales = tuple(scale_args) if all(provided) else None

        result = prefetch_weight.prefetch_weight(
            ctx,
            plan.experts_to_copy,
            weights,
            self._epn,
            local_scales=scales,
            slot=int(slot),
            slots=self._slots,
            async_finish=async_finish,
        )
        if async_finish:
            (views, scale_views), event = result
        else:
            views, scale_views = result
        out = (
            (views[0], views[1], views[2]) if len(views) == 3 else (views[0], views[1])
        )
        if async_finish:
            return (out, scale_views), event
        return out, scale_views

    def combine(
        self,
        *,
        plan: MoonepPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, object]:
        """专家输出回收。

        传 ``route_weights_nvs`` 为前向加权；不传时跟随最近一次 dispatch
        的方向（支持前向不传 weights / 推理 combine 不传 weights 的形态）。
        """
        self._require_alive()
        self._ensure_eb()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(f"combine 需要 MoonepPlan，got {type(plan).__name__}")
        if not isinstance(hidden_nvsh, torch.Tensor):
            raise TypeError(
                f"hidden_nvsh 须为 torch.Tensor，got {type(hidden_nvsh).__name__}"
            )
        if route_weights_nvs is not None and not isinstance(
            route_weights_nvs, torch.Tensor
        ):
            raise TypeError(
                "route_weights_nvs 须为 torch.Tensor 或 None，"
                f"got {type(route_weights_nvs).__name__}"
            )
        if hidden_nvsh.dim() != 2 or hidden_nvsh.shape[1] != self._H:
            raise ValueError(
                f"hidden_nvsh.shape={tuple(hidden_nvsh.shape)} 须为 [cap, H={self._H}]"
            )
        if hidden_nvsh.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"hidden_nvsh dtype 须为 bfloat16/float16，got {hidden_nvsh.dtype}"
            )
        if not hidden_nvsh.is_contiguous():
            raise ValueError("hidden_nvsh 须为连续张量")
        if route_weights_nvs is not None:
            if (
                route_weights_nvs.dim() != 1
                or route_weights_nvs.dtype != torch.float32
                or not route_weights_nvs.is_contiguous()
            ):
                raise ValueError(
                    "route_weights_nvs 须为连续 1D float32，且行数等于对应 dispatch 的 recv 行数"
                )
            if route_weights_nvs.numel() != hidden_nvsh.shape[0]:
                raise ValueError(
                    f"route_weights_nvs.numel()={route_weights_nvs.numel()} "
                    f"!= hidden_nvsh.shape[0]={hidden_nvsh.shape[0]}"
                )
        combine_input = hidden_nvsh
        if route_weights_nvs is not None and self._rank_size >= 16:
            # 大组 combine 先把路由权重乘进去；gathered 视图仍保留原始权重。
            combine_input = (hidden_nvsh.float() * route_weights_nvs.unsqueeze(1)).to(
                hidden_nvsh.dtype
            )
        have_fwd = self._handle_fwd is not None
        have_bwd = self._handle_bwd is not None
        if route_weights_nvs is not None:
            fwd = True
        elif have_fwd and have_bwd:
            raise RuntimeError(
                "combine 方向歧义：前向/反向都有未消费的 dispatch handle，"
                "请先 combine 其中一个方向（或传 route_weights_nvs 指明前向）"
            )
        elif have_fwd:
            fwd = True
        elif have_bwd:
            fwd = False
        else:
            raise RuntimeError("combine 前需先调用一次 dispatch（前向或反向）")
        handle = self._handle_fwd if fwd else self._handle_bwd
        bound_plan = self._handle_fwd_plan if fwd else self._handle_bwd_plan
        if handle is None:
            raise RuntimeError(
                "combine 找不到对应 dispatch 的 handle——请先调用同方向的 "
                "dispatch（单方向单飞行）"
            )
        if plan is not bound_plan:
            raise RuntimeError(
                "combine 的 plan 与最近一次同方向 dispatch 不匹配"
                "（同方向 dispatch→combine 之间不能插另一 plan）"
            )
        out, out_w = self._eb.combine(
            combine_input, handle, topk_weights=route_weights_nvs
        )
        # 用后置空：重复 combine / 旧 handle 复用立即报错
        if fwd:
            self._handle_fwd = None
            self._handle_fwd_plan = None
        else:
            self._handle_bwd = None
            self._handle_bwd_plan = None
        return out, out_w, None

    def _combine_cached(
        self,
        *,
        plan: MoonepPlan,
        hidden_nvsh: torch.Tensor,
        route_weights_nvs: torch.Tensor | None,
        handle,
    ) -> tuple[torch.Tensor, torch.Tensor | None, object]:
        """Combine with a caller-managed cached dispatch handle."""
        self._require_alive()
        self._ensure_eb()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(f"combine 需要 MoonepPlan，got {type(plan).__name__}")
        if not isinstance(hidden_nvsh, torch.Tensor):
            raise TypeError(
                f"hidden_nvsh 须为 torch.Tensor，got {type(hidden_nvsh).__name__}"
            )
        if route_weights_nvs is not None and not isinstance(
            route_weights_nvs, torch.Tensor
        ):
            raise TypeError(
                "route_weights_nvs 须为 torch.Tensor 或 None，"
                f"got {type(route_weights_nvs).__name__}"
            )
        if hidden_nvsh.dim() != 2 or hidden_nvsh.shape[1] != self._H:
            raise ValueError(
                f"hidden_nvsh.shape={tuple(hidden_nvsh.shape)} 须为 [cap, H={self._H}]"
            )
        if hidden_nvsh.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                f"hidden_nvsh dtype 须为 bfloat16/float16，got {hidden_nvsh.dtype}"
            )
        if not hidden_nvsh.is_contiguous():
            raise ValueError("hidden_nvsh 须为连续张量")
        if route_weights_nvs is not None:
            if (
                route_weights_nvs.dim() != 1
                or route_weights_nvs.dtype != torch.float32
                or not route_weights_nvs.is_contiguous()
            ):
                raise ValueError(
                    "route_weights_nvs 须为连续 1D float32，且行数等于对应 dispatch 的 recv 行数"
                )
            if route_weights_nvs.numel() != hidden_nvsh.shape[0]:
                raise ValueError(
                    f"route_weights_nvs.numel()={route_weights_nvs.numel()} "
                    f"!= hidden_nvsh.shape[0]={hidden_nvsh.shape[0]}"
                )
        combine_input = hidden_nvsh
        if route_weights_nvs is not None and self._rank_size >= 16:
            combine_input = (hidden_nvsh.float() * route_weights_nvs.unsqueeze(1)).to(
                hidden_nvsh.dtype
            )
        out, out_w = self._eb.combine(
            combine_input, handle, topk_weights=route_weights_nvs
        )
        return out, out_w, None

    def reduce_grad(
        self,
        *,
        plan: MoonepPlan,
        local_gate_grad: torch.Tensor | None = None,
        local_up_grad: torch.Tensor | None = None,
        local_down_grad: torch.Tensor | None = None,
        local_gate_up_grad: torch.Tensor | None = None,
        async_finish: bool = False,
    ):
        """副本梯度归并回 home。

        3 组形态返回 ``(gate, up, down)``；W13 交织返回 ``(gate_up, down)``；
        ``async_finish=True`` 时再追加发起流事件。
        """
        ctx = self._require_alive()
        if not isinstance(plan, MoonepPlan):
            raise TypeError(f"reduce_grad 需要 MoonepPlan，got {type(plan).__name__}")
        if local_gate_up_grad is None:
            if any(
                x is None for x in (local_gate_grad, local_up_grad, local_down_grad)
            ):
                raise ValueError(
                    "reduce_grad: 3 组形态须同时传 gate/up/down；"
                    "W13 形态传 local_gate_up_grad + local_down_grad"
                )
            grads = (local_gate_grad, local_up_grad, local_down_grad)
        else:
            if local_gate_grad is not None or local_up_grad is not None:
                raise ValueError("reduce_grad: W13 形态不接受 gate/up 独立参数")
            if local_down_grad is None:
                raise ValueError("reduce_grad: W13 形态须同时传 local_down_grad")
            grads = (local_gate_up_grad, local_down_grad)

        result = reduce_grad.reduce_grad(
            ctx,
            plan.experts_to_copy,
            grads,
            self._epn,
            slots=self._slots,
            async_finish=async_finish,
        )
        if async_finish:
            home_views, event = result
        else:
            home_views = result
        out = (
            (home_views[0], home_views[1], home_views[2])
            if len(home_views) == 3
            else (home_views[0], home_views[1])
        )
        return (out, event) if async_finish else out

    def barrier(self) -> None:
        """workspace flag 自旋 barrier：返回即全组此前流上工作全部完成。"""
        ctx = self._require_alive()
        sync_transport_stream(ctx)
        sym_barrier(ctx)

    # --------------------------------------------------------------------------
    # 只读 property
    # --------------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """本 rank 在通信组内的 id。"""
        return self._rank

    @property
    def rank_size(self) -> int:
        """通信组 rank 数。"""
        return self._rank_size

    @property
    def epn(self) -> int:
        """每 rank 本地专家数（E // R）。"""
        return self._epn

    @property
    def Hp(self) -> int:
        """专家 FFN 中间维（W = H × Hp）。"""
        return self._Hp

    @property
    def weight_dtype(self) -> torch.dtype:
        """权重 dtype 定型值。"""
        return self._weight_dtype

    @property
    def layout_groups(self) -> int | None:
        """运行期组数（首次 prefetch/reduce 后定型）。"""
        ctx = self._require_alive()
        return ctx.layout_groups

    @property
    def slots(self) -> int:
        """多槽窗口池层数。"""
        return self._slots

    @property
    def grad_dtype(self) -> torch.dtype:
        """梯度 dtype 定型值。"""
        return self._grad_dtype

    @property
    def H(self) -> int:
        """hidden size。"""
        return self._H

    @property
    def num_sms(self) -> int | None:
        """预留参数：计划开放为 MoonEP kernel 使用的 AIV 核数；当前恒为 None。"""
        return self._num_sms

    @property
    def token_padding(self) -> int:
        """token padding 参数（当前仅保留，不生效）。"""
        return self._token_padding

    @property
    def heap_required_bytes(self) -> int:
        """构造期布局需求（HEAP_BASE + reduce_region_end + LAYOUT_TAIL）。"""
        return self._heap_need

    @property
    def heap_bytes(self) -> int:
        """workspace 实际字节数（构造期精算分配）。"""
        return self._heap_bytes

    @property
    def destroyed(self) -> bool:
        """是否已显式销毁。"""
        return self._destroyed

    # --------------------------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------------------------

    def destroy(self, *, release_group: bool = False, release_eb: bool = False) -> None:
        """显式释放逻辑资源（幂等）。

        通信组由调用方创建与销毁，本方法不会销毁它（release_group 仅为兼容保留）；
        ``release_eb=False`` 不销毁 EB（其 Destroy 有分钟级屏障，缺省跳过），``True`` 才显式回收。
        """
        if self._destroyed:
            return
        self._destroyed = True
        if self._ctx is not None:
            self._ctx.close()
        self._ctx = None
        if release_eb and self._eb is not None:
            self._eb.destroy()

    def __del__(self):
        if getattr(self, "_destroyed", True):
            return
        warnings.warn(
            "Buffer 未 destroy() 即被回收：子组/通信连接与 EB 留待进程退出或 GC"
            " 回收（__del__ 内不调用 dist/comm API；EB 析构的硬退代价见"
            " destroy 的 release_eb 说明）",
            ResourceWarning,
            stacklevel=2,
        )
