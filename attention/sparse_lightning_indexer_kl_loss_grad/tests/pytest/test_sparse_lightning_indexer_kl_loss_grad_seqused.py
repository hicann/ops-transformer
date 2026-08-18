# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Standalone CPU-reference precision test for arch22 seqused_q/seqused_k."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "build.sh").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
TORCH_EXTENSION_PATH = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_PATH.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))


N1 = 8
N2 = 1
HEAD_DIM = 128
TOPK = 512
METADATA_TOTAL_SIZE_INDEX = 1
RTOL = 0.08
ATOL = 0.01


@dataclass(frozen=True)
class SeqUsedCase:
    name: str
    layout: str
    alloc_q: Tuple[int, ...]
    alloc_k: Tuple[int, ...]
    used_q: Optional[Tuple[int, ...]]
    used_k: Optional[Tuple[int, ...]]
    dtype: torch.dtype = torch.float16
    mask_mode: int = 0
    cmp_ratio: int = 1
    cmp_residual_k: Optional[Tuple[int, ...]] = None

    @property
    def batch_size(self) -> int:
        return len(self.alloc_q)

    @property
    def effective_q(self) -> Tuple[int, ...]:
        return self.alloc_q if self.used_q is None else self.used_q

    @property
    def effective_k(self) -> Tuple[int, ...]:
        return self.alloc_k if self.used_k is None else self.used_k

    @property
    def residual_k(self) -> Tuple[int, ...]:
        if self.cmp_residual_k is None:
            return (0,) * self.batch_size
        return self.cmp_residual_k

    def validate(self) -> None:
        if len(self.alloc_k) != self.batch_size:
            raise ValueError("alloc_q and alloc_k must have the same batch size")
        if self.used_q is not None and len(self.used_q) != self.batch_size:
            raise ValueError("used_q must have one element per batch")
        if self.used_k is not None and len(self.used_k) != self.batch_size:
            raise ValueError("used_k must have one element per batch")
        if len(self.residual_k) != self.batch_size:
            raise ValueError("cmp_residual_k must have one element per batch")
        if self.layout == "BSND":
            if len(set(self.alloc_q)) != 1 or len(set(self.alloc_k)) != 1:
                raise ValueError("BSND allocation lengths must be uniform")
        elif self.layout != "TND":
            raise ValueError(f"unsupported layout: {self.layout}")
        for used, allocated in zip(self.effective_q, self.alloc_q):
            if used < 0 or used > allocated:
                raise ValueError(f"used_q {used} is outside [0, {allocated}]")
        for used, allocated in zip(self.effective_k, self.alloc_k):
            if used < 0 or used > allocated:
                raise ValueError(f"used_k {used} is outside [0, {allocated}]")


CASES = (
    SeqUsedCase(
        name="tnd_q_only",
        layout="TND",
        alloc_q=(12, 9),
        alloc_k=(5, 4),
        used_q=(7, 5),
        used_k=None,
    ),
    SeqUsedCase(
        name="tnd_k_only_including_zero",
        layout="TND",
        alloc_q=(8, 7),
        alloc_k=(5, 6),
        used_q=None,
        used_k=(0, 3),
    ),
    SeqUsedCase(
        name="tnd_q_and_k_causal_bf16",
        layout="TND",
        alloc_q=(16, 15),
        alloc_k=(6, 6),
        used_q=(10, 13),
        used_k=(3, 4),
        dtype=torch.bfloat16,
        mask_mode=3,
        cmp_ratio=4,
        cmp_residual_k=(1, 2),
    ),
    SeqUsedCase(
        name="bsnd_q_and_k",
        layout="BSND",
        alloc_q=(10, 10),
        alloc_k=(6, 6),
        used_q=(6, 8),
        used_k=(4, 3),
    ),
    SeqUsedCase(
        name="bsnd_q_and_k_zero",
        layout="BSND",
        alloc_q=(10, 10),
        alloc_k=(6, 6),
        used_q=(1, 0),
        used_k=(0, 2),
    ),
)


def init_npu_device(device_id: int) -> Tuple[torch.device, object]:
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError(
            "torch_npu is required; load the CANN environment and use an "
            "Ascend PyTorch environment before running this script"
        ) from error

    if not torch.npu.is_available():
        raise RuntimeError("NPU device is unavailable")
    torch_npu.npu.set_device(device_id)
    try:
        import cann_ops_transformer  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "cann_ops_transformer is required; install or load the operator "
            "package built from the current branch"
        ) from error
    return torch.device(f"npu:{device_id}"), torch_npu


def cumulative_lengths(lengths: Tuple[int, ...]) -> Tuple[int, ...]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)


def make_cu_seqlens(lengths: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(cumulative_lengths(lengths), dtype=torch.int32, device=device)


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division truncated toward zero, matching the kernel behavior."""
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def valid_k_count(
    case: SeqUsedCase,
    batch_index: int,
    local_q_index: int,
) -> int:
    used_q = case.effective_q[batch_index]
    used_k = case.effective_k[batch_index]
    if case.mask_mode == 0:
        sparse_len = used_k
    elif case.mask_mode == 3:
        pre_compress_k = used_k * case.cmp_ratio + case.residual_k[batch_index]
        sparse_len = trunc_div(
            pre_compress_k - used_q + local_q_index + 1,
            case.cmp_ratio,
        )
    else:
        raise ValueError(f"mask_mode must be 0 or 3, got {case.mask_mode}")
    return max(0, min(sparse_len, used_k, TOPK))


def cpu_reference_one_batch(
    batch: Dict[str, torch.Tensor],
    used_q: int,
    used_k: int,
    *,
    mask_mode: int,
    cmp_ratio: int,
    cmp_residual_k: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute dq, dk, dw and softmax_out for one physical batch.

    All arithmetic is performed on CPU in FP32, except that dS is rounded to
    the q/k input dtype before the dq/dk matmuls. This reproduces the arch22
    kernel's low-precision reluGrad intermediate while keeping the reference
    implementation independent from the NPU operator.
    """
    required = {"q", "k", "w", "sparse_indices", "softmax_l1"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"missing reference inputs: {sorted(missing)}")

    q = batch["q"].detach().cpu()
    k = batch["k"].detach().cpu()
    w = batch["w"].detach().cpu()
    sparse_indices = batch["sparse_indices"].detach().cpu()
    softmax_l1 = batch["softmax_l1"].detach().cpu()

    if q.ndim != 3 or k.ndim != 3 or w.ndim != 2:
        raise ValueError("q/k/w must have shapes [S,N,D], [S,N,D], [S,N]")
    if sparse_indices.ndim != 3 or softmax_l1.ndim != 3:
        raise ValueError("sparse_indices/softmax_l1 must have shape [S,N2,K]")
    if k.shape[1] != 1 or sparse_indices.shape[1] != 1 or softmax_l1.shape[1] != 1:
        raise ValueError("the arch22 CPU reference currently supports N2 == 1")
    if q.shape[1] != w.shape[1] or q.shape[2] != k.shape[2]:
        raise ValueError("q/k/w dimensions are inconsistent")
    if sparse_indices.shape != softmax_l1.shape:
        raise ValueError("sparse_indices and softmax_l1 shapes must match")
    if not 0 <= used_q <= q.shape[0]:
        raise ValueError(f"used_q {used_q} is outside [0, {q.shape[0]}]")
    if not 0 <= used_k <= k.shape[0]:
        raise ValueError(f"used_k {used_k} is outside [0, {k.shape[0]}]")

    alloc_q, n1, head_dim = q.shape
    alloc_k, n2, _ = k.shape
    topk = sparse_indices.shape[-1]
    outputs = {
        "dq": torch.zeros((alloc_q, n1, head_dim), dtype=torch.float32),
        "dk": torch.zeros((alloc_k, n2, head_dim), dtype=torch.float32),
        "dw": torch.zeros((alloc_q, n1), dtype=torch.float32),
        "softmax_out": torch.zeros((alloc_q, n2, topk), dtype=torch.float32),
    }

    if used_q == 0 or used_k == 0:
        return outputs

    q_fp32 = q[:used_q].float()
    k_fp32 = k[:used_k, 0].float()
    w_fp32 = w[:used_q].float()
    sparse = sparse_indices[:used_q, 0].long()
    target = softmax_l1[:used_q, 0].float()

    for local_q_index in range(used_q):
        if mask_mode == 0:
            real_k = used_k
        elif mask_mode == 3:
            pre_compress_k = used_k * cmp_ratio + cmp_residual_k
            real_k = trunc_div(
                pre_compress_k - used_q + local_q_index + 1,
                cmp_ratio,
            )
        else:
            raise ValueError(f"mask_mode must be 0 or 3, got {mask_mode}")
        real_k = max(0, min(real_k, used_k, topk))
        if real_k == 0:
            continue

        key_ids = sparse[local_q_index, :real_k]
        if torch.any(key_ids < 0) or torch.any(key_ids >= used_k):
            raise ValueError(
                f"sparse index outside [0, {used_k}) in q row {local_q_index}: "
                f"{key_ids.tolist()}"
            )

        selected_k = k_fp32.index_select(0, key_ids)
        score = torch.matmul(q_fp32[local_q_index], selected_k.transpose(0, 1))
        relu_score = torch.relu(score)
        logits = torch.sum(relu_score * w_fp32[local_q_index].unsqueeze(1), dim=0)
        softmax_value = torch.softmax(logits, dim=0)
        target_value = target[local_q_index, :real_k]

        # dI = softmax(I) * sum(P) - P. When P is normalized this reduces
        # to the familiar softmax(I) - P expression.
        d_logits = softmax_value * target_value.sum() - target_value
        outputs["softmax_out"][local_q_index, 0, :real_k] = softmax_value
        outputs["dw"][local_q_index] = torch.matmul(relu_score, d_logits)

        d_score = (
            w_fp32[local_q_index].unsqueeze(1)
            * d_logits.unsqueeze(0)
            * (score > 0).float()
        )
        d_score_kernel_precision = d_score.to(q.dtype).float()
        outputs["dq"][local_q_index] = torch.matmul(
            d_score_kernel_precision,
            selected_k,
        )
        dk_gather = torch.matmul(
            d_score_kernel_precision.transpose(0, 1),
            q_fp32[local_q_index],
        )
        outputs["dk"][:used_k, 0].index_add_(0, key_ids, dk_gather)

    return outputs


def make_batch_inputs(
    case: SeqUsedCase,
) -> List[Dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20260723)
    batches = []
    for batch_index, (alloc_q, alloc_k) in enumerate(zip(case.alloc_q, case.alloc_k)):
        q = (
            torch.rand(
                alloc_q,
                N1,
                HEAD_DIM,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(0.25)
            .to(case.dtype)
        )
        k = (
            torch.rand(
                alloc_k,
                N2,
                HEAD_DIM,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(0.25)
            .to(case.dtype)
        )
        w = (
            torch.rand(
                alloc_q,
                N1,
                generator=generator,
                dtype=torch.float32,
            )
            .mul_(0.02)
            .add_(0.01)
        )

        # Unused Q rows deliberately contain nonzero inputs. Zero output there
        # must therefore come from seqused handling, not from the test data.
        sparse_indices = torch.zeros(alloc_q, N2, TOPK, dtype=torch.int32)
        softmax_l1 = torch.rand(
            alloc_q,
            N2,
            TOPK,
            generator=generator,
            dtype=torch.float32,
        )
        softmax_l1 /= softmax_l1.sum(dim=-1, keepdim=True)

        used_q = case.effective_q[batch_index]
        for local_q_index in range(used_q):
            count = valid_k_count(case, batch_index, local_q_index)
            sparse_indices[local_q_index].fill_(-1)
            softmax_l1[local_q_index].zero_()
            if count == 0:
                continue
            sparse_indices[local_q_index, 0, :count] = torch.arange(
                count,
                dtype=torch.int32,
            )
            values = torch.rand(count, generator=generator, dtype=torch.float32)
            softmax_l1[local_q_index, 0, :count] = values / values.sum()

        batches.append(
            {
                "q": q.contiguous(),
                "k": k.contiguous(),
                "w": w.contiguous(),
                "sparse_indices": sparse_indices.contiguous(),
                "softmax_l1": softmax_l1.contiguous(),
            }
        )
    return batches


def pack_batches(
    batches: List[Dict[str, torch.Tensor]],
    layout: str,
) -> Dict[str, torch.Tensor]:
    packed = {}
    for name in batches[0]:
        tensors = [batch[name] for batch in batches]
        packed[name] = (
            torch.cat(tensors, dim=0).contiguous()
            if layout == "TND"
            else torch.stack(tensors, dim=0).contiguous()
        )
    return packed


def unpack_output(
    tensor: torch.Tensor,
    layout: str,
    alloc_lengths: Tuple[int, ...],
) -> List[torch.Tensor]:
    if layout == "BSND":
        return [tensor[index] for index in range(tensor.shape[0])]

    offsets = cumulative_lengths(alloc_lengths)
    return [
        tensor[offsets[index] : offsets[index + 1]]
        for index in range(len(alloc_lengths))
    ]


def format_tensor_info(tensor: torch.Tensor, *, show_values: bool = False) -> str:
    info = (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, contiguous={tensor.is_contiguous()}"
    )
    if show_values:
        info += f", values={tensor.detach().cpu().tolist()}"
    return info


def print_case_info(case: SeqUsedCase) -> None:
    print("[ CASE INFO ]", flush=True)
    print(
        f"  name={case.name}, layout={case.layout}, dtype={case.dtype}, "
        f"batch_size={case.batch_size}",
        flush=True,
    )
    print(
        f"  N1={N1}, N2={N2}, head_dim={HEAD_DIM}, topk={TOPK}, "
        f"mask_mode={case.mask_mode}, cmp_ratio={case.cmp_ratio}",
        flush=True,
    )
    print(
        f"  alloc_q={case.alloc_q}, alloc_k={case.alloc_k}, "
        f"used_q={case.used_q}, used_k={case.used_k}, "
        f"effective_q={case.effective_q}, effective_k={case.effective_k}",
        flush=True,
    )
    print(f"  cmp_residual_k={case.cmp_residual_k}", flush=True)


def print_torch_inputs(
    inputs: Dict[str, torch.Tensor],
    optional_inputs: Dict[str, torch.Tensor],
    metadata: torch.Tensor,
) -> None:
    print("[ TORCH INPUTS ]", flush=True)
    for name, tensor in inputs.items():
        print(f"  {name}: {format_tensor_info(tensor)}", flush=True)
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
    ):
        tensor = optional_inputs.get(name)
        if tensor is None:
            print(f"  {name}: None", flush=True)
        else:
            print(
                f"  {name}: {format_tensor_info(tensor, show_values=True)}",
                flush=True,
            )
    print(f"  metadata: {format_tensor_info(metadata)}", flush=True)


def run_operator(
    inputs: Dict[str, torch.Tensor],
    layout: str,
    alloc_q: Tuple[int, ...],
    alloc_k: Tuple[int, ...],
    device: torch.device,
    *,
    used_q: Optional[Tuple[int, ...]] = None,
    used_k: Optional[Tuple[int, ...]] = None,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    cmp_residual_k: Optional[Tuple[int, ...]] = None,
    torch_npu_module: object,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    batch_size = len(alloc_q)
    optional_inputs = {}
    if layout == "TND":
        optional_inputs["cu_seqlens_q"] = make_cu_seqlens(alloc_q, device)
        optional_inputs["cu_seqlens_k"] = make_cu_seqlens(alloc_k, device)
    if used_q is not None:
        optional_inputs["seqused_q"] = torch.tensor(
            used_q,
            dtype=torch.int32,
            device=device,
        )
    if used_k is not None:
        optional_inputs["seqused_k"] = torch.tensor(
            used_k,
            dtype=torch.int32,
            device=device,
        )
    if cmp_residual_k is not None:
        optional_inputs["cmp_residual_k"] = torch.tensor(
            cmp_residual_k,
            dtype=torch.int32,
            device=device,
        )

    metadata = (
        torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            N1,
            N2,
            HEAD_DIM,
            batch_size=batch_size,
            max_seqlen_q=max(alloc_q),
            max_seqlen_k=max(alloc_k),
            topk=TOPK,
            layout_q=layout,
            layout_k=layout,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            **optional_inputs,
        )
    )

    npu_inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    print_torch_inputs(npu_inputs, optional_inputs, metadata)
    outputs = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
        npu_inputs["q"],
        npu_inputs["k"],
        npu_inputs["w"],
        npu_inputs["sparse_indices"],
        npu_inputs["softmax_l1"],
        metadata=metadata,
        layout_q=layout,
        layout_k=layout,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        **optional_inputs,
    )
    torch_npu_module.npu.synchronize()
    return metadata.cpu(), tuple(output.cpu() for output in outputs)


def assert_output_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    if actual_fp32.numel() == 0:
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        mismatch_count = 0
    else:
        abs_diff = torch.abs(actual_fp32 - expected_fp32)
        max_abs_diff = float(abs_diff.max().item())
        nonzero_expected = expected_fp32 != 0
        max_rel_diff = (
            float(
                (abs_diff[nonzero_expected] / expected_fp32[nonzero_expected].abs())
                .max()
                .item()
            )
            if bool(nonzero_expected.any())
            else 0.0
        )
        mismatch_count = int(
            torch.count_nonzero(
                ~torch.isclose(
                    actual_fp32,
                    expected_fp32,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=True,
                )
            ).item()
        )

    status = "PASS" if mismatch_count == 0 else "FAIL"
    print(
        f"[ COMPARE  ] {name}: {status}, shape={tuple(actual.shape)}, "
        f"max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}, "
        f"mismatch={mismatch_count}/{actual.numel()}, rtol={RTOL}, atol={ATOL}",
        flush=True,
    )
    torch.testing.assert_close(
        actual_fp32,
        expected_fp32,
        rtol=RTOL,
        atol=ATOL,
        msg=lambda message: f"{name} mismatch: {message}",
    )


def assert_exact_zero(name: str, tensor: torch.Tensor) -> None:
    nonzero = int(torch.count_nonzero(tensor).item())
    status = "PASS" if nonzero == 0 else "FAIL"
    print(
        f"[ ZERO     ] {name}: {status}, shape={tuple(tensor.shape)}, "
        f"nonzero={nonzero}/{tensor.numel()}",
        flush=True,
    )
    assert nonzero == 0, f"{name} contains {nonzero} nonzero elements"


def run_case(
    case: SeqUsedCase,
    npu_device: torch.device,
    torch_npu_module: object,
) -> None:
    case.validate()

    batches = make_batch_inputs(case)
    packed_inputs = pack_batches(batches, case.layout)
    print_case_info(case)
    metadata, actual_outputs = run_operator(
        packed_inputs,
        case.layout,
        case.alloc_q,
        case.alloc_k,
        npu_device,
        used_q=case.used_q,
        used_k=case.used_k,
        mask_mode=case.mask_mode,
        cmp_ratio=case.cmp_ratio,
        cmp_residual_k=case.cmp_residual_k,
        torch_npu_module=torch_npu_module,
    )

    expected_total_tasks = (
        sum(case.used_q) if case.used_q is not None else sum(case.alloc_q)
    )
    actual_total_tasks = int(metadata[METADATA_TOTAL_SIZE_INDEX])
    metadata_status = "PASS" if actual_total_tasks == expected_total_tasks else "FAIL"
    print(
        f"[ METADATA ] {metadata_status}, {format_tensor_info(metadata)}, "
        f"total_size={actual_total_tasks}, expected_total_size={expected_total_tasks}",
        flush=True,
    )
    assert actual_total_tasks == expected_total_tasks

    output_names = ("dq", "dk", "dw", "softmax_out")
    print("[ TORCH OUTPUTS ]", flush=True)
    for name, tensor in zip(output_names, actual_outputs):
        print(f"  {name}: {format_tensor_info(tensor)}", flush=True)

    dq_batches = unpack_output(actual_outputs[0], case.layout, case.alloc_q)
    dk_batches = unpack_output(actual_outputs[1], case.layout, case.alloc_k)
    dw_batches = unpack_output(actual_outputs[2], case.layout, case.alloc_q)
    softmax_batches = unpack_output(actual_outputs[3], case.layout, case.alloc_q)

    for batch_index, (used_q, used_k) in enumerate(
        zip(case.effective_q, case.effective_k)
    ):
        expected = cpu_reference_one_batch(
            batches[batch_index],
            used_q,
            used_k,
            mask_mode=case.mask_mode,
            cmp_ratio=case.cmp_ratio,
            cmp_residual_k=case.residual_k[batch_index],
        )
        assert_output_close(
            f"batch {batch_index} dq",
            dq_batches[batch_index][:used_q],
            expected["dq"][:used_q],
        )
        assert_output_close(
            f"batch {batch_index} dk",
            dk_batches[batch_index][:used_k],
            expected["dk"][:used_k],
        )
        assert_output_close(
            f"batch {batch_index} dw",
            dw_batches[batch_index][:used_q],
            expected["dw"][:used_q],
        )
        assert_output_close(
            f"batch {batch_index} softmax_out",
            softmax_batches[batch_index][:used_q],
            expected["softmax_out"][:used_q],
        )

        assert_exact_zero(
            f"batch {batch_index} dq unused Q tail",
            dq_batches[batch_index][used_q:],
        )
        assert_exact_zero(
            f"batch {batch_index} dw unused Q tail",
            dw_batches[batch_index][used_q:],
        )
        assert_exact_zero(
            f"batch {batch_index} softmax_out unused Q tail",
            softmax_batches[batch_index][used_q:],
        )
        assert_exact_zero(
            f"batch {batch_index} dk unused K tail",
            dk_batches[batch_index][used_k:],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SparseLightningIndexerKLLossGrad seqused_q/seqused_k "
            "precision cases against an independent CPU reference."
        )
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Ascend NPU device ID (default: 0)",
    )
    parser.add_argument(
        "--case",
        choices=("all", *(case.name for case in CASES)),
        default="all",
        help="case to run (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    npu_device, torch_npu_module = init_npu_device(args.device_id)
    selected_cases = (
        CASES
        if args.case == "all"
        else tuple(case for case in CASES if case.name == args.case)
    )

    for case in selected_cases:
        print(f"[ RUN      ] {case.name}", flush=True)
        run_case(case, npu_device, torch_npu_module)
        print(f"[       OK ] {case.name}", flush=True)
    print(f"[  PASSED  ] {len(selected_cases)} case(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
