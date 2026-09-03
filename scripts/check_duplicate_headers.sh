#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 拦截暂存区中「新增(A)/重命名(R)」的头文件：若其 basename 在仓内已有同名文件则失败。
#
# 收集范围：git diff --cached --diff-filter=AR 的 *.h / *.hpp
# 判定规则：find <仓库根> -name <basename>，非白名单命中数 > 1 则拦截
# 退出码  ：0 = 通过；1 = 发现重名
#
# 白名单：仓库相对路径前缀（不要加 './'）。命中白名单的暂存文件跳过检查；
# find 会直接 prune 白名单目录（及 .git），不计入重名。
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# 预留白名单路径前缀（相对仓库根）。示例：
WHITELIST_PATH_PREFIXES=(
  "build/"
  "build_out/"
  # "third_party/"
  # "3rdparty/"
  # "experimental/"
)

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "[dup-header] Not inside a git repository; skipping."
  exit 0
fi
cd "$ROOT"

is_whitelisted_path() {
  local rel="${1#./}"
  local prefix
  if [[ ${#WHITELIST_PATH_PREFIXES[@]} -eq 0 ]]; then
    return 1
  fi
  for prefix in "${WHITELIST_PATH_PREFIXES[@]}"; do
    [[ -z "$prefix" ]] && continue
    prefix="${prefix#./}"
    if [[ "$rel" == "$prefix"* ]]; then
      return 0
    fi
  done
  return 1
}

# 构造 find 的 prune 参数：跳过 .git 与白名单目录
find_prune_paths() {
  local prefix p
  printf '%s\n' "-path" "*/.git" "-o" "-path" "*/.git/*"
  if [[ ${#WHITELIST_PATH_PREFIXES[@]} -eq 0 ]]; then
    return 0
  fi
  for prefix in "${WHITELIST_PATH_PREFIXES[@]}"; do
    [[ -z "$prefix" ]] && continue
    prefix="${prefix#./}"
    prefix="${prefix%/}"
    p="${ROOT}/${prefix}"
    printf '%s\n' "-o" "-path" "$p" "-o" "-path" "${p}/*"
  done
}

basename_exists() {
  local name="$1"
  local hit
  local -a _prune=()
  # 与查重同一套 prune，避免仅因白名单路径下有同名就换推荐名
  mapfile -t _prune < <(find_prune_paths)
  hit="$(
    find "$ROOT" \( "${_prune[@]}" \) -prune -o \
      -type f -name "$name" -print -quit 2>/dev/null
  )"
  [[ -n "$hit" ]]
}

# 推荐唯一 basename：按路径段由浅到深加前缀
# 例：attention/common/op_kernel/buffer.h -> attention_buffer.h，
# 再试 attention_common_buffer.h、attention_common_op_kernel_buffer.h 等
suggest_unique_basename() {
  local rel="$1"
  local base stem ext dir IFS
  local -a parts=()
  local candidate prefix i

  base="$(basename "$rel")"
  if [[ "$base" == *.hpp ]]; then
    stem="${base%.hpp}"
    ext=".hpp"
  else
    stem="${base%.h}"
    ext=".h"
  fi

  dir="$(dirname "$rel")"
  if [[ "$dir" != "." ]]; then
    IFS='/' read -r -a parts <<< "$dir"
  fi

  # 从仓库相对路径顶层目录起逐步加长前缀
  # 例：attention/common/op_kernel/buffer.h -> attention_buffer.h，
  # 再试 attention_common_buffer.h、attention_common_op_kernel_buffer.h
  prefix=""
  for ((i = 0; i < ${#parts[@]}; i++)); do
    if [[ -z "$prefix" ]]; then
      prefix="${parts[i]}"
    else
      prefix="${prefix}_${parts[i]}"
    fi
    candidate="${prefix}_${stem}${ext}"
    if ! basename_exists "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  # 回退：整段路径下划线拼接，仍冲突则加 _vN 后缀
  if [[ ${#parts[@]} -gt 0 ]]; then
    candidate="$(IFS=_; echo "${parts[*]}")_${stem}${ext}"
  else
    candidate="${stem}_unique${ext}"
  fi
  if ! basename_exists "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  i=2
  while basename_exists "${stem}_v${i}${ext}"; do
    i=$((i + 1))
  done
  printf '%s\n' "${stem}_v${i}${ext}"
}

# 仅暂存区：新增(A) / 重命名(R) 的头文件（重命名取新路径）
mapfile -t CANDIDATES < <(
  git diff --cached --diff-filter=AR --name-only --relative -- '*.h' '*.hpp' \
    | sed '/^$/d' \
    | sort -u
)

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[dup-header] No staged Added/Renamed .h/.hpp files; skipping."
  exit 0
fi

FAIL=0
echo "[dup-header] Checking ${#CANDIDATES[@]} staged Added/Renamed header(s) for duplicate basenames..."
if [[ ${#WHITELIST_PATH_PREFIXES[@]} -gt 0 ]]; then
  echo "[dup-header] Whitelist prefixes: ${WHITELIST_PATH_PREFIXES[*]}"
fi

for rel in "${CANDIDATES[@]}"; do
  rel="${rel#./}"
  if is_whitelisted_path "$rel"; then
    echo "[dup-header] SKIP (whitelisted): ${rel}"
    continue
  fi

  base="$(basename "$rel")"
  # find 直接 prune .git 与白名单目录
  mapfile -t _prune < <(find_prune_paths)
  mapfile -t HITS < <(
    find "$ROOT" \( "${_prune[@]}" \) -prune -o \
      -type f -name "$base" -print 2>/dev/null \
      | sed "s|^${ROOT}/||" \
      | sed 's|^\./||' \
      | sort -u
  )

  count="${#HITS[@]}"
  if [[ "$count" -gt 1 ]]; then
    suggested="$(suggest_unique_basename "$rel")"
    parent="$(dirname "$rel")"
    if [[ "$parent" == "." ]]; then
      new_path="${suggested}"
    else
      new_path="${parent}/${suggested}"
    fi
    echo "[dup-header] FAIL: duplicate header basename '${base}'"
    echo "[dup-header] Reason: staged file '${rel}' introduces or keeps basename '${base}',"
    echo "[dup-header]         but find '${base}' returned ${count} non-whitelist paths:"
    printf '[dup-header]   - %s\n' "${HITS[@]}"
    echo "[dup-header] How to fix:"
    echo "[dup-header]   Naming guidance: prefer an operator/component name as prefix"
    echo "[dup-header]     (e.g. moe_init_routing_buffer.h), or an architecture as suffix"
    echo "[dup-header]     (e.g. buffer_arch35.h / buffer_arch22.h)."
    echo "[dup-header]   Recommended name: ${suggested}"
    echo "[dup-header]   Rename: ${rel}  =>  ${new_path}"
    echo "[dup-header]   Steps:"
    echo "[dup-header]     1. git mv '${rel}' '${new_path}'"
    echo "[dup-header]     2. Update #include and build references to use '${suggested}'"
    echo "[dup-header]     3. git add -u && bash scripts/check_duplicate_headers.sh"
    FAIL=1
  else
    echo "[dup-header] OK: ${rel}"
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[dup-header] Commit blocked: duplicate header basename(s) are not allowed."
  exit 1
fi

echo "[dup-header] Passed."
exit 0
