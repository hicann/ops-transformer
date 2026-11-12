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
# =====================================================================
# 一键离线三方件下载脚本
#   仓库: ops-transformer    CANN_CMAKE_TAG: master-059
#   适用: Linux 及 Windows(已安装 Git Bash)
#   用法: bash scripts/download_3rd_party.sh [--cann_3rd_lib_path <目录>|--cann_3rd_lib_path=<目录>] [-f] [-h]
# 说明:
#   - 三方件源码包默认存放于 <业务仓根目录>/third_party (可通过 --cann_3rd_lib_path 指定其它目录),
#     所有下载的 patch 统一存放于 <三方件目录>/patch 下;
#   - 每个文件均带 SHA256 校验: 二次调用时先做缓存校验, hash 一致则直接复用, 无需重复下载;
#   - 下载失败或校验失败时保留临时文件、响应头和 curl 日志, 便于排查代理/认证问题;
#   - 本脚本的 URL/SHA256 与本地构建(cann-cmake master-059 third_party/*.cmake)保持一致。
# =====================================================================
set -euo pipefail

REPO="ops-transformer"
CANN_CMAKE_TAG="master-059"
CMAKE_TAR_URL="https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-${CANN_CMAKE_TAG}.tar.gz"

# ---- 定位业务仓根目录(本脚本位于 scripts/ 下) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CANN_3RD_LIB_PATH=""
FORCE=0

usage() {
cat <<EOF
用法: bash $0 [选项]

选项:
  --cann_3rd_lib_path <路径>   指定三方件存放根目录(默认: <业务仓根目录>/third_party)
  --cann_3rd_lib_path=<路径>   同上, 等号形式(与 build.sh 参数风格一致)
  -f, --force                  忽略已存在文件, 强制重新下载; git 仓库重新 checkout 到目标提交
  -h, --help                   显示本帮助

说明:
  - 默认在业务仓根目录下生成 third_party 文件夹, 三方源包放入其中, 所有 patch 放入 third_party/patch;
  - 已存在的同名文件会先做 SHA256 校验: hash 一致则复用(不重复下载), 不一致则下载到临时文件并在校验通过后替换;
  - 校验所需的 SHA256 已写入本脚本, 下载后亦会复核完整性;
  - 离线编译时, 将 <三方件目录> 通过 build.sh 的 --cann_3rd_lib_path=<目录> 传入即可。
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --cann_3rd_lib_path=*)
      CANN_3RD_LIB_PATH="${1#*=}"
      if [ -z "$CANN_3RD_LIB_PATH" ]; then echo "[ERROR] --cann_3rd_lib_path 缺少参数"; usage; exit 1; fi
      shift;;
    --cann_3rd_lib_path)
      CANN_3RD_LIB_PATH="${2:-}"
      if [ -z "$CANN_3RD_LIB_PATH" ]; then echo "[ERROR] --cann_3rd_lib_path 缺少参数"; usage; exit 1; fi
      shift 2;;
    -f|--force) FORCE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] 未知参数: $1"; usage; exit 1;;
  esac
done

if [ -z "$CANN_3RD_LIB_PATH" ]; then
  CANN_3RD_LIB_PATH="$DEFAULT_REPO_ROOT/third_party"
fi

# ---------- 下载工具检测: Windows(Git Bash) 与 Linux 通用 ----------
if command -v curl >/dev/null 2>&1; then
  # 仅在当前 curl 声明支持时启用, 避免 Linux/macOS 上因不识别 Windows 专用参数导致下载直接失败。
  curl_help="$(curl --help all 2>/dev/null || true)"
  # TLS 证书链错误等不可重试错误应立即返回,
  # 避免同一条证书错误被重复输出多次并延长失败等待时间。
  CURL_ARGS=(--show-error --location --retry 3 --connect-timeout 30 --max-time 600)
  if [[ "$curl_help" == *"--fail-with-body"* ]]; then
    CURL_ARGS+=(--fail-with-body)
  else
    CURL_ARGS+=(--fail)
  fi
  if [[ "$curl_help" == *"--ssl-no-revoke"* ]]; then
    CURL_ARGS+=(--ssl-no-revoke)
  fi
  if [[ "$curl_help" == *"--no-progress-meter"* ]]; then
    CURL_ARGS+=(--no-progress-meter)
  fi
  dl() {
    local url="$1" dest="$2" headers="$3" curl_log="$4"
    curl "${CURL_ARGS[@]}" -D "$headers" -o "$dest" "$url" 2>"$curl_log"
  }
elif command -v wget >/dev/null 2>&1; then
  dl() {
    local url="$1" dest="$2" headers="$3" curl_log="$4"
    wget --server-response --tries=3 --timeout=30 -O "$dest" "$url" \
      >"$curl_log" 2>"$headers"
  }
else
  echo "[ERROR] 未找到 curl 或 wget(Windows 请先安装 Git Bash 或 curl)"
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}' || true; }
elif command -v sha256 >/dev/null 2>&1; then
  sha_of() { sha256 "$1" 2>/dev/null | awk '{print $1}' || true; }
else
  echo "[ERROR] 未找到 sha256sum 或 sha256, 无法校验三方件完整性"
  exit 1
fi

# fetch <url> <subdir:.|patch> <expected_sha256>
fetch() {
  local url="$1" sub="$2" exp="$3"
  local fname dest tmp headers curl_log act2 status content_type transfer_error
  fname="$(basename "$url")"
  dest="$CANN_3RD_LIB_PATH/$sub/$fname"
  mkdir -p "$(dirname "$dest")"

  # 缓存复用: 已存在同名文件
  if [ -f "$dest" ] && [ "$FORCE" = "0" ]; then
    if [ -n "$exp" ]; then
      local act; act="$(sha_of "$dest")"
      if [ -n "$act" ] && [ "$act" = "$exp" ]; then
        echo "[REUSE] $fname (SHA256 匹配, 跳过下载)"
        return 0
      else
        echo "[INFO] $fname 已存在但 SHA256 不一致, 将重新下载"
        echo "       期望=$exp 实际=${act:-<空>}"
      fi
    else
      echo "[INFO] $fname 已存在但没有参考 hash, 将重新下载"
    fi
  fi

  echo "==> 下载 $fname  <-  $url"
  tmp="${dest}.tmp.$$.$RANDOM"
  headers="${tmp}.headers"
  curl_log="${tmp}.curl.log"
  if ! dl "$url" "$tmp" "$headers" "$curl_log"; then
    status="$(grep -E '^HTTP/' "$headers" 2>/dev/null | grep -Eiv '^[[:space:]]*HTTP/[^ ]+[[:space:]]+200[[:space:]]+Connection[[:space:]]+Established' | tail -1 || true)"
    content_type="$(grep -i '^Content-Type:' "$headers" 2>/dev/null | tail -1 | tr -d '\r' || true)"
    transfer_error="$({ cat "$curl_log" "$headers" 2>/dev/null || true; } | grep -Ei '(^curl:|^wget:|certificate|ssl|issuer|proxy|authentication|error|failed)' | awk '!seen[$0]++' | tail -5 || true)"
    echo "[ERROR] 下载失败: $fname"
    echo "        URL: $url"
    echo "        ${status:-源站 HTTP 状态未知}"
    [ -n "$content_type" ] && echo "        $content_type"
    [ -n "$transfer_error" ] && echo "$transfer_error" | sed 's/^/        /'
    echo "        curl 日志: $curl_log"
    echo "        下载文件: $tmp"
    return 1
  fi

  # 下载后完整性校验: 计算结果为空同样视为失败。校验失败时保留文件和诊断信息。
  if [ -n "$exp" ]; then
    act2="$(sha_of "$tmp")"
    if [ -z "$act2" ] || [ "$act2" != "$exp" ]; then
      status="$(grep -E '^HTTP/' "$headers" 2>/dev/null | grep -Eiv '^[[:space:]]*HTTP/[^ ]+[[:space:]]+200[[:space:]]+Connection[[:space:]]+Established' | tail -1 || true)"
      content_type="$(grep -i '^Content-Type:' "$headers" 2>/dev/null | tail -1 | tr -d '\r' || true)"
      echo "[ERROR] $fname SHA256 校验失败, 文件已保留"
      echo "        URL: $url"
      echo "        期望=$exp 实际=${act2:-<空>}"
      echo "        ${status:-源站 HTTP 状态未知}"
      [ -n "$content_type" ] && echo "        $content_type"
      echo "        响应头: $headers"
      echo "        下载文件: $tmp"
      echo "        curl 日志: $curl_log"
      return 1
    fi
  fi

  mv -f "$tmp" "$dest"
  rm -f "$headers" "$curl_log"
}

echo "==> ops-transformer | CANN_CMAKE_TAG=${CANN_CMAKE_TAG}"
echo "==> 三方件存放目录: $CANN_3RD_LIB_PATH"
mkdir -p "$CANN_3RD_LIB_PATH/patch"

# ---- cann-cmake 构建框架 ----
echo "==> 下载 cann-cmake ${CANN_CMAKE_TAG}"
fetch "${CMAKE_TAR_URL}" "." "9e59939b5c97c9498293969df22ab2dd40e450700c51f84f117d065880f40d25"

# ---- 三方件源码包(URL 与 cann-cmake master-059 third_party/*.cmake 一致) ----
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.12.0.tar.gz" "." "4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187"
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/googletest/googletest-1.14.0.tar.gz" "." "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7"
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/eigen/eigen-5.0.0.tar.gz" "." "93f7f0462988b934e632a9fba58af55192ffceae38e8f46233f4f62cb1e79370"
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/abseil-cpp/abseil-cpp-20230802.1.tar.gz" "." "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed"
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/protobuf/protobuf-25.1.tar.gz" "." "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42"
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/makeself/makeself-release-2.5.0.tar.gz" "." "705d0376db9109a8ef1d4f3876c9997ee6bed454a23619e1dbc03d25033e46ea"
fetch "https://gitcode.com/cann-src-third-party/pybind11/releases/download/v2.13.6/pybind11-2.13.6.tar.gz" "." "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20"

# ---- patch(统一存放于 <三方件目录>/patch) ----
# abseil-cpp: CVE-2025-0838 修复补丁(在线构建需应用, 离线场景需提前下载)
fetch "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1-h0/backport-CVE-2025-0838.patch" "patch" "a4e4ef6fc4f4c1ed871a3472c5e650068aaadaad2adfef2c8ccb00918bb36d8c"
# makeself: 修复补丁(在线构建为 源码包+patch 方式)
fetch "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/makeself/fix/makeself-2.5.0.patch" "patch" "0ecac3fe200221353322626900d5257a59d98347f83653b98456e08475994486"

# ---- git 仓库(opbase/ops-tensor/catlass, 离线场景需提前 clone 到 CANN_3RD_LIB_PATH 下) ----
# clone_repo <git_url> <dir_name> <tag_id>
clone_repo() {
  local url="$1" name="$2" tag="$3" init_submodules="${4:-0}"
  local dest="$CANN_3RD_LIB_PATH/$name"
  local git_log="${dest}.git.log"

  # 用 rev-parse 识别仓库, 兼容 .git 为文件的 worktree
  if git -C "$dest" rev-parse --git-dir >/dev/null 2>&1; then
    local remote head fetch_ok=1
    remote="$(git -C "$dest" remote get-url origin 2>/dev/null || true)"
    head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"

    if [ "$FORCE" = "0" ] && [ "$head" = "$tag" ]; then
      if [ "$remote" != "$url" ]; then
        echo "[WARN] $name 远端为 $remote, 与预期 $url 不同, 但 HEAD 已匹配 $tag"
      fi
      if [ "$init_submodules" = "1" ] && ! git -C "$dest" submodule update --init --recursive >"$git_log" 2>&1; then
        echo "[ERROR] 初始化 $name 的 submodule 失败"
        sed 's/^/        /' "$git_log"
        echo "        Git 日志: $git_log"
        return 1
      fi
      rm -f "$git_log"
      echo "[REUSE] $name (HEAD=$tag)"
      return 0
    fi

    echo "==> 更新 $name 到 $tag"
    if ! git -C "$dest" fetch --tags origin >"$git_log" 2>&1; then
      fetch_ok=0
      echo "[WARN] 更新 $name 的远端 refs 失败, 将尝试使用本地 commit"
    fi
    if ! git -C "$dest" checkout --detach "$tag" >>"$git_log" 2>&1; then
      if [ "$fetch_ok" = "0" ]; then
        echo "[ERROR] $name 无法更新到目标 commit $tag: 远端 fetch 失败, 可能是网络、代理或证书链问题"
      elif git -C "$dest" cat-file -e "$tag^{commit}" >/dev/null 2>&1; then
        echo "[ERROR] checkout 目标 commit $tag 失败: $name"
      else
        echo "[ERROR] $name 本地缺少目标 commit $tag, 无法在离线环境继续"
      fi
      sed 's/^/        /' "$git_log"
      echo "        Git 日志: $git_log"
      return 1
    fi
    if [ "$init_submodules" = "1" ] && ! git -C "$dest" submodule update --init --recursive >>"$git_log" 2>&1; then
      echo "[ERROR] 初始化 $name 的 submodule 失败"
      sed 's/^/        /' "$git_log"
      echo "        Git 日志: $git_log"
      return 1
    fi
    head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
    if [ "$head" != "$tag" ]; then
      echo "[ERROR] $name HEAD=$head, 与目标提交 $tag 不一致"
      echo "        Git 日志: $git_log"
      return 1
    fi
    if [ "$fetch_ok" = "0" ]; then
      echo "[WARN] $name 已使用本地目标 commit $tag, 远端 fetch 失败详情如下"
      sed 's/^/        /' "$git_log"
    fi
    rm -f "$git_log"
    echo "[OK] $name checkout $tag"
    return 0
  fi

  if [ -e "$dest" ]; then
    if [ -d "$dest" ] && [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
      rmdir "$dest" || { echo "[ERROR] 无法删除空目录: $dest"; return 1; }
    else
      echo "[ERROR] $dest 已存在且不是可复用的 git 仓库"
      echo "        为避免删除已有内容, 请手动删除该目录或更换 --cann_3rd_lib_path"
      return 1
    fi
  fi

  echo "==> clone $name  <-  $url"
  if ! git clone "$url" "$dest" >"$git_log" 2>&1; then
    echo "[ERROR] clone 失败: $name"
    sed 's/^/        /' "$git_log"
    echo "        Git 日志: $git_log"
    return 1
  fi
  if ! (cd "$dest" && git checkout --detach "$tag") >>"$git_log" 2>&1; then
    echo "[ERROR] checkout $tag 失败: $name"
    sed 's/^/        /' "$git_log"
    echo "        Git 日志: $git_log"
    return 1
  fi
  if [ "$init_submodules" = "1" ] && ! git -C "$dest" submodule update --init --recursive >>"$git_log" 2>&1; then
    echo "[ERROR] 初始化 $name 的 submodule 失败"
    sed 's/^/        /' "$git_log"
    echo "        Git 日志: $git_log"
    return 1
  fi
  head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
  if [ "$head" != "$tag" ]; then
    echo "[ERROR] $name HEAD=$head, 与目标提交 $tag 不一致"
    echo "        Git 日志: $git_log"
    return 1
  fi
  rm -f "$git_log"
  echo "[OK] $name checkout $tag"
}

if command -v git >/dev/null 2>&1; then
  echo "==> 下载 git 仓库(opbase/ops-tensor/catlass)"
  clone_repo "https://atomgit.com/cann/opbase.git"     "opbase"     "de414d06cfb7bf2728a555bfb9fc1d26a4c83792"
  clone_repo "https://atomgit.com/cann/ops-tensor.git" "ops-tensor" "fb30752ed5524744cee0cd21f26db5d52e54b6e4" 1
  clone_repo "https://atomgit.com/cann/catlass.git"     "catlass"    "769cd40a8716b28650b6bebb08db4834eea4462f"
else
  echo "[WARN] 未找到 git, 跳过 opbase/ops-tensor/catlass 仓库下载"
  echo "       请在有网环境手动执行:"
  echo "         git clone https://atomgit.com/cann/opbase.git     \$CANN_3RD_LIB_PATH/opbase"
  echo "         git clone https://atomgit.com/cann/ops-tensor.git \$CANN_3RD_LIB_PATH/ops-tensor"
  echo "         git clone https://atomgit.com/cann/catlass.git     \$CANN_3RD_LIB_PATH/catlass"
fi

echo "==> 全部完成, 存放位置: $CANN_3RD_LIB_PATH"
find "$CANN_3RD_LIB_PATH" -type f 2>/dev/null | sort
