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
  - 已存在的同名文件会先做 SHA256 校验: hash 一致则复用(不重复下载), 不一致则删除后重新下载;
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
  # --ssl-no-revoke: Windows 内置 curl 在无外网吊销服务器时可能报 CRYPT_E_REVOCATION_OFFLINE,
  # 仅跳过证书吊销列表查询, 不影响证书有效性校验。
  dl() { curl -fSL --ssl-no-revoke --retry 3 --connect-timeout 30 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
  dl() { wget -q --tries=3 --timeout=30 -O "$2" "$1"; }
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
  local fname dest
  fname="$(basename "$url")"
  dest="$CANN_3RD_LIB_PATH/$sub/$fname"
  mkdir -p "$(dirname "$dest")"

  # 缓存复用: 已存在同名文件
  if [ -f "$dest" ] && [ "$FORCE" = "0" ]; then
    if [ -n "$exp" ]; then
      local act; act="$(sha_of "$dest")"
      if [ -n "$act" ] && [ "$act" = "$exp" ]; then
        echo "[复用] $fname (SHA256 匹配, 跳过下载)"
        return 0
      else
        echo "[删除] $fname (SHA256 不一致, 期望=$exp 实际=$act)"
        rm -f "$dest"
      fi
    else
      echo "[删除] $fname (无参考 hash, 重新下载)"
      rm -f "$dest"
    fi
  fi

  echo "==> 下载 $fname  <-  $url"
  dl "$url" "$dest" || { echo "[ERROR] 下载失败: $fname"; return 1; }

  # 下载后完整性校验: 计算结果为空同样视为失败
  if [ -n "$exp" ]; then
    local act2; act2="$(sha_of "$dest")"
    if [ -z "$act2" ] || [ "$act2" != "$exp" ]; then
      echo "[ERROR] $fname SHA256 校验失败, 已删除（期望=$exp 实际=${act2:-<空>}）"
      rm -f "$dest"
      return 1
    fi
  fi
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
  local url="$1" name="$2" tag="$3"
  local dest="$CANN_3RD_LIB_PATH/$name"

  # 用 rev-parse 识别仓库, 兼容 .git 为文件的 worktree
  if git -C "$dest" rev-parse --git-dir >/dev/null 2>&1; then
    local remote head
    remote="$(git -C "$dest" remote get-url origin 2>/dev/null || true)"
    head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"

    if [ "$FORCE" = "0" ] && [ "$head" = "$tag" ]; then
      if [ "$remote" != "$url" ]; then
        echo "[WARN] $name 远端为 $remote, 与预期 $url 不同, 但 HEAD 已匹配 $tag"
      fi
      echo "[复用] $name (HEAD=$tag)"
      return 0
    fi

    echo "==> 更新 $name 到 $tag"
    git -C "$dest" fetch --tags origin >/dev/null 2>&1 || true
    git -C "$dest" checkout "$tag" || { echo "[ERROR] checkout $tag 失败: $name"; return 1; }
    head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
    if [ "$head" != "$tag" ]; then
      echo "[ERROR] $name HEAD=$head, 与目标提交 $tag 不一致"
      return 1
    fi
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
  git clone "$url" "$dest" || { echo "[ERROR] clone 失败: $name"; return 1; }
  (cd "$dest" && git checkout "$tag") || { echo "[ERROR] checkout $tag 失败: $name"; return 1; }
  echo "[OK] $name checkout $tag"
}

if command -v git >/dev/null 2>&1; then
  echo "==> 下载 git 仓库(opbase/ops-tensor/catlass)"
  clone_repo "https://atomgit.com/cann/opbase.git"     "opbase"     "0a83b915a6f6be49a03b907713d6f000b3dea1ea"
  clone_repo "https://atomgit.com/cann/ops-tensor.git" "ops-tensor" "deea03a7d112b002a986783e39d4f91c92949518"
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
