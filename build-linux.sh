#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Linux counterpart of build-fast.ps1: configures a CMake preset (clang/clang++ + ld.lld, x86-64-v3, -O3, ThinLTO with the
# lld cache), builds gow3 and gow3_debug with ninja -k 0, checks that both are x86-64 ELF executables and copies them to
# build/linux (and to $GOW3_GAME_DIR when set).
#   ./build-linux.sh                 linux-relwithdebinfo (the default of build-fast.ps1)
#   ./build-linux.sh --release       linux-release          ./build-linux.sh --ci     linux-ci (full LTO, LTO -O3)
#   ./build-linux.sh --debug         linux-debug            ./build-linux.sh --asan   linux-debug-asan
#   options: --ivybridge (AVX1 CPUs, GOW3_IVY_BRIDGE), --jobs N, --configure-only, --no-deploy, --fresh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
preset=linux-relwithdebinfo
jobs="$(nproc)"
configure_only=0
deploy=1
fresh=()
cpu=()
while (($#)); do
    case "$1" in
        --release) preset=linux-release ;;
        --ci) preset=linux-ci ;;
        --debug) preset=linux-debug ;;
        --asan) preset=linux-debug-asan ;;
        --jobs) jobs="$2"; shift ;;
        --configure-only) configure_only=1 ;;
        --no-deploy) deploy=0 ;;
        --fresh) fresh=(--fresh) ;;
        --ivybridge) cpu=(-DGOW3_IVY_BRIDGE=ON) ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

missing=()
for tool in cmake ninja clang clang++ ld.lld nasm pkg-config; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
pkg-config --exists vulkan || missing+=("vulkan (headers + loader)")
if ((${#missing[@]})); then
    echo "BUILD FAILED: missing ${missing[*]}" >&2
    exit 1
fi

launcher=()
if command -v sccache >/dev/null; then
    launcher=(-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache)
elif command -v ccache >/dev/null; then
    launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
# A static shaderc from vcpkg (GOW3_SHADERC_PREFIX=<vcpkg>/installed/x64-linux) is preferred; otherwise the system's.
prefix=()
if [[ -n "${GOW3_SHADERC_PREFIX:-}" ]]; then
    prefix=("-DCMAKE_PREFIX_PATH=$GOW3_SHADERC_PREFIX")
    [[ -x "$GOW3_SHADERC_PREFIX/tools/shaderc/glslc" ]] && prefix+=("-DVulkan_GLSLC_EXECUTABLE=$GOW3_SHADERC_PREFIX/tools/shaderc/glslc")
fi

cd "$root"
cmake --preset "$preset" "${fresh[@]}" "${cpu[@]}" "${launcher[@]}" "${prefix[@]}"
((configure_only)) && exit 0
build="$root/build-linux/$preset"
if ! cmake --build "$build" --parallel "$jobs" --target gow3 gow3_debug -- -k 0; then
    echo "BUILD FAILED: $preset" >&2
    exit 1
fi

out="$root/build/linux"
mkdir -p "$out"
for name in gow3 gow3_debug; do
    binary="$build/bin/$name"
    # ELF magic, 64-bit (EI_CLASS=2), little endian, e_machine 0x3e (x86-64).
    if [[ "$(od -An -tx1 -N20 "$binary" | tr -d ' \n')" != 7f454c460201* ]] ||
       [[ "$(od -An -tx1 -j18 -N2 "$binary" | tr -d ' \n')" != 3e00 ]]; then
        echo "BUILD FAILED: $name is not an x86-64 ELF executable" >&2
        exit 1
    fi
    install -m 755 "$binary" "$out/$name"
done
if ((deploy)) && [[ -n "${GOW3_GAME_DIR:-}" ]]; then
    [[ -d "$GOW3_GAME_DIR" ]] || { echo "BUILD FAILED: no such directory: $GOW3_GAME_DIR" >&2; exit 1; }
    install -m 755 "$out/gow3" "$out/gow3_debug" "$GOW3_GAME_DIR/"
fi
echo "BUILD SUCCEEDED: gow3, gow3_debug ($preset)"
