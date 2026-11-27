#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
bin=$(realpath -- "$1")
output=$(realpath -s -- "$2")
source_url=$3
tag=${4:-}
work=$(mktemp -d -- "$(dirname -- "$output")/.appimage.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT

package="$work/package"
mkdir -p "$package/licenses" "$package/_rehooked_/shader_cache"
cp "$root"/licenses/* "$package/licenses/"
cp "$root/assets/shader_cache/recipes-hash2-aes.bin" "$package/_rehooked_/shader_cache/"

for exe in gow3 gow3_debug; do
    name=GOW3rehooked
    [[ "$exe" != gow3_debug ]] || name+=' Debug'
    cat > "$work/$exe.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$name
Exec=$exe
Icon=gow3
Categories=Game;
EOF
    appdir="$work/$exe.AppDir"
    APPIMAGE_EXTRACT_AND_RUN=1 linuxdeploy --appdir "$appdir" --executable "$bin/$exe" \
        --desktop-file "$work/$exe.desktop" --icon-file "$root/tools/appimage/gow3.svg" \
        --exclude-library 'libvulkan.so*'
    if [[ "$exe" == gow3 && -n "$tag" ]]; then
        [[ "$tag" =~ ^v[0-9]+(\.[0-9]+){1,2}(-(alpha|beta|rc)(\.[0-9]+)?)?$ ]]
        asset=$(basename -- "$output")
        [[ "$asset" == GOW3rehooked-Linux-x86_64.zip || "$asset" == GOW3rehooked-Linux-x86_64-AVX1.zip ]]
        mv -- "$appdir/AppRun" "$appdir/AppRun.game"
        CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath \
            -ldflags "-s -w -X main.currentTag=$tag -X main.assetName=$asset" \
            -o "$appdir/AppRun" "$root/tools/appimage/update.go"
    fi
    ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 appimagetool --no-appstream "$appdir" "$package/$exe.AppImage"
done

(
    cd "$root"
    printf '%s\n\n' "GOW3rehooked is licensed under GPL-2.0-or-later and builds on shadPS4. Source code: $source_url"
    for notice in \
        src/externals/{ffmpeg/LICENSE.md,sdl3/LICENSE.txt,sdl3/src/hidapi/LICENSE-orig.txt,imgui/LICENSE.txt,stb/LICENSE,zydis/LICENSE} \
        src/externals/{zydis/dependencies/zycore/LICENSE,xbyak/COPYRIGHT,xxhash/LICENSE,zstd/LICENSE,fmt/LICENSE} \
        src/externals/{magic_enum/LICENSE,LibAtrac9/LICENSE} src/ui/fonts/OFL.txt licenses/OFL-Noto-CJK.txt src/present/fsr/LICENSE.txt; do
        printf '\n=== %s ===\n\n' "$notice"
        cat "$notice"
    done
    printf '\n=== Intel CMAA2, Copyright Intel Corporation ===\n\n'
    cat licenses/Apache-2.0.txt
    printf '\n=== Fast Adaptive AA (based on NVIDIA FXAA 3.11) ===\n\n'
    cat licenses/Fast-Adaptive-AA.txt
    printf '\n=== AppImage Type 2 runtime ===\n%s\n' 'https://github.com/AppImage/type2-runtime#license'
    if [[ -n "$tag" ]]; then
        printf '\n=== Go standard library ===\n'
        goroot=$(go env GOROOT)
        if [[ -f "$goroot/LICENSE" ]]; then cat "$goroot/LICENSE"; else cat "$goroot/src/vendor/golang.org/x/net/LICENSE"; fi
    fi
    {
        printf '%s\n' libshaderc-dev glslang-dev spirv-tools libvulkan-dev
        find "$work" -path '*/usr/lib/*.so*' -printf '%f\n' | sort -u | while read -r name; do
            { dpkg-query -S "/usr/lib/x86_64-linux-gnu/$name" 2>/dev/null || true; } | sed -n 's/: .*//p'
        done
    } | sed 's/:.*//' | sort -u | while read -r component; do
        notice="/usr/share/doc/$component/copyright"
        if [[ -f "$notice" ]]; then
            printf '\n=== %s ===\n\n' "$component"
            cat "$notice"
        fi
    done
) > "$package/licenses/THIRD-PARTY-NOTICES.txt"

cd "$package"
zip -q -r "$work/release.zip" gow3.AppImage gow3_debug.AppImage licenses _rehooked_
mv -fT -- "$work/release.zip" "$output"
