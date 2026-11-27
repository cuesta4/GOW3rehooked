cmake_minimum_required(VERSION 3.24)
project(gow3_recomp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
option(GOW3_IVY_BRIDGE "Build for Intel Ivy Bridge CPUs" OFF)
set(GOW3_CPU_ARCH x86-64-v3)
if(GOW3_IVY_BRIDGE)
    set(GOW3_CPU_ARCH ivybridge)
endif()
if(WIN32)
    option(GOW3_NO_UPDATER "Build without the Windows autoupdater" OFF)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "Use build-fast.ps1 with the Windows EWDK/LLVM toolchain")
    endif()
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
    # Omitted aggregate members are value-initialized on purpose (designated initializers).
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/clang:-Wno-missing-field-initializers> $<$<COMPILE_LANGUAGE:CXX>:/clang:-Wno-missing-designated-field-initializers>)
    # Keep implicit FMA contraction off and use a ThinLTO cache for incremental builds. The resource compiler gets none of it.
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/clang:-march=${GOW3_CPU_ARCH};/clang:-ffp-contract=off;-flto=thin>")
    add_link_options(/lldltocache:${CMAKE_BINARY_DIR}/lto-cache)
    find_package(Vulkan REQUIRED) # gpu/device.cpp
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        message(FATAL_ERROR "The Linux build requires Clang targeting x86-64")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-march=${GOW3_CPU_ARCH};-ffp-contract=off;-flto=thin>")
    add_link_options(-fuse-ld=lld -flto=thin LINKER:--thinlto-cache-dir=${CMAKE_BINARY_DIR}/lto-cache)
    find_package(PkgConfig REQUIRED)
    find_package(Threads REQUIRED)
    find_package(Vulkan 1.4.328 REQUIRED)
    if(NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "glslc is required to build the compute shaders")
    endif()
else()
    message(FATAL_ERROR "Supported systems are Windows and Linux")
endif()
set(ARCHITECTURE x86_64)
set(SRC ${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src/externals)

# The game runtime: main (launcher, bootstrap, game icon), orbis (emulated PS4: core and HLE libraries), gpu (Vulkan
# renderer and shaders), present (window, swapchain, scalers), ui (ImGui screens), input, patches (EBOOT patches and hooks)
# and debug (the diagnostics that GOW3_DEBUG turns on). The GPU thread (gpu/gpu_thread.cpp) draws each submit with the Vulkan renderer (vk_gpu.cpp), then applies its fences
# and memory writes. patches/reference holds the former engine hooks (gnm_hooks.cpp) for reference; it is not built.
file(GLOB_RECURSE gow3_sources CONFIGURE_DEPENDS ${SRC}/main/*.cpp ${SRC}/orbis/*.cpp ${SRC}/gpu/*.cpp ${SRC}/present/*.cpp
     ${SRC}/ui/*.cpp ${SRC}/input/*.cpp ${SRC}/patches/*.cpp ${SRC}/debug/*.cpp)
list(FILTER gow3_sources EXCLUDE REGEX "/src/(patches/reference|gpu/shaders/legacy)/|/src/main/gow3_debug\\.cpp$")
if(NOT WIN32 OR GOW3_RELEASE_TAG OR CMAKE_BUILD_TYPE STREQUAL "Release")
    list(FILTER gow3_sources EXCLUDE REGEX "/src/main/exe_icon\\.cpp$")
else()
    # Development builds write the game's icon into gow3.exe; releases never rewrite their executable.
    set_property(SOURCE src/main/game.cpp APPEND PROPERTY COMPILE_DEFINITIONS GOW3_EXE_ICON)
endif()
if(NOT WIN32 OR GOW3_NO_UPDATER)
    list(FILTER gow3_sources EXCLUDE REGEX "/src/main/update\\.cpp$") # the AppImage updates itself (tools/appimage/update.go)
endif()

# Each Windows binary names itself and its release in a version resource (Explorer's file properties). Antivirus
# heuristics also distrust executables without one.
function(gow3_version_resource target file_name description)
    if(NOT WIN32)
        return()
    endif()
    set(numbers 0,0,0,0)
    set(text "development build")
    set(flags 0)
    if(GOW3_RELEASE_TAG MATCHES "^v([0-9]+)\\.([0-9]+)(\\.([0-9]+))?")
        set(patch 0)
        if(CMAKE_MATCH_4)
            set(patch ${CMAKE_MATCH_4})
        endif()
        set(numbers "${CMAKE_MATCH_1},${CMAKE_MATCH_2},${patch},0")
        string(SUBSTRING "${GOW3_RELEASE_TAG}" 1 -1 text)
        if(GOW3_RELEASE_TAG MATCHES "-")
            set(flags VS_FF_PRERELEASE)
        endif()
    endif()
    set(type VFT_APP)
    if(file_name MATCHES "\\.dll$")
        set(type VFT_DLL)
    endif()
    file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version/${target}.rc CONTENT [[
#include <winver.h>
VS_VERSION_INFO VERSIONINFO
FILEVERSION @numbers@
PRODUCTVERSION @numbers@
FILEFLAGSMASK VS_FFI_FILEFLAGSMASK
FILEFLAGS @flags@
FILEOS VOS_NT_WINDOWS32
FILETYPE @type@
FILESUBTYPE VFT2_UNKNOWN
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
            VALUE "CompanyName", "GOW3rehooked"
            VALUE "FileDescription", "@description@"
            VALUE "FileVersion", "@text@"
            VALUE "InternalName", "@file_name@"
            VALUE "LegalCopyright", "GPL-2.0-or-later, https://github.com/cuesta4/GOW3rehooked"
            VALUE "OriginalFilename", "@file_name@"
            VALUE "ProductName", "GOW3rehooked"
            VALUE "ProductVersion", "@text@"
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END
]] @ONLY)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/version/${target}.rc)
endfunction()

# One executable on Windows: antivirus machine learning flagged the small launchers of v0.1.5 to v0.2.1, never the
# executable that holds the whole game. gow3_debug.cmd starts it with --debug.
add_executable(gow3 WIN32 ${gow3_sources})
gow3_version_resource(gow3 gow3.exe "GOW3rehooked")
if(NOT WIN32)
    # The debug launcher starts the same game with GOW3_DEBUG=1.
    add_executable(gow3_debug ${SRC}/main/gow3_debug.cpp)
endif()
set(easu_dir ${SRC}/present/fsr)
set(easu_spv ${CMAKE_CURRENT_BINARY_DIR}/easu_spv.inc)
add_custom_command(OUTPUT ${easu_spv}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -mfmt=num -I ${easu_dir} ${easu_dir}/easu.comp -o ${easu_spv}
    DEPENDS ${easu_dir}/easu.comp ${easu_dir}/ffx_a.h ${easu_dir}/ffx_fsr1.h VERBATIM)
target_sources(gow3 PRIVATE ${easu_spv})
foreach(shader cmaa2/cmaa2_0 cmaa2/cmaa2_1 cmaa2/cmaa2_2 cmaa2/cmaa2_3 faaa/faaa)
    get_filename_component(name ${shader} NAME)
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/${name}_spv.inc)
    add_custom_command(OUTPUT ${spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -mfmt=num ${SRC}/present/${shader}.comp -o ${spv}
        DEPENDS ${SRC}/present/${shader}.comp VERBATIM)
    target_sources(gow3 PRIVATE ${spv})
endforeach()
set(ui_font ${SRC}/ui/fonts/NotoSansCJKtc-RehookedSubset.otf)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ui_font})
file(READ ${ui_font} ui_font_hex HEX)
string(REGEX REPLACE "(..)" "0x\\1," ui_font_bytes "${ui_font_hex}")
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ui_font.inc CONTENT "@ui_font_bytes@\n" @ONLY)
target_include_directories(gow3 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if(WIN32)
    target_include_directories(gow3 PRIVATE ${SRC}) # every project include is a path from src/
else()
    # -iquote keeps src/debug/debug.h from shadowing libstdc++'s <debug/debug.h>.
    target_compile_options(gow3 PRIVATE -iquote ${SRC})
endif()
target_compile_definitions(gow3 PRIVATE $<$<BOOL:${GOW3_IVY_BRIDGE}>:GOW3_IVY_BRIDGE>)
target_link_libraries(gow3 PRIVATE fmt::fmt Zydis xbyak
    magic_enum::magic_enum SDL3::SDL3 LibAtrac9
    ffmpeg Dear_ImGui Vulkan::Vulkan xxhash zstd runtime_headers)
if(WIN32)
    file(GLOB clang_builtins "$ENV{LLVM_ROOT}/lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
    list(SORT clang_builtins COMPARE NATURAL ORDER DESCENDING)
    list(GET clang_builtins 0 clang_builtins_library)
    if(GOW3_NO_UPDATER)
        set_property(SOURCE src/main/game.cpp APPEND PROPERTY COMPILE_DEFINITIONS GOW3_NO_UPDATER)
    else()
        target_link_libraries(gow3 PRIVATE winhttp)
        if(GOW3_RELEASE_TAG)
            set_property(SOURCE src/main/game.cpp APPEND PROPERTY COMPILE_DEFINITIONS "GOW3_RELEASE_TAG=L\"${GOW3_RELEASE_TAG}\"")
        endif()
    endif()
    set_property(SOURCE src/main/game.cpp src/main/exe_icon.cpp APPEND PROPERTY COMPILE_DEFINITIONS "UNICODE;_UNICODE")
    target_compile_definitions(gow3 PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS)
    target_compile_options(gow3 PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/Gy;/Gw;/GS-;/clang:-Wno-unknown-warning-option>")
    target_link_libraries(gow3 PRIVATE ${clang_builtins_library} OneCore ws2_32 Psapi
        Synchronization dbghelp bcrypt shell32 user32 windowsapp)
    set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    find_package(unofficial-shaderc CONFIG REQUIRED)
    target_link_libraries(gow3 PRIVATE unofficial::shaderc::shaderc)
else()
    pkg_check_modules(SHADERC REQUIRED IMPORTED_TARGET shaderc_combined)
    pkg_check_modules(UNWIND REQUIRED libunwind)
    find_package(glslang CONFIG REQUIRED)
    find_package(SPIRV-Tools CONFIG REQUIRED)
    find_package(SPIRV-Tools-opt CONFIG REQUIRED)
    target_include_directories(gow3 SYSTEM PRIVATE ${UNWIND_INCLUDE_DIRS})
    target_link_libraries(gow3 PRIVATE PkgConfig::SHADERC glslang::SPIRV glslang::glslang SPIRV-Tools-opt
        SPIRV-Tools-static Threads::Threads ${CMAKE_DL_LIBS})
    target_link_options(gow3 PRIVATE -pie -Wl,--export-dynamic)
endif()
