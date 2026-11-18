cmake_minimum_required(VERSION 3.24)
project(gow3_recomp LANGUAGES C CXX)
if(WIN32)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "Use build-fast.ps1 with the Windows EWDK/LLVM toolchain")
    endif()
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "Use build-linux.sh (clang/clang++ and ld.lld) on Linux")
endif()
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
add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
# Omitted aggregate members are value-initialized on purpose (designated initializers).
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/clang:-Wno-missing-field-initializers> $<$<COMPILE_LANGUAGE:CXX>:/clang:-Wno-missing-designated-field-initializers>)
# Keep implicit FMA contraction off and use a ThinLTO cache for incremental builds.
add_compile_options(/clang:-march=${GOW3_CPU_ARCH} /clang:-ffp-contract=off -flto=thin)
add_link_options(/lldltocache:${CMAKE_BINARY_DIR}/lto-cache)
else()
# Linux: the code generation of the clang-cl build. GOW3_CPU_ARCH without FMA contraction; ThinLTO (GOW3_LTO=full in CI) with
# the lld cache. clang-cl also implies no strict aliasing (-relaxed-aliasing), MSVC volatile semantics (-fms-volatile) and
# function sections (/Gy, part of /O2). clang-cl /O2 is -O3, so the Linux presets pass -O3.
set(GOW3_LTO thin CACHE STRING "Linux LTO mode: thin, full or off")
set_property(CACHE GOW3_LTO PROPERTY STRINGS thin full off)
option(GOW3_ASAN "Linux: AddressSanitizer, without LTO (debug builds)" OFF)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers> $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-designated-field-initializers>)
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-march=${GOW3_CPU_ARCH};-ffp-contract=off;-fno-strict-aliasing;-fms-volatile;-ffunction-sections>")
add_link_options(-fuse-ld=lld -Wl,--gc-sections)
if(GOW3_ASAN)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address;-fno-omit-frame-pointer>")
    add_link_options(-fsanitize=address)
elseif(NOT GOW3_LTO STREQUAL "off")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-flto=${GOW3_LTO}>")
    add_link_options(-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/lto-cache)
endif()
endif()
find_package(Vulkan REQUIRED) # gpu/device.cpp
set(ARCHITECTURE x86_64)
set(SRC ${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src/externals)

if(WIN32)
file(GLOB clang_builtins "$ENV{LLVM_ROOT}/lib/clang/*/lib/windows/clang_rt.builtins-x86_64.lib")
list(SORT clang_builtins COMPARE NATURAL ORDER DESCENDING)
list(GET clang_builtins 0 clang_builtins_library)
endif()

# gow3.exe, one executable: main (launcher, bootstrap, game icon), orbis (emulated PS4: core and HLE libraries), gpu (Vulkan
# renderer and shaders), present (window, swapchain, scalers), ui (ImGui screens), input, patches (EBOOT patches and hooks)
# and debug (the diagnostics that GOW3_DEBUG turns on). The GPU thread (gpu/gpu_thread.cpp) draws each submit with the Vulkan renderer (vk_gpu.cpp), then applies its fences
# and memory writes. patches/reference holds the former engine hooks (gnm_hooks.cpp) for reference; it is not built.
file(GLOB_RECURSE gow3_sources CONFIGURE_DEPENDS ${SRC}/main/*.cpp ${SRC}/orbis/*.cpp ${SRC}/gpu/*.cpp ${SRC}/present/*.cpp
     ${SRC}/ui/*.cpp ${SRC}/input/*.cpp ${SRC}/patches/*.cpp ${SRC}/debug/*.cpp)
list(FILTER gow3_sources EXCLUDE REGEX "/src/(patches/reference|gpu/shaders/legacy)/|/src/main/gow3_debug\\.cpp$")
add_executable(gow3 WIN32 ${gow3_sources})
# gow3_debug.exe ships beside gow3.exe and starts it with GOW3_DEBUG=1 (the log and the diagnostics) for users who report problems.
add_executable(gow3_debug WIN32 ${SRC}/main/gow3_debug.cpp)
if(WIN32)
target_link_libraries(gow3_debug PRIVATE user32)
endif()
# The launcher sources use the wide Windows API; the runtime keeps its own choice.
set_source_files_properties(src/main/game.cpp src/main/exe_icon.cpp PROPERTIES COMPILE_DEFINITIONS "UNICODE;_UNICODE")
set(easu_dir ${SRC}/present/fsr)
set(easu_spv ${CMAKE_CURRENT_BINARY_DIR}/easu_spv.inc)
add_custom_command(OUTPUT ${easu_spv}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -mfmt=num -I ${easu_dir} ${easu_dir}/easu.comp -o ${easu_spv}
    DEPENDS ${easu_dir}/easu.comp ${easu_dir}/ffx_a.h ${easu_dir}/ffx_fsr1.h VERBATIM)
target_sources(gow3 PRIVATE ${easu_spv})
foreach(pass cmaa2_0 cmaa2_1 cmaa2_2 cmaa2_3)
    set(cmaa_spv ${CMAKE_CURRENT_BINARY_DIR}/${pass}_spv.inc)
    add_custom_command(OUTPUT ${cmaa_spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -mfmt=num ${SRC}/present/cmaa2/${pass}.comp -o ${cmaa_spv}
        DEPENDS ${SRC}/present/cmaa2/${pass}.comp VERBATIM)
    target_sources(gow3 PRIVATE ${cmaa_spv})
endforeach()
target_include_directories(gow3 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if(WIN32)
target_include_directories(gow3 PRIVATE ${SRC}) # every project include is a path from src/
else()
# Every project include is a quoted path from src/; with -I, src/debug/debug.h would shadow libstdc++'s <debug/debug.h>.
target_compile_options(gow3 PRIVATE "-iquote${SRC}")
target_compile_options(gow3_debug PRIVATE "-iquote${SRC}")
endif()
target_compile_definitions(gow3 PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE _SCL_SECURE_NO_WARNINGS)
target_compile_definitions(gow3 PRIVATE $<$<BOOL:${GOW3_IVY_BRIDGE}>:GOW3_IVY_BRIDGE>)
if(WIN32)
target_compile_options(gow3 PRIVATE /Gy /Gw /GS- /clang:-Wno-unknown-warning-option)
else()
target_compile_options(gow3 PRIVATE -fdata-sections -fno-stack-protector -Wno-unknown-warning-option) # /Gw /GS-
endif()
target_link_libraries(gow3 PRIVATE fmt::fmt Zydis xbyak
    magic_enum::magic_enum SDL3::SDL3 LibAtrac9
    ffmpeg Dear_ImGui Vulkan::Vulkan xxhash zstd runtime_headers)
if(WIN32)
target_link_libraries(gow3 PRIVATE ${clang_builtins_library} OneCore ws2_32 Psapi
    Synchronization dbghelp bcrypt shell32 user32 windowsapp)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
find_package(unofficial-shaderc CONFIG REQUIRED)
target_link_libraries(gow3 PRIVATE unofficial::shaderc::shaderc)
else()
# shaderc: the static vcpkg package when CMAKE_PREFIX_PATH has it (release builds, as on Windows); otherwise the
# system's shared libshaderc (development builds). Some distributions' libshaderc_combined.a lack glslang (Arch).
find_package(Threads REQUIRED)
find_package(unofficial-shaderc CONFIG QUIET)
if(TARGET unofficial::shaderc::shaderc)
    target_link_libraries(gow3 PRIVATE unofficial::shaderc::shaderc)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(shaderc REQUIRED IMPORTED_TARGET shaderc)
    target_link_libraries(gow3 PRIVATE PkgConfig::shaderc)
endif()
target_link_libraries(gow3 PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()
