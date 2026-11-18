# FFmpeg 9.0.2, vendored by tools/ffmpeg/vendor.ps1: the sources of the objects the port links (H.264 decoder for the
# cinematics, MP3 decoder + parser and swresample for sceAjm) and configure's output (config.h, config_components.h, config.asm).
find_program(NASM nasm REQUIRED)
set(CMAKE_ASM_NASM_COMPILER ${NASM})
enable_language(ASM_NASM)
set_directory_properties(PROPERTIES COMPILE_OPTIONS "") # the global clang-cl flags would reach NASM
add_library(ffmpeg STATIC
    libavcodec/allcodecs.c
    libavcodec/aom_film_grain.c
    libavcodec/atsc_a53.c
    libavcodec/avcodec.c
    libavcodec/bitstream_filters.c
    libavcodec/bsf.c
    libavcodec/cabac.c
    libavcodec/codec_desc.c
    libavcodec/codec_par.c
    libavcodec/dct32_fixed.c
    libavcodec/dct32_float.c
    libavcodec/decode.c
    libavcodec/dovi_rpu.c
    libavcodec/dovi_rpudec.c
    libavcodec/dynamic_hdr_vivid.c
    libavcodec/encode.c
    libavcodec/exif.c
    libavcodec/get_buffer.c
    libavcodec/golomb.c
    libavcodec/h264_cabac.c
    libavcodec/h264_cavlc.c
    libavcodec/h264_direct.c
    libavcodec/h264_loopfilter.c
    libavcodec/h264_mb.c
    libavcodec/h264_parse.c
    libavcodec/h264_picture.c
    libavcodec/h264_ps.c
    libavcodec/h264_refs.c
    libavcodec/h264_sei.c
    libavcodec/h264_slice.c
    libavcodec/h2645_parse.c
    libavcodec/h2645_sei.c
    libavcodec/h2645_vui.c
    libavcodec/h2645data.c
    libavcodec/h264chroma.c
    libavcodec/h264data.c
    libavcodec/h264dec.c
    libavcodec/h264dsp.c
    libavcodec/h264idct.c
    libavcodec/h264pred.c
    libavcodec/h264qpel.c
    libavcodec/h274.c
    libavcodec/itut35.c
    libavcodec/mathtables.c
    libavcodec/mpegaudio_parser.c
    libavcodec/mpegaudio.c
    libavcodec/mpegaudiodata.c
    libavcodec/mpegaudiodec_common.c
    libavcodec/mpegaudiodec_fixed.c
    libavcodec/mpegaudiodecheader.c
    libavcodec/mpegaudiodsp_data.c
    libavcodec/mpegaudiodsp_fixed.c
    libavcodec/mpegaudiodsp_float.c
    libavcodec/mpegaudiodsp.c
    libavcodec/mpegaudiotabs.c
    libavcodec/options.c
    libavcodec/packet.c
    libavcodec/parser.c
    libavcodec/parsers.c
    libavcodec/profiles.c
    libavcodec/pthread_frame.c
    libavcodec/pthread_slice.c
    libavcodec/pthread.c
    libavcodec/startcode.c
    libavcodec/threadprogress.c
    libavcodec/tiff_common.c
    libavcodec/utils.c
    libavcodec/videodsp.c
    libavcodec/vlc.c
    libavcodec/x86/constants.c
    libavcodec/x86/dct32.asm
    libavcodec/x86/fpel.asm
    libavcodec/x86/h264_chromamc_10bit.asm
    libavcodec/x86/h264_chromamc.asm
    libavcodec/x86/h264_deblock_10bit.asm
    libavcodec/x86/h264_deblock.asm
    libavcodec/x86/h264_idct_10bit.asm
    libavcodec/x86/h264_idct.asm
    libavcodec/x86/h264_intrapred_10bit.asm
    libavcodec/x86/h264_intrapred_init.c
    libavcodec/x86/h264_intrapred.asm
    libavcodec/x86/h264_qpel_10bit.asm
    libavcodec/x86/h264_qpel_8bit.asm
    libavcodec/x86/h264_qpel.c
    libavcodec/x86/h264_weight_10bit.asm
    libavcodec/x86/h264_weight.asm
    libavcodec/x86/h264chroma_init.c
    libavcodec/x86/h264dsp_init.c
    libavcodec/x86/imdct36.asm
    libavcodec/x86/mpegaudiodsp.c
    libavcodec/x86/qpel.asm
    libavcodec/x86/videodsp_init.c
    libavcodec/x86/videodsp.asm
    libavutil/ambient_viewing_environment.c
    libavutil/avsscanf.c
    libavutil/avstring.c
    libavutil/bprint.c
    libavutil/buffer.c
    libavutil/channel_layout.c
    libavutil/container_fifo.c
    libavutil/cpu.c
    libavutil/crc.c
    libavutil/dict.c
    libavutil/display.c
    libavutil/dovi_meta.c
    libavutil/error.c
    libavutil/eval.c
    libavutil/fifo.c
    libavutil/film_grain_params.c
    libavutil/frame.c
    libavutil/hdr_dynamic_metadata.c
    libavutil/hdr_dynamic_vivid_metadata.c
    libavutil/hwcontext.c
    libavutil/imgutils.c
    libavutil/log.c
    libavutil/mastering_display_metadata.c
    libavutil/mathematics.c
    libavutil/md5.c
    libavutil/mem.c
    libavutil/opt.c
    libavutil/parseutils.c
    libavutil/pixdesc.c
    libavutil/random_seed.c
    libavutil/rational.c
    libavutil/refstruct.c
    libavutil/reverse.c
    libavutil/samplefmt.c
    libavutil/sha.c
    libavutil/side_data.c
    libavutil/slicethread.c
    libavutil/stereo3d.c
    libavutil/time.c
    libavutil/timecode_internal.c
    libavutil/timecode.c
    libavutil/utils.c
    libavutil/video_enc_params.c
    libavutil/x86/cpu.c
    libavutil/x86/cpuid.asm
    libavutil/x86/crc.asm
    libavutil/x86/imgutils_init.c
    libavutil/x86/imgutils.asm
    libswresample/audioconvert.c
    libswresample/dither.c
    libswresample/options.c
    libswresample/rematrix.c
    libswresample/resample_dsp.c
    libswresample/resample.c
    libswresample/swresample_frame.c
    libswresample/swresample.c
    libswresample/x86/audio_convert_init.c
    libswresample/x86/audio_convert.asm
    libswresample/x86/rematrix_init.c
    libswresample/x86/rematrix.asm
    libswresample/x86/resample_init.c
    libswresample/x86/resample.asm
)
target_include_directories(ffmpeg SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(WIN32)
target_compile_definitions(ffmpeg PRIVATE $<$<COMPILE_LANGUAGE:C>:HAVE_AV_CONFIG_H;_ISOC11_SOURCE;_FILE_OFFSET_BITS=64;_LARGEFILE_SOURCE;_USE_MATH_DEFINES;_CRT_NONSTDC_NO_WARNINGS;_WIN32_WINNT=0x0600;PIC>)
# FFmpeg relies on the optimizer dropping calls under disabled ARCH_*/HAVE_* checks; no LTO (inline asm names symbols).
target_compile_options(ffmpeg PRIVATE $<$<COMPILE_LANGUAGE:C>:/clang:-march=${GOW3_CPU_ARCH};/std:c17;-w>
    $<$<COMPILE_LANGUAGE:ASM_NASM>:-I${CMAKE_CURRENT_SOURCE_DIR}/;-I${CMAKE_CURRENT_SOURCE_DIR}/libavutil/x86/;-P${CMAKE_CURRENT_SOURCE_DIR}/config.asm>)
target_link_libraries(ffmpeg PUBLIC bcrypt)
else()
# Linux: linux/config.h and linux/config.asm come from the same configure options with --target-os=linux --enable-pthreads
# (config_components.* are identical). They are searched before the Windows ones; -fno-strict-aliasing and -fms-volatile
# keep the clang-cl semantics.
target_include_directories(ffmpeg BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/linux)
# The POSIX side of av_random_bytes (/dev/urandom) also needs avpriv_fopen_utf8; Windows uses BCrypt instead.
target_sources(ffmpeg PRIVATE libavutil/file_open.c)
target_compile_definitions(ffmpeg PRIVATE $<$<COMPILE_LANGUAGE:C>:HAVE_AV_CONFIG_H;_ISOC11_SOURCE;_FILE_OFFSET_BITS=64;_LARGEFILE_SOURCE;_POSIX_C_SOURCE=200112;_XOPEN_SOURCE=600;PIC>)
target_compile_options(ffmpeg PRIVATE $<$<COMPILE_LANGUAGE:C>:-march=${GOW3_CPU_ARCH};-std=c17;-fno-strict-aliasing;-fms-volatile;-w>
    $<$<COMPILE_LANGUAGE:ASM_NASM>:-I${CMAKE_CURRENT_SOURCE_DIR}/;-I${CMAKE_CURRENT_SOURCE_DIR}/libavutil/x86/;-P${CMAKE_CURRENT_SOURCE_DIR}/linux/config.asm>)
find_package(Threads REQUIRED)
target_link_libraries(ffmpeg PUBLIC Threads::Threads m)
endif()
