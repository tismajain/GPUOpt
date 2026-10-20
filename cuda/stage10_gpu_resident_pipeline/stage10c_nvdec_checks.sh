#!/usr/bin/env bash
# Extracted from the Stage 10C notebook shell cells.

ffmpeg -hide_banner -decoders 2>/dev/null | grep -Ei "cuda|cuvid|nvdec|h264|hevc" | head -50
ffmpeg -hide_banner -hwaccels
ffprobe -v error \
-show_entries stream=codec_name,codec_long_name,width,height,r_frame_rate \
-of default=noprint_wrappers=1 \
"/kaggle/input/datasets/tismajain/boxing/GH088416.mp4"
ffmpeg -hide_banner \
-c:v h264_cuvid \
-i "/kaggle/input/datasets/tismajain/boxing/GH088416.mp4" \
-frames:v 100 \
-f null - 2>&1 | tail -30
# CPU decode benchmark
ffmpeg -benchmark -hide_banner \
-i "/kaggle/input/datasets/tismajain/boxing/GH088416.mp4" \
-frames:v 300 \
-f null - 2>&1 | tail -20
# NVDEC benchmark
ffmpeg -benchmark -hide_banner \
-c:v h264_cuvid \
-i "/kaggle/input/datasets/tismajain/boxing/GH088416.mp4" \
-frames:v 300 \
-f null - 2>&1 | tail -20
ffprobe -v error \
-select_streams v:0 \
-show_entries stream=color_range,color_space,color_transfer,color_primaries,pix_fmt \
-of default=noprint_wrappers=1 \
"/kaggle/input/datasets/tismajain/boxing/GH088416.mp4"