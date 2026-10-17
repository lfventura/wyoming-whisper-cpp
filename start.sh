#!/bin/sh

export LD_LIBRARY_PATH=/wd/whisper.cpp/build/bin:/wd/whisper.cpp/build/src:/wd/whisper.cpp/build/ggml/src:/wd/whisper.cpp/build/ggml/src/ggml-vulkan:$LD_LIBRARY_PATH

python3 -m wyoming_whisper_cpp \
    --whisper-cpp-dir ./whisper.cpp/ \
    --uri 'tcp://0.0.0.0:10300' \
    --data-dir /data \
    --download-dir /data \
    "$@"
