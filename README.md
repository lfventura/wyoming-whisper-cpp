# wyoming-whisper-cpp

Simple STT server using wyoming protocol and whisper.cpp.

"""
This code is presented as-is, without any warranties or guarantees of functionality. 
It serves as a simple implementation that can surely be improved or optimized 
based on specific requirements or use cases. Users are encouraged to review, 
test, and modify the code as needed to suit their needs.
"""

In the project ther's the code to build a docker image using vulkan acceleration for devices that use the mesa drivers. To use other frameworks to accelerate the workload, follow instructions on the whisper.cpp github to compile whisper.cpp and copy the necessary files. New dockerfiles for different frameworks are welcome. 

## Project Components

- **simple-server**: lightweight C++ server based on the stream example provided by whisper.cpp.
- **wyoming_whisper_cpp**: python module implementing a Wyoming protocol server.


## Build Docker image

Build docker image:

```shell
docker build -f dockerfiles/ubuntu-vulkan-mesa -t wyoming-whisper-cpp:vulkan .
```
Prebuild images published on dockerhub: https://hub.docker.com/r/franc2002/wyoming-whisper-cpp

If you don't want to use docker, just follow all the command used in the dockerfile 

## Run with docker-compose

```shell
docker-compose up -d
```

## Fork changes (lfventura)

- `dockerfiles/ubuntu-vulkan-mesa` pins whisper.cpp to a release tag (`ARG WHISPER_CPP_REF`) instead of building whatever is on upstream HEAD, and clones from `ggml-org/whisper.cpp`.
- Installs `spirv-headers`, which newer whisper.cpp releases require for the Vulkan backend (`find_package(SPIRV-Headers CONFIG REQUIRED)`); without it `cmake -B build` fails.
- Since whisper.cpp v1.9 every shared library is built into `build/bin`; the old `COPY` globs matched nothing and the image shipped without `libwhisper`/`libggml*` (BuildKit does not fail on an empty glob). The libraries are now copied from `build/bin`, and the build runs `ldd` on `simple-server` and fails if anything is missing.
- Sets `HOME=/data` and `XDG_CACHE_HOME=/data/.cache` so the container can run as a non-root UID and still persist the Mesa shader cache.

To update whisper.cpp, bump `WHISPER_CPP_REF` and push: the GHCR workflow rebuilds the image.

## Prebuilt image (GHCR)

`.github/workflows/ghcr-vulkan-mesa.yml` builds `dockerfiles/ubuntu-vulkan-mesa` on every push to `main` that touches the image sources and publishes:

- `ghcr.io/lfventura/wyoming-whisper-cpp:vulkan` (latest build)
- `ghcr.io/lfventura/wyoming-whisper-cpp:vulkan-<whisper.cpp tag>` (e.g. `vulkan-v1.9.5`)
- `ghcr.io/lfventura/wyoming-whisper-cpp:vulkan-sha-<commit>`

## TrueNAS SCALE (Install via YAML)

`truenas-custom-app.yaml` is a ready-to-paste compose file for Apps -> Discover -> Custom App -> Install via YAML. It pulls the GHCR image, runs as a dedicated non-root UID/GID and passes through a single GPU. Edit the UID/GID, render/video GIDs, device nodes and data path to match your host (see the comments at the top of the file).
