## Equirectangular to cubemap converter
Originally based on [this stackoverflow post](https://stackoverflow.com/questions/29678510/convert-21-equirectangular-panorama-to-cube-map) and [this C++ implementation](https://github.com/denivip/panorama)

Reworked to run on CUDA and use 4x4 trilinear sampling instead of 2x2 bilinear sampling

### Compilation
#### Windows
- Install libpng through NuGet
- Install CUDA 9.2 toolkit
- Install [MSVC v140](https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/) and switch to that (Properties -> General -> Platform Toolkit)
- Add $(CUDA_INC_PATH) to VC++ Include Directories and CUDA/C++ custom CUDA path

Intellisense will spit some errors about you, specifically about Kernel calls and CUDA math code.  Ignore it, NVCC will link through MSVC and take care of it.

Usage: ```Panorama.exe -i <Input File> -o <Output Prefix> -r <Cubeface output resolution> [-c]```

The -c flag will enable CUDA.  Without this flag on Windows, the process will be single threaded.

TODO: Get TBB working on Windows, get CUDA compiling on Linux

#### Linux
TODO