# EquiShifter
Tool for transforming existing spherical equirectangular images and rotating or otherwise warping them to yield new, aligned versions.

Useful for panoramas that are not aligned stereoscopically or otherwise need warping.

## Dependencies (Windows)
- Freeglut (64bit DLL and Lib)
- Glew32 (64bit DLL and Lib)
- GLM (0.9.9 used at the time of writing)
- libpng (NuGet installation recommended)
- libtiff (NuGet installation recommended)
- Visual Studio 2015 (v140) Toolkit (Required for libpng/libtiff compatibility)

## Usage
w/a/s/d/q/e will warp the panorama, scrollwheel zooms in/out, arrow keys move the panorama around the screen.

F5 will save the panorama as it appears on the screen.

This program does not copy anything out of the framebuffer, the viewscreen is rendered by fragment shader manipulations and all image data is retained in CPU-side memory throughout.  The warping math is repeated CPU-side when saving, preventing any expensive calls to pull information out of the GPU.