@echo off
REM Build script for Stable Fluids simulation
REM Requires: CMake, Visual Studio or MinGW, GLFW, and glad

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
REM Adjust paths below to match your setup:
REM   -DGLFW_INCLUDE_DIR=path/to/glfw/include
REM   -DGLFW_LIBRARY=path/to/glfw3.lib
REM   -DGLAD_DIR=path/to/glad

cmake .. -G "Visual Studio 17 2022" -A x64 -DGLFW_INCLUDE_DIR=C:\JangaFX\Claude\Fluid Simulation\glfw\include\GLFW -DGLFW_LIBRARY=C:\JangaFX\Claude\Fluid Simulation\glfw\lib-vc2022\glfw3.lib -DGLAD_DIR=glad

REM Build
cmake --build . --config Release

echo.
echo Build complete! Run build\Release\StableFluids.exe
echo Make sure shaders folder is in the same directory as the executable.
pause
