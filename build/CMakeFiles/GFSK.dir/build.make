# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "D:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\Users\Gtupxx\Desktop\GFSK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\Users\Gtupxx\Desktop\GFSK\build

# Include any dependencies generated for this target.
include CMakeFiles/GFSK.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GFSK.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GFSK.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GFSK.dir/flags.make

CMakeFiles/GFSK.dir/main.cpp.obj: CMakeFiles/GFSK.dir/flags.make
CMakeFiles/GFSK.dir/main.cpp.obj: CMakeFiles/GFSK.dir/includes_CXX.rsp
CMakeFiles/GFSK.dir/main.cpp.obj: D:/Users/Gtupxx/Desktop/GFSK/main.cpp
CMakeFiles/GFSK.dir/main.cpp.obj: CMakeFiles/GFSK.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\Users\Gtupxx\Desktop\GFSK\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GFSK.dir/main.cpp.obj"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GFSK.dir/main.cpp.obj -MF CMakeFiles\GFSK.dir\main.cpp.obj.d -o CMakeFiles\GFSK.dir\main.cpp.obj -c D:\Users\Gtupxx\Desktop\GFSK\main.cpp

CMakeFiles/GFSK.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GFSK.dir/main.cpp.i"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\Users\Gtupxx\Desktop\GFSK\main.cpp > CMakeFiles\GFSK.dir\main.cpp.i

CMakeFiles/GFSK.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GFSK.dir/main.cpp.s"
	D:\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\Users\Gtupxx\Desktop\GFSK\main.cpp -o CMakeFiles\GFSK.dir\main.cpp.s

# Object files for target GFSK
GFSK_OBJECTS = \
"CMakeFiles/GFSK.dir/main.cpp.obj"

# External object files for target GFSK
GFSK_EXTERNAL_OBJECTS =

GFSK.exe: CMakeFiles/GFSK.dir/main.cpp.obj
GFSK.exe: CMakeFiles/GFSK.dir/build.make
GFSK.exe: D:/Python312/libs/python312.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/fftw3.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/fftw3f.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/fftw3l.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/opencv_ml4d.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/opencv_dnn4d.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/opencv_flann4d.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/tinyfiledialogs.lib
GFSK.exe: D:/Python312/libs/python312.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/opencv_imgproc4d.lib
GFSK.exe: D:/vcpkg/installed/x64-windows/debug/lib/opencv_core4d.lib
GFSK.exe: CMakeFiles/GFSK.dir/linkLibs.rsp
GFSK.exe: CMakeFiles/GFSK.dir/objects1.rsp
GFSK.exe: CMakeFiles/GFSK.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=D:\Users\Gtupxx\Desktop\GFSK\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable GFSK.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\GFSK.dir\link.txt --verbose=$(VERBOSE)
	C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -noprofile -executionpolicy Bypass -file D:/vcpkg/scripts/buildsystems/msbuild/applocal.ps1 -targetBinary D:/Users/Gtupxx/Desktop/GFSK/build/GFSK.exe -installedDir D:/vcpkg/installed/x64-windows/debug/bin -OutVariable out

# Rule to build all files generated by this target.
CMakeFiles/GFSK.dir/build: GFSK.exe
.PHONY : CMakeFiles/GFSK.dir/build

CMakeFiles/GFSK.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\GFSK.dir\cmake_clean.cmake
.PHONY : CMakeFiles/GFSK.dir/clean

CMakeFiles/GFSK.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\Users\Gtupxx\Desktop\GFSK D:\Users\Gtupxx\Desktop\GFSK D:\Users\Gtupxx\Desktop\GFSK\build D:\Users\Gtupxx\Desktop\GFSK\build D:\Users\Gtupxx\Desktop\GFSK\build\CMakeFiles\GFSK.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/GFSK.dir/depend

