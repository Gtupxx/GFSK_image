cmake_minimum_required(VERSION 3.5.0)
project(GFSK VERSION 0.1.0 LANGUAGES C CXX)

# Add the executable
# add_executable(GFSK main.cpp)
add_executable(GFSK GFSK.cpp)

# Find and include matplotlibcpp
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
if (MATPLOTLIB_CPP_INCLUDE_DIRS)
    target_include_directories(GFSK PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Could not find matplotlibcpp. Please ensure it is installed and the path is correct.")
endif()

# Find and include Python3 with NumPy
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
if (Python3_FOUND)
    target_include_directories(GFSK PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(GFSK PRIVATE Python3::Python Python3::NumPy)
else()
    message(FATAL_ERROR "Could not find Python3 or NumPy. Please ensure they are installed.")
endif()

  # this is heuristically generated, and may not be correct
  find_package(FFTW3 CONFIG REQUIRED)
  target_link_libraries(GFSK PRIVATE FFTW3::fftw3)

  find_package(FFTW3f CONFIG REQUIRED)
  target_link_libraries(GFSK PRIVATE FFTW3::fftw3f)

  find_package(FFTW3l CONFIG REQUIRED)
  target_link_libraries(GFSK PRIVATE FFTW3::fftw3l)

# this is heuristically generated, and may not be correct
find_package(OpenCV CONFIG REQUIRED)
# note: 10 additional targets are not displayed.
target_link_libraries(GFSK PRIVATE opencv_ml opencv_dnn opencv_core opencv_flann)

  # this is heuristically generated, and may not be correct
  find_package(tinyfiledialogs CONFIG REQUIRED)
  target_link_libraries(GFSK PRIVATE tinyfiledialogs::tinyfiledialogs)