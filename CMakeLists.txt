cmake_minimum_required(VERSION 3.26)

if (WIN32)
  # Workaround for VSCode CMake Tools on Windows does not activate pyhton venv
  # Manually add .venv/Scripts to PATH
  if (EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts")
    set(ENV{PATH} "${CMAKE_SOURCE_DIR}/.venv/Scripts;$ENV{PATH}")
    unset(ENV{PYTHONHOME} CACHE)
    message(STATUS "Activated Python venv from .venv/Scripts")
  endif()

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Debug mode may not work with PyTorch on Windows. Consider using Release mode.")
  endif()
endif()

message("Build type: ${CMAKE_BUILD_TYPE}")

file(READ "${CMAKE_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENTS)
function (parse_pyproject key)
  string(REGEX MATCH "${key} = \"([^\"]+)\"" _match "${PYPROJECT_CONTENTS}")
  if (_match)
    string(REGEX REPLACE "${key} = \"([^\"]+)\"" "\\1" value "${_match}")
    message(STATUS "Found ${key} in pyproject.toml: ${value}")
    set(${key} "${value}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "${key} not found in pyproject.toml")
  endif()
endfunction()

if (NOT TORCH_CUDA_ARCH_LIST)
  parse_pyproject(TORCH_CUDA_ARCH_LIST)
endif()

if (NOT MODULE_DIR)
  parse_pyproject(MODULE_DIR)
endif()

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

project(ops LANGUAGES C CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" 
    OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH 
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
set(USE_SYSTEM_NVTX ON)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# PyTorch includes are already added
include_directories(${Python_INCLUDE_DIRS})

file(GLOB_RECURSE SRC_FILES
    csrc/*.c
    csrc/*.cpp
    csrc/*.cc
    csrc/*.cu
)

add_library(ops SHARED ${SRC_FILES})
target_link_libraries(ops ${Python_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(ops PRIVATE TORCH_EXTENSION_NAME=libops)
if (MSVC)
  target_compile_options(ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4264> $<$<COMPILE_LANGUAGE:CXX>:/wd4267>)
endif()

install(TARGETS ops LIBRARY DESTINATION ${MODULE_DIR})

# Add a custom target to copy the shared library to the python source directory
add_custom_target(copy_ops ALL DEPENDS ops)
if (WIN32)
    set(OUTPUT_FILE "${CMAKE_SOURCE_DIR}/${MODULE_DIR}/libops.pyd")
else()
    set(OUTPUT_FILE "${CMAKE_SOURCE_DIR}/${MODULE_DIR}/libops.so")
endif()

add_custom_command(
    TARGET copy_ops POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ops> ${OUTPUT_FILE}
    COMMENT "Copying shared library to python source directory"
)