cmake_minimum_required(VERSION 3.26)

message("Build type: ${CMAKE_BUILD_TYPE}")

set(MODULE_DIR "example")

if (NOT TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "8.6")
endif()

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

if (MSVC)
  # suppress C4624 for PyTorch header files
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4624")
endif()

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