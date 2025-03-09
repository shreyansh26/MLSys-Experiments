cmake_minimum_required(VERSION 3.19) # Specify the minimum CMake version required
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA) # Define the project name and the languages used (C++ and CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Enable the generation of compile_commands.json for tooling support

find_package(CUDA REQUIRED) # Find and include the CUDA package, marking it as required

# Ensure CUDA is available
include(CheckLanguage) # Include the module to check for language support
check_language(CUDA) # Verify that CUDA language support is available

set(CMAKE_CXX_STANDARD 20) # Set the C++ standard to C++20
set(CUDA_COMPUTE_CAPABILITY 90) # Define the CUDA compute capability version (e.g., 9.0)

# In debug mode, add debug symbols to device code
# IF (Configuration is Debug) AND (Language is CUDA)
#     THEN add these compiler flags: "-G" and "-src-in-ptx"
# ELSE
#     Don't add these flags
# This disables most optimizations and affects performance
# add_compile_options(
#     "$<$<AND:                           
#         $<CONFIG:Debug>,                
#         $<COMPILE_LANGUAGE:CUDA>        
#     >:
#     -G;                                 
#     -src-in-ptx                         
#     >"
# ) # Add compile options for CUDA in Debug configuration
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v") # Optionally add PTXAS verbosity options

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS}) # Include CUDA's include directories
include_directories(${PROJECT_SOURCE_DIR}/src) # Include the project's source directory

# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC) # Gather all source files from the src directory into the SRC variable

message(STATUS "Source files found: ${SRC}")

# Set Precision option
set(BUILD_PRECISION "FP32" CACHE STRING "Precision to build (FP32 or FP16 or BF16)")
set_property(CACHE BUILD_PRECISION PROPERTY STRINGS FP32 FP16 BF16)

if(BUILD_PRECISION STREQUAL "FP32")
    add_executable(sgemm_fp32 sgemm_fp32.cu ${SRC})
    set_target_properties(sgemm_fp32 PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
    target_link_libraries(sgemm_fp32 ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
elseif(BUILD_PRECISION STREQUAL "FP16")
    add_executable(sgemm_fp16 sgemm_fp16.cu ${SRC})
    set_target_properties(sgemm_fp16 PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
    target_link_libraries(sgemm_fp16 ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
elseif(BUILD_PRECISION STREQUAL "BF16")
    add_executable(sgemm_bf16 sgemm_bf16.cu ${SRC})
    set_target_properties(sgemm_bf16 PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
    target_link_libraries(sgemm_bf16 ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
else()
    message(FATAL_ERROR "Unknown PRECISION: ${BUILD_PRECISION}. Use FP32 or FP16 or BF16.")
endif()