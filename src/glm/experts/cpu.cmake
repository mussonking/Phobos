# Include after ggml-base is defined. This target is CPU-only and never inherits
# the Qwen kernel's AVX512VBMI flags. GCC/Clang use per-function ISA attributes.
if(NOT TARGET ggml-base)
  message(FATAL_ERROR "GLM IQ4 CPU experts require the ggml-base target")
endif()
add_library(strata_glm_experts_cpu STATIC
  ${CMAKE_CURRENT_LIST_DIR}/iq4_cpu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/experts_cpu.cpp)
target_compile_features(strata_glm_experts_cpu PUBLIC cxx_std_17)
target_include_directories(strata_glm_experts_cpu PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../../../include)
target_link_libraries(strata_glm_experts_cpu PUBLIC ggml-base)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Preserve a separate rounding of the weight scale before FMA with F32 x;
  # the scalar reference remains a sequential non-fused F32 matmul.
  target_compile_options(strata_glm_experts_cpu PRIVATE -ffp-contract=off)
endif()
add_executable(glm_cpu_parity ${CMAKE_CURRENT_LIST_DIR}/cpu_parity.cpp)
target_link_libraries(glm_cpu_parity PRIVATE strata_glm_experts_cpu)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(glm_cpu_parity PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
  add_test(NAME glm_cpu_iq4_parity COMMAND glm_cpu_parity --quick)
endif()
