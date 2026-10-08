# Include after ggml-base is defined; standalone from Qwen kernels/MMQ.
if(STRATA_ENABLE_CUDA)
  add_library(strata_glm_experts_gpu STATIC
    ${CMAKE_CURRENT_LIST_DIR}/experts_gpu.cu)
  target_include_directories(strata_glm_experts_gpu PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(strata_glm_experts_gpu PUBLIC CUDA::cudart)
  target_compile_features(strata_glm_experts_gpu PUBLIC cxx_std_17)
  set_target_properties(strata_glm_experts_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  # Do not inherit -use_fast_math: the oracle contract uses F32 activations and expf.
  target_compile_options(strata_glm_experts_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>)
  if(TARGET ggml-base)
    add_executable(glm_experts_gpu_parity ${CMAKE_CURRENT_LIST_DIR}/gpu_parity.cpp)
    target_link_libraries(glm_experts_gpu_parity PRIVATE strata_glm_experts_gpu ggml-base)
    target_compile_features(glm_experts_gpu_parity PRIVATE cxx_std_17)
    target_compile_options(glm_experts_gpu_parity PRIVATE
      $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off>)
  endif()
endif()
