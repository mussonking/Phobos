# Optional byte-preserving eight-row IQ4 layout; include after vnni.cmake.
if(NOT TARGET strata_glm_experts_vnni OR NOT TARGET ggml-base)
  message(FATAL_ERROR "GLM R8 experts require VNNI experts and ggml-base")
endif()
add_library(strata_glm_experts_r8 STATIC ${CMAKE_CURRENT_LIST_DIR}/r8.cpp)
target_compile_features(strata_glm_experts_r8 PUBLIC cxx_std_17)
target_include_directories(strata_glm_experts_r8 PUBLIC
  ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../../include
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../third_party/ggml)
target_link_libraries(strata_glm_experts_r8 PUBLIC strata_glm_experts_vnni ggml-base)
add_executable(glm_r8_parity ${CMAKE_CURRENT_LIST_DIR}/r8_parity.cpp)
target_include_directories(glm_r8_parity PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../../third_party/ggml)
target_link_libraries(glm_r8_parity PRIVATE strata_glm_experts_r8 strata_glm_experts_cpu)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(strata_glm_experts_r8 PRIVATE -ffp-contract=off)
  target_compile_options(glm_r8_parity PRIVATE -ffp-contract=off)
endif()
if(BUILD_TESTING)
  # GLM CPU parity can run without the unpublished global microbench sources.
  # With STRATA_BUILD_TESTS=OFF, invoke CTest in build/src/glm/experts.
  enable_testing()
  add_test(NAME glm_r8_quantized_parity COMMAND glm_r8_parity --quick)
endif()
