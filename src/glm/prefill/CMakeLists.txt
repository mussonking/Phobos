# GLM-5.3-Flash prompt prefill with streamed experts (session kda).
add_executable(glm_pcie_probe pcie_probe.cu)
target_link_libraries(glm_pcie_probe PRIVATE strata_warnings CUDA::cudart)
add_executable(glm_ctx_gather_probe ctx_gather_probe.cu)
target_link_libraries(glm_ctx_gather_probe PRIVATE strata_warnings CUDA::cudart)

add_library(strata_glm_prefill STATIC prefill_experts.cu prefill_cpu.cpp prefill_moe.cu prefill_prompt.cpp)
target_include_directories(strata_glm_prefill PUBLIC ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}
                           ${PROJECT_SOURCE_DIR}/src/glm/engine)
target_link_libraries(strata_glm_prefill PUBLIC strata_warnings strata_glm_engine strata_glm_experts strata_glm_kda
                      strata_glm_mla strata_glm_mtp CUDA::cudart CUDA::cublas)

add_executable(glm_prefill_bench prefill_bench.cu)
target_link_libraries(glm_prefill_bench PRIVATE strata_glm_prefill strata_glm_experts_gpu CUDA::cudart)

add_executable(glm_kv_quality kv_quality.cu)
target_link_libraries(glm_kv_quality PRIVATE strata_glm_prefill strata_glm_engine strata_glm_kda strata_glm_mla
                      CUDA::cudart)

add_executable(glm_prefill_test prefill_test.cpp)
target_link_libraries(glm_prefill_test PRIVATE strata_glm_prefill strata_glm_engine strata_glm_kda strata_glm_mla strata_glm_mtp
                      CUDA::cudart)
