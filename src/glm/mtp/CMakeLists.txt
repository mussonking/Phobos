# GLM-5.3-Flash MTP drafter, NextN block 45 (session kda).
add_library(strata_glm_mtp STATIC mtp.cu)
target_include_directories(strata_glm_mtp PUBLIC ${PROJECT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../engine)
target_link_libraries(strata_glm_mtp PUBLIC strata_glm_engine strata_glm_mla strata_glm_ops strata_warnings CUDA::cudart)

add_executable(mtp_test mtp_test.cpp ../tests/ref_experts.cpp)
target_include_directories(mtp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../engine ${CMAKE_CURRENT_SOURCE_DIR}/../tests
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(mtp_test PRIVATE strata_glm_mtp CUDA::cudart ggml-cpu ggml-base)
