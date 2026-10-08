# GLM-5.3-Flash KDA attention branch (session kda).
add_library(strata_glm_kda STATIC kda.cu kda_chunk.cu)
target_include_directories(strata_glm_kda PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(strata_glm_kda PUBLIC strata_glm_ops strata_warnings CUDA::cudart)

add_executable(kda_test kda_test.cpp)
target_include_directories(kda_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(kda_test PRIVATE strata_glm_kda strata_artifact CUDA::cudart)
