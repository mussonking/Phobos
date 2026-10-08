# GLM-5.3-Flash MLA attention + DSA indexer. Included by src/glm/CMakeLists.txt.
add_library(strata_glm_mla STATIC mla.cu)
target_include_directories(strata_glm_mla PUBLIC ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(strata_glm_mla PUBLIC strata_glm_ops strata_warnings CUDA::cudart)

add_executable(glm_mla_parity mla_parity.cpp)
target_link_libraries(glm_mla_parity PRIVATE strata_glm_mla strata_artifact)

add_executable(glm_mla_bench mla_bench.cpp)
target_link_libraries(glm_mla_bench PRIVATE strata_glm_mla strata_artifact)

add_executable(glm_mla_select_test mla_select_test.cpp)
target_link_libraries(glm_mla_select_test PRIVATE strata_glm_mla strata_artifact)
