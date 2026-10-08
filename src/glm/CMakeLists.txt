# GLM-5.3-Flash engine (strata-glm). Built only with CUDA; the Qwen targets do not depend on anything here.
#
# Components delivered by the other port sessions add their own CMakeLists.txt in their directory
# (kda/, mla/, experts/, mtp/): each defines a static library named strata_glm_<dir> that the engine links.

add_library(strata_glm_ops STATIC ops/matmul.cu ops/trunk_ops.cu)
target_include_directories(strata_glm_ops PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(strata_glm_ops PUBLIC strata_warnings CUDA::cudart CUDA::cublas)

foreach(_c kda mla experts mtp prefill)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_c}/CMakeLists.txt)
    add_subdirectory(${_c})
  endif()
endforeach()

add_executable(glm_matmul_parity tests/matmul_parity.cpp)
target_link_libraries(glm_matmul_parity PRIVATE strata_glm_ops strata_artifact)

# The engine: loader, shared ops, decode/prefill loops.
add_library(strata_glm_engine STATIC engine/model.cpp engine/forward.cpp engine/decode.cpp engine/routed.cu engine/r8_gpu.cu)
target_include_directories(strata_glm_engine PUBLIC ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(strata_glm_engine PUBLIC strata_glm_ops strata_glm_experts strata_artifact CUDA::cudart)

add_executable(glm_load_check tests/load_check.cpp)
target_include_directories(glm_load_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(glm_load_check PRIVATE strata_glm_engine)

add_executable(glm_trunk_parity tests/trunk_parity.cpp tests/ref_experts.cpp)
target_include_directories(glm_trunk_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(glm_trunk_parity PRIVATE strata_glm_engine strata_glm_kda strata_glm_mla ggml-cpu ggml-base)

# Generation session (graphs + MTP speculative decoding) on top of the engine and the component libraries.
add_library(strata_glm_session STATIC engine/session.cpp engine/prefix_cache.cpp)
target_include_directories(strata_glm_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(strata_glm_session PUBLIC strata_glm_engine strata_glm_kda strata_glm_mla strata_glm_mtp
                      strata_glm_prefill)

add_executable(glm_decode tests/decode_loop.cpp tests/ref_experts.cpp)
target_include_directories(glm_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(glm_decode PRIVATE strata_glm_session strata_glm_prefill strata_glm_engine strata_glm_kda
                      strata_glm_mla ggml-cpu ggml-base)

add_executable(glm_eval tests/eval.cpp tests/ref_experts.cpp)
target_include_directories(glm_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
target_link_libraries(glm_eval PRIVATE strata_glm_engine strata_glm_kda strata_glm_mla ggml-cpu ggml-base)

# Miss path error vs F64 on the real experts, AVX2 vs VNNI (CPU only)
add_executable(glm_miss_error experts/miss_error.cpp)
target_link_libraries(glm_miss_error PRIVATE strata_glm_experts strata_artifact)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(glm_miss_error PRIVATE -ffp-contract=off)
endif()

# The engine behind Strata's server (serve/server.py --engine strata): the `strata --serve` line protocol.
add_executable(glm_serve serve.cpp)
target_link_libraries(glm_serve PRIVATE strata_glm_session)
# its INFO line (header-only, no GPU)
add_executable(glm_serve_info_test tests/serve_info_test.cpp)
add_test(NAME glm_serve_info_test COMMAND glm_serve_info_test)
# its default --threads (pool.hpp, header-only, no GPU)
add_executable(glm_pool_cores_test tests/pool_cores_test.cpp)
target_include_directories(glm_pool_cores_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
add_test(NAME glm_pool_cores_test COMMAND glm_pool_cores_test)

# glm_serve's page-locked budget arithmetic (engine/pinned_budget.hpp), no GPU
add_executable(glm_pinned_budget_test tests/pinned_budget_test.cpp)
target_include_directories(glm_pinned_budget_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)

add_executable(glm_ops_bench tests/ops_bench.cpp)
target_link_libraries(glm_ops_bench PRIVATE strata_glm_ops)
