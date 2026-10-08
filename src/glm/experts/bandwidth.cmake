# Opt-in CPU-only benchmark. Include after cpu.cmake and vnni.cmake.
# No root build integration, test registration, or global ISA flags.
if(NOT TARGET strata_glm_experts_cpu OR NOT TARGET strata_glm_experts_vnni)
  message(FATAL_ERROR "GLM bandwidth requires cpu.cmake and vnni.cmake first")
endif()
find_package(Threads REQUIRED)
add_executable(glm_cpu_bandwidth ${CMAKE_CURRENT_LIST_DIR}/cpu_bandwidth.cpp)
target_compile_features(glm_cpu_bandwidth PRIVATE cxx_std_17)
target_link_libraries(glm_cpu_bandwidth PRIVATE
  strata_glm_experts_cpu strata_glm_experts_vnni Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(glm_cpu_bandwidth PRIVATE -ffp-contract=off)
endif()
