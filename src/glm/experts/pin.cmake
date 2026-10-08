if(STRATA_ENABLE_CUDA)
  add_executable(glm_pin_probe ${CMAKE_CURRENT_LIST_DIR}/pin_probe.cpp)
  target_link_libraries(glm_pin_probe PRIVATE CUDA::cudart)
endif()
