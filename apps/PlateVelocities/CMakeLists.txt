if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/output)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output")
endif()

add_executable( PlateVelocityDemo
                PlateVelocityDemo.cpp )

target_include_directories(PlateVelocityDemo PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/apps/mantlecirculation/ )

target_link_libraries(PlateVelocityDemo
    PRIVATE terra
)

