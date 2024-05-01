

find_package(Eigen3 CONFIG REQUIRED)

add_library(lbm
  Boltzman.cpp
  Lattice.cpp
)

target_link_libraries(lbm
  PRIVATE 
  fmt::fmt
)


target_include_directories(lbm 
  PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)