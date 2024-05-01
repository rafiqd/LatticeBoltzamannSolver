cmake_minimum_required(VERSION 3.12)


set(VCPKG_ROOT C:/Users/Rafiq/Documents/code/vcpkg)
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

project(lattice-boltzman-fluid-sim)

find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_subdirectory(cnpy)
add_subdirectory(lbm)

add_executable(lbm-fluid-sim
  main.cpp
)



target_link_libraries(lbm-fluid-sim 
  PRIVATE 
  fmt::fmt
  lbm
  cnpy
)

target_include_directories(lbm-fluid-sim  PRIVATE ${MATPLOTLIB_CPP_INCLUDE_DIRS})