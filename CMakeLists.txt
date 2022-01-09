cmake_minimum_required (VERSION 3.16)
project(ceres-factors)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_TESTS "Build Tests" ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)
find_package(manif-geom-cpp REQUIRED)

add_library(ceres-factors INTERFACE)
target_include_directories(ceres-factors INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(ceres-factors  
    INTERFACE
    Eigen3::Eigen
    Ceres::ceres
    manif-geom-cpp
)

set(UNIT_TEST unit-tests)
add_executable(${UNIT_TEST}
    tests/MainTest.cpp
    tests/FactorTests.cpp
)
target_link_libraries(${UNIT_TEST}
    ceres-factors
    boost_unit_test_framework
)
add_test(NAME ${UNIT_TEST} COMMAND ${UNIT_TEST})

add_custom_command(
    TARGET ${UNIT_TEST}
    COMMENT "Run unit tests"
    POST_BUILD
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${UNIT_TEST}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/ceres-factorsConfigVersion.cmake"
    VERSION 0.0
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS ceres-factors
    EXPORT ceres-factorsTargets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/ceres-factorsConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/ceres-factorsConfig.cmake"
    INSTALL_DESTINATION lib/cmake/ceres-factors
)

install(EXPORT ceres-factorsTargets DESTINATION lib/cmake/ceres-factors)
install(FILES "${PROJECT_BINARY_DIR}/ceres-factorsConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/ceres-factorsConfig.cmake"
        DESTINATION lib/cmake/ceres-factors)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)