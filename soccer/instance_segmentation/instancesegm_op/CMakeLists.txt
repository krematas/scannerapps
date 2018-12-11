# To build your custom op, you can either use our CMake convenience methods
# or do it the slightly harder way in normal Make (see the Makefile).

cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

execute_process(
  OUTPUT_VARIABLE SCANNER_CMAKE_PATH
  COMMAND python3 -c "import scannerpy.stdlib.build_flags as b; b.print_cmake()")
include(${SCANNER_CMAKE_PATH})

# build_op will create a shared library called lib${LIB_NAME}.so that builds
# from all of the CPP_SRCS. You can also optionally specify a PROTO_SRC that
# points to a Protobuf file and will generate the C++ and Python bindings.
build_op(
  LIB_NAME instancesegm_op
  CPP_SRCS instancesegm_op.cpp
  PROTO_SRC instancesegm.proto)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Eigen
set(MIN_EIGEN_VERSION 3.2.0)
find_package(Eigen ${MIN_EIGEN_VERSION} REQUIRED)
if (EIGEN_FOUND)
    message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

# The library specified in build_op is a normal CMake target, so you can use all
# the normal CMake functions with it.
find_package(OpenCV REQUIRED)
find_package(Boost 1.45.0 COMPONENTS filesystem regex program_options)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(instancesegm_op PUBLIC "${OpenCV_LIBRARIES}")
