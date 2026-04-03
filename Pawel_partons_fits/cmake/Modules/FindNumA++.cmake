# - Try to find NumA++ (both in system folders if it was installed, and in the current workspace)
#
# Once done this will define
#
#  NUMA_FOUND - system has NumA++ lib
#  NUMA_LIBRARIES - the path of the library
#  NUMA_INCLUDE_DIR - the include directory

if (NUMA_INCLUDE_DIR AND NUMA_LIBRARIES)

  # in cache already
  set(NUMA_FOUND TRUE)

else (NUMA_INCLUDE_DIR AND NUMA_LIBRARIES)

  find_path(NUMA_INCLUDE_DIR NAMES NumA/integration/Integrator.h NumA++/integration/Integrator.h
      PATHS
      ${CMAKE_INSTALL_PREFIX}
      ${CMAKE_SOURCE_DIR}/../numa
      ${CMAKE_SOURCE_DIR}/../NumA
      ${CMAKE_SOURCE_DIR}/../NumA++
      ${CMAKE_SOURCE_DIR}/numa
      ${CMAKE_SOURCE_DIR}/NumA
      ${CMAKE_SOURCE_DIR}/NumA++
      ${CMAKE_SOURCE_DIR}
      PATH_SUFFIXES include/PARTONS include
    )
    
  find_library(NUMA_LIBRARIES
                     NAMES NumA NumA++
                     PATHS
                     ${CMAKE_INSTALL_PREFIX}
                     ${CMAKE_SOURCE_DIR}/../NumA++
                     ${CMAKE_SOURCE_DIR}/../NumA
                     ${CMAKE_SOURCE_DIR}/../numa
                     ${CMAKE_SOURCE_DIR}/NumA++
                     ${CMAKE_SOURCE_DIR}/NumA
                     ${CMAKE_SOURCE_DIR}/numa
                     ${CMAKE_SOURCE_DIR}
                     PATH_SUFFIXES lib/PARTONS lib64/PARTONS bin/PARTONS lib lib64 bin)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_INCLUDE_DIR NUMA_LIBRARIES)

endif(NUMA_INCLUDE_DIR AND NUMA_LIBRARIES)

