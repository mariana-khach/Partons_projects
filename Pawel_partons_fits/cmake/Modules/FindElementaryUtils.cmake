# - Try to find ElementaryUtils (both in system folders if it was installed, and in the current workspace)
#
# Once done this will define
#
#  ELEMENTARY_UTILS_FOUND - system has Elementary_Utils lib
#  ELEMENTARY_UTILS_LIBRARIES - the path of the library
#  ELEMENTARY_UTILS_INCLUDE_DIR - the include directory

if (ELEMENTARY_UTILS_INCLUDE_DIR AND ELEMENTARY_UTILS_LIBRARIES)

  # in cache already
  set(ELEMENTARY_UTILS_FOUND TRUE)

else (ELEMENTARY_UTILS_INCLUDE_DIR AND ELEMENTARY_UTILS_LIBRARIES)

  find_path(ELEMENTARY_UTILS_INCLUDE_DIR ElementaryUtils/PropertiesManager.h
      PATHS
      ${CMAKE_INSTALL_PREFIX}
      ${CMAKE_SOURCE_DIR}/../ElementaryUtils
      ${CMAKE_SOURCE_DIR}/ElementaryUtils
      ${CMAKE_SOURCE_DIR}/../elementary-utils
      ${CMAKE_SOURCE_DIR}/elementary-utils
      ${CMAKE_SOURCE_DIR}
      PATH_SUFFIXES include/PARTONS include
    )
    
  find_library(ELEMENTARY_UTILS_LIBRARIES
                     NAMES ElementaryUtils
                     PATHS
                     ${CMAKE_INSTALL_PREFIX}
                     ${CMAKE_SOURCE_DIR}/../ElementaryUtils
                     ${CMAKE_SOURCE_DIR}/ElementaryUtils
                     ${CMAKE_SOURCE_DIR}/../elementary-utils
                     ${CMAKE_SOURCE_DIR}/elementary-utils
                     ${CMAKE_SOURCE_DIR}
                     PATH_SUFFIXES lib/PARTONS lib64/PARTONS bin/PARTONS lib lib64 bin)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(ELEMENTARY_UTILS DEFAULT_MSG ELEMENTARY_UTILS_INCLUDE_DIR ELEMENTARY_UTILS_LIBRARIES)

endif(ELEMENTARY_UTILS_INCLUDE_DIR AND ELEMENTARY_UTILS_LIBRARIES)

