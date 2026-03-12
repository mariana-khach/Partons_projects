# BASIC DEFINITIONS ========================================================================

# define minimum version of cmake
cmake_minimum_required(VERSION 3.5)

# define project name and its language
project(DVCS_analysis CXX)

# define c++ standard and issue all the warning demanded by this standard
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_definitions(-std=c++17 -pedantic -Wno-vla-extension)
else()
	add_definitions(-std=c++17 -pedantic -Wno-vla-extension -fext-numeric-literals)
endif()
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED CMAKE_MACOSX_RPATH)
	set(CMAKE_MACOSX_RPATH 0)
endif()

# FIND LIBRARIES ===========================================================================

# find libraries: additional modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# find libraries: ElementaryUtils
find_package(ElementaryUtils REQUIRED)

# find libraries: NumA++
find_package(NumA++ REQUIRED)

# find libraries: PARTONS
find_package(PARTONS REQUIRED)

# find libraries: SFML
find_package(SFML COMPONENTS system REQUIRED)

# find libraries: CLN
find_package(CLN REQUIRED)

# find libraries: GSL
find_package(GSL REQUIRED)

# find libraries: Apfel++
find_package(Apfel++ REQUIRED)

# find libraries: LHAPDF
find_package(LHAPDF REQUIRED)

# find libxml2
find_package(LibXml2 REQUIRED)

# find libraries: Torch (libtorch)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

# directories containing headers
include_directories(${ElementaryUtils_INCLUDE_DIR} ${NumA++_INCLUDE_DIR} ${PARTONS_INCLUDE_DIR} ${SFML_INCLUDE_DIR} ${CLN_INCLUDE_DIR} ${GSL_INCLUDE_DIRS} ${Apfel++_INCLUDE_DIR} ${LHAPDF_INCLUDE_DIR} ${LIBXML2_INCLUDE_DIRS})

# FINALIZE ==================================================================================

# generate list of source files
file(

    GLOB_RECURSE

    source_files

    src/*
)

# exclude dcgan.cpp from the main executable (it has its own main)
list(REMOVE_ITEM source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/dcgan.cpp)

# exclude ObsCalc_CFFNNReplicas.cpp from the main executable (it has its own main)
list(REMOVE_ITEM source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/ObsCalc_CFFNNReplicas.cpp)

# define target executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(

	DVCS_analysis

    ${source_files}
		src/Compute_obs.cpp
		include/Compute_obs.h
)

# define libraries to be linked
target_link_libraries(

	DVCS_analysis

    ${SFML_LIBRARIES}

	${CLN_LIBRARIES}

	${ElementaryUtils_LIBRARIES}

	${NumA++_LIBRARIES}

	${PARTONS_LIBRARIES}

	${GSL_LIBRARIES}

	${Apfel++_LIBRARIES}

	${LHAPDF_LIBRARIES}

	${LIBXML2_LIBRARIES}
)

# ObsCalc_CFFNNReplicas executable
add_executable(ObsCalc_CFFNNReplicas src/ObsCalc_CFFNNReplicas.cpp)
target_link_libraries(
	ObsCalc_CFFNNReplicas
	${SFML_LIBRARIES}
	${CLN_LIBRARIES}
	${ElementaryUtils_LIBRARIES}
	${NumA++_LIBRARIES}
	${PARTONS_LIBRARIES}
	${GSL_LIBRARIES}
	${Apfel++_LIBRARIES}
	${LHAPDF_LIBRARIES}
	${LIBXML2_LIBRARIES}
)
set_property(TARGET ObsCalc_CFFNNReplicas PROPERTY CXX_STANDARD 17)

# dcgan executable (libtorch test)
add_executable(dcgan src/dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)