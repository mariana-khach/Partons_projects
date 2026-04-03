# BASIC DEFINITIONS ========================================================================

cmake_minimum_required(VERSION 3.5...3.31)

project(PARTONS_Fits CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FIND LIBRARIES ===========================================================================

# find libraries: additional modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# find libraries: SFML
find_package(SFML COMPONENTS system REQUIRED)

# find libraries: Root
find_package(ROOT COMPONENTS Core Hist Minuit REQUIRED)

# find libraries: LHAPDF
find_package(LHAPDF COMPONENTS system REQUIRED)

# find libraries: ElementaryUtils
find_package(ElementaryUtils REQUIRED)

# find libraries: NumA++
find_package(NumA++ REQUIRED)

# find libraries: PARTONS
find_package(PARTONS REQUIRED)

# directories containing headers
include_directories(${SFML_INCLUDE_DIR} ${ROOT_INCLUDE_DIR} ${LHAPDF_INCLUDE_DIR} ${ELEMENTARY_UTILS_INCLUDE_DIR} ${NUMA_INCLUDE_DIR} ${PARTONS_INCLUDE_DIR})

# FINALIZE ==================================================================================

# generate list of source files
file(

        GLOB_RECURSE

        source_files

        src/*
        
        dependancies/*
)

# define target executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(

        PARTONS_Fits

        ${source_files}
)

# define libraries to be linked
target_link_libraries(

        PARTONS_Fits
        
        ${SFML_LIBRARIES}
          
        ${ROOT_LIBRARIES}

 		${LHAPDF_LIBRARIES}
        
        ${ELEMENTARY_UTILS_LIBRARIES}

        ${NUMA_LIBRARIES}   
      
        ${PARTONS_LIBRARIES}       
)