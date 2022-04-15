cmake_minimum_required(VERSION 3.0)
project(pisoFoamPC_CNN)




link_directories($ENV{FOAM_LIBBIN})
#set(CMAKE_CXX_FLAGS "-arch x86_64 -std=c++14 -Ddarwin64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wno-unused-parameter -Wno-overloaded-virtual -Wno-unused-variable -Wno-unused-local-typedef -Wno-invalid-offsetof -Wno-deprecated-register -Wno-undefined-var-template -O3  -DNoRepository -ftemplate-depth-100" )
set(CMAKE_CXX_FLAGS " -arch x86_64 -std=c++14 -Ddarwin64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wno-unused-parameter -Wno-overloaded-virtual -Wno-unused-variable -Wno-unused-local-typedef -Wno-invalid-offsetof -Wno-deprecated-register -Wno-undefined-var-template -O3  -DNoRepository -ftemplate-depth-100 -m64 -fPIC" )
set(CMAKE_PREFIX_PATH $ENV{TORCH_PATH})
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)


find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

set(SOURCE_FILES
	pisoFoamPC_CNN.C
	InferenceEngine/InferenceEngine.cpp
	InferenceEngine/CustomNormalizer.cpp
    )
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

#set_target_properties(${PROJECT_NAME}
#	PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")


################################################### FOR OPENFOAM INTEGRATION   	#################################################@@
#
## cxx11
set( CMAKE_CXX_STANDARD 14 )
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker --no-as-needed -Xlinker --add-needed")
#add_definitions(-Ddarwin64 -DWM_ARCH_OPTION=64 -DWM_DP -DWM_LABEL_SIZE=32 -DNoRepository -m64 -fPIC )

# Make/files
# Add source to this project's executable.

# Make/options
#MAYBE NEED TO ADD PyTORCH INCLUDE PATHS 
include_directories("/usr/local/include")
include_directories(${PROJECT_SOURCE_DIR})

include_directories($ENV{JSON_PATH}/include )
include_directories( InferenceEngine )

include_directories( $ENV{FOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude )
include_directories( $ENV{FOAM_SRC}/TurbulenceModels/incompressible/lnInclude )
include_directories( $ENV{FOAM_SRC}/transportModels )
include_directories( $ENV{FOAM_SRC}/transportModels/incompressible/singlePhaseTransportModel )
include_directories( $ENV{FOAM_SRC}/finiteVolume/lnInclude )
include_directories( $ENV{FOAM_SRC}/meshTools/lnInclude )
include_directories( $ENV{FOAM_SRC}/sampling )
include_directories( $ENV{FOAM_SRC}/OpenFOAM/lnInclude )
include_directories( $ENV{FOAM_SRC}/OSspecific/POSIX/lnInclude )

set(EXTRA_LIBS dl m)
set(LIBS turbulenceModels incompressibleTurbulenceModels incompressibleTransportModels finiteVolume meshTools fvOptions sampling OpenFOAM ${EXTRA_LIBS})

target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}"  nlohmann_json::nlohmann_json ${LIBS})

########################################################################################################################@@@




