cmake_minimum_required(VERSION 3.0)
project(Installation VERSION 1.0)

set(OPENCV_VERSION opencv4.1.0)
set(OPENCV_DIR "/mnt/d/PaddleDetection/deploy/lite/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv/demo/cxx/lite/third_party/opencv4.1.0/arm64-v8a/")
set(ARM_PLAT arm64-v8a)



set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_BUILD_TYPE Debug)

PROJECT(onnxandroidbin)

#add xtensor library
set(xtl_DIR /mnt/h/xtl-0.7.5/build)
set(xtensor_DIR /mnt/h/xtensor-master/build)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)






include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories("/mnt/d/PaddleDetection/deploy/lite/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv/demo/cxx/lite/third_party/opencv4.1.0/arm64-v8a/include")
#include_directories(${PROJECT_SOURCE_DIR}/eigen-3.4.0)


aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)





get_property( dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES )

foreach( dir ${dirs} )
    message( STATUS "include dir=${dir}" )
endforeach()



message(STATUS "SRC_FILES=${SRC_FILES}")

#product the exe files
add_executable(${PROJECT_NAME} ${SRC_FILES})


target_link_libraries(${PROJECT_NAME} -latomic -pthread -ldl -llog -lz -static-libstdc++)

target_link_libraries(${PROJECT_NAME} xtensor)

target_link_libraries(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/libs/libonnxruntime.so)

target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_imgcodecs.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_imgproc.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_xfeatures2d.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_features2d.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_highgui.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_calib3d.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_flann.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/libs/libopencv_core.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/liblibjasper.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/libtegra_hal.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/liblibjpeg-turbo.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/liblibwebp.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/liblibpng.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/liblibtiff.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/libIlmImf.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/libtbb.a")
target_link_libraries(${PROJECT_NAME} "${OPENCV_DIR}/3rdparty/libs/libcpufeatures.a")
