cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(HBP_ISP LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_library(CRTrees STATIC CRTrees.cu CRTrees.hpp)
target_compile_features(CRTrees PUBLIC cxx_std_11)
target_compile_options(CRTrees PUBLIC -O3)

add_library(HSS STATIC segment.cu segment.hpp imutil.cu util.h)
target_compile_features(HSS PUBLIC cxx_std_11)
target_compile_options(HSS PUBLIC -O3)
target_link_libraries(HSS PUBLIC ${OpenCV_LIBS} CRTrees)

add_library(stereo_cost STATIC stereo_cost.cu stereo_cost.hpp cvutils.hpp)
target_compile_features(stereo_cost PUBLIC cxx_std_11)
target_compile_options(stereo_cost PUBLIC -O3)
target_link_libraries(stereo_cost PUBLIC ${OpenCV_LIBS})

add_library(post_process STATIC post_process.cu post_process.hpp util.h)
target_compile_features(post_process PUBLIC cxx_std_11)
target_compile_options(post_process PUBLIC -O3) 
target_link_libraries(post_process PUBLIC ${OpenCV_LIBS})

add_library(hbp_isp STATIC hbp_isp.cu hbp_isp.hpp)
target_link_libraries(hbp_isp ${OpenCV_LIBS} HSS stereo_cost)

add_executable(stereo_hbp_isp main_stereo_hbp_isp.cu)
target_link_libraries(stereo_hbp_isp PUBLIC ${OpenCV_LIBS} hbp_isp HSS stereo_cost post_process)

add_executable(bench_middv3 main_bench_middv3.cu ArgsParser.h Evaluator.h)
target_link_libraries(bench_middv3 PUBLIC ${OpenCV_LIBS} hbp_isp HSS stereo_cost post_process)

add_executable(bench_middv3_vis main_bench_middv3_vis.cu ArgsParser.h Evaluator.h)
target_link_libraries(bench_middv3_vis PUBLIC ${OpenCV_LIBS} hbp_isp HSS stereo_cost post_process)

add_executable(eval_middv3 eval_algo.cpp ArgsParser.h Evaluator.h)
target_link_libraries(eval_middv3 PUBLIC ${OpenCV_LIBS})