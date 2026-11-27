cmake_minimum_required(VERSION 3.18)
project(bevfusion_board_demo LANGUAGES CXX)

set(UCP_SDK_ROOT "" CACHE PATH "OpenExplorer ucp directory containing include/ and lib/")
if(UCP_SDK_ROOT)
    set(UCP_INCLUDE_HINTS "${UCP_SDK_ROOT}/include")
    set(UCP_LIBRARY_HINTS "${UCP_SDK_ROOT}/lib" "${UCP_SDK_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
    set(UCP_SEARCH_MODE NO_DEFAULT_PATH)
else()
    set(UCP_INCLUDE_HINTS /usr/hobot/include /opt/hobot/include /usr/local/hobot/include)
    set(UCP_LIBRARY_HINTS /usr/hobot/lib /usr/hobot/lib64 /usr/lib/hobot
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/hobot" /opt/hobot/lib /usr/local/hobot/lib)
endif()

find_path(UCP_INCLUDE_DIR NAMES hobot/dnn/hb_dnn.h HINTS ${UCP_INCLUDE_HINTS} ${UCP_SEARCH_MODE})
find_library(DNN_LIBRARY NAMES dnn libdnn.so.1 libdnn.so.2 libdnn.so.3
    HINTS ${UCP_LIBRARY_HINTS} ${UCP_SEARCH_MODE})
find_library(UCP_LIBRARY NAMES hbucp libhbucp.so.1 libhbucp.so.2 libhbucp.so.3
    HINTS ${UCP_LIBRARY_HINTS} ${UCP_SEARCH_MODE})
if(NOT UCP_INCLUDE_DIR OR NOT DNN_LIBRARY OR NOT UCP_LIBRARY)
    message(FATAL_ERROR "Cannot find the UCP SDK (hobot/dnn/hb_dnn.h, libdnn, libhbucp). "
        "Set -DUCP_SDK_ROOT=/path/to/ucp, or set UCP_INCLUDE_DIR, DNN_LIBRARY and UCP_LIBRARY explicitly.")
endif()

add_executable(board_demo board_demo.cpp)
target_compile_features(board_demo PRIVATE cxx_std_17)
target_include_directories(board_demo PRIVATE "${UCP_INCLUDE_DIR}")
target_link_libraries(board_demo PRIVATE "${DNN_LIBRARY}" "${UCP_LIBRARY}")
if(UCP_SDK_ROOT)
    # An offline SDK can omit target OS dependencies, which are supplied on the board.
    foreach(library IN ITEMS "${DNN_LIBRARY}" "${UCP_LIBRARY}")
        get_filename_component(library_dir "${library}" DIRECTORY)
        target_link_options(board_demo PRIVATE "LINKER:-rpath-link,${library_dir}")
    endforeach()
    target_link_options(board_demo PRIVATE "LINKER:--allow-shlib-undefined")
    set_target_properties(board_demo PROPERTIES SKIP_BUILD_RPATH TRUE)
endif()
