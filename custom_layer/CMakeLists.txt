# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

cmake_minimum_required(VERSION 2.8)

# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_compile_flags())" OUTPUT_VARIABLE TF_COMPILE_FLAGS)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_link_flags())" OUTPUT_VARIABLE TF_LINK_FLAGS)

if(WIN32)
    string(REPLACE "\r\n" "" TF_INC "${TF_INC}")
    string(REPLACE "\r\n" "" TF_LIB "${TF_LIB}")
    string(REPLACE "\r\n" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
    string(REPLACE "\r\n" "" TF_LINK_FLAGS "${TF_LINK_FLAGS}")
else(WIN32)
    string(REPLACE "\n" "" TF_INC "${TF_INC}")
    string(REPLACE "\n" "" TF_LIB "${TF_LIB}")
    string(REPLACE "\n" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
    string(REPLACE "\n" "" TF_LINK_FLAGS "${TF_LINK_FLAGS}")
endif(WIN32)

string(REPLACE "[" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
string(REPLACE "]" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
string(REPLACE "'" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
string(REPLACE "," "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")

string(REPLACE "[" "" TF_LINK_FLAGS "${TF_LINK_FLAGS}")
string(REPLACE "]" "" TF_LINK_FLAGS "${TF_LINK_FLAGS}")
string(REPLACE "'" "" TF_LINK_FLAGS "${TF_LINK_FLAGS}")
string(REPLACE "," "" TF_LINK_FLAGS "${TF_LINK_FLAGS}")

message("TF_INC: ${TF_INC}")
message("TF_LIB: ${TF_LIB}")
message("TF_COMPILE_FLAGS: ${TF_COMPILE_FLAGS}")
message("TF_LINK_FLAGS: ${TF_LINK_FLAGS}")

find_package(CUDA)

set(TF_NSYNC_INC "${TF_INC}/external/nsync/public/")
message(${TF_NSYNC_INC})
include_directories(${TF_NSYNC_INC})

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC ${CMAKE_CXX_FLAGS} ${TF_COMPILE_FLAGS} -DGOOGLE_CUDA")
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

#pass flags to c++ compiler
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -DNDEBUG")

# build the actual operation which can be used directory
include_directories(${TF_INC})
link_directories(${TF_LIB})
include_directories("/usr/local/")

cuda_add_library(compute_depth SHARED compute_depth.cu compute_depth.cc
    OPTIONS -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr)
target_link_libraries(compute_depth ${TF_LINK_FLAGS})
