###################################################################################
# Copyright (c) 2018 Habana Labs.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# *   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# *   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################
include(ExternalProject)

#required for 'ExternalProject_Add' command
set(TEST_TARGET hbits)

include_directories(
    /usr/lib/habanatools/include
    .
)


# Select files to compile
file(GLOB TESTS_FILES *.cpp)

# Create test executable
add_executable(${TEST_TARGET} ${TESTS_FILES})

# link with libraries
target_link_libraries(
    ${TEST_TARGET}
    rt
    pthread
    ${TPC_TEST_CORE_LIBRARY} 
    ${TPCSIM_SHARED_LIBRARY}
    ${TPC_ELF_READER_LIBRARY}
)


# This routine does the following:
# 1. Copies the TPC-C source file to 'bin' for reference.
# 2. Compiles sources using TPC-C compiler (two times - one is for testing of compiler regression).
# 3. Wraps compiler output in X86 elf.
# 4. Links the X86 elf into the executable
# 5. Disassemble the file for debug purposes
function(CompileTPCCWithClang TPC_C_SOURCES_TO_EMBED TPC_HEADERS deviceName)
    foreach(SOURCE_FILE  ${TPC_C_SOURCES_TO_EMBED})
        get_filename_component(SOURCE_NAME_NO_PATH_NO_SUFFIX ${SOURCE_FILE} NAME_WE)
        get_filename_component(SOURCE_FILE_NAME_NO_PATH ${SOURCE_FILE} NAME)
        if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND "${SOURCE_FILE_NAME_NO_PATH}" STREQUAL "filter_2d_i8_w33_s11.c")
            set (OPT_FLAG "-O1")
        elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set (OPT_FLAG "-O0")
        endif()
        add_custom_command(OUTPUT "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o"
            DEPENDS ${SOURCE_FILE} ${TPC_COMPILER_PATH} ${TPC_HEADERS}
                COMMAND cp ${SOURCE_FILE}  ${CMAKE_CURRENT_BINARY_DIR}
                COMMAND ${TPC_COMPILER_PATH} -x c++ -Wall -march=${deviceName} ${OPT_FLAG} ${DEBUG_FLAG} -I${TPC_INCLUDE_DIR}/ -I${TPC_CURRENT_LIST_DIR} ./${SOURCE_FILE_NAME_NO_PATH} -c
                COMMAND objcopy -I binary -O elf64-x86-64 -B i386:x86-64 ./${SOURCE_NAME_NO_PATH_NO_SUFFIX}.o "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o"
				COMMAND  ${TPC_DISASSEMBLER_PATH} --triple tpc -d -j .text --no-show-raw-insn --no-leading-addr --mcpu=${deviceName} ./${RELATIVE_DIRECTORY}/${SOURCE_NAME_NO_PATH_NO_SUFFIX}.o > ${CMAKE_CURRENT_BINARY_DIR}/${RELATIVE_DIRECTORY}/${SOURCE_NAME_NO_PATH_NO_SUFFIX}.tpcasm)
        add_library(${SOURCE_NAME_NO_PATH_NO_SUFFIX} STATIC "${SOURCE_NAME_NO_PATH_NO_SUFFIX}_x86.o")
        set_target_properties( ${SOURCE_NAME_NO_PATH_NO_SUFFIX} PROPERTIES LINKER_LANGUAGE C  )
        target_link_libraries(${TEST_TARGET} ${SOURCE_NAME_NO_PATH_NO_SUFFIX})
        target_compile_definitions(${TEST_TARGET} PRIVATE KERNEL_LIB=${SOURCE_NAME_NO_PATH_NO_SUFFIX})
    endforeach()
endfunction(CompileTPCCWithClang)

file(GLOB TPC_C_TEST_SOURCES_TO_EMBED ./my_kern.c)
CompileTPCCWithClang("${TPC_C_TEST_SOURCES_TO_EMBED}" "" gaudi)

