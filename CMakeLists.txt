cmake_minimum_required(VERSION 3.20)
project(pennis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- Vulkan ----------------
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# ---------------- Shaders ----------------
function(add_shaders TARGET_NAME)
    set(SHADER_SOURCE_FILES ${ARGN})

    list(LENGTH SHADER_SOURCE_FILES FILE_COUNT)
    if(FILE_COUNT EQUAL 0)
        message(FATAL_ERROR "Cannot create a shaders target without any source files")
    endif()

    set(SHADER_COMMANDS)
    set(SHADER_PRODUCTS)
    set(SHADER_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/shaders")

    foreach(SHADER_SOURCE IN LISTS SHADER_SOURCE_FILES)
        get_filename_component(SHADER_SOURCE "${SHADER_SOURCE}" ABSOLUTE)
        get_filename_component(SHADER_BASENAME "${SHADER_SOURCE}" NAME_WE)

        set(SHADER_OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER_BASENAME}.spv")

        list(APPEND SHADER_COMMANDS
            COMMAND ${GLSLC_EXECUTABLE}
            "${SHADER_SOURCE}"
            "-o"
            "${SHADER_OUTPUT_FILE}"
        )

        list(APPEND SHADER_PRODUCTS "${SHADER_OUTPUT_FILE}")
    endforeach()

    add_custom_target(${TARGET_NAME} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
        ${SHADER_COMMANDS}
        COMMENT "Compiling Shaders [${TARGET_NAME}]"
        SOURCES ${SHADER_SOURCE_FILES}
        BYPRODUCTS ${SHADER_PRODUCTS}
    )
endfunction()

# Compile shader files
add_shaders(shaders
    shaders/forward.comp
    shaders/backprop.comp
    shaders/reduce.comp
    shaders/adam.comp
)

# ---------------- Raylib ----------------
add_subdirectory("${CMAKE_SOURCE_DIR}/raylib" ${CMAKE_BINARY_DIR}/raylib)

# ---------------- Executable ----------------
add_executable(pennis
    src/main.cpp
    src/pennis.cpp
)

# Ensure shaders build first
add_dependencies(pennis shaders)

# Link Vulkan + raylib
target_include_directories(pennis PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(pennis PRIVATE ${Vulkan_LIBRARIES} raylib)

# Windows extras (needed by raylib)
if (MSVC)
    target_link_libraries(pennis PRIVATE winmm.lib gdi32.lib opengl32.lib)
endif()
