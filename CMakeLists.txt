cmake_minimum_required(VERSION 3.20)
project(pennis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- Vulkan ----------------
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# ---------------- Shaders ----------------
function(add_shaders TARGET_NAME)
    # If caller passed shader files, use them. Otherwise glob all shader files in shaders/
    set(SHADER_SOURCE_FILES ${ARGN})

    list(LENGTH SHADER_SOURCE_FILES FILE_COUNT)
    if(FILE_COUNT EQUAL 0)
        # Automatically find common shader extensions in the project's shaders/ directory
        file(GLOB SHADER_SOURCE_FILES
            "${CMAKE_SOURCE_DIR}/shaders/*.comp"
        )
    endif()

    if(NOT SHADER_SOURCE_FILES)
        message(FATAL_ERROR "Cannot create a shaders target: no shader source files found")
    endif()

    set(SHADER_COMMANDS)
    set(SHADER_PRODUCTS)
    set(SHADER_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/shaders")  # keep outputs next to sources (same as original)

    foreach(SHADER_SOURCE IN LISTS SHADER_SOURCE_FILES)
        # Normalize to absolute path
        get_filename_component(SHADER_SOURCE_ABS "${SHADER_SOURCE}" ABSOLUTE)

        # Human-friendly relative path for echoing
        file(RELATIVE_PATH SHADER_SOURCE_REL "${CMAKE_SOURCE_DIR}" "${SHADER_SOURCE_ABS}")

        get_filename_component(SHADER_BASENAME "${SHADER_SOURCE_ABS}" NAME_WE)
        set(SHADER_OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER_BASENAME}.spv")

        # Print which file is being compiled, then invoke glslc to produce .spv
        list(APPEND SHADER_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E echo "Compiling: ${SHADER_SOURCE_REL}"
            COMMAND ${GLSLC_EXECUTABLE} "${SHADER_SOURCE_ABS}" -o "${SHADER_OUTPUT_FILE}"
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

add_shaders(shaders)

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
