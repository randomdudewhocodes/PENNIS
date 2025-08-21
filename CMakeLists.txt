cmake_minimum_required(VERSION 3.20)
project(pennis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- Vulkan ----------------
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# ---------------- GLFW ----------------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/glfw ${CMAKE_BINARY_DIR}/glfw)

# ---------------- Shaders ----------------
function(add_shaders TARGET_NAME)
    set(SHADER_SOURCE_FILES ${ARGN})
    list(LENGTH SHADER_SOURCE_FILES FILE_COUNT)

    if(FILE_COUNT EQUAL 0)
        file(GLOB SHADER_SOURCE_FILES
            "${CMAKE_SOURCE_DIR}/shaders/*.comp"
            "${CMAKE_SOURCE_DIR}/shaders/*.vert"
            "${CMAKE_SOURCE_DIR}/shaders/*.frag"
            "${CMAKE_SOURCE_DIR}/shaders/*.geom"
            "${CMAKE_SOURCE_DIR}/shaders/*.tesc"
            "${CMAKE_SOURCE_DIR}/shaders/*.tese"
            "${CMAKE_SOURCE_DIR}/shaders/*.glsl"
        )
    endif()

    if(NOT SHADER_SOURCE_FILES)
        message(FATAL_ERROR "No shader source files found")
    endif()

    set(SHADER_COMMANDS)
    set(SHADER_PRODUCTS)
    set(SHADER_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/shaders")

    foreach(SHADER_SOURCE IN LISTS SHADER_SOURCE_FILES)
        get_filename_component(SHADER_SOURCE_ABS "${SHADER_SOURCE}" ABSOLUTE)
        file(RELATIVE_PATH SHADER_SOURCE_REL "${CMAKE_SOURCE_DIR}" "${SHADER_SOURCE_ABS}")

        get_filename_component(SHADER_BASENAME "${SHADER_SOURCE_ABS}" NAME_WE)
        set(SHADER_OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER_BASENAME}.spv")

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

# Compile shader files
add_shaders(shaders)

# ---------------- PENNIS core library ----------------
add_library(pennis STATIC
    src/pennis.cpp
    # ImGui core
    ${CMAKE_SOURCE_DIR}/imgui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/imgui/imgui_tables.cpp
    ${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
    # ImGui backends
    ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_vulkan.cpp
    ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(pennis PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/imgui
    ${CMAKE_SOURCE_DIR}/imgui/backends
)

target_link_libraries(pennis PUBLIC
    ${Vulkan_LIBRARIES}
    glfw
)

if (MSVC)
    target_link_libraries(pennis PUBLIC winmm.lib gdi32.lib opengl32.lib)
else()
    target_link_libraries(pennis PUBLIC GL)
endif()

# ---------------- Examples ----------------
file(GLOB EXAMPLE_SOURCES "${CMAKE_SOURCE_DIR}/src/examples/*.cpp")

foreach(EXAMPLE_FILE IN LISTS EXAMPLE_SOURCES)
    get_filename_component(EXAMPLE_NAME "${EXAMPLE_FILE}" NAME_WE)

    add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE pennis)
    add_dependencies(${EXAMPLE_NAME} shaders)
endforeach()
