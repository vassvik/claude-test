cmake_minimum_required(VERSION 3.10)
project(StableFluids C)

set(CMAKE_C_STANDARD 11)

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW - adjust path as needed or use find_package if installed system-wide
# Option 1: If GLFW is installed via vcpkg or system package manager
find_package(glfw3 CONFIG QUIET)

if(NOT glfw3_FOUND)
    # Option 2: Manual path specification
    # Set these paths to your GLFW installation
    set(GLFW_INCLUDE_DIR "" CACHE PATH "Path to GLFW include directory")
    set(GLFW_LIBRARY "" CACHE FILEPATH "Path to GLFW library")

    if(GLFW_INCLUDE_DIR AND GLFW_LIBRARY)
        add_library(glfw STATIC IMPORTED)
        set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION ${GLFW_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${GLFW_INCLUDE_DIR}
        )
    else()
        message(FATAL_ERROR "GLFW not found. Please install GLFW or set GLFW_INCLUDE_DIR and GLFW_LIBRARY")
    endif()
endif()

# glad - we include it directly in the project
# You need to generate glad files from https://glad.dav1d.de/
# Settings: Language=C, Specification=OpenGL, Profile=Core, gl=4.3, Generate a loader=yes
set(GLAD_DIR "${CMAKE_SOURCE_DIR}/glad" CACHE PATH "Path to glad directory")

add_executable(StableFluids
    main.c
    ${GLAD_DIR}/src/glad.c
)

target_include_directories(StableFluids PRIVATE
    ${GLAD_DIR}/include
)

target_link_libraries(StableFluids
    OpenGL::GL
    glfw
)

# Copy shaders to build directory
add_custom_command(TARGET StableFluids POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:StableFluids>/shaders
)

# Windows-specific settings
if(WIN32)
    target_link_libraries(StableFluids gdi32 user32 shell32)
endif()
