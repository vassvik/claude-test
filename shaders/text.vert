#version 430 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform vec2 screenSize;

void main() {
    // Convert pixel coordinates to NDC (-1 to 1)
    vec2 ndc = (aPos / screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y so origin is top-left
    gl_Position = vec4(ndc, 0.0, 1.0);
    TexCoord = aTexCoord;
}
