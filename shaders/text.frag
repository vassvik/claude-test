#version 430 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D fontTex;
uniform vec3 textColor;

void main() {
    float alpha = texture(fontTex, TexCoord).r;
    if (alpha < 0.5) discard;
    FragColor = vec4(textColor, alpha);
}
