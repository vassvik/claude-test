#version 430 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D densityTex;
uniform int showVelocity;

void main() {
    if (showVelocity != 0) {
        // Visualize velocity field
        vec2 vel = texture(densityTex, TexCoord).xy;
        float mag = length(vel);
        float angle = atan(vel.y, vel.x);

        // Color by direction, brightness by magnitude
        vec3 color;
        color.r = 0.5 + 0.5 * cos(angle);
        color.g = 0.5 + 0.5 * cos(angle + 2.094);
        color.b = 0.5 + 0.5 * cos(angle + 4.189);

        // Scale magnitude for visibility (adjust as needed)
        float brightness = mag * 0.01;
        color *= brightness;

        FragColor = vec4(color, 1.0);
    } else {
        vec3 density = texture(densityTex, TexCoord).rgb;

        // Apply some tone mapping for nicer visuals
        density = density / (1.0 + density);

        // Gamma correction
        density = pow(density, vec3(1.0 / 2.2));

        FragColor = vec4(density, 1.0);
    }
}
