#version 430 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D densityTex;
uniform sampler2D divergenceTex;
uniform sampler2D uVelocityTex;
uniform sampler2D vVelocityTex;
uniform int displayMode;  // 0=density, 1=velocity, 2=divergence, 3=pressure

// Map divergence magnitude to color using log10 scale
// One distinct color per power of ten from 1e-6 to 1e1
vec3 divergenceToColor(float div) {
    float absDiv = abs(div);
    if (absDiv < 1e-7) return vec3(0.0);  // Near zero = black

    // Use log10 scale
    float logVal = log(absDiv) / log(10.0);  // log10(absDiv)

    // Map from -6 to +1 (7 orders of magnitude)
    float t = clamp((logVal + 6.0) / 7.0, 0.0, 1.0);

    // 8 colors for 8 powers of ten
    vec3 colors[8] = vec3[8](
        vec3(0.0, 0.0, 0.4),   // 1e-6: dark blue
        vec3(0.5, 0.0, 0.5),   // 1e-5: purple
        vec3(0.0, 0.5, 0.5),   // 1e-4: cyan
        vec3(0.0, 0.7, 0.0),   // 1e-3: green
        vec3(0.8, 0.8, 0.0),   // 1e-2: yellow
        vec3(1.0, 0.5, 0.0),   // 1e-1: orange
        vec3(1.0, 0.0, 0.0),   // 1e0:  red
        vec3(1.0, 1.0, 1.0)    // 1e1:  white
    );

    float idx = t * 7.0;
    int i0 = int(floor(idx));
    int i1 = min(i0 + 1, 7);

    return mix(colors[i0], colors[i1], fract(idx));
}

void main() {
    if (displayMode == 1) {
        // Visualize velocity field from separate u/v textures
        float u = texture(uVelocityTex, TexCoord).r;
        float v = texture(vVelocityTex, TexCoord).r;
        vec2 vel = vec2(u, v);
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
    } else if (displayMode == 2) {
        // Visualize divergence
        float div = texture(divergenceTex, TexCoord).r;
        vec3 color = divergenceToColor(div);
        FragColor = vec4(color, 1.0);
    } else if (displayMode == 3) {
        // Visualize pressure with wider range to see spatial variation
        float p = texture(divergenceTex, TexCoord).r;
        // Show positive as red, negative as blue, intensity by magnitude
        // Scale down by 100 to see variation in high-pressure fields
        float scaledP = p / 100.0;
        vec3 color;
        if (scaledP > 0.0) {
            color = vec3(min(scaledP, 1.0), 0.0, 0.0);  // Red for positive
        } else {
            color = vec3(0.0, 0.0, min(-scaledP, 1.0)); // Blue for negative
        }
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
