#version 430 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D densityTex;
uniform sampler2D divergenceTex;
uniform sampler2D uVelocityTex;  // 513x512
uniform sampler2D vVelocityTex;  // 512x513
uniform int displayMode;  // 0=density, 1=velocity, 2=divergence, 3=pressure

// Map divergence magnitude to color using log10 scale
// New Tableau 10 palette: gray (small) -> blue (large), white for [100, 1000)
vec3 divergenceToColor(float div) {
    float absDiv = abs(div);
    if (absDiv < 1e-8) return vec3(0.0);  // Near zero = black

    // Clipping: anything >= 1000 is solid white
    if (absDiv >= 1000.0) return vec3(1.0);

    // Use log10 scale
    float logVal = log(absDiv) / log(10.0);  // log10(absDiv)

    // Map from -7 to +3 (10 orders of magnitude, shifted down by 1)
    float t = clamp((logVal + 7.0) / 10.0, 0.0, 1.0);

    // New Tableau 10: gray (bottom) to blue (top), then white
    vec3 colors[11] = vec3[11](
        vec3(0.729, 0.690, 0.675),  // 1e-7: gray (#bab0ac)
        vec3(0.612, 0.459, 0.373),  // 1e-6: brown (#9c755f)
        vec3(1.000, 0.616, 0.655),  // 1e-5: pink (#ff9da7)
        vec3(0.690, 0.478, 0.631),  // 1e-4: purple (#b07aa1)
        vec3(0.929, 0.788, 0.282),  // 1e-3: yellow (#edc948)
        vec3(0.349, 0.631, 0.310),  // 1e-2: green (#59a14f)
        vec3(0.463, 0.718, 0.698),  // 1e-1: teal (#76b7b2)
        vec3(0.882, 0.341, 0.349),  // 1e0:  red (#e15759)
        vec3(0.949, 0.557, 0.169),  // 1e1:  orange (#f28e2b)
        vec3(0.306, 0.475, 0.655),  // 1e2:  blue (#4e79a7)
        vec3(1.000, 1.000, 1.000)   // 1e3:  white
    );

    float idx = t * 10.0;
    int i0 = int(floor(idx));
    int i1 = min(i0 + 1, 10);

    return mix(colors[i0], colors[i1], fract(idx));
}

void main() {
    if (displayMode == 1) {
        // Visualize velocity field from separate u/v textures with proper staggered dimensions
        // Screen TexCoord (0-1) maps to the 512x512 cell grid
        // u texture is 513x512, v texture is 512x513

        // For u (513 wide): to get velocity at cell center position (x*512 + 0.5, y*512 + 0.5),
        // we need to sample at u-texture UV that accounts for the offset
        // u faces are at x positions 0,1,2,...,512, so cell center x*512+0.5 maps to u-texture x = (x*512+0.5)/513
        vec2 uTexCoord = vec2((TexCoord.x * 512.0 + 0.5) / 513.0, TexCoord.y);
        float u = texture(uVelocityTex, uTexCoord).r;

        // For v (513 tall): similarly
        vec2 vTexCoord = vec2(TexCoord.x, (TexCoord.y * 512.0 + 0.5) / 513.0);
        float v = texture(vVelocityTex, vTexCoord).r;

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
