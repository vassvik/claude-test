#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// Force discrete GPU on Optimus/PowerXpress systems
#ifdef _WIN32
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
__declspec(dllexport) unsigned long AmdPowerXpressRequestHighPerformance = 1;
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define SIM_WIDTH 512
#define SIM_HEIGHT 512
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

// Solver parameters
int pressureIterations = 128;
float pressureOmega = 1.8f;

typedef struct {
    unsigned int histogram[32 * 32];
} DivergenceStats2D;

// Shader programs
GLuint advectProgram;
GLuint advectDensityProgram;
GLuint divergenceProgram;
GLuint pressureProgram;
GLuint gradientSubtractProgram;
GLuint addForceProgram;
GLuint renderProgram;
GLuint divergenceStatsProgram;

// Stats buffer
GLuint statsBuffer;

// Textures for simulation
GLuint velocityTex[2];
GLuint pressureTex[2];
GLuint divergenceTex;
GLuint postDivergenceTex;
GLuint densityTex[2];

// Stats timing
double lastStatsPrintTime = 0.0;

// Render resources
GLuint quadVAO, quadVBO;

// Simulation state
int currentVel = 0;
int currentPressure = 0;
int currentDensity = 0;

// Clear texture data buffers (allocated once)
float* clearDataR = NULL;
float* clearDataRG = NULL;
float* clearDataRGBA = NULL;

// Mouse state
double lastMouseX = 0, lastMouseY = 0;
int mousePressed = 0;

// Debug visualization
int showVelocity = 0;

// Function prototypes
char* loadShaderSource(const char* filename);
GLuint createComputeShader(const char* filename);
GLuint createRenderProgram(const char* vertFile, const char* fragFile);
void createTextures(void);
void createQuad(void);
void simulate(float dt);
void render(void);

char* loadShaderSource(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(length + 1);
    fread(source, 1, length, file);
    source[length] = '\0';

    fclose(file);
    return source;
}

GLuint createComputeShader(const char* filename) {
    char* source = loadShaderSource(filename);
    if (!source) return 0;

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, (const char**)&source, NULL);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        fprintf(stderr, "Compute shader compilation failed (%s):\n%s\n", filename, log);
        free(source);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, NULL, log);
        fprintf(stderr, "Program linking failed (%s):\n%s\n", filename, log);
    }

    glDeleteShader(shader);
    free(source);
    return program;
}

GLuint createRenderProgram(const char* vertFile, const char* fragFile) {
    char* vertSource = loadShaderSource(vertFile);
    char* fragSource = loadShaderSource(fragFile);
    if (!vertSource || !fragSource) return 0;

    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertShader, 1, (const char**)&vertSource, NULL);
    glCompileShader(vertShader);

    int success;
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vertShader, 512, NULL, log);
        fprintf(stderr, "Vertex shader compilation failed:\n%s\n", log);
    }

    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragShader, 1, (const char**)&fragSource, NULL);
    glCompileShader(fragShader);

    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fragShader, 512, NULL, log);
        fprintf(stderr, "Fragment shader compilation failed:\n%s\n", log);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, NULL, log);
        fprintf(stderr, "Render program linking failed:\n%s\n", log);
    }

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    free(vertSource);
    free(fragSource);

    return program;
}

void initClearData(void) {
    // Allocate zero-filled buffers for clearing textures
    clearDataR = (float*)calloc(SIM_WIDTH * SIM_HEIGHT, sizeof(float));
    clearDataRG = (float*)calloc(SIM_WIDTH * SIM_HEIGHT * 2, sizeof(float));
    clearDataRGBA = (float*)calloc(SIM_WIDTH * SIM_HEIGHT * 4, sizeof(float));
}

void clearTextureR(GLuint tex) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SIM_WIDTH, SIM_HEIGHT, GL_RED, GL_FLOAT, clearDataR);
}

void clearTextureRG(GLuint tex) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SIM_WIDTH, SIM_HEIGHT, GL_RG, GL_FLOAT, clearDataRG);
}

void clearTextureRGBA(GLuint tex) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SIM_WIDTH, SIM_HEIGHT, GL_RGBA, GL_FLOAT, clearDataRGBA);
}

void createTextures(void) {
    // Velocity textures (RG32F for 2D velocity)
    glGenTextures(2, velocityTex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, velocityTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, SIM_WIDTH, SIM_HEIGHT, 0, GL_RG, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    // Pressure textures (R32F)
    glGenTextures(2, pressureTex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, pressureTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SIM_WIDTH, SIM_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    // Divergence texture (R32F) - stores pre-projection divergence
    glGenTextures(1, &divergenceTex);
    glBindTexture(GL_TEXTURE_2D, divergenceTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SIM_WIDTH, SIM_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Post-divergence texture (R32F) - stores post-projection divergence
    glGenTextures(1, &postDivergenceTex);
    glBindTexture(GL_TEXTURE_2D, postDivergenceTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SIM_WIDTH, SIM_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Density textures (R32F for scalar density/dye)
    glGenTextures(2, densityTex);
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, densityTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SIM_WIDTH, SIM_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
}

void clearStats2D(void) {
    DivergenceStats2D zero = {{0}};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(DivergenceStats2D), &zero);
}

void computeStats2D(GLuint preTex, GLuint postTex) {
    glUseProgram(divergenceStatsProgram);
    glBindImageTexture(0, preTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, postTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, statsBuffer);
    glDispatchCompute((SIM_WIDTH+15)/16, (SIM_HEIGHT+15)/16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void printStats2DTable(void) {
    DivergenceStats2D stats;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(DivergenceStats2D), &stats);

    // Find active column range (pre bins with any data)
    int minCol = 31, maxCol = 0;
    for (int pre = 0; pre < 32; pre++) {
        for (int post = 0; post < 32; post++) {
            if (stats.histogram[post * 32 + pre] > 0) {
                if (pre < minCol) minCol = pre;
                if (pre > maxCol) maxCol = pre;
            }
        }
    }
    if (minCol > maxCol) return;  // No data

    // Print header
    printf("\n=== Divergence Transition (rows=post, cols=pre) ===\n");
    printf("post\\pre");
    for (int pre = minCol; pre <= maxCol; pre++) {
        printf(" %4d", pre - 24);
    }
    printf("\n");

    // Print rows (only non-empty)
    for (int post = 0; post < 32; post++) {
        int hasData = 0;
        for (int pre = minCol; pre <= maxCol; pre++) {
            if (stats.histogram[post * 32 + pre] > 0) hasData = 1;
        }
        if (!hasData) continue;

        printf("%4d   ", post - 24);
        for (int pre = minCol; pre <= maxCol; pre++) {
            unsigned int count = stats.histogram[post * 32 + pre];
            if (count == 0) printf("    .");
            else if (count < 10000) printf(" %4u", count);
            else printf(" %4uk", count / 1000);
        }
        printf("\n");
    }
}

void createQuad(void) {
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

void simulate(float dt) {
    int groupsX = (SIM_WIDTH + 15) / 16;
    int groupsY = (SIM_HEIGHT + 15) / 16;

    // 1. Advect velocity
    glUseProgram(advectProgram);
    glUniform1f(glGetUniformLocation(advectProgram, "dt"), dt);
    glUniform2f(glGetUniformLocation(advectProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glUniform1f(glGetUniformLocation(advectProgram, "dissipation"), 1.0f);
    glUniform1i(glGetUniformLocation(advectProgram, "fieldTex"), 0);
    glBindImageTexture(0, velocityTex[currentVel], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, velocityTex[1 - currentVel], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, velocityTex[currentVel]);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    currentVel = 1 - currentVel;

    // 2. Compute divergence
    glUseProgram(divergenceProgram);
    glUniform2f(glGetUniformLocation(divergenceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glBindImageTexture(0, velocityTex[currentVel], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, divergenceTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // 3. Pressure solve (Red-Black SOR)
    clearTextureR(pressureTex[currentPressure]);
    glUseProgram(pressureProgram);
    glUniform1f(glGetUniformLocation(pressureProgram, "omega"), pressureOmega);
    glBindImageTexture(0, pressureTex[currentPressure], 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
    glBindImageTexture(1, divergenceTex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    for (int i = 0; i < pressureIterations; i++) {
        // Red pass
        glUniform1i(glGetUniformLocation(pressureProgram, "redPass"), 1);
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        // Black pass
        glUniform1i(glGetUniformLocation(pressureProgram, "redPass"), 0);
        glDispatchCompute(groupsX, groupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    // 4. Gradient subtraction (projection)
    glUseProgram(gradientSubtractProgram);
    glUniform2f(glGetUniformLocation(gradientSubtractProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glBindImageTexture(0, pressureTex[currentPressure], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, velocityTex[currentVel], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(2, velocityTex[1 - currentVel], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    currentVel = 1 - currentVel;

    // Compute post-divergence into separate texture (pre-divergence preserved in divergenceTex)
    glUseProgram(divergenceProgram);
    glBindImageTexture(0, velocityTex[currentVel], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, postDivergenceTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Compute 2D stats (table printed by caller when needed)
    clearStats2D();
    computeStats2D(divergenceTex, postDivergenceTex);

    // 5. Advect density using divergence-free velocity
    glUseProgram(advectDensityProgram);
    glUniform1f(glGetUniformLocation(advectDensityProgram, "dt"), dt);
    glUniform2f(glGetUniformLocation(advectDensityProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glUniform1f(glGetUniformLocation(advectDensityProgram, "dissipation"), 0.999f);
    glUniform1i(glGetUniformLocation(advectDensityProgram, "densityIn"), 0);
    glBindImageTexture(0, velocityTex[currentVel], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, densityTex[1 - currentDensity], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, densityTex[currentDensity]);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    currentDensity = 1 - currentDensity;
}

void addForce(float x, float y, float dx, float dy) {
    int groupsX = (SIM_WIDTH + 15) / 16;
    int groupsY = (SIM_HEIGHT + 15) / 16;

    // Convert screen-space delta to grid-space velocity (grid cells per second)
    // dx/dy are in normalized screen coords per frame, scale to reasonable velocity
    float forceScale = 300.0f * SIM_WIDTH;  // Scale factor for force
    float fx = dx * forceScale;
    float fy = dy * forceScale;

    // Generate color based on direction
    float angle = atan2f(fy, fx);
    float r = 0.5f + 0.5f * cosf(angle);
    float g = 0.5f + 0.5f * cosf(angle + 2.094f);  // 120 degrees
    float b = 0.5f + 0.5f * cosf(angle + 4.189f);  // 240 degrees

    glUseProgram(addForceProgram);
    glUniform2f(glGetUniformLocation(addForceProgram, "point"), x, y);
    glUniform2f(glGetUniformLocation(addForceProgram, "force"), fx, fy);
    glUniform1f(glGetUniformLocation(addForceProgram, "radius"), 0.02f);
    glUniform3f(glGetUniformLocation(addForceProgram, "dyeColor"), r, g, b);

    glBindImageTexture(0, velocityTex[currentVel], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG32F);
    glBindImageTexture(1, densityTex[currentDensity], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    glDispatchCompute(groupsX, groupsY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(renderProgram);
    glActiveTexture(GL_TEXTURE0);
    if (showVelocity) {
        glBindTexture(GL_TEXTURE_2D, velocityTex[currentVel]);
    } else {
        glBindTexture(GL_TEXTURE_2D, densityTex[currentDensity]);
    }
    glUniform1i(glGetUniformLocation(renderProgram, "densityTex"), 0);
    glUniform1i(glGetUniformLocation(renderProgram, "showVelocity"), showVelocity);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void setupImpulseTest(void) {
    // Clear all textures first
    clearTextureRG(velocityTex[0]);
    clearTextureRG(velocityTex[1]);
    clearTextureRGBA(densityTex[0]);
    clearTextureRGBA(densityTex[1]);
    clearTextureR(pressureTex[0]);
    clearTextureR(pressureTex[1]);
    currentVel = 0;
    currentPressure = 0;
    currentDensity = 0;

    // Set a single point impulse at center
    int cx = SIM_WIDTH / 2;
    int cy = SIM_HEIGHT / 2;
    float impulse[2] = {1.0f, 0.0f};  // Velocity pointing right

    glBindTexture(GL_TEXTURE_2D, velocityTex[currentVel]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, cx, cy, 1, 1, GL_RG, GL_FLOAT, impulse);
}

// Returns: largest non-zero bin index (0-31), and count in that bin via pointer
int evaluateConvergence(int* worstBinCount) {
    DivergenceStats2D stats;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(DivergenceStats2D), &stats);

    // Find the largest (worst) post-divergence bin with data
    // Post bins are rows, we want the highest row index with any count
    int worstBin = 0;
    int worstCount = 0;
    for (int post = 31; post >= 0; post--) {
        unsigned int total = 0;
        for (int pre = 0; pre < 32; pre++) {
            total += stats.histogram[post * 32 + pre];
        }
        if (total > 0) {
            worstBin = post;
            worstCount = total;
            break;
        }
    }
    *worstBinCount = worstCount;
    return worstBin;
}

void testOmega(float omega, int* worstBin, int* worstCount) {
    pressureOmega = omega;

    // Setup fresh impulse
    setupImpulseTest();

    // Run one simulation step
    simulate(0.016f);

    // Evaluate convergence
    *worstBin = evaluateConvergence(worstCount);
}

void runOmegaSearch(float omegaMin, float omegaMax, int numBins) {
    printf("\nSearching omega in [%.4f, %.4f] with %d samples, %d iterations\n",
           omegaMin, omegaMax, numBins, pressureIterations);
    printf("%-10s %-12s %-12s\n", "Omega", "WorstBin", "Count");
    printf("--------------------------------------\n");

    float bestOmega = omegaMin;
    int bestWorstBin = 31;
    int bestWorstCount = INT_MAX;

    for (int i = 0; i < numBins; i++) {
        float omega = omegaMin + (omegaMax - omegaMin) * i / (numBins - 1);
        int worstBin, worstCount;
        testOmega(omega, &worstBin, &worstCount);

        printf("%.4f     %-12d %-12d", omega, worstBin - 24, worstCount);

        // Check if this is better (lower bin, or same bin with fewer entries)
        if (worstBin < bestWorstBin || (worstBin == bestWorstBin && worstCount < bestWorstCount)) {
            bestOmega = omega;
            bestWorstBin = worstBin;
            bestWorstCount = worstCount;
            printf(" *");
        }
        printf("\n");
    }

    printf("--------------------------------------\n");
    printf("Best: omega=%.4f, worst_bin=%d, count=%d\n", bestOmega, bestWorstBin - 24, bestWorstCount);
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        float x = (float)xpos / width;
        float y = 1.0f - (float)ypos / height;
        float fdx = (float)dx / width;
        float fdy = -(float)dy / height;

        addForce(x, y, fdx, fdy);
    }
    lastMouseX = xpos;
    lastMouseY = ypos;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed = (action == GLFW_PRESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        // Reset simulation
        clearTextureRG(velocityTex[0]);
        clearTextureRG(velocityTex[1]);
        clearTextureRGBA(densityTex[0]);
        clearTextureRGBA(densityTex[1]);
        clearTextureR(pressureTex[0]);
        clearTextureR(pressureTex[1]);
    }
    if (key == GLFW_KEY_V && action == GLFW_PRESS) {
        showVelocity = !showVelocity;
        printf("Showing: %s\n", showVelocity ? "velocity" : "density");
    }
}

int main(void) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Stable Fluids 2D", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    printf("OpenGL %s\n", glGetString(GL_VERSION));

    // Set callbacks
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);

    // Load shaders
    advectProgram = createComputeShader("shaders/advect.comp");
    advectDensityProgram = createComputeShader("shaders/advect_density.comp");
    divergenceProgram = createComputeShader("shaders/divergence.comp");
    pressureProgram = createComputeShader("shaders/pressure.comp");
    gradientSubtractProgram = createComputeShader("shaders/gradient_subtract.comp");
    addForceProgram = createComputeShader("shaders/add_force.comp");
    divergenceStatsProgram = createComputeShader("shaders/divergence_stats.comp");
    renderProgram = createRenderProgram("shaders/quad.vert", "shaders/render.frag");

    if (!advectProgram || !advectDensityProgram || !divergenceProgram || !pressureProgram ||
        !gradientSubtractProgram || !addForceProgram || !divergenceStatsProgram || !renderProgram) {
        fprintf(stderr, "Failed to load shaders\n");
        glfwTerminate();
        return -1;
    }

    // Create resources
    initClearData();
    createTextures();
    createQuad();

    // Create stats buffer
    glGenBuffers(1, &statsBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(DivergenceStats2D), NULL, GL_DYNAMIC_READ);

    // Initialize all simulation textures to zero
    clearTextureRG(velocityTex[0]);
    clearTextureRG(velocityTex[1]);
    clearTextureRGBA(densityTex[0]);
    clearTextureRGBA(densityTex[1]);
    clearTextureR(pressureTex[0]);
    clearTextureR(pressureTex[1]);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Set solver parameters for interactive use
    pressureIterations = 100;
    pressureOmega = 1.9f;

    printf("Controls:\n");
    printf("  Left mouse + drag: Add velocity and dye\n");
    printf("  R: Reset simulation\n");
    printf("  V: Toggle velocity visualization\n");
    printf("  ESC: Quit\n");

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float dt = (float)(currentTime - lastTime);
        lastTime = currentTime;

        // Clamp dt to avoid instability
        if (dt > 0.1f) dt = 0.1f;

        simulate(dt);
        render();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteProgram(advectProgram);
    glDeleteProgram(advectDensityProgram);
    glDeleteProgram(divergenceProgram);
    glDeleteProgram(pressureProgram);
    glDeleteProgram(gradientSubtractProgram);
    glDeleteProgram(addForceProgram);
    glDeleteProgram(divergenceStatsProgram);
    glDeleteProgram(renderProgram);

    glDeleteBuffers(1, &statsBuffer);

    glDeleteTextures(2, velocityTex);
    glDeleteTextures(2, pressureTex);
    glDeleteTextures(1, &divergenceTex);
    glDeleteTextures(1, &postDivergenceTex);
    glDeleteTextures(2, densityTex);

    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);

    free(clearDataR);
    free(clearDataRG);
    free(clearDataRGBA);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
