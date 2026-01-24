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
GLuint textProgram;

// Text rendering
GLuint fontTexture;
GLuint textVAO, textVBO;

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
int showConvergence = 0;

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

// Get top 3 bins for pre and post divergence
void getTopBins(int* preBins, int* preCounts, int* postBins, int* postCounts) {
    DivergenceStats2D stats;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, statsBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(DivergenceStats2D), &stats);

    // Sum pre-divergence bins (columns)
    unsigned int preSums[32] = {0};
    unsigned int postSums[32] = {0};

    for (int post = 0; post < 32; post++) {
        for (int pre = 0; pre < 32; pre++) {
            unsigned int count = stats.histogram[post * 32 + pre];
            preSums[pre] += count;
            postSums[post] += count;
        }
    }

    // Find top 3 for pre (highest bin index with counts, i.e., worst divergence)
    for (int i = 0; i < 3; i++) {
        preBins[i] = -1;
        preCounts[i] = 0;
        postBins[i] = -1;
        postCounts[i] = 0;
    }

    for (int b = 31; b >= 0; b--) {
        if (preSums[b] > 0) {
            for (int i = 0; i < 3; i++) {
                if (preBins[i] == -1) {
                    preBins[i] = b - 24;
                    preCounts[i] = preSums[b];
                    break;
                }
            }
        }
        if (postSums[b] > 0) {
            for (int i = 0; i < 3; i++) {
                if (postBins[i] == -1) {
                    postBins[i] = b - 24;
                    postCounts[i] = postSums[b];
                    break;
                }
            }
        }
    }
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

// Simple 8x8 bitmap font (ASCII 32-127, 16 chars per row, 6 rows)
// Each character is 8x8 pixels, stored as 8 bytes (1 bit per pixel)
static const unsigned char font8x8[96][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 32: space
    {0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x00}, // 33: !
    {0x6C,0x6C,0x24,0x00,0x00,0x00,0x00,0x00}, // 34: "
    {0x6C,0xFE,0x6C,0x6C,0xFE,0x6C,0x00,0x00}, // 35: #
    {0x18,0x7E,0xC0,0x7C,0x06,0xFC,0x18,0x00}, // 36: $
    {0xC6,0xCC,0x18,0x30,0x66,0xC6,0x00,0x00}, // 37: %
    {0x38,0x6C,0x38,0x76,0xDC,0xCC,0x76,0x00}, // 38: &
    {0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00}, // 39: '
    {0x0C,0x18,0x30,0x30,0x30,0x18,0x0C,0x00}, // 40: (
    {0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x00}, // 41: )
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, // 42: *
    {0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00}, // 43: +
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x30}, // 44: ,
    {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00}, // 45: -
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00}, // 46: .
    {0x06,0x0C,0x18,0x30,0x60,0xC0,0x00,0x00}, // 47: /
    {0x7C,0xC6,0xCE,0xD6,0xE6,0xC6,0x7C,0x00}, // 48: 0
    {0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00}, // 49: 1
    {0x7C,0xC6,0x0C,0x18,0x30,0x60,0xFE,0x00}, // 50: 2
    {0x7C,0xC6,0x06,0x3C,0x06,0xC6,0x7C,0x00}, // 51: 3
    {0x0C,0x1C,0x3C,0x6C,0xFE,0x0C,0x0C,0x00}, // 52: 4
    {0xFE,0xC0,0xFC,0x06,0x06,0xC6,0x7C,0x00}, // 53: 5
    {0x7C,0xC0,0xFC,0xC6,0xC6,0xC6,0x7C,0x00}, // 54: 6
    {0xFE,0x06,0x0C,0x18,0x30,0x30,0x30,0x00}, // 55: 7
    {0x7C,0xC6,0xC6,0x7C,0xC6,0xC6,0x7C,0x00}, // 56: 8
    {0x7C,0xC6,0xC6,0x7E,0x06,0x06,0x7C,0x00}, // 57: 9
    {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00}, // 58: :
    {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x30}, // 59: ;
    {0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x00}, // 60: <
    {0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00}, // 61: =
    {0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x00}, // 62: >
    {0x7C,0xC6,0x0C,0x18,0x18,0x00,0x18,0x00}, // 63: ?
    {0x7C,0xC6,0xDE,0xDE,0xDC,0xC0,0x7C,0x00}, // 64: @
    {0x38,0x6C,0xC6,0xC6,0xFE,0xC6,0xC6,0x00}, // 65: A
    {0xFC,0xC6,0xC6,0xFC,0xC6,0xC6,0xFC,0x00}, // 66: B
    {0x7C,0xC6,0xC0,0xC0,0xC0,0xC6,0x7C,0x00}, // 67: C
    {0xF8,0xCC,0xC6,0xC6,0xC6,0xCC,0xF8,0x00}, // 68: D
    {0xFE,0xC0,0xC0,0xFC,0xC0,0xC0,0xFE,0x00}, // 69: E
    {0xFE,0xC0,0xC0,0xFC,0xC0,0xC0,0xC0,0x00}, // 70: F
    {0x7C,0xC6,0xC0,0xCE,0xC6,0xC6,0x7C,0x00}, // 71: G
    {0xC6,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0x00}, // 72: H
    {0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00}, // 73: I
    {0x06,0x06,0x06,0x06,0xC6,0xC6,0x7C,0x00}, // 74: J
    {0xC6,0xCC,0xD8,0xF0,0xD8,0xCC,0xC6,0x00}, // 75: K
    {0xC0,0xC0,0xC0,0xC0,0xC0,0xC0,0xFE,0x00}, // 76: L
    {0xC6,0xEE,0xFE,0xD6,0xC6,0xC6,0xC6,0x00}, // 77: M
    {0xC6,0xE6,0xF6,0xDE,0xCE,0xC6,0xC6,0x00}, // 78: N
    {0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00}, // 79: O
    {0xFC,0xC6,0xC6,0xFC,0xC0,0xC0,0xC0,0x00}, // 80: P
    {0x7C,0xC6,0xC6,0xC6,0xD6,0xDE,0x7C,0x06}, // 81: Q
    {0xFC,0xC6,0xC6,0xFC,0xD8,0xCC,0xC6,0x00}, // 82: R
    {0x7C,0xC6,0xC0,0x7C,0x06,0xC6,0x7C,0x00}, // 83: S
    {0xFF,0x18,0x18,0x18,0x18,0x18,0x18,0x00}, // 84: T
    {0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00}, // 85: U
    {0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x10,0x00}, // 86: V
    {0xC6,0xC6,0xC6,0xD6,0xFE,0xEE,0xC6,0x00}, // 87: W
    {0xC6,0xC6,0x6C,0x38,0x6C,0xC6,0xC6,0x00}, // 88: X
    {0xC3,0xC3,0x66,0x3C,0x18,0x18,0x18,0x00}, // 89: Y
    {0xFE,0x06,0x0C,0x18,0x30,0x60,0xFE,0x00}, // 90: Z
    {0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00}, // 91: [
    {0xC0,0x60,0x30,0x18,0x0C,0x06,0x00,0x00}, // 92: backslash
    {0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00}, // 93: ]
    {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00}, // 94: ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF}, // 95: _
    {0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00}, // 96: `
    {0x00,0x00,0x7C,0x06,0x7E,0xC6,0x7E,0x00}, // 97: a
    {0xC0,0xC0,0xFC,0xC6,0xC6,0xC6,0xFC,0x00}, // 98: b
    {0x00,0x00,0x7C,0xC6,0xC0,0xC6,0x7C,0x00}, // 99: c
    {0x06,0x06,0x7E,0xC6,0xC6,0xC6,0x7E,0x00}, // 100: d
    {0x00,0x00,0x7C,0xC6,0xFE,0xC0,0x7C,0x00}, // 101: e
    {0x1C,0x30,0x7C,0x30,0x30,0x30,0x30,0x00}, // 102: f
    {0x00,0x00,0x7E,0xC6,0xC6,0x7E,0x06,0x7C}, // 103: g
    {0xC0,0xC0,0xFC,0xC6,0xC6,0xC6,0xC6,0x00}, // 104: h
    {0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00}, // 105: i
    {0x06,0x00,0x0E,0x06,0x06,0x06,0xC6,0x7C}, // 106: j
    {0xC0,0xC0,0xCC,0xD8,0xF0,0xD8,0xCC,0x00}, // 107: k
    {0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00}, // 108: l
    {0x00,0x00,0xEC,0xFE,0xD6,0xC6,0xC6,0x00}, // 109: m
    {0x00,0x00,0xFC,0xC6,0xC6,0xC6,0xC6,0x00}, // 110: n
    {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0x7C,0x00}, // 111: o
    {0x00,0x00,0xFC,0xC6,0xC6,0xFC,0xC0,0xC0}, // 112: p
    {0x00,0x00,0x7E,0xC6,0xC6,0x7E,0x06,0x06}, // 113: q
    {0x00,0x00,0xDC,0xE6,0xC0,0xC0,0xC0,0x00}, // 114: r
    {0x00,0x00,0x7E,0xC0,0x7C,0x06,0xFC,0x00}, // 115: s
    {0x30,0x30,0x7C,0x30,0x30,0x30,0x1C,0x00}, // 116: t
    {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0x7E,0x00}, // 117: u
    {0x00,0x00,0xC6,0xC6,0xC6,0x6C,0x38,0x00}, // 118: v
    {0x00,0x00,0xC6,0xC6,0xD6,0xFE,0x6C,0x00}, // 119: w
    {0x00,0x00,0xC6,0x6C,0x38,0x6C,0xC6,0x00}, // 120: x
    {0x00,0x00,0xC6,0xC6,0xC6,0x7E,0x06,0x7C}, // 121: y
    {0x00,0x00,0xFE,0x0C,0x38,0x60,0xFE,0x00}, // 122: z
    {0x0E,0x18,0x18,0x70,0x18,0x18,0x0E,0x00}, // 123: {
    {0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00}, // 124: |
    {0x70,0x18,0x18,0x0E,0x18,0x18,0x70,0x00}, // 125: }
    {0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00}, // 126: ~
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 127: DEL
};

void createFontTexture(void) {
    // Create 128x64 texture (16 chars x 8 rows, each char 8x8)
    unsigned char* pixels = (unsigned char*)calloc(128 * 64, 1);

    for (int c = 0; c < 96; c++) {
        int cx = (c % 16) * 8;
        int cy = (c / 16) * 8;
        for (int y = 0; y < 8; y++) {
            unsigned char row = font8x8[c][y];
            for (int x = 0; x < 8; x++) {
                if (row & (0x80 >> x)) {
                    pixels[(cy + y) * 128 + cx + x] = 255;
                }
            }
        }
    }

    glGenTextures(1, &fontTexture);
    glBindTexture(GL_TEXTURE_2D, fontTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 128, 64, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    free(pixels);
}

void createTextBuffers(void) {
    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);
    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4 * 256, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

void renderText(const char* text, float x, float y, float scale, float r, float g, float b) {
    glUseProgram(textProgram);
    glUniform2f(glGetUniformLocation(textProgram, "screenSize"), (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    glUniform3f(glGetUniformLocation(textProgram, "textColor"), r, g, b);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fontTexture);
    glUniform1i(glGetUniformLocation(textProgram, "fontTex"), 0);

    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);

    float vertices[6 * 4 * 256];
    int vertexCount = 0;
    float charW = 8.0f * scale;
    float charH = 8.0f * scale;

    for (int i = 0; text[i] && i < 256; i++) {
        char c = text[i];
        if (c < 32 || c > 127) c = '?';
        int idx = c - 32;

        float u0 = (idx % 16) * 8.0f / 128.0f;
        float v0 = (idx / 16) * 8.0f / 64.0f;
        float u1 = u0 + 8.0f / 128.0f;
        float v1 = v0 + 8.0f / 64.0f;

        float x0 = x + i * charW;
        float y0 = y;
        float x1 = x0 + charW;
        float y1 = y0 + charH;

        float quad[24] = {
            x0, y0, u0, v0,
            x1, y0, u1, v0,
            x1, y1, u1, v1,
            x0, y0, u0, v0,
            x1, y1, u1, v1,
            x0, y1, u0, v1,
        };
        memcpy(vertices + vertexCount * 24, quad, sizeof(quad));
        vertexCount++;
    }

    glBufferSubData(GL_ARRAY_BUFFER, 0, vertexCount * 24 * sizeof(float), vertices);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount * 6);
    glDisable(GL_BLEND);
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
    if (key == GLFW_KEY_C && action == GLFW_PRESS) {
        showConvergence = !showConvergence;
        printf("Convergence stats: %s\n", showConvergence ? "on" : "off");
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
    textProgram = createRenderProgram("shaders/text.vert", "shaders/text.frag");

    if (!advectProgram || !advectDensityProgram || !divergenceProgram || !pressureProgram ||
        !gradientSubtractProgram || !addForceProgram || !divergenceStatsProgram || !renderProgram || !textProgram) {
        fprintf(stderr, "Failed to load shaders\n");
        glfwTerminate();
        return -1;
    }

    // Create resources
    initClearData();
    createTextures();
    createQuad();
    createFontTexture();
    createTextBuffers();

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
    printf("  C: Toggle convergence stats\n");
    printf("  ESC: Quit\n");

    double lastTime = glfwGetTime();
    double fpsTime = lastTime;
    int frameCount = 0;
    float fps = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float dt = (float)(currentTime - lastTime);
        lastTime = currentTime;

        // Update FPS counter
        frameCount++;
        if (currentTime - fpsTime >= 1.0) {
            fps = frameCount / (float)(currentTime - fpsTime);
            frameCount = 0;
            fpsTime = currentTime;
        }

        // Clamp dt to avoid instability
        if (dt > 0.1f) dt = 0.1f;

        simulate(dt);
        render();

        // Render stats overlay
        char buf[64];
        snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        renderText(buf, 10, 10, 2.0f, 1.0f, 1.0f, 1.0f);

        snprintf(buf, sizeof(buf), "Iterations: %d", pressureIterations);
        renderText(buf, 10, 30, 2.0f, 1.0f, 1.0f, 1.0f);

        snprintf(buf, sizeof(buf), "Omega: %.3f", pressureOmega);
        renderText(buf, 10, 50, 2.0f, 1.0f, 1.0f, 1.0f);

        snprintf(buf, sizeof(buf), "Grid: %dx%d", SIM_WIDTH, SIM_HEIGHT);
        renderText(buf, 10, 70, 2.0f, 1.0f, 1.0f, 1.0f);

        if (showConvergence) {
            int preBins[3], preCounts[3], postBins[3], postCounts[3];
            getTopBins(preBins, preCounts, postBins, postCounts);

            renderText("Pre-projection (worst bins):", 10, 110, 2.0f, 1.0f, 0.8f, 0.5f);
            for (int i = 0; i < 3 && preBins[i] != -1; i++) {
                snprintf(buf, sizeof(buf), "  bin %d: %d cells", preBins[i], preCounts[i]);
                renderText(buf, 10, 130 + i * 20, 2.0f, 1.0f, 0.8f, 0.5f);
            }

            renderText("Post-projection (worst bins):", 10, 210, 2.0f, 0.5f, 1.0f, 0.5f);
            for (int i = 0; i < 3 && postBins[i] != -1; i++) {
                snprintf(buf, sizeof(buf), "  bin %d: %d cells", postBins[i], postCounts[i]);
                renderText(buf, 10, 230 + i * 20, 2.0f, 0.5f, 1.0f, 0.5f);
            }
        }

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
    glDeleteProgram(textProgram);

    glDeleteTextures(1, &fontTexture);
    glDeleteVertexArrays(1, &textVAO);
    glDeleteBuffers(1, &textVBO);

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
