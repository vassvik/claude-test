# 2D Stable Fluids with GPU-Based Pressure Solver

A real-time 2D fluid simulation using OpenGL compute shaders, implementing the Stable Fluids algorithm with a Red-Black Gauss-Seidel (RBGS) pressure solver with Successive Over-Relaxation (SOR).

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Or on Windows, simply run `build.bat`.

## Algorithm Overview

The simulation follows the standard Stable Fluids pipeline:

1. **Advection** - Semi-Lagrangian advection of velocity field
2. **Force Application** - External forces (mouse interaction)
3. **Pressure Projection** - Make velocity field divergence-free
4. **Density Advection** - Advect dye/density for visualization

### Pressure Solver Details

The pressure projection step solves the Poisson equation:

```
∇²p = ∇·v
```

where `p` is pressure and `v` is the velocity field. After solving, we subtract the pressure gradient to obtain a divergence-free velocity field.

#### Discretization

We use a standard 5-point stencil for the Laplacian:

```
∇²p ≈ (p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - 4·p[i,j]) / h²
```

With grid spacing h=1, the Gauss-Seidel update becomes:

```
p_new = (p_L + p_R + p_B + p_T - div) / 4
```

#### Red-Black Ordering

Standard Gauss-Seidel is inherently sequential - each cell update depends on its neighbors. Red-Black ordering enables parallelism by splitting cells into two groups based on a checkerboard pattern:

- **Red cells**: (x + y) is even
- **Black cells**: (x + y) is odd

Within each color, all cells can be updated simultaneously since no two cells of the same color are adjacent. One full iteration consists of a Red pass followed by a Black pass.

#### Successive Over-Relaxation (SOR)

SOR accelerates convergence by extrapolating the Gauss-Seidel update:

```
p_SOR = p_old + ω·(p_GS - p_old)
```

where ω is the relaxation factor:
- ω = 1.0: Standard Gauss-Seidel
- 1.0 < ω < 2.0: Over-relaxation (faster convergence)
- ω ≥ 2.0: Unstable

#### Boundary Conditions

The solver uses **Dirichlet boundary conditions** (p = 0 at domain boundaries). When sampling neighbors outside the domain, we return 0:

```glsl
float pL = (pos.x > 0) ? imageLoad(pressure, pos - ivec2(1,0)).r : 0.0;
```

## Choosing the Optimal ω

### Theoretical Optimal

For SOR on a Poisson problem with Dirichlet BCs on an N×N grid, the theoretical optimal is:

```
ω_opt = 2 / (1 + sin(π/N))
```

For N=512, this gives ω_opt ≈ 1.988.

### Practical Considerations

The theoretical optimal assumes convergence to machine precision (infinite iterations). With a **finite iteration budget**, the practical optimal differs:

| Iterations | Half-passes | Optimal ω | Notes |
|------------|-------------|-----------|-------|
| 512        | 1024        | ~1.916    | 2× grid coverage |
| 256        | 512         | ~1.96     | 1× grid coverage |
| 128        | 256         | ~1.924    | 0.5× grid coverage |

Key insight: Each half-pass propagates information by roughly one cell. With 512 half-passes on a 512-wide grid, information from the center can just reach the boundaries. This is the threshold where iteration count matches grid size.

- **More iterations than grid size**: Can use ω closer to theoretical optimal
- **Fewer iterations than grid size**: Higher ω causes instability without improving convergence (information physically cannot propagate across the grid)

### Measuring Convergence

The code includes a divergence statistics system that bins pre- and post-projection divergence values into a 2D histogram. This allows analysis of how well the solver is eliminating divergence across the domain.

The histogram uses log-scale bins (powers of 2) with an offset, so bin -24 represents near-zero divergence, while higher bins (-17, -16, etc.) represent increasing residual divergence.

## File Structure

```
├── main.c                      # Main simulation loop and setup
├── shaders/
│   ├── advect.comp            # Velocity self-advection
│   ├── advect_density.comp    # Density/dye advection
│   ├── divergence.comp        # Compute velocity divergence
│   ├── pressure.comp          # Red-Black SOR pressure solver
│   ├── gradient_subtract.comp # Pressure gradient subtraction
│   ├── add_force.comp         # Mouse force injection
│   ├── divergence_stats.comp  # Convergence analysis
│   ├── quad.vert              # Fullscreen quad vertex shader
│   └── render.frag            # Visualization fragment shader
├── glad/                       # OpenGL loader
├── glfw/                       # Windowing library
├── CMakeLists.txt
└── build.bat
```

## Controls (Interactive Mode)

- **Left mouse + drag**: Add velocity and dye
- **R**: Reset simulation
- **V**: Toggle velocity visualization
- **ESC**: Quit

## Current State

The code is currently configured as a test harness for analyzing pressure solver convergence. It runs an omega search to find the optimal SOR parameter for a given iteration budget, using a single-point impulse as a test case.

To restore interactive mode, modify `main()` to use the window loop instead of `runOmegaSearch()`.

## References

- Stam, J. (1999). "Stable Fluids". SIGGRAPH 1999.
- Harris, M. (2004). "Fast Fluid Dynamics Simulation on the GPU". GPU Gems.
