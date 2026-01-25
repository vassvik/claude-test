# 2D Stable Fluids with GPU-Based Pressure Solver

A real-time 2D fluid simulation using OpenGL compute shaders, implementing the Stable Fluids algorithm with a Red-Black Gauss-Seidel (RBGS) pressure solver with Successive Over-Relaxation (SOR).

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Or on Windows, simply run `build.bat`.

## Controls

- **Left mouse + drag**: Add velocity and dye
- **R**: Reset simulation
- **V**: Cycle display mode (density → velocity → pre-divergence → post-divergence → pressure)
- **C**: Toggle convergence stats overlay
- **T**: Toggle debug test mode (fixed impulse for pressure solver analysis)
- **ESC**: Quit

## Algorithm Overview

The simulation follows the standard Stable Fluids pipeline:

1. **Density Advection** - Advect dye/density for visualization
2. **Velocity Advection** - Semi-Lagrangian advection of velocity field
3. **Force Application** - External forces (mouse interaction) with velocity clamping
4. **Pressure Projection** - Make velocity field divergence-free

### MAC Grid Implementation

The simulation uses a proper **MAC (Marker-and-Cell) staggered grid** with separate textures for each velocity component:

- **u-velocity**: 513×512 texture (horizontal velocity at vertical cell faces)
- **v-velocity**: 512×513 texture (vertical velocity at horizontal cell faces)
- **pressure/density**: 512×512 textures (cell centers)

This staggered arrangement naturally avoids checkerboard instabilities and ensures consistent discrete operators.

### Pressure Solver Details

The pressure projection step solves the Poisson equation:

```
∇²p = ∇·v
```

We use the **narrow 5-point Laplacian stencil**:

```
∇²p ≈ (p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - 4·p[i,j]) / h²
```

#### Red-Black Ordering with SOR

Red-Black ordering enables parallel updates by splitting cells into two groups based on a checkerboard pattern. SOR accelerates convergence:

```
p_SOR = p_old + ω·(p_GS - p_old)
```

Default parameters:
- **Iterations**: 512 (1024 half-passes)
- **ω (omega)**: 1.9

#### Operator Consistency

For the projection to be exact, the discrete operators must satisfy:

```
Laplacian[p] = Divergence[Gradient[p]]
```

This is achieved by using:
- **Divergence**: Forward differences from MAC face velocities
- **Gradient**: Backward differences to MAC face velocities

### Velocity Clamping

Injected velocities are clamped to **±3840 cells/second** (equivalent to 64 cells per timestep at 60fps) to prevent numerical instability from excessive force injection.

## Divergence Visualization

Press **V** to cycle through display modes. The divergence views use a **Tableau 10** color scale:

| Divergence | Color |
|------------|-------|
| 1e-7 | Gray |
| 1e-6 | Brown |
| 1e-5 | Pink |
| 1e-4 | Purple |
| 1e-3 | Yellow |
| 1e-2 | Green |
| 1e-1 | Teal |
| 1e0 | Red |
| 1e1 | Orange |
| 1e2 | Blue |
| ≥1e3 | White (clipping) |

## Convergence Statistics

Press **C** to toggle the convergence stats overlay, which displays a 2D histogram of divergence values:

- **Pre-divergence**: 36 bins (2^-24 to 2^11, covering up to ~2048)
- **Post-divergence**: 32 bins (2^-24 to 2^7, covering up to ~128)

The histogram shows how divergence is distributed before and after the pressure projection step. Well-converged simulations show post-divergence concentrated in low bins (negative exponents).

Press **C** again in the terminal to print the full histogram table.

## File Structure

```
├── main.c                        # Main simulation loop and setup
├── shaders/
│   ├── advect_u.comp             # U-velocity advection (513×512)
│   ├── advect_v.comp             # V-velocity advection (512×513)
│   ├── advect_density.comp       # Density/dye advection
│   ├── divergence.comp           # Compute velocity divergence
│   ├── pressure.comp             # Red-Black SOR pressure solver
│   ├── gradient_subtract_u.comp  # Pressure gradient for u
│   ├── gradient_subtract_v.comp  # Pressure gradient for v
│   ├── add_force_u.comp          # Force injection for u (with clamping)
│   ├── add_force_v.comp          # Force injection for v (with clamping)
│   ├── add_force_density.comp    # Dye injection
│   ├── divergence_stats.comp     # Convergence histogram
│   ├── quad.vert                 # Fullscreen quad vertex shader
│   ├── render.frag               # Visualization fragment shader
│   └── text.vert/frag            # Text overlay shaders
├── glad/                         # OpenGL loader
├── glfw/                         # Windowing library
├── CMakeLists.txt
└── build.bat
```

## Technical Notes

### Optimal SOR Parameter

For SOR on a Poisson problem with Dirichlet BCs on an N×N grid, the theoretical optimal is:

```
ω_opt = 2 / (1 + sin(π/N))
```

For N=512, this gives ω_opt ≈ 1.988. However, with finite iterations, a slightly lower value (~1.9) often works better in practice.

### Boundary Conditions

The solver uses **Dirichlet boundary conditions** (p = 0 at domain boundaries) with **open/outflow** velocity boundaries where flow can exit freely.

## Future Directions

See [Vertex_Grid.md](Vertex_Grid.md) for an alternative grid formulation where velocity lives at cell centers and pressure at vertices (the dual of MAC). This document derives the consistent 27-point Laplacian stencil for 3D and an iterative solution strategy using the dominant 9-point corner stencil as a preconditioner.

## References

- Stam, J. (1999). "Stable Fluids". SIGGRAPH 1999.
- Harris, M. (2004). "Fast Fluid Dynamics Simulation on the GPU". GPU Gems.
- Harlow, F.H. & Welch, J.E. (1965). "Numerical Calculation of Time-Dependent Viscous Incompressible Flow". Physics of Fluids.
