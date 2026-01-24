# Session Notes for Claude

## Project Overview

2D Stable Fluids simulation using OpenGL 4.3 compute shaders. Implements a proper MAC (Marker-and-Cell) staggered grid with Red-Black SOR pressure solver.

## Current State (January 2025)

The simulation is fully functional with:
- Proper MAC grid: separate u (513×512) and v (512×513) velocity textures
- Red-Black Gauss-Seidel with SOR (ω=1.9, 512 iterations)
- Velocity clamping at ±3840 cells/sec to prevent instability
- Tableau 10 color scale for divergence visualization
- 2D histogram for convergence analysis (36 pre-bins, 32 post-bins)

## Key Files

- `main.c` - All simulation logic, rendering, UI
- `shaders/advect_u.comp`, `advect_v.comp` - Velocity self-advection (separate for MAC)
- `shaders/divergence.comp` - Computes ∇·v from MAC faces
- `shaders/pressure.comp` - Red-Black SOR solver
- `shaders/gradient_subtract_u.comp`, `gradient_subtract_v.comp` - Pressure projection
- `shaders/render.frag` - Visualization with Tableau 10 divergence colors

## Architecture Decisions

1. **MAC Grid**: Velocity components stored at face centers, pressure/density at cell centers. This avoids checkerboard instability and ensures operator consistency.

2. **Operator Consistency**: Divergence uses forward differences, gradient uses backward differences. Composing them yields the narrow 5-point Laplacian used in the pressure solve.

3. **Velocity Clamping**: Force injection clamps to ±3840 cells/sec (64 cells/step at 60fps). This prevents numerical blowup from aggressive mouse input.

4. **Histogram Bins**: Pre-divergence has more bins (36) than post (32) because input divergence can be much larger than residual after projection.

## Potential Next Steps

- **Viscosity**: Add diffusion step (either explicit or implicit)
- **Multigrid**: Accelerate pressure solve for larger grids
- **Vorticity confinement**: Re-inject lost small-scale vortices
- **3D extension**: Would need 3D textures and more complex MAC grid
- **Different boundary conditions**: Solid obstacles, inflow/outflow regions
- **BFECC or MacCormack advection**: Reduce numerical diffusion

## Common Tasks

**Run simulation:**
```bash
cd "C:\JangaFX\Claude\Fluid Simulation"
./build/Release/StableFluids.exe
```

**Rebuild:**
```bash
cmake --build build --config Release
```

**Controls:**
- V: Cycle display modes (density/velocity/pre-div/post-div/pressure)
- C: Toggle convergence stats (also prints histogram to console)
- T: Debug test mode (fixed impulse)
- R: Reset

## Known Quirks

- The histogram printing to console happens when you toggle C twice (on then off triggers print)
- Force scale is 100 × SIM_WIDTH = 51,200 - can adjust in `addForce()` if needed
- Color scale runs from 1e-7 (gray) to 1e3 (white clipping)
