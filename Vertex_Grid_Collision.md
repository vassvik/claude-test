# Solid Obstacles and Boundary Conditions for Vertex Grid

> **Note:** This document extends the vertex grid formulation described in [Vertex_Grid.md](Vertex_Grid.md) to handle solid obstacles and boundary conditions. The same caveats apply: these derivations were developed for [EmberGen](https://jangafx.com/software/embergen/) and produced with AI assistance (Claude).

> **EXPERIMENTAL - UNVERIFIED:** This document is even more tentative than the main Vertex_Grid.md. The content was generated in a Claude session and has NOT been read, verified, or validated by the author. Derivations may contain errors. Treat everything here as a starting point for further investigation, not as authoritative reference material. (January 2025)

This document derives consistent boundary condition treatment for solid obstacles in the vertex grid formulation, where **velocity lives at cell centers** and **pressure lives at vertices**.

## Table of Contents

1. [Cell Classification](#cell-classification)
2. [Velocity Boundary Conditions](#velocity-boundary-conditions)
3. [Pressure Boundary Conditions](#pressure-boundary-conditions)
4. [Modified Divergence Operator](#modified-divergence-operator)
5. [Modified Gradient Operator](#modified-gradient-operator)
6. [Modified Laplacian Stencil Near Obstacles](#modified-laplacian-stencil-near-obstacles)
7. [Impact on Sublattice Decoupling](#impact-on-sublattice-decoupling)
8. [Ghost Cell Methods](#ghost-cell-methods)
9. [Immersed Boundary Methods](#immersed-boundary-methods)
   - [Signed Distance Fields (SDFs)](#signed-distance-fields-sdfs)
   - [Binarized SDFs and Voxelization](#binarized-sdfs-and-voxelization)
   - [Moving Obstacles](#moving-obstacles)
   - [Forcing Methods](#forcing-methods)
   - [SDF-Based Boundary Condition Enforcement](#sdf-based-boundary-condition-enforcement)
10. [Vertex Grid vs MAC Grid: Obstacle Handling](#vertex-grid-vs-mac-grid-obstacle-handling)
11. [Summary](#summary)

---

## Cell Classification

### Three-State Classification

Each cell in the grid is classified based on its relationship to solid obstacles:

| State | Symbol | Description |
|-------|--------|-------------|
| **FLUID** | F | Entirely within fluid domain |
| **SOLID** | S | Entirely within solid obstacle |
| **BOUNDARY** | B | Cut by obstacle surface (partial fluid) |

```
    Obstacle (shaded region)
    ╔═══════════════╗
    ║███████████████║
    ║███████████████║
    ║███████████████║
    ╚═══════════════╝

Grid overlay:

    p───────p───────p───────p───────p
    │       │       │       │       │
    │   S   │   S   │   B   │   F   │
    │       │       │▓▓▓    │       │
    p───────p───────p───────p───────p
    │       │       │▓▓▓▓▓▓▓│       │
    │   S   │   B   │▓▓▓▓▓▓▓│   F   │
    │       │▓▓▓▓▓▓▓│▓▓▓▓▓▓▓│       │
    p───────p───────p───────p───────p
    │       │▓▓▓▓▓▓▓│▓▓▓▓▓▓▓│       │
    │   F   │   B   │▓▓▓▓▓▓▓│   F   │
    │       │       │▓▓▓▓▓▓▓│       │
    p───────p───────p───────p───────p
    │       │       │       │       │
    │   F   │   F   │   F   │   F   │
    │       │       │       │       │
    p───────p───────p───────p───────p

▓ = solid region within cells
```

### Volume Fractions

For boundary cells, we define the **volume fraction** α_c as the fraction of the cell occupied by fluid:

```
α_c = V_fluid / V_cell
```

| Cell Type | α_c Value |
|-----------|-----------|
| FLUID | 1.0 |
| SOLID | 0.0 |
| BOUNDARY | 0 < α_c < 1 |

**Computing volume fractions:**

For a cell with corners at positions (x₀, y₀) to (x₁, y₁) and a signed distance function φ(x,y) where φ < 0 inside the solid:

1. **Simple approximation:** Sample φ at cell corners, count fluid corners
2. **Linear approximation:** Interpolate φ to find intersection points, compute polygon area
3. **Higher-order:** Use quadrature or analytic formulas for specific shapes

### Face Area Fractions

For flux computation, we also need **face area fractions** β_f representing the fraction of each cell face open to fluid flow:

```
    ┌─────────────────┐
    │                 │
    │     β_left      │
    │        ↓        │
    │   ┌────────┐    │
    │ β_│▓▓▓▓▓▓▓▓│    │
    │ bot▓▓▓▓▓▓▓▓│ β_right = 0.7
    │   │▓▓solid▓│    │
    │   └────────┘    │
    │        ↑        │
    │    β_bottom     │
    └─────────────────┘

β_left = 0.4, β_right = 0.7, β_top = 1.0, β_bottom = 0.6
```

Face area fractions are computed by finding where the obstacle surface intersects each face.

---

## Velocity Boundary Conditions

### Types of Velocity Boundary Conditions

| Condition | Description | Mathematical Form |
|-----------|-------------|-------------------|
| **No-slip** | Fluid velocity matches wall velocity | v_fluid = v_wall (usually 0) |
| **Free-slip** | No tangential stress, no normal flow | v·n = v_wall·n, (∂v/∂n)_tangent = 0 |
| **No-penetration** | Only normal component constrained | v·n = v_wall·n |

### The Challenge: Cell-Centered Velocity

In the vertex grid, velocity is stored at **cell centers**. When an obstacle cuts through the domain:

1. Some velocity samples fall inside the solid (invalid)
2. Valid fluid velocities may be arbitrarily close to the boundary
3. Boundary conditions must be enforced at the obstacle surface, not at grid points

```
    p───────p───────p───────p
    │       │▓▓▓▓▓▓▓│▓▓▓▓▓▓▓│
    │   ●   │▓▓ ×  ▓│▓▓ × ▓▓│  ● = valid velocity
    │ fluid │▓▓solid│▓▓solid▓│  × = invalid (inside solid)
    p───────p───────p───────p
    │       │       │▓▓▓▓▓▓▓│
    │   ●   │   ●   │▓▓ × ▓▓│
    │ fluid │ fluid │▓▓solid▓│
    p───────p───────p───────p
```

### Approach 1: Ghost Velocity Extrapolation

For cells inside the solid, compute a **ghost velocity** that enforces the boundary condition when interpolated with neighboring fluid velocities.

**No-slip condition:**

The boundary velocity should be zero (for stationary walls). If we linearly interpolate between the fluid cell and ghost cell, the zero should occur at the boundary:

```
    fluid cell        boundary        ghost cell
        ●────────────────|────────────────○
      v_fluid           v=0            v_ghost

        d_fluid                d_ghost
    ←───────────→        ←───────────→
```

Linear interpolation at boundary:
```
0 = (d_ghost · v_fluid + d_fluid · v_ghost) / (d_fluid + d_ghost)
```

Solving for ghost velocity:
```
v_ghost = -(d_ghost / d_fluid) · v_fluid
```

For the common case where the boundary is at the cell face (d_ghost = d_fluid):
```
v_ghost = -v_fluid    (no-slip, symmetric ghost)
```

**Free-slip condition:**

The normal component should be zero, tangential component unchanged:

```
v_ghost = v_fluid - 2(v_fluid · n̂)n̂
```

This is a **reflection** of the velocity about the tangent plane.

**Derivation of reflection formula:**

Let n̂ be the unit normal pointing into the fluid. Decompose velocity:
```
v_fluid = v_tangent + v_normal
v_normal = (v_fluid · n̂)n̂
v_tangent = v_fluid - (v_fluid · n̂)n̂
```

For free-slip, we want the interpolated velocity at the boundary to have:
- Normal component = 0 (no penetration)
- Tangential component = v_tangent (no shear stress)

The ghost velocity that achieves this:
```
v_ghost = v_tangent - v_normal
        = v_fluid - (v_fluid · n̂)n̂ - (v_fluid · n̂)n̂
        = v_fluid - 2(v_fluid · n̂)n̂
```

**Reflection matrix form:**

Define the reflection matrix R = I - 2n̂n̂ᵀ:
```
R = I - 2n̂n̂ᵀ = ⎛1-2nₓ²    -2nₓnᵧ   -2nₓnᵤ⎞
                ⎜-2nₓnᵧ   1-2nᵧ²   -2nᵧnᵤ⎟
                ⎝-2nₓnᵤ   -2nᵧnᵤ   1-2nᵤ²⎠
```

Then: `v_ghost = R · v_fluid`

### Approach 2: Velocity Extrapolation from Valid Cells

Instead of mirroring, extrapolate velocity values from the nearest valid fluid cells into solid cells:

```
v_solid = extrapolate(v_fluid_neighbors)
```

Methods:
1. **Constant extrapolation:** Copy nearest fluid velocity
2. **Linear extrapolation:** Use gradient from fluid region
3. **PDE-based:** Solve ∂v/∂n = 0 propagating outward from fluid

This approach is simpler but less accurate for enforcing specific boundary conditions.

### Approach 3: Volume-Weighted Averaging

For boundary cells (partially solid), weight the velocity by the fluid volume fraction:

```
v_effective = α_c · v_cell + (1 - α_c) · v_boundary
```

Where v_boundary is the desired velocity at the solid surface (zero for no-slip walls).

### Summary: Velocity BC Methods

| Method | Pros | Cons |
|--------|------|------|
| Ghost mirroring | Accurate BC enforcement | Requires normal computation |
| Extrapolation | Simple implementation | Less accurate BCs |
| Volume weighting | Smooth near boundaries | Doesn't enforce BC at surface |

---

## Pressure Boundary Conditions

### The Neumann Condition

At solid boundaries, the pressure gradient normal to the wall must balance any wall acceleration:

```
∂p/∂n = -ρ(∂v_wall/∂t · n̂)
```

For **stationary walls**, this simplifies to:
```
∂p/∂n = 0    (homogeneous Neumann)
```

**Derivation from momentum equation:**

The momentum equation at the boundary:
```
∂v/∂t = -∇p/ρ + (other terms)
```

At a solid wall, v = v_wall, so:
```
∂v_wall/∂t = -∇p/ρ + ...
```

Taking the normal component:
```
∂v_wall/∂t · n̂ = -(∂p/∂n)/ρ + ...
```

For inviscid flow with a stationary wall:
```
0 = -∂p/∂n
∂p/∂n = 0
```

### Modified Gradient Stencil Near Obstacles

When a pressure vertex is inside a solid obstacle, the gradient at neighboring fluid cells must be modified.

**Standard gradient (2D):**
```
∂p/∂x at vel[i,j] = (p[i+1,j+1] + p[i+1,j] - p[i,j+1] - p[i,j]) / (2h)
```

**With blocked vertex p[i,j]:**

If p[i,j] is inside a solid, we cannot use it directly. Options:

1. **Constant extrapolation (Neumann):** Set p[i,j] = p_nearest_fluid
2. **Modified stencil:** Redistribute weight to remaining valid vertices
3. **One-sided difference:** Use only valid vertices

**Example: p[i,j] blocked, extrapolate from p[i+1,j]:**
```
p[i,j] ≈ p[i+1,j]    (constant extrapolation = Neumann BC)
```

Modified gradient:
```
∂p/∂x at vel[i,j] = (p[i+1,j+1] + p[i+1,j] - p[i,j+1] - p[i+1,j]) / (2h)
                  = (p[i+1,j+1] - p[i,j+1]) / (2h)
```

The stencil now uses only 2 vertices instead of 4.

### Ghost Pressure Values

For pressure vertices inside solids, define ghost values:

**Constant extrapolation (enforces ∂p/∂n = 0):**
```
p_ghost = p_nearest_fluid
```

**Linear extrapolation (for non-zero ∂p/∂n):**
```
p_ghost = p_fluid - d · (∂p/∂n)_boundary
```

Where d is the distance from the fluid vertex to the ghost vertex location.

---

## Modified Divergence Operator

### Standard Divergence (Review)

At vertex p[i,j], the divergence samples the 4 surrounding cells:
```
∇·v[i,j] = (1/2h) × [ (u+v)[i,j] + (u-v)[i,j-1] + (v-u)[i-1,j] - (u+v)[i-1,j-1] ]
```

### Volume-Weighted Divergence

When cells have different fluid volume fractions, weight each contribution:

```
∇·v[i,j] = (1/2h) × [ α[i,j]·(u+v)[i,j]
                     + α[i,j-1]·(u-v)[i,j-1]
                     + α[i-1,j]·(v-u)[i-1,j]
                     - α[i-1,j-1]·(u+v)[i-1,j-1] ]
```

Where α[i,j] is the volume fraction of cell [i,j].

**Physical interpretation:** We're computing the net flux out of the fluid portion of the control volume around the vertex.

### Flux-Based Interpretation with Area Fractions

More accurately, divergence measures net flux through faces. Using face area fractions β:

```
    ┌───────────────┐
    │               │
    │   vel[i-1,j]  β_e  vel[i,j]
    │       ↑       │       ↑
    │      flux     │      flux
    p[i-1,j]──β_n───p[i,j]──β_n───p[i+1,j]
    │      flux     │      flux
    │       ↓       │       ↓
    │  vel[i-1,j-1] β_e  vel[i,j-1]
    │               │
    └───────────────┘
```

The flux through each face segment:
```
flux_east  = β_e · (u velocity component)
flux_north = β_n · (v velocity component)
```

Net outward flux at vertex p[i,j]:
```
∇·v[i,j] = (1/2h) × [ β_ne·(u+v)[i,j] + β_se·(u-v)[i,j-1]
                     + β_nw·(v-u)[i-1,j] - β_sw·(u+v)[i-1,j-1] ]
```

Where β_ne, β_se, β_nw, β_sw are effective area fractions for the four quadrants.

### Consistency Requirement

For the pressure projection to work correctly:
```
modified_div · modified_grad = modified_Laplacian
```

When we modify divergence and gradient operators near obstacles, the composed operator must remain consistent with whatever Laplacian we use in the pressure solve.

---

## Modified Gradient Operator

### Standard Gradient (Review)

At cell vel[i,j], the gradient uses 4 surrounding pressure vertices:
```
∂p/∂x = (p[i+1,j+1] + p[i+1,j] - p[i,j+1] - p[i,j]) / (2h)
∂p/∂y = (p[i+1,j+1] + p[i,j+1] - p[i+1,j] - p[i,j]) / (2h)
```

### Gradient with Blocked Vertices

When pressure vertices lie inside solid obstacles, we must modify the stencil.

**Single blocked corner (e.g., p[i,j] blocked):**

```
    p[i,j+1]────p[i+1,j+1]
        │           │
        │  vel[i,j] │
        │     ●     │
    p[i,j]──────p[i+1,j]
      ╳ (blocked)
```

Options:

1. **Neumann extrapolation:** p[i,j] = average of valid neighbors
   ```
   p[i,j] ≈ (p[i+1,j] + p[i,j+1]) / 2
   ```

2. **One-sided stencil:** Exclude blocked vertex, renormalize
   ```
   ∂p/∂x ≈ (p[i+1,j+1] + p[i+1,j] - 2·p[i,j+1]) / (2h)
   ```

3. **Distance-weighted:** Weight by distance from cell center to valid vertices

**Two blocked corners (half the stencil blocked):**

```
    p[i,j+1]────p[i+1,j+1]
        │           │
        │  vel[i,j] │
        │     ●     │
    p[i,j]──────p[i+1,j]
      ╳             ╳
   (both blocked)
```

Gradient becomes one-sided:
```
∂p/∂x = (p[i+1,j+1] - p[i,j+1]) / (2h)    [or use h if one-sided]
∂p/∂y = (p[i+1,j+1] + p[i,j+1]) / (2h) - p_extrapolated / h
```

### Velocity Cells Inside Solids

If the velocity cell center itself is inside a solid, we typically:

1. Don't compute gradient (velocity is constrained by BC)
2. Use the ghost velocity approach instead
3. Set velocity to wall velocity (usually zero)

---

## Modified Laplacian Stencil Near Obstacles

### 2D: Modified Diagonal 5-Point Stencil

The standard 2D vertex grid Laplacian uses diagonal neighbors:

```
        p[i-1,j+1]          p[i+1,j+1]
             (+1)            (+1)
               \              /
                \            /
                  p[i,j]
                   (-4)
                /          \
               /            \
             (+1)            (+1)
        p[i-1,j-1]          p[i+1,j-1]
```

**One corner blocked (e.g., p[i+1,j+1] inside solid):**

We must remove the blocked vertex and adjust the center coefficient to maintain row sum = 0.

```
        p[i-1,j+1]          p[i+1,j+1]
             (+1)              ╳ (blocked)
               \
                \
                  p[i,j]
                   (-3)    ← was -4, now -3
                /          \
               /            \
             (+1)            (+1)
        p[i-1,j-1]          p[i+1,j-1]
```

**Modified stencil coefficients:**

| Neighbor | Standard | One corner blocked |
|----------|----------|-------------------|
| Center | -4 | -3 |
| Remaining corners | +1 each | +1 each |
| Blocked corner | +1 | 0 |
| **Sum** | 0 | 0 ✓ |

**General rule:** For each blocked neighbor with coefficient c, set that coefficient to 0 and add -c to the center.

### 3D: Modified 27-Point Stencil

The 3D vertex grid Laplacian has coefficients (×1/16h²):

| Type | Coefficient | Count |
|------|-------------|-------|
| Center | -24 | 1 |
| Face | -4 | 6 |
| Edge | +2 | 12 |
| Corner | +3 | 8 |

**Blocking rules:**

When a neighbor is blocked (inside solid), set its coefficient to 0 and adjust the center:

| Blocked Type | Original Coeff | Center Adjustment |
|--------------|----------------|-------------------|
| Face | -4 | +4 to center |
| Edge | +2 | -2 to center |
| Corner | +3 | -3 to center |

**Example: One corner blocked**

Original center: -24
Blocked corner coefficient: +3 → 0
Adjusted center: -24 + (-3) = **-27**

Wait, that's wrong! Let me reconsider...

**Correct blocking rule:** Remove the blocked term and maintain sum = 0.

Original sum: -24 + 6×(-4) + 12×(+2) + 8×(+3) = -24 - 24 + 24 + 24 = 0 ✓

If we block a corner (+3):
- Remove: +3 → 0
- To maintain sum = 0: center goes from -24 to -24 - 3 = **-21**

If we block a face (-4):
- Remove: -4 → 0
- To maintain sum = 0: center goes from -24 to -24 + 4 = **-20**

If we block an edge (+2):
- Remove: +2 → 0
- To maintain sum = 0: center goes from -24 to -24 - 2 = **-22**

**Corrected table:**

| Blocked Type | Original Coeff c | New Center = old + (-c) |
|--------------|------------------|-------------------------|
| Face | -4 | -24 - (-4) = -20 |
| Edge | +2 | -24 - (+2) = -26 |
| Corner | +3 | -24 - (+3) = -21 |

Hmm, I had the signs backwards. The rule is: **center_new = center_old - coefficient_removed**.

### SPD Preservation

The modified Laplacian must remain **symmetric positive semi-definite (SPSD)** for standard iterative solvers to converge.

**Symmetry:** If we block the connection from vertex A to vertex B, we must also block B to A (same coefficient removed from both rows).

**Semi-definiteness:** The stencil should have:
- Non-positive center coefficient
- Non-positive row sum (≤ 0)
- The matrix L should satisfy xᵀLx ≤ 0 for all x

For the vertex grid Laplacian with obstacles, SPD is preserved as long as:
1. Blocking is symmetric
2. We don't create isolated fluid regions

**Potential issue:** The 3D 27-point stencil has **positive edge and corner coefficients**. Blocking too many negative (face) terms while leaving positive terms can break definiteness. In practice, this rarely occurs because solid obstacles tend to block spatially connected sets of neighbors.

---

## Impact on Sublattice Decoupling

### Why Obstacles Break Decoupling

The corner stencil in 3D has perfect sublattice structure: all 8 corners of a cell flip all three parities, so each sublattice is independent.

**But the full 27-point stencil couples sublattices** through face and edge terms:
- Face neighbors flip 0 parities (same sublattice)
- Edge neighbors flip 2 parities (different sublattice)

In the obstacle-free case, the face terms sum to zero and don't break the decoupling (they're purely repulsive). But near obstacles:

### Mathematical Derivation of Coupling

Consider a pressure vertex p at position (even, even, even). Its face neighbor p_face at (odd, even, even) is on a different sublattice.

In the interior, the face contribution to vertex p is:
```
Face contribution = (-4/16h²) × [p(i+1,j,k) + p(i-1,j,k) + ... ] = -24/16h² × p_face_avg
```

But the divergence term (right-hand side) also involves face-neighbor cells, so the net coupling cancels.

**Near an obstacle:** If some face neighbors are blocked, the cancellation is incomplete:

```
    Blocked solid region
    ▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓▓▓ p_face ▓▓
    ───────────────────
          │
          │
          p (even,even,even)
          │
          │
        p_other_face
```

If p_face is blocked, the stencil for p no longer includes that face term. But the edge and corner terms that connected p to the opposite sublattice remain. This creates **asymmetric coupling** between sublattices near the obstacle.

### Quantifying the Coupling

Define the "coupling strength" C as the magnitude of cross-sublattice terms near obstacles.

For a pressure vertex with n_blocked neighbors blocked:

```
C ∝ (blocked face terms) - (modified corner/edge balance)
```

**Interior (n_blocked = 0):** C = 0 (perfect decoupling)
**Single blocked face:** C ≈ 4/16h² (one unbalanced face term)
**Multiple blocked:** C accumulates

### Practical Classification

| Scenario | Coupling | Strategy |
|----------|----------|----------|
| No obstacles | None | Full sublattice decoupling |
| Sparse obstacles | Localized | Decoupled interior + coupled boundary layer |
| Dense obstacles | Global | Full coupled solve |

**Hybrid Strategy:**

```
1. Classify vertices:
   - INTERIOR: no blocked neighbors within 1-2 cells
   - BOUNDARY_LAYER: near obstacles

2. For INTERIOR vertices:
   - Use decoupled corner-stencil solve per sublattice

3. For BOUNDARY_LAYER vertices:
   - Use coupled 27-point stencil
   - Or use iterative correction to handle coupling

4. Iterate between regions until converged
```

### Sublattice Coupling Illustration

```
Sublattice A (even parity sum)    Sublattice B (odd parity sum)
    ●───────────●                     ○───────────○
    │           │                     │           │
    │           │                     │           │
    │           │                     │           │
    ●───────────●                     ○───────────○

Interior: A and B solve independently

    ●───────────●                     ○───────────○
    │           │                     │    ▓▓▓    │
    │           │         →→→         │    ▓▓▓    │
    │           │       coupling      │    ▓▓▓    │
    ●───────────●                     ○───────────○

Near obstacle: A needs values from B (and vice versa)
```

---

## Ghost Cell Methods

Ghost cell methods extend the computational domain into solid regions with artificial values that enforce boundary conditions.

### Ghost Velocity Computation

**For no-slip walls:**

```
v_ghost = 2·v_wall - v_fluid = -v_fluid    (if v_wall = 0)
```

This ensures linear interpolation gives v = v_wall at the boundary.

**Diagram:**

```
    SOLID          │ BOUNDARY        FLUID
                   │
    ○ ghost        │                 ● fluid cell
    v_ghost        │    v = 0        v_fluid
                   │    at wall
    ←───── d ─────→│←───── d ─────→

    v_ghost = -v_fluid (reflection through wall)
```

**For free-slip walls:**

```
v_ghost = R · v_fluid    where R = I - 2n̂n̂ᵀ
```

### Ghost Pressure Computation

**For Neumann BC (∂p/∂n = 0):**

```
p_ghost = p_fluid    (constant extrapolation)
```

This gives zero normal gradient at the boundary.

**For non-zero Neumann (accelerating walls):**

```
p_ghost = p_fluid - 2d · (∂p/∂n)_boundary
```

Where d is the distance from the boundary to the ghost location.

### Advantages and Limitations

| Aspect | Advantage | Limitation |
|--------|-----------|------------|
| **Simplicity** | Easy to implement | Requires ghost region storage |
| **Accuracy** | Good for aligned boundaries | Less accurate for curved/diagonal |
| **Consistency** | Maintains stencil structure | Ghost values may be inconsistent at corners |
| **Parallelism** | Ghost fill is parallel | Requires halo exchange |
| **Cut cells** | Works with volume fractions | Complex geometry needs care |

---

## Immersed Boundary Methods

Immersed boundary methods handle obstacles without modifying the underlying grid structure, instead using level set representations (signed distance fields) and forcing terms to enforce boundary conditions.

### Signed Distance Fields (SDFs)

A **signed distance field** φ(x) represents geometry implicitly:

```
φ(x) = ±dist(x, ∂Ω)

where:
  φ(x) < 0   inside solid
  φ(x) = 0   on boundary (zero level set)
  φ(x) > 0   outside solid (in fluid)
```

**Properties of a true SDF:**

| Property | Mathematical Form | Meaning |
|----------|-------------------|---------|
| Eikonal | \|∇φ\| = 1 | Gradient magnitude is unity everywhere |
| Normal | n̂ = ∇φ/\|∇φ\| = ∇φ | Gradient gives outward surface normal |
| Distance | \|φ(x)\| | Magnitude gives distance to nearest surface |
| Projection | x_surface = x - φ(x)·∇φ(x) | Projects any point onto surface |

**Diagram: SDF around a circular obstacle**

```
    +3  +2  +2  +2  +3
    +2  +1  +1  +1  +2
    +2  +1  -1  +1  +2      ← negative inside
    +2  +1  +1  +1  +2
    +3  +2  +2  +2  +3

    Contour lines:
    ┌─────────────────────┐
    │     ╭───────╮       │  ← φ = +2
    │   ╭─│───────│─╮     │  ← φ = +1
    │   │ │ ●●●●● │ │     │  ← φ = 0 (boundary)
    │   │ │ ●●●●● │ │     │  ← φ = -1
    │   ╰─│───────│─╯     │
    │     ╰───────╯       │
    └─────────────────────┘
```

**Computing SDFs:**

For common primitives, analytic SDFs exist:

| Shape | SDF Formula |
|-------|-------------|
| Sphere | φ = \|x - c\| - r |
| Box (axis-aligned) | φ = \|max(abs(x-c) - half_size, 0)\| + min(max_component, 0) |
| Plane | φ = (x - p)·n̂ |
| Cylinder | φ = \|x_perp - c_perp\| - r (for infinite cylinder) |
| Torus | φ = \|(r_major - \|x_xy\|, x_z)\| - r_minor |

**SDF Operations (CSG):**

| Operation | Formula | Description |
|-----------|---------|-------------|
| Union | min(φ_A, φ_B) | A ∪ B |
| Intersection | max(φ_A, φ_B) | A ∩ B |
| Difference | max(φ_A, -φ_B) | A \ B |
| Smooth union | -ln(e^{-k·φ_A} + e^{-k·φ_B})/k | Blended join |
| Shell | abs(φ) - thickness | Hollow shell |
| Offset | φ - offset | Dilate/erode |

**Discretizing SDFs on the grid:**

Store φ values at either:
- **Cell centers** (collocated with velocity in vertex grid)
- **Vertices** (collocated with pressure in vertex grid)
- **Both** (for different operations)

```
Vertex-stored SDF:             Cell-stored SDF:
    φ───────φ───────φ              ┌───────┬───────┐
    │       │       │              │   φ   │   φ   │
    │       │       │              │       │       │
    │       │       │              ├───────┼───────┤
    φ───────φ───────φ              │   φ   │   φ   │
    │       │       │              │       │       │
    │       │       │              └───────┴───────┘
    φ───────φ───────φ
```

**Normal computation from discrete SDF:**

Central differences on the SDF give the (unnormalized) normal:

```
∇φ ≈ (φ[i+1,j,k] - φ[i-1,j,k],
      φ[i,j+1,k] - φ[i,j-1,k],
      φ[i,j,k+1] - φ[i,j,k-1]) / (2h)

n̂ = ∇φ / |∇φ|
```

### Binarized SDFs and Voxelization

For simpler or faster implementations, the continuous SDF can be **binarized** into a solid mask:

```
solid[i,j,k] = (φ[i,j,k] < 0) ? 1 : 0
```

**Voxelization methods:**

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Point sampling | Check if cell center is inside | Fast, simple | Misses thin features |
| Conservative | Mark solid if ANY part of cell is inside | No missed solids | Thickens geometry |
| 6-neighbor flood | Propagate from known exterior | Handles closed meshes | Slow, sequential |
| Scanline | Ray-cast along axes, toggle inside/outside | GPU-friendly | Needs watertight mesh |

**Point sampling voxelization:**

```
for each cell (i,j,k):
    x_center = (i + 0.5) * h
    if sdf(x_center) < 0:
        solid[i,j,k] = 1
    else:
        solid[i,j,k] = 0
```

**Conservative voxelization:**

Sample at cell corners (8 points in 3D):

```
for each cell (i,j,k):
    any_inside = false
    for each corner (di,dj,dk) in {0,1}³:
        x_corner = (i + di, j + dj, k + dk) * h
        if sdf(x_corner) < 0:
            any_inside = true
    solid[i,j,k] = any_inside ? 1 : 0
```

**Diagram: Point vs Conservative voxelization**

```
Actual obstacle boundary: curved line
    ╲
     ╲
      ╲

Point sampling:              Conservative:
┌───┬───┬───┐               ┌───┬───┬───┐
│   │   │   │               │▓▓▓│▓▓▓│   │
├───┼───┼───┤               ├───┼───┼───┤
│   │ ▓ │   │               │▓▓▓│▓▓▓│▓▓▓│
├───┼───┼───┤               ├───┼───┼───┤
│   │   │   │               │   │▓▓▓│▓▓▓│
└───┴───┴───┘               └───┴───┴───┘

Only center inside →        Any corner inside →
misses boundary cells       includes boundary cells
```

**Limitations of binarization:**

1. **Loss of sub-cell information:** Cannot compute accurate normals or distances
2. **Staircase boundaries:** Diagonal surfaces become stair-stepped
3. **No volume fractions:** Cannot do cut-cell methods
4. **Resolution-dependent:** Features smaller than cell size disappear

**Hybrid approach: Binary mask + narrow-band SDF**

Store full SDF only near the boundary (narrow band), use binary mask elsewhere:

```
if |φ[i,j,k]| < bandwidth:
    store φ[i,j,k]  (full precision)
else:
    store sign(φ)    (just inside/outside)
```

This saves memory while preserving boundary accuracy.

### Moving Obstacles

When obstacles move, we must track:
1. The time-varying geometry (SDF or mask)
2. The obstacle velocity field (for boundary conditions)
3. Fresh cells (fluid entering previously solid regions)

**Time-varying SDF:**

For a rigidly moving obstacle with position c(t) and rotation R(t):

```
φ(x, t) = φ_reference(R(t)ᵀ · (x - c(t)))
```

Where φ_reference is the SDF in the obstacle's local coordinates.

**Obstacle velocity field:**

For rigid body motion with center c, velocity v_c, and angular velocity ω:

```
v_obstacle(x) = v_c + ω × (x - c)
```

This is the velocity that boundary conditions should enforce (not necessarily zero).

**Diagram: Moving obstacle velocity field**

```
        ω (rotation)
         ↺
    ┌─────────┐
    │    ●────│→ v_c (translation)
    │   /│    │
    │  / │    │
    └─────────┘

Velocity at point x on surface:
v_obstacle = v_c + ω × (x - center)
```

**Fresh cell problem:**

When an obstacle moves, cells transition between states:

| Transition | Issue | Solution |
|------------|-------|----------|
| FLUID → SOLID | Cell is now blocked | Zero out velocity, exclude from solve |
| SOLID → FLUID | "Fresh" cell needs valid values | Extrapolate from neighbors |
| BOUNDARY → FLUID | Partial to full | Adjust volume fractions |
| FLUID → BOUNDARY | Full to partial | Adjust volume fractions |

**Fresh cell velocity initialization:**

When a cell becomes fluid (obstacle moving away), we need valid velocity values:

```
Option 1: Obstacle velocity
    v_fresh = v_obstacle(x_cell)

Option 2: Extrapolation from existing fluid
    v_fresh = weighted_average(v_fluid_neighbors)

Option 3: Solve for consistent velocity
    Include fresh cells in pressure projection
    Let projection determine divergence-free velocity
```

**Algorithm for moving obstacles:**

```
Each timestep:

1. Update obstacle geometry:
   φ^{n+1}(x) = φ_ref(R^{n+1}ᵀ · (x - c^{n+1}))

2. Identify cell transitions:
   for each cell:
       was_solid = (φ^n < 0)
       now_solid = (φ^{n+1} < 0)
       if was_solid and not now_solid:
           mark as FRESH
       if not was_solid and now_solid:
           mark as NEWLY_SOLID

3. Initialize fresh cells:
   for each FRESH cell:
       v = v_obstacle(x_cell)  # or extrapolate

4. Advection (skip solid cells):
   v* = advect(v, Δt)

5. Apply obstacle velocity BC:
   for each cell inside solid:
       v* = v_obstacle(x_cell)

6. Pressure projection (modified for obstacles):
   Solve ∇²p = ∇·v* with Neumann BC at solid boundaries
   v^{n+1} = v* - ∇p

7. Final BC enforcement:
   for each cell inside solid:
       v^{n+1} = v_obstacle(x_cell)
```

**CFL condition with moving obstacles:**

The timestep must satisfy:

```
Δt < h / max(|v_fluid|, |v_obstacle|)
```

Fast-moving obstacles require smaller timesteps.

**Two-way coupling (optional):**

For obstacles that respond to fluid forces:

```
F_fluid→solid = -∫_∂Ω p·n̂ dA + ∫_∂Ω τ·n̂ dA

τ_fluid→solid = ∫_∂Ω (x - c) × (-p·n̂ + τ·n̂) dA

Update obstacle motion:
m · dv_c/dt = F_fluid→solid + F_external
I · dω/dt = τ_fluid→solid + τ_external
```

### Forcing Methods

Immersed boundary methods use forcing terms rather than grid modification.

#### Penalty Method (Soft Boundary)

Add a large penalty force that drives velocity toward the wall velocity:

```
f_penalty = -β · (v - v_wall)    for points inside solid
```

Where β is a large penalty coefficient.

**Implementation:**
```
v* = v + Δt · (advection + diffusion + external forces)
v** = v* + Δt · f_penalty    (only inside solid)
Solve pressure, project
```

**Choosing β:**

| β value | Effect |
|---------|--------|
| Small | Soft boundary, significant penetration |
| ~ 1/Δt | Moderate enforcement |
| >> 1/Δt | Stiff, may need smaller Δt |

Practical choice: β ≈ 1/Δt² gives reasonable enforcement without excessive stiffness.

**Issues:**
- Boundary is "soft" - some penetration always occurs
- Trade-off between accuracy and stability
- Difficult to achieve machine-precision BC enforcement

#### Direct Forcing (Sharp Interface)

Compute the exact force needed to achieve the boundary velocity:

```
f_IBM = (v_target - v*) / Δt    at boundary cells
```

This is applied **after** computing v* but **before** pressure projection.

**For cells fully inside solid:**
```
v_corrected = v_obstacle    (direct assignment, not forcing)
```

**For boundary cells (cut cells):**

Interpolate to find the effective boundary location, then apply forcing:

```
1. Find boundary location x_b where φ = 0
2. Interpolate v* to x_b
3. Compute f = (v_target(x_b) - v*(x_b)) / Δt
4. Spread f back to grid points
```

#### Feedback Forcing

For steady-state problems, use integral feedback:

```
f^{n+1} = f^n + α · (v^n - v_target)
```

This accumulates until the error is eliminated.

**PID-style enhancement:**
```
f^{n+1} = Kp·(v - v_target) + Ki·∫(v - v_target)dt + Kd·d(v - v_target)/dt
```

### SDF-Based Boundary Condition Enforcement

Using the SDF, we can enforce boundary conditions with sub-cell accuracy.

**Velocity correction using SDF:**

For a cell at position x with velocity v and obstacle velocity v_obs:

```
if φ(x) < 0:  # Inside solid
    v_corrected = v_obs(x)

elif φ(x) < h:  # In boundary layer (one cell width)
    # Blend based on distance to surface
    t = φ(x) / h  # 0 at surface, 1 at one cell away
    v_corrected = lerp(v_obs(x_surface), v, t)

else:  # In fluid
    v_corrected = v
```

**Normal velocity enforcement (no-penetration):**

Only constrain the normal component:

```
n̂ = ∇φ / |∇φ|
v_normal = (v · n̂) · n̂
v_tangent = v - v_normal

v_obs_normal = (v_obs · n̂) · n̂

if φ(x) < threshold:
    v_corrected = v_tangent + v_obs_normal
```

**Extrapolation into solid region:**

For ghost values or fresh cells, extrapolate φ along its gradient:

```
repeat N times:
    for each solid cell adjacent to fluid:
        φ_neighbors = average of fluid neighbor φ values
        solid cell φ = φ_neighbors - h  # Extend SDF
        solid cell v = average of fluid neighbor v values
```

This propagates information from fluid into solid along the normal direction.

### Comparison: Methods Summary

| Method | BC Accuracy | Implementation | Moving Obstacles | Stability |
|--------|-------------|----------------|------------------|-----------|
| Ghost cell | High | Moderate (modify stencils) | Rebuild ghosts each step | Good |
| Penalty forcing | Low-Medium | Simple (add force term) | Easy (update v_target) | May need small Δt |
| Direct forcing | High | Moderate (interpolation) | Easy | Good |
| Feedback forcing | High (steady state) | Simple | Slower adaptation | Good |
| SDF-based blending | Medium | Moderate | Easy | Good |
| Binary mask | Low | Simple | Easy | Good |

### Practical Recommendations

**For real-time applications (games, VFX):**
- Use binary voxelization for speed
- Direct velocity assignment inside solid
- Simple extrapolation for fresh cells
- Accept some boundary smearing

**For accuracy-critical applications:**
- Use narrow-band SDF with sub-cell resolution
- Ghost cell method with proper normal computation
- Cut-cell volume fractions for boundary cells
- Careful fresh cell handling

**For moving obstacles:**
- Store obstacle velocity field (not just geometry)
- Track cell state transitions explicitly
- Initialize fresh cells from obstacle velocity
- Consider CFL implications of fast-moving obstacles

---

## Vertex Grid vs MAC Grid: Obstacle Handling

### Structural Differences

| Aspect | MAC Grid | Vertex Grid |
|--------|----------|-------------|
| Velocity location | Face centers | Cell centers |
| Flux computation | Direct (face velocities) | Interpolated |
| Ghost cells | Separate u,v,w ghosts | Single vector ghost |
| Normal velocity | Natural at faces | Must interpolate |
| Tangent velocity | Must interpolate | Natural at centers |

### MAC Grid Advantages for Obstacles

1. **Face velocities define flux directly:**
   - The velocity component normal to a face IS the flux through that face
   - Blocking a face = setting one velocity component to zero
   - No interpolation needed for flux computation

2. **Clear blocking semantics:**
   - Block a face → that velocity component is wall-constrained
   - Each velocity component has independent boundary conditions

3. **Conservation is natural:**
   - Divergence counts fluxes through faces
   - Blocking a face removes its flux contribution exactly

### Vertex Grid Advantages for Obstacles

1. **Symmetric ghost cell treatment:**
   - All velocity components at same location
   - Single reflection operation for free-slip
   - Consistent ghost cell fill pattern

2. **Collocated storage:**
   - Velocity vector stored together → better cache behavior
   - Single ghost cell region (not separate u/v/w halos)
   - Simpler data structures for complex geometry

3. **Interpolation to boundary:**
   - Can interpolate full velocity vector to any point
   - Easier to compute wall shear stress, etc.

### When to Prefer Each

| Scenario | Preferred Grid | Reason |
|----------|----------------|--------|
| Axis-aligned obstacles | Either | Both handle well |
| Complex curved geometry | Vertex | Simpler ghost handling |
| Strict conservation needed | MAC | Direct flux control |
| Moving obstacles | Vertex | Easier velocity updates |
| Thin shell obstacles | MAC | Can block single face |
| Coupled multiphysics | Vertex | Collocated data |

---

## Summary

### Key Formulas

**Ghost velocity (no-slip):**
```
v_ghost = -v_fluid
```

**Ghost velocity (free-slip):**
```
v_ghost = v_fluid - 2(v_fluid · n̂)n̂ = (I - 2n̂n̂ᵀ) · v_fluid
```

**Ghost pressure (Neumann):**
```
p_ghost = p_fluid
```

**Modified 2D Laplacian (one blocked corner):**
```
L[p] = (1/2h²) × [Σ(unblocked corners) - 3·p_center]
```

**Modified 3D Laplacian coefficient rules:**

| Blocked Type | Center Adjustment |
|--------------|-------------------|
| Face (-4) | -24 → -20 |
| Edge (+2) | -24 → -26 |
| Corner (+3) | -24 → -21 |

### Summary Table

| Property | 2D Vertex Grid | 3D Vertex Grid |
|----------|----------------|----------------|
| **Base Laplacian** | Diagonal 5-point | 27-point |
| **Obstacle blocking** | Modify diagonal stencil | Modify 27-point, breaks decoupling |
| **Ghost cell fill** | Simple reflection | Same, but 3D normal |
| **Volume fractions** | 4 corners per cell | 8 corners per cell |
| **Face fractions** | 4 faces | 6 faces |
| **Sublattice impact** | Still decoupled | Boundary layer couples sublattices |

### Implementation Checklist

1. **Cell classification:** Determine FLUID/SOLID/BOUNDARY for each cell
2. **Volume/face fractions:** Compute α_c and β_f for boundary cells
3. **Ghost velocity fill:** Populate ghost cells with reflected values
4. **Ghost pressure fill:** Constant extrapolation for Neumann BC
5. **Modified divergence:** Weight by volume fractions
6. **Modified gradient:** Handle blocked vertices
7. **Modified Laplacian:** Adjust coefficients, maintain symmetry
8. **Solver adjustment:** Account for coupling if using sublattice decomposition

---

## References

For the base vertex grid formulation, see [Vertex_Grid.md](Vertex_Grid.md).

Additional references on boundary conditions in CFD:

- Fedkiw, R., Aslam, T., Merriman, B., & Osher, S. (1999). A non-oscillatory Eulerian approach to interfaces in multimaterial flows (the ghost fluid method). *Journal of Computational Physics*, 152(2), 457-492.

- Mittal, R., & Iaccarino, G. (2005). Immersed boundary methods. *Annual Review of Fluid Mechanics*, 37, 239-261.

- Peskin, C. S. (2002). The immersed boundary method. *Acta Numerica*, 11, 479-517.
