# Vertex Grid (Dual Grid) Formulation for Fluid Simulation

This document derives the consistent discrete Laplacian for a "vertex grid" (also called dual grid) arrangement where **velocity lives at cell centers** and **pressure lives at vertices**. This is the dual of the standard MAC grid.

## Table of Contents

1. [Grid Setup](#grid-setup)
2. [2D Vertex Grid](#2d-vertex-grid)
3. [3D Vertex Grid](#3d-vertex-grid)
4. [Stencil Decomposition](#stencil-decomposition)
5. [Taylor Expansion Analysis](#taylor-expansion-analysis)
6. [Operator Consistency vs Truncation Error](#operator-consistency-vs-truncation-error)
7. [Iterative Solution Strategy](#iterative-solution-strategy)
8. [Lattice Decoupling and Cache Optimization](#lattice-decoupling-and-cache-optimization)

---

## Grid Setup

### Standard MAC Grid (for comparison)
- Pressure at cell centers
- Velocity components at cell faces (u at vertical faces, v at horizontal faces)
- Natural for divergence/gradient operators

### Vertex Grid (Dual Grid)
- **Velocity at cell centers** (both components collocated)
- **Pressure at vertices** (corners of cells)
- Dye/density naturally collocated with velocity

This arrangement is attractive because velocity and dye live together, simplifying advection. The challenge is deriving consistent discrete operators.

---

## 2D Vertex Grid

### Setup

Consider a 2D grid with spacing h:
- Velocity (u,v) stored at cell centers: (i+0.5, j+0.5)·h
- Pressure p stored at vertices: (i, j)·h

```
    p-------p-------p
    |       |       |
    |  vel  |  vel  |
    |       |       |
    p-------p-------p
    |       |       |
    |  vel  |  vel  |
    |       |       |
    p-------p-------p
```

### Divergence Operator

The divergence at vertex (i,j) samples the 4 surrounding velocity cells:

```
∇·v[i,j] = (1/2h) * ( u[i,j] - u[i-1,j-1] + u[i,j-1] - u[i-1,j]
                    + v[i,j] - v[i-1,j-1] + v[i-1,j] - v[i,j-1] )
```

This uses central differences across the diagonals.

### Gradient Operator

The pressure gradient at cell center (i+0.5, j+0.5) samples the 4 corner vertices:

```
∂p/∂x = (1/2h) * ( p[i+1,j+1] + p[i+1,j] - p[i,j+1] - p[i,j] )
∂p/∂y = (1/2h) * ( p[i+1,j+1] + p[i,j+1] - p[i+1,j] - p[i,j] )
```

### Consistent Laplacian (2D)

For the projection to exactly remove divergence, we need:

```
∇²p = ∇·(∇p)
```

Composing the divergence of the gradient gives the **diagonal 5-point stencil**:

```
        p[i-1,j+1]          p[i+1,j+1]
               \            /
                \          /
                 \        /
                   p[i,j]
                 /        \
                /          \
               /            \
        p[i-1,j-1]          p[i+1,j-1]
```

**Coefficients (×1/2h²):**
- Center p[i,j]: -4
- Corners p[i±1,j±1]: +1 each

This is equivalent to the standard 5-point Laplacian rotated 45° and scaled by √2.

**Key insight:** The face neighbors (p[i±1,j] and p[i,j±1]) cancel out in the composition, leaving only diagonal neighbors.

---

## 3D Vertex Grid

### Setup

Extend to 3D with spacing h:
- Velocity (u,v,w) at cell centers: (i+0.5, j+0.5, k+0.5)·h
- Pressure p at vertices: (i, j, k)·h

Each pressure vertex has 8 neighboring velocity cells (the corners of a cube centered on the vertex).

### Consistent Laplacian (3D)

Unlike 2D, the face and edge neighbors do **not** cancel. The consistent Laplacian is a **full 27-point stencil**.

**Neighbor types and distances:**
- 6 face neighbors at distance h: (±1,0,0), (0,±1,0), (0,0,±1)
- 12 edge neighbors at distance √2·h: (±1,±1,0), (±1,0,±1), (0,±1,±1)
- 8 corner neighbors at distance √3·h: (±1,±1,±1)

**Coefficients (×1/16h²):**

| Neighbor Type | Count | Coefficient | Total Contribution |
|---------------|-------|-------------|-------------------|
| Center        | 1     | -24         | -24               |
| Face          | 6     | -4          | -24               |
| Edge          | 12    | +2          | +24               |
| Corner        | 8     | +3          | +24               |
| **Sum**       |       |             | **0** ✓           |

**Why doesn't 3D simplify like 2D?**

In 2D, the "45° rotation" maps the square lattice to itself, making the diagonal stencil natural. In 3D, there's no analogous rotation that maps the cubic lattice to itself while aligning with body diagonals. The face and edge neighbors contribute genuinely different information that cannot be eliminated.

---

## Stencil Decomposition

The 27-point stencil can be decomposed into three basis Laplacians:

### Basis Stencils (properly normalized to approximate ∇²)

**Face Laplacian (7-point):**
```
L_face = (1/h²)[Σ faces - 6·center]
```

**Edge Laplacian (13-point):**
```
L_edge = (1/4h²)[Σ edges - 12·center]
```

**Corner Laplacian (9-point, BCC-like):**
```
L_corner = (1/4h²)[Σ corners - 8·center]
```

### Decomposition Weights

The full 27-point dual-grid Laplacian decomposes as:

```
L_full = -1/4 · L_face + 1/2 · L_edge + 3/4 · L_corner
```

Or in ratio form: **-1 : 2 : 3**

**Observations:**
- The corner stencil has the **dominant weight** (3/4)
- The face contribution is **negative** (subtracting axis-aligned information)
- The edge contribution is positive but secondary

The corner (BCC) stencil is the "natural" Laplacian for this grid arrangement, with face and edge terms providing geometric corrections.

---

## Taylor Expansion Analysis

Expanding each basis stencil to 4th order:

```
L = ∇²u + h²·E₄ + O(h⁴)
```

Where E₄ contains 4th derivative terms.

### Error Coefficients (coefficient of h²)

| Stencil | u_xxxx + u_yyyy + u_zzzz | u_xxyy + u_xxzz + u_yyzz |
|---------|--------------------------|--------------------------|
| Face    | 1/12                     | 0                        |
| Edge    | 1/12                     | 1/4                      |
| Corner  | 1/12                     | 1/2                      |

### Full 27-point Stencil Error

Combining with weights (-1/4, 1/2, 3/4):

**Pure 4th derivatives:** (-1/4 + 1/2 + 3/4) × 1/12 = **1/12**

**Mixed 4th derivatives:** (-1/4 × 0) + (1/2 × 1/4) + (3/4 × 1/2) = **1/2**

This exactly matches the corner stencil error:

```
L_full = ∇²u + (h²/12)(u_xxxx + u_yyyy + u_zzzz) + (h²/2)(u_xxyy + u_xxzz + u_yyzz) + O(h⁴)

L_corner = ∇²u + (h²/12)(u_xxxx + u_yyyy + u_zzzz) + (h²/2)(u_xxyy + u_xxzz + u_yyzz) + O(h⁴)
```

**The corner stencil and full 27-point stencil have identical 4th-order truncation error.**

The face and edge contributions only affect 6th-order and higher terms.

---

## Operator Consistency vs Truncation Error

### The Critical Distinction

Although the corner stencil has the same truncation error as the full stencil, **you cannot use it for the pressure solve**.

**Truncation error** measures how well a stencil approximates the continuous ∇² for smooth functions.

**Operator consistency** requires that the discrete Laplacian equals the discrete divergence of the discrete gradient:

```
L[p] = Divergence[Gradient[p]]
```

### Why Corner-Only Fails

If you solve:
```
L_corner[p] = ∇·v
```

You get a pressure field that makes L_corner[p] equal the divergence. But when you compute:
```
v_new = v - ∇p
```

The result is **not divergence-free** because:
```
∇·v_new = ∇·v - ∇·(∇p) = ∇·v - L_full[p] ≠ 0
```

The discrete divergence operator "sees" velocity through its specific stencil. The pressure correction must go through the exact composition ∇·∇ = L_full for the divergence to vanish.

### Residual in Null Space

The corner stencil has a different null space than the full stencil. Divergence components that lie in the null space of L_corner but not L_full cannot be removed by solving the corner system. These modes would persist as spurious divergence.

---

## Iterative Solution Strategy

Since the corner stencil dominates (weight 3/4) and has the same truncation error, it can be used as an efficient **preconditioner**.

### Algorithm

```
1. Initial solve:    p₀ = L_corner⁻¹[d]        where d = ∇·v

2. Iterate:
   repeat until converged:
       r = d - L_full[p]                        (residual)
       e = L_corner⁻¹[r]                        (correction)
       p = p + ω·e                              (update with relaxation)
```

### Convergence Analysis

The error evolves as:
```
e_{k+1} = (I - ω·L_corner⁻¹·L_full) e_k
```

Since L_full = 3/4·L_corner + L_remainder:
```
L_corner⁻¹·L_full ≈ 3/4·I + (small correction)
```

The iteration matrix:
```
M = I - ω·L_corner⁻¹·L_full ≈ (1 - 3ω/4)·I
```

### Optimal Relaxation Parameter

For fastest convergence, set (1 - 3ω/4) = 0:

```
ω_optimal = 4/3
```

This compensates for the corner stencil being "3/4 of" the full operator.

### Expected Behavior

- **Without relaxation (ω=1):** Convergence rate ~1/4 per iteration
- **With ω=4/3:** Much faster convergence, potentially just a few iterations

The corner solve removes the bulk of the divergence (capturing the dominant BCC-like behavior), while the iterations clean up the geometric consistency terms from face and edge contributions.

### Practical Considerations

1. **Corner solve is cheaper:** 9-point vs 27-point stencil
2. **Good initial guess:** First corner solve gets most of the way there
3. **Fast convergence:** With ω=4/3, few iterations needed for full consistency
4. **Parallelization:** Corner stencil may have better parallelization properties

---

## Lattice Decoupling and Cache Optimization

A key insight for efficient implementation is that both the 2D diagonal stencil and the 3D corner stencil exhibit **sublattice decoupling**, which can be exploited for better cache utilization and parallelization.

### 2D Sublattice Structure

The diagonal stencil only samples corners at (i±1, j±1). Consider the parity of vertices:

- **Even vertices:** (i+j) mod 2 = 0
- **Odd vertices:** (i+j) mod 2 = 1

Diagonal neighbors of an even vertex (i,j):
- (i+1, j+1): parity = i+j+2 = **even**
- (i+1, j-1): parity = i+j = **even**
- (i-1, j+1): parity = i+j = **even**
- (i-1, j-1): parity = i+j-2 = **even**

**The diagonal stencil only couples even-to-even and odd-to-odd.** The pressure grid splits into two independent sublattices!

### Why 2D Projection Still Works

Although the Laplacian decouples, the full projection pipeline couples the sublattices:

```
v* ──→ d_even ──→ p_even ──┐
   │                       ├──→ ∇p ──→ v_new
   └→ d_odd  ──→ p_odd  ──┘
```

1. **Divergence:** v* produces divergence on ALL vertices (both parities)
2. **Laplacian solve:** Two independent Poisson problems with consistent RHS
3. **Gradient:** ∇p at each velocity cell samples 4 vertices (2 even + 2 odd)
4. **Projection:** Both pressure parities contribute to velocity correction

The even/odd split is an internal detail of the pressure solve, invisible to the velocity field.

### 2D Deinterlacing for Cache Efficiency

The diagonal stencil on the full grid equals the **standard 5-point stencil on a rotated sublattice**.

Using rotated coordinates (a,b) where a = (i+j)/2, b = (i-j)/2:
- Original diagonal neighbors (i±1, j±1) become face neighbors (a±1, b) and (a, b±1)

```
Original grid (strided access):      Deinterlaced (contiguous):

    X   o   X   o                         X   X   X
    o   X   o   X          →              X   X   X
    X   o   X   o                         X   X   X
    o   X   o   X
```

**Benefits:**
- Each sublattice is a contiguous ~(N/√2) × (N/√2) array
- Standard 5-point stencil with unit stride
- No skipping neighbors - perfect cache line utilization
- Two smaller solves instead of one large strided solve

### 3D Sublattice Structure

The corner stencil samples 8 vertices at (i±1, j±1, k±1). Using 3-bit parity (i mod 2, j mod 2, k mod 2):

Corner neighbors flip ALL three parities:
- (0,0,0) ↔ (1,1,1)
- (0,0,1) ↔ (1,1,0)
- (0,1,0) ↔ (1,0,1)
- (0,1,1) ↔ (1,0,0)

**Four independent bipartite pairs!** Each pair forms a separate subproblem on 1/4 of the grid.

### 3D Deinterlacing

Similar to 2D, the corner stencil becomes a standard stencil on rotated sublattices:
- Original corner neighbors (±1,±1,±1) become face neighbors in the rotated basis
- The 9-point BCC stencil becomes a standard 7-point stencil on each sublattice

```
512³ grid → 4 independent ~(256)³ subproblems
Each with standard 7-point stencil, contiguous memory access
```

### Recoupling Through Residual

For the 3D iterative preconditioner, the sublattices decouple during the corner solve but recouple during residual computation:

```
1. Decouple: Corner solve (4 parallel sublattice solves)
           ↓
2. Recouple: 27-point residual (mixes all 8 colors)
           ↓
3. Iterate: Cross-sublattice information accumulates
```

The full 27-point stencil has face, edge, AND corner neighbors - it couples all 8 parity classes. This is analogous to how divergence/gradient couple the 2D sublattices.

| Dimension | Decoupled Solve | Recoupling Mechanism |
|-----------|-----------------|---------------------|
| 2D | Diagonal (2 sublattices) | Divergence/Gradient operators |
| 3D | Corner (4 sublattices) | 27-point residual |

### Multigrid Integration

The sublattice structure integrates naturally with Full Multigrid (FMG):

```
1. Initial corner solve (4 parallel sublattice solves)
2. Compute full 27-point residual (recouples)
3. FMG V-cycle on residual equation
4. Apply correction with ω = 4/3
5. Usually converged in 1-2 outer iterations
```

**Why this works well:**
- Corner solve removes bulk of divergence (O(N) complexity)
- Residual is smooth - exactly what multigrid handles efficiently
- FMG achieves O(N) complexity for the correction
- Sublattice structure can be maintained through coarse levels

### Boundary Considerations

The clean decoupling holds for **homogeneous domains**. Boundaries can recouple:

**What breaks decoupling:**
- Internal boundaries (obstacles) with modified stencils
- Neumann BCs that reference across parities
- Immersed boundary methods with interpolation
- Irregular domain boundaries

**Practical approach:**
- Homogeneous interior: Full decoupling (fast, cache-friendly)
- Boundary regions: Handle coupling as correction or fall back to coupled solve
- Complex geometry: May lose clean separation entirely

The optimization is most valuable for large homogeneous regions where the decoupled solve dominates the work.

---

## Summary

| Property | 2D Vertex Grid | 3D Vertex Grid |
|----------|---------------|----------------|
| Stencil | 5-point diagonal | 27-point full |
| Dominant term | Corners only | Corners (weight 3/4) |
| Face contribution | Cancels (0) | Negative (-1/4) |
| Edge contribution | N/A | Positive (1/2) |
| 4th-order error | Same as corners | Same as corners |

The vertex/dual grid formulation has an elegant structure where the BCC-like corner stencil captures the essential behavior, with face and edge terms providing geometric corrections for consistency on the cubic lattice.

---

## References

- Harlow, F.H. & Welch, J.E. (1965). "Numerical Calculation of Time-Dependent Viscous Incompressible Flow"
- Stam, J. (1999). "Stable Fluids"
- Bridson, R. (2015). "Fluid Simulation for Computer Graphics"
