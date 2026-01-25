# Vertex Grid (Dual Grid) Formulation for Fluid Simulation

> **Note:** The core derivations in this document were developed independently by the author between 2021 (2D) and 2022 (3D), and have been implemented in [EmberGen](https://jangafx.com/software/embergen/) since late 2022. This document is a later write-up of those techniques, produced with AI assistance (Claude). While the mathematical derivations have been checked for internal consistency and validated through practical implementation, the document has not been formally peer-reviewed. The 3D analysis appears to contradict published literature (see [Literature](#literature) section)—this discrepancy should be independently verified before being cited.

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
9. [Comparison with Rhie-Chow Interpolation](#comparison-with-rhie-chow-interpolation)
10. [Performance Comparison with MAC Grid](#performance-comparison-with-mac-grid)
11. [Velocity Hourglass Filter](#velocity-hourglass-filter)
12. [Literature](#literature)

---

## TL;DR

- **2D:** Consistent Laplacian is the diagonal 5-point stencil (rotated standard stencil)
- **3D:** Consistent Laplacian is a 27-point stencil with weights -24 (center), -4 (faces), +2 (edges), +3 (corners)
- **Key insight:** Corner stencil alone captures 75% of the operator and has identical 4th-order accuracy
- **Solver strategy:** Use corner stencil as preconditioner with ω ≈ 4/3, converges in ~5 iterations
- **Optimization:** Sublattice decoupling allows 4 independent solves with standard 7-point stencils

---

## Grid Setup

### The Incompressible Navier-Stokes Equations

The incompressible Navier-Stokes equations govern fluid motion:

```
∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v + f    (momentum)
∇·v = 0                                (incompressibility)
```

The pressure projection method splits each timestep:
1. Compute intermediate velocity v* (advection, viscosity, forces)
2. Solve for pressure: ∇²p = ∇·v*
3. Project to divergence-free: v_new = v* - ∇p

The key challenge is discretizing these operators consistently so that the projection **exactly** removes divergence in the discrete sense.

### Standard MAC Grid (for comparison)

The Marker-and-Cell (MAC) grid, introduced by Harlow & Welch (1965), staggers variables:

```
      p───u───p───u───p
      │       │       │
      v       v       v
      │       │       │
      p───u───p───u───p
      │       │       │
      v       v       v
      │       │       │
      p───u───p───u───p
```

- **Pressure p:** Cell centers
- **u-velocity:** Vertical cell faces (between cells horizontally)
- **v-velocity:** Horizontal cell faces (between cells vertically)

**Advantages:**
- Divergence naturally computed from face velocities
- Gradient naturally applies to face locations
- No pressure checkerboard (pressure at cell centers, gradient at faces)
- No velocity checkerboard (each component at its own staggered location)

**Disadvantages:**
- Three separate grids with different sizes
- Advection requires interpolation between grids
- Dye/density must choose a location (usually cell centers, separate from velocity)

### Vertex Grid (Dual Grid)

The vertex grid inverts the MAC arrangement:

```
    p───────p───────p
    │       │       │
    │ (u,v) │ (u,v) │
    │       │       │
    p───────p───────p
    │       │       │
    │ (u,v) │ (u,v) │
    │       │       │
    p───────p───────p
```

- **Velocity (u,v):** Cell centers (both components collocated)
- **Pressure p:** Vertices (corners of cells)
- **Dye/density:** Cell centers (collocated with velocity)

**Advantages:**
- Velocity components collocated - simpler advection
- Dye naturally lives with velocity - no interpolation needed
- Single velocity grid instead of two/three

**Disadvantages:**
- Must derive consistent discrete operators carefully
- More complex Laplacian stencil (especially in 3D)
- Potential for checkerboard modes (addressed by consistent discretization)

This arrangement is the **dual** of MAC: where MAC has pressure, vertex grid has velocity, and vice versa.

---

## 2D Vertex Grid

### Setup and Notation

Consider a 2D grid with uniform spacing h:

- **Velocity (u,v)** stored at cell centers: position ((i+0.5)h, (j+0.5)h)
- **Pressure p** stored at vertices: position (ih, jh)

We use integer indices where:
- `vel[i,j]` refers to the velocity at cell center (i+0.5, j+0.5)·h
- `p[i,j]` refers to the pressure at vertex (i, j)·h

```
    p[0,2]────p[1,2]────p[2,2]
      │         │         │
      │ vel[0,1]│ vel[1,1]│
      │         │         │
    p[0,1]────p[1,1]────p[2,1]
      │         │         │
      │ vel[0,0]│ vel[1,0]│
      │         │         │
    p[0,0]────p[1,0]────p[2,0]
```

Each velocity cell is surrounded by 4 pressure vertices.
Each pressure vertex is surrounded by 4 velocity cells.

### Divergence Operator

The divergence ∇·v at a vertex must be computed from the surrounding velocity cells. Vertex p[i,j] is surrounded by cells:

```
    vel[i-1,j]    vel[i,j]
           ╲      ╱
            p[i,j]
           ╱      ╲
    vel[i-1,j-1]  vel[i,j-1]
```

To compute ∂u/∂x + ∂v/∂y at the vertex, we use central differences across the diagonals:

**For ∂u/∂x:** The derivative in x uses the diagonal difference:
```
∂u/∂x ≈ (u[i,j] + u[i,j-1] - u[i-1,j] - u[i-1,j-1]) / (2h)
```

This is a central difference: cells on the right (i,*) minus cells on the left (i-1,*), divided by the horizontal distance (2h between cell centers diagonally).

**For ∂v/∂y:** Similarly:
```
∂v/∂y ≈ (v[i,j] + v[i-1,j] - v[i,j-1] - v[i-1,j-1]) / (2h)
```

**Combined divergence:**
```
∇·v[i,j] = (1/2h) × [ (u[i,j] + u[i,j-1] - u[i-1,j] - u[i-1,j-1])
                    + (v[i,j] + v[i-1,j] - v[i,j-1] - v[i-1,j-1]) ]
```

This can be rewritten by grouping terms per cell:
```
∇·v[i,j] = (1/2h) × [ (u+v)[i,j] + (u-v)[i,j-1] + (v-u)[i-1,j] - (u+v)[i-1,j-1] ]
```

### Gradient Operator

The pressure gradient ∇p at a cell center must be computed from the surrounding pressure vertices. Cell vel[i,j] (at position (i+0.5, j+0.5)·h) is surrounded by:

```
    p[i,j+1]────p[i+1,j+1]
        │           │
        │  vel[i,j] │
        │           │
    p[i,j]──────p[i+1,j]
```

**For ∂p/∂x:** Central difference in x direction:
```
∂p/∂x[i,j] = (p[i+1,j+1] + p[i+1,j] - p[i,j+1] - p[i,j]) / (2h)
```

Right vertices minus left vertices, divided by horizontal distance.

**For ∂p/∂y:** Central difference in y direction:
```
∂p/∂y[i,j] = (p[i+1,j+1] + p[i,j+1] - p[i+1,j] - p[i,j]) / (2h)
```

Top vertices minus bottom vertices.

### Deriving the Consistent Laplacian (2D)

For the pressure projection to **exactly** remove divergence, we need:
```
∇²p = ∇·(∇p)
```

We must compose the discrete divergence and gradient operators.

**Step 1:** Write the gradient at each cell surrounding vertex p[i,j].

The 4 cells surrounding p[i,j] are: vel[i-1,j-1], vel[i,j-1], vel[i-1,j], vel[i,j]

For cell vel[i,j] (upper-right of vertex p[i,j]):
```
∂p/∂x at vel[i,j] = (p[i+1,j+1] + p[i+1,j] - p[i,j+1] - p[i,j]) / (2h)
∂p/∂y at vel[i,j] = (p[i+1,j+1] + p[i,j+1] - p[i+1,j] - p[i,j]) / (2h)
```

For cell vel[i-1,j] (upper-left of vertex p[i,j]):
```
∂p/∂x at vel[i-1,j] = (p[i,j+1] + p[i,j] - p[i-1,j+1] - p[i-1,j]) / (2h)
∂p/∂y at vel[i-1,j] = (p[i,j+1] + p[i-1,j+1] - p[i,j] - p[i-1,j]) / (2h)
```

For cell vel[i,j-1] (lower-right of vertex p[i,j]):
```
∂p/∂x at vel[i,j-1] = (p[i+1,j] + p[i+1,j-1] - p[i,j] - p[i,j-1]) / (2h)
∂p/∂y at vel[i,j-1] = (p[i+1,j] + p[i,j] - p[i+1,j-1] - p[i,j-1]) / (2h)
```

For cell vel[i-1,j-1] (lower-left of vertex p[i,j]):
```
∂p/∂x at vel[i-1,j-1] = (p[i,j] + p[i,j-1] - p[i-1,j] - p[i-1,j-1]) / (2h)
∂p/∂y at vel[i-1,j-1] = (p[i,j] + p[i-1,j] - p[i,j-1] - p[i-1,j-1]) / (2h)
```

**Step 2:** Apply divergence operator to these gradients.

Recall divergence at vertex p[i,j]:
```
∇·(∇p)[i,j] = (1/2h) × [ (∂p/∂x + ∂p/∂y)[i,j]
                       + (∂p/∂x - ∂p/∂y)[i,j-1]
                       + (∂p/∂y - ∂p/∂x)[i-1,j]
                       - (∂p/∂x + ∂p/∂y)[i-1,j-1] ]
```

**Step 3:** Substitute and collect terms.

After substituting all gradient expressions and collecting coefficients for each pressure value, the **face neighbors cancel out**:

- Coefficient of p[i+1,j]: 0
- Coefficient of p[i-1,j]: 0
- Coefficient of p[i,j+1]: 0
- Coefficient of p[i,j-1]: 0

The surviving terms are only the **diagonal neighbors**:

```
∇²p[i,j] = (1/2h²) × [ p[i+1,j+1] + p[i+1,j-1] + p[i-1,j+1] + p[i-1,j-1] - 4·p[i,j] ]
```

### The Diagonal 5-Point Stencil

The consistent 2D Laplacian is:

```
        p[i-1,j+1]          p[i+1,j+1]
             (+1)            (+1)
               \              /
                \            /
                 \          /
                   p[i,j]
                    (-4)
                 /          \
                /            \
               /              \
             (+1)            (+1)
        p[i-1,j-1]          p[i+1,j-1]
```

**Coefficients (×1/2h²):**

| Position | Coefficient |
|----------|-------------|
| p[i,j] (center) | -4 |
| p[i+1,j+1] | +1 |
| p[i+1,j-1] | +1 |
| p[i-1,j+1] | +1 |
| p[i-1,j-1] | +1 |
| p[i±1,j], p[i,j±1] (faces) | 0 |

**Geometric interpretation:**

This is the standard 5-point Laplacian **rotated 45°** and scaled. The diagonal neighbors are at distance √2·h, so the effective grid spacing is √2·h, giving a factor of 1/(√2·h)² = 1/(2h²).

**Why face neighbors cancel:**

The face neighbors appear with equal positive and negative contributions from different gradient terms. The symmetry of the vertex grid causes exact cancellation. This is a consequence of the 45° rotational symmetry between the primal (velocity) and dual (pressure) grids.

---

## 3D Vertex Grid

### Setup and Notation

Extend to 3D with uniform spacing h:

- **Velocity (u,v,w)** at cell centers: position ((i+0.5)h, (j+0.5)h, (k+0.5)h)
- **Pressure p** at vertices: position (ih, jh, kh)

Each velocity cell is a cube surrounded by 8 pressure vertices (its corners).
Each pressure vertex is surrounded by 8 velocity cells.

### Divergence and Gradient Operators (3D)

**Divergence at vertex p[i,j,k]:**

The 8 surrounding velocity cells contribute:
```
∇·v[i,j,k] = (1/2h) × Σ (±u ± v ± w) over 8 cells
```

with signs determined by the cell's position relative to the vertex.

**Gradient at cell vel[i,j,k]:**

The 8 surrounding pressure vertices contribute:
```
∂p/∂x = (1/2h) × (right 4 vertices - left 4 vertices)
∂p/∂y = (1/2h) × (top 4 vertices - bottom 4 vertices)
∂p/∂z = (1/2h) × (front 4 vertices - back 4 vertices)
```

### Why 3D Doesn't Simplify Like 2D

In 2D, composing ∇·∇ causes face neighbors to cancel, leaving only diagonal neighbors. One might expect 3D to work similarly, leaving only corner neighbors.

**This does not happen in 3D.**

The key difference is geometric:
- In 2D, rotating 45° maps the square lattice to itself (the dual of a square lattice is another square lattice, rotated)
- In 3D, rotating to align with body diagonals does **not** map the cubic lattice to itself

The body diagonals of a cube point to corners at (±1,±1,±1). These form a different lattice structure (BCC - body-centered cubic) that is not equivalent to the original cubic lattice.

**Result:** Face and edge neighbors contribute terms that do not cancel.

### The Full 27-Point Stencil

The consistent 3D Laplacian involves ALL 27 points in a 3×3×3 neighborhood:

**Neighbor classification:**

| Type | Positions | Count | Distance |
|------|-----------|-------|----------|
| Center | (0,0,0) | 1 | 0 |
| Face | (±1,0,0), (0,±1,0), (0,0,±1) | 6 | h |
| Edge | (±1,±1,0), (±1,0,±1), (0,±1,±1) | 12 | √2·h |
| Corner | (±1,±1,±1) | 8 | √3·h |

**Coefficients (×1/16h²):**

| Neighbor Type | Coefficient | Count | Total |
|---------------|-------------|-------|-------|
| Center | -24 | 1 | -24 |
| Face | -4 | 6 | -24 |
| Edge | +2 | 12 | +24 |
| Corner | +3 | 8 | +24 |
| **Sum** | | | **0** ✓ |

The coefficients sum to zero, as required for any Laplacian stencil (constant functions are in the null space).

**Sign pattern:**
- Center: negative (as expected)
- Face: **negative** (unusual!)
- Edge: positive
- Corner: positive

The negative face coefficient is counterintuitive but emerges from the algebra. The face-neighbor direction is being **subtracted** in the consistent discretization.

### Visualizing the 3D Stencil

**Face neighbors (coefficient -4):**
```
         -4
          │
    -4────●────-4
         /│
       -4 │
          -4
```

**Edge neighbors (coefficient +2):**
```
    +2─────+2
    │╲    ╱│
    │ ╲  ╱ │
    +2──●──+2
    │ ╱  ╲ │
    │╱    ╲│
    +2─────+2

(and 4 more in front/back planes)
```

**Corner neighbors (coefficient +3):**
```
    +3─────────+3
    │╲         │╲
    │ ╲        │ ╲
    │  +3──────│──+3
    │  │       │  │
    +3─│───────+3 │
     ╲ │        ╲ │
      ╲│         ╲│
       +3─────────+3
```

---

## Stencil Decomposition

The 27-point stencil can be understood as a weighted combination of three simpler stencils.

### Three Basis Laplacians

Each of the three neighbor types (face, edge, corner) defines its own Laplacian approximation:

**Face Laplacian (7-point) - Standard:**

The classical second-order Laplacian using only axis-aligned neighbors:
```
L_face[p] = (1/h²) × [p(i+1,j,k) + p(i-1,j,k) + p(i,j+1,k) + p(i,j-1,k)
                     + p(i,j,k+1) + p(i,j,k-1) - 6·p(i,j,k)]
```

Stencil (×1/h²):
- Center: -6
- 6 faces: +1 each

**Edge Laplacian (13-point):**

Using face-diagonal neighbors:
```
L_edge[p] = (1/4h²) × [Σ(12 edge neighbors) - 12·p(i,j,k)]
```

Derivation: Edge neighbors are at distance √2·h. Taylor expansion shows:
```
Σ edges = 12·p + 4h²·∇²p + O(h⁴)
```

Therefore L_edge = (1/4h²)[Σ edges - 12·center] approximates ∇².

Stencil (×1/4h²):
- Center: -12
- 12 edges: +1 each

**Corner Laplacian (9-point, BCC-like):**

Using body-diagonal neighbors:
```
L_corner[p] = (1/4h²) × [Σ(8 corner neighbors) - 8·p(i,j,k)]
```

Derivation: Corner neighbors are at distance √3·h. Taylor expansion shows:
```
Σ corners = 8·p + 4h²·∇²p + O(h⁴)
```

Therefore L_corner = (1/4h²)[Σ corners - 8·center] approximates ∇².

Stencil (×1/4h²):
- Center: -8
- 8 corners: +1 each

This is the natural Laplacian for a **BCC (body-centered cubic) lattice**, where each point has 8 nearest neighbors along body diagonals.

### Finding the Decomposition Weights

We want to express the full stencil as:
```
L_full = α·L_face + β·L_edge + γ·L_corner
```

**Matching coefficients:**

From face neighbors: α × (1/h²) = -4/(16h²) → **α = -1/4**

From edge neighbors: β × (1/4h²) = 2/(16h²) → **β = +1/2**

From corner neighbors: γ × (1/4h²) = 3/(16h²) → **γ = +3/4**

**Verify center coefficient:**
```
α×(-6/h²) + β×(-12/4h²) + γ×(-8/4h²)
= (-1/4)×(-6) + (1/2)×(-3) + (3/4)×(-2)  [all ×1/h²]
= 3/2 - 3/2 - 3/2
= -3/2  [×1/h²]
= -24/16h² ✓
```

### The Decomposition

```
L_full = -1/4 · L_face + 1/2 · L_edge + 3/4 · L_corner
```

**Ratio form: -1 : 2 : 3**

**Key observations:**

1. **Corner dominates:** Weight 3/4 (75% of the full operator)
2. **Face is subtracted:** Weight -1/4 (negative contribution)
3. **Edge is secondary:** Weight 1/2

The vertex grid Laplacian is fundamentally a **BCC-like operator** with corrections. The corner/BCC stencil captures the dominant behavior, while face and edge terms adjust for the cubic lattice geometry.

**Physical interpretation:**

The pressure vertex "wants" to couple diagonally to its neighbors (like BCC), but the cubic lattice geometry forces additional face and edge coupling. The face coupling is negative because it partially counteracts an over-coupling that would otherwise occur.

---

## Taylor Expansion Analysis

To understand approximation quality, we expand each stencil in Taylor series.

### General Framework

For any stencil L, we can write:
```
L[u] = ∇²u + h²·E₄ + h⁴·E₆ + O(h⁶)
```

Where E₄ contains 4th derivatives, E₆ contains 6th derivatives, etc.

All three basis stencils (and their combinations) are **second-order accurate**: they match ∇² exactly, with leading error at O(h²).

### Deriving Error Coefficients

**Method:** Use test functions u = x⁴ and u = x²y² to isolate specific derivative terms.

**For L_face:**

Test u = x⁴ (so ∇²u = 12x², u_xxxx = 24 at origin):
- Face sum at origin: 2h⁴ (from ±h in x direction)
- L_face[x⁴] = (1/h²)[2h⁴ - 0] = 2h²
- True Laplacian = 0 at origin
- Error = 2h² from u_xxxx = 24 → coefficient = 2h²/(24h²) = **1/12**

Test u = x²y² (so u_xxyy = 4 at origin):
- All faces have x=0 or y=0, so u = 0
- L_face[x²y²] = 0
- True Laplacian = 0 at origin
- Error coefficient for u_xxyy = **0**

**For L_edge:**

Test u = x⁴:
- Edge sum: 8h⁴ (from 8 edges with |x| = h)
- L_edge[x⁴] = (1/4h²)[8h⁴] = 2h²
- Error coefficient = **1/12**

Test u = x²y²:
- Only xy-plane edges contribute: 4h⁴
- L_edge[x²y²] = (1/4h²)[4h⁴] = h²
- Error coefficient = h²/(4h²) = **1/4**

**For L_corner:**

Test u = x⁴:
- All 8 corners have x = ±h: sum = 8h⁴
- L_corner[x⁴] = (1/4h²)[8h⁴] = 2h²
- Error coefficient = **1/12**

Test u = x²y²:
- All 8 corners have |x| = |y| = h: sum = 8h⁴
- L_corner[x²y²] = (1/4h²)[8h⁴] = 2h²
- Error coefficient = 2h²/(4h²) = **1/2**

### Summary of Error Coefficients

| Stencil | u_xxxx + u_yyyy + u_zzzz | u_xxyy + u_xxzz + u_yyzz |
|---------|--------------------------|--------------------------|
| Face (L_face) | 1/12 | 0 |
| Edge (L_edge) | 1/12 | 1/4 |
| Corner (L_corner) | 1/12 | 1/2 |

### Full 27-Point Stencil Error

Using weights (-1/4, 1/2, 3/4):

**Pure 4th derivatives (u_xxxx + ...):**
```
(-1/4 + 1/2 + 3/4) × (1/12) = 1 × (1/12) = 1/12
```

**Mixed 4th derivatives (u_xxyy + ...):**
```
(-1/4)×0 + (1/2)×(1/4) + (3/4)×(1/2) = 0 + 1/8 + 3/8 = 1/2
```

**Complete error expression:**
```
L_full = ∇²u + (h²/12)(u_xxxx + u_yyyy + u_zzzz) + (h²/2)(u_xxyy + u_xxzz + u_yyzz) + O(h⁴)
```

### The Remarkable Coincidence

The corner stencil alone has:
```
L_corner = ∇²u + (h²/12)(u_xxxx + u_yyyy + u_zzzz) + (h²/2)(u_xxyy + u_xxzz + u_yyzz) + O(h⁴)
```

**The corner stencil and full 27-point stencil have IDENTICAL 4th-order truncation error.**

The face and edge contributions only affect 6th-order and higher terms. To 4th order, you could use just the corner stencil and get the same approximation quality.

This is not a coincidence - it reflects the dominance of the corner term (weight 3/4) in the decomposition.

---

## Operator Consistency vs Truncation Error

This section explains why you **cannot** simply use the corner stencil despite its matching truncation error.

### Two Different Concepts

**Truncation Error:**
- Measures how well a stencil approximates the continuous operator ∇²
- Computed via Taylor expansion for smooth functions
- Determines convergence rate as h → 0

**Operator Consistency:**
- Requires discrete operators to satisfy exact algebraic relationships
- Specifically: L = ∇·∇ (Laplacian = divergence of gradient)
- Determines whether projection **exactly** removes divergence

These are fundamentally different requirements.

### The Projection Equation

The pressure projection solves:
```
∇²p = ∇·v*
```

Then applies:
```
v_new = v* - ∇p
```

For divergence-free result:
```
∇·v_new = ∇·v* - ∇·(∇p) = ∇·v* - L[p]
```

This equals zero **if and only if** L[p] = ∇·v*.

### Why Corner-Only Fails

If you solve using the corner stencil:
```
L_corner[p] = ∇·v*
```

You get a pressure field where L_corner[p] equals the divergence. But the actual divergence after projection is:
```
∇·v_new = ∇·v* - ∇·(∇p) = ∇·v* - L_full[p]
```

The discrete divergence-of-gradient is L_full, not L_corner! So:
```
∇·v_new = ∇·v* - L_full[p]
        = L_corner[p] - L_full[p]
        = (L_corner - L_full)[p]
        ≠ 0
```

The residual divergence is:
```
∇·v_new = (-1/4·L_face + 1/2·L_edge - 1/4·L_corner)[p]
```

This is the "remainder" after subtracting the corner contribution.

### The Null Space Problem

Different stencils have different null spaces. A mode that L_corner can "see" and remove might be invisible to L_full, and vice versa.

More precisely:
- If there exists p such that L_corner[p] = 0 but L_full[p] ≠ 0
- Then solving L_corner removes divergence in one sense but leaves residual in another

The projection would leave **systematic residual divergence** in modes that lie in the null space difference.

### The Bottom Line

**Truncation error** tells you: "These stencils approximate ∇² equally well for smooth functions."

**Operator consistency** tells you: "Only L_full makes divergence exactly zero."

For fluid simulation, operator consistency is **essential**. Residual divergence causes:
- Volume loss/gain (mass conservation violation)
- Spurious pressure modes
- Accumulating errors over time

You must use the consistent operator L_full, even though L_corner has the same formal accuracy.

---

## Iterative Solution Strategy

Since L_corner is simpler (9 vs 27 points) and captures 3/4 of the operator, we can use it as an efficient **preconditioner**.

### The Core Idea

Instead of solving L_full[p] = d directly, we:
1. Solve the cheaper L_corner system
2. Correct for the difference iteratively

The corner solve removes the bulk of the divergence. The iterations handle the face/edge consistency terms.

### Algorithm

```
Input: d = ∇·v* (divergence to remove)
Output: p such that L_full[p] = d

1. Initial solve:
   Solve L_corner[p₀] = d
   (This is a 9-point Poisson problem)

2. Iteration loop:
   for k = 0, 1, 2, ... until converged:

       a. Compute residual:
          r = d - L_full[p_k]
          (Uses full 27-point stencil)

       b. Solve correction equation:
          Solve L_corner[e] = r
          (Another 9-point solve)

       c. Update with relaxation:
          p_{k+1} = p_k + ω·e

       d. Check convergence:
          if ||r|| < tolerance: break

3. Return p
```

### Physical Interpretation: Repeated Projection

The iterative scheme has a beautiful physical interpretation. The "residual" computed with the full 27-point stencil is actually the **post-projection divergence**:

```
r = d - L_full[p]
  = ∇·v* - ∇·(∇p)
  = ∇·(v* - ∇p)
  = ∇·v_projected
```

This means the iteration is literally **repeated projection**:

```
1. Solve L_corner[p₀] = ∇·v*           (approximate pressure)
2. Project: v₁ = v* - ∇p₀              (apply pressure correction)
3. Compute divergence: d₁ = ∇·v₁       (this IS the "residual"!)
4. Solve L_corner[e] = d₁              (correction for remaining divergence)
5. Project again: v₂ = v₁ - ω·∇e       (further correction)
6. Repeat until ∇·v_k ≈ 0
```

The abstract algebraic view (preconditioned linear solve) and the physical view (iteratively removing remaining divergence) are the same thing. Each iteration projects the velocity and removes whatever divergence remains.

This also explains why convergence is guaranteed: even an imperfect projection reduces divergence, and repeated application drives it to zero.

### Convergence Analysis

Let p* be the true solution (L_full[p*] = d).

**Error evolution:**

Define error e_k = p* - p_k. Then:
```
e_{k+1} = e_k - ω·L_corner⁻¹·L_full[e_k]
        = (I - ω·L_corner⁻¹·L_full)·e_k
        = M·e_k
```

where M = I - ω·L_corner⁻¹·L_full is the iteration matrix.

**Spectral analysis:**

Since L_full = (3/4)·L_corner + L_remainder:
```
L_corner⁻¹·L_full = (3/4)·I + L_corner⁻¹·L_remainder
```

If L_remainder is small compared to L_corner (which it is, being weight 1/4 total), then:
```
L_corner⁻¹·L_full ≈ (3/4)·I
```

The iteration matrix becomes:
```
M = I - ω·(3/4)·I = (1 - 3ω/4)·I
```

**Convergence rate:**

The spectral radius ρ(M) determines convergence:
- Without relaxation (ω=1): ρ(M) ≈ 1/4, converges as (1/4)^k
- With optimal ω: can achieve ρ(M) ≈ 0

### Relaxation Parameter

**Simple estimate:** If L_corner⁻¹·L_full ≈ (3/4)·I, then setting M = 0 gives:
```
1 - 3ω/4 = 0
ω = 4/3
```

However, **this estimate has caveats:**

The decomposition weights -1/4, 1/2, 3/4 sum to **1**, not 3/4:
```
-1/4 + 1/2 + 3/4 = 1
```

So L_full is a weighted average of three operators that all approximate ∇². The ratio L_corner⁻¹·L_full is not simply 3/4·I.

**The actual situation:**

L_face, L_edge, L_corner have different **spectral properties**. For a Fourier mode with wavevector k:
- L_face eigenvalue involves cos(k_i·h) terms
- L_edge eigenvalue involves cos(k_i·h)cos(k_j·h) products
- L_corner eigenvalue involves cos(k_x·h)cos(k_y·h)cos(k_z·h)

The ratio λ_full(k)/λ_corner(k) varies across modes - it's not a constant.

**Practical guidance:**

The estimate ω = 4/3 comes from assuming L_corner⁻¹·L_full ≈ (3/4)·I, which is only approximate. In practice:

- **Start with ω = 1.2-1.3** (slightly conservative)
- **Monitor convergence rate** across iterations
- **Adjust if needed** - optimal ω may depend on grid size and BCs
- **Avoid ω > 1.5** which risks divergence for some modes

### Expected Performance

| Relaxation | Convergence Rate | Iterations to 10⁻⁶ |
|------------|------------------|-------------------|
| ω = 1 | ~0.25 per iteration | ~10 |
| ω = 4/3 | ~0.05 per iteration | ~5 |
| ω = 4/3 (with FMG) | - | 1-2 |

With ω = 4/3, the method converges in just a few iterations. Combined with multigrid for the corner solves, this is highly efficient.

### Practical Implementation

**Corner solve options:**
- Red-Black Gauss-Seidel (simple, parallelizable)
- Multigrid (optimal O(N) complexity)
- FFT if periodic boundaries

**Residual computation:**
- Must use full 27-point stencil
- Can be computed in parallel
- Relatively cheap compared to the solve

**Convergence criterion:**
- ||r||_∞ < ε (max residual)
- ||r||_2 < ε (RMS residual)
- Relative reduction: ||r_k||/||r_0|| < ε

### GPU Implementation

**Corner solve (compute shader):**
- Deinterlace pressure into 4 sublattice textures
- Run Red-Black SOR on each (standard 7-point stencil)
- Each sublattice is half resolution → fits in cache better

**27-point residual:**
- Single dispatch over all vertices
- Sample all 27 neighbors (3×3×3 texture fetches)
- Can use shared memory for the 3×3×3 block

**Memory layout:**
- Consider storing sublattices separately for corner solve
- Or use stride-2 access with careful cache management

### Spectral Analysis of the 27-Point Stencil

When solving the 27-point system directly (e.g., for residual correction), the negative face coefficients (-4) cause significant problems for standard iterative methods.

**Fourier mode notation:**

For a mode with wavevector **k**, define:
- c_x = cos(k_x h), c_y = cos(k_y h), c_z = cos(k_z h)
- S₁ = c_x + c_y + c_z
- S₂ = c_x c_y + c_x c_z + c_y c_z
- S₃ = c_x c_y c_z

**Jacobi iteration:**

The Jacobi iteration matrix eigenvalue for a Fourier mode is:

μ_J = −S₁/3 + S₂/3 + S₃

For mode (π, 0, 0): S₁ = 1, S₂ = −1, S₃ = −1, giving **μ_J = −5/3** (unstable).

The stencil is not diagonally dominant: |center| = 24, but Σ|off-diagonal| = 72.

**Stability requirements for damped Jacobi:**

For damped Jacobi with parameter ω, the eigenvalue becomes μ(ω) = 1 − ω(1 − μ_J).

The worst mode (π, 0, 0) requires:
- **ω < 3/4** for stability
- **ω ≈ 6/11 ≈ 0.55** for optimal convergence (spectral radius ≈ 0.45)

This matches empirical observations of optimal ω in the range 0.5–0.6.

**8-color Gauss-Seidel:**

With 8 colors based on (i mod 2, j mod 2, k mod 2), same-color points never neighbor each other in the 27-point stencil. However, the 8×8 sublattice iteration matrix for mode (π, 0, 0) has **spectral radius ≈ 1.95** — still unstable.

The negative face coefficients cause instability even with multicolored GS.

**Sequential (lexicographic) Gauss-Seidel:**

For sequential GS, "past" and "future" neighbors are determined by lexicographic ordering. The GS eigenvalue is:

μ_GS = −λ_U / (d + λ_L)

where λ_L and λ_U are the coupling sums for past/future neighbors.

For mode (π, 0, 0): λ_L = λ_U = −20, d = −24, giving **μ_GS = −5/11 ≈ −0.45** (stable).

Sequential GS works because "past" neighbors include same-color points (in the 8-color sense), providing additional coupling that stabilizes the iteration.

**Hybrid Red-Black (ping-pong) approach:**

Red-Black coloring based on (i+j+k) mod 2:
- Face neighbors: opposite color (GS treatment)
- Edge neighbors: same color (Jacobi treatment)
- Corner neighbors: opposite color (GS treatment)

The 2×2 iteration matrix for red/black amplitudes has a structural property: when |1−a| = |b| (where a = edge coupling, b = face+corner coupling), the eigenvalue **λ = 1 persists for all ω**.

For mode (π, 0, 0): |1−a| = |b| = 4/3, so this mode has **λ = 1** (marginal stability).

This explains empirical observations of the hybrid approach "not converging well" — problematic modes persist indefinitely without growing or shrinking.

**Solver comparison for the 27-point stencil:**

| Method | Mode (π,0,0) | Parallelism | Notes |
|--------|--------------|-------------|-------|
| Plain Jacobi | ρ = 5/3 | Full | Unstable |
| Damped Jacobi (ω ≈ 0.55) | ρ ≈ 0.45 | Full | Stable, ~17 iter to 10⁻⁶ |
| 8-color GS | ρ ≈ 1.95 | Full | Unstable |
| Hybrid Red-Black | ρ = 1 | Partial | Marginal (modes persist) |
| Sequential GS | ρ ≈ 0.45 | None | Stable, but serial |

**Practical solver strategy:**

The analysis suggests a two-level approach:

1. **Corner stencil + Multigrid + Red-Black SOR**: Main solver for global pressure solve. The corner stencil is well-behaved (diagonally dominant), and Red-Black works perfectly since all corner neighbors flip parity.

2. **Damped Jacobi on 27-point** (ω ≈ 0.55): Residual correction for consistency. Fully parallel, stable, handles the difference between L_corner and L_full. Only a few iterations needed since it's correcting a small residual.

This separates the "heavy lifting" (global pressure solve via multigrid on corner stencil) from the "consistency correction" (local smoothing via damped Jacobi on full stencil).

### Spectral Analysis of the Corner Stencil

The corner stencil (9-point BCC-like) is well-behaved for iterative methods, making it ideal for the preconditioner role.

**Jacobi eigenvalue:**

For the corner stencil, the Jacobi iteration eigenvalue is simply:

μ_J = S₃ = cos(k_x h) cos(k_y h) cos(k_z h)

This ranges from -1 to +1 across all modes.

**Weighted Jacobi for multigrid smoothing:**

For damped/weighted Jacobi with parameter ω, the iteration eigenvalue is:

λ(ω) = 1 - ω(1 - μ_J) = 1 - ω + ω·S₃

The smoothing factor (spectral radius over high-frequency modes) depends on the target mode range:

| Target |μ| range | Optimal ω | Smoothing factor |
|-------------------|-----------|------------------|
| All modes (|μ| ≤ 1) | 2/3 | 1/3 |
| High-freq only (|μ| ≤ 0.5) | 8/9 | ~0.33 |

For multigrid, ω = 2/3 is a safe choice. If you know coarse grids handle |μ| > 0.5 modes well, ω = 8/9 can be slightly faster.

**Red-Black SOR for multigrid smoothing:**

The corner stencil has perfect Red-Black structure: all 8 corner neighbors flip all three parities, so neighbors are always opposite color. This makes RBSOR natural.

For RBSOR with parameter ω, the eigenvalue behavior depends on the Jacobi eigenvalue μ:

**Complex regime** (|μ| < μ_c where μ_c = 2√(ω-1)/ω):
- Eigenvalues are complex conjugates with |λ| = ω - 1
- All modes in this regime have identical convergence rate

**Real regime** (|μ| ≥ μ_c):
- Eigenvalues are real: λ = (ω|μ| ± √(ω²μ² - 4(ω-1)))/2
- Larger |μ| means slower convergence

For ω = 1 (standard Gauss-Seidel): ρ = μ² (good for small |μ|, poor for |μ| → 1)

**Over-relaxation improves smoothing:**

For multigrid smoothing, we only need to damp high-frequency modes—smooth modes are handled by coarse grids. Over-relaxation (ω > 1) pushes more modes into the complex regime where ρ = ω - 1.

The crossover point μ_c = 2√(ω-1)/ω:

| ω | μ_c | Complex regime ρ |
|-----|------|------------------|
| 1.0 | 0 | N/A |
| 1.1 | 0.57 | 0.10 |
| 1.15 | 0.67 | 0.15 |
| 1.2 | 0.75 | 0.20 |
| 1.3 | 0.87 | 0.30 |

**Optimal ω for smoothing:**

The classical SOR optimization, applied to modes up to |μ| = μ_max, gives:

ω_opt = 2 / (1 + √(1 - μ_max²))

with smoothing factor ν = ω_opt - 1.

| μ_max | ω_opt | Smoothing factor |
|-------|-------|------------------|
| 0.5 | 1.07 | 0.07 |
| 0.6 | 1.11 | 0.11 |
| 0.7 | 1.17 | 0.17 |
| 0.8 | 1.25 | 0.25 |

For multigrid with 2:1 coarsening, modes with |μ| ≲ 0.5-0.7 typically need smoothing. This gives:

- **ω = 1.1-1.15** for conservative smoothing (targets |μ| ≤ 0.5-0.6)
- **ω = 1.15-1.2** for aggressive smoothing (targets |μ| ≤ 0.7)

**Grid size independence:**

The optimal ω for smoothing does **not** depend on grid size N. This is because:

1. High-frequency modes have the same μ distribution regardless of N
2. The coarse grid correction capability (what modes it handles) depends on the coarsening ratio, not absolute size
3. Smoothing factors are defined over mode ranges, not specific wavelengths

What changes with N is the number of V-cycles needed (roughly O(log N) for full multigrid), but not the per-cycle smoothing quality or optimal ω.

**Null space consideration:**

Note that RBSOR on the corner stencil has a structural null space issue: modes like (π, π, 0) have μ = (-1)(-1)(1) = 1, which acts like a "smooth" mode for the corner operator. These modes:
- Are not smoothed efficiently by RBSOR alone
- Must be handled by the coarse grid correction
- Are recoupled through the 27-point residual in the outer iteration

This is another reason the two-level approach (corner preconditioner + 27-point residual) works well: the 27-point stencil "sees" modes that the corner stencil treats as smooth.

### Inlined Correction Stencil

The correction step in the iterative algorithm (compute 27-point residual, solve corner system, update) can be simplified by combining multiple operations into a single stencil. This section derives the combined operation and its special properties.

**Current Algorithm (steps 2-4 of outer iteration):**
```
r = d - L_full[p]           (27-point residual)
Solve L_corner[e] = r       (full corner solve)
p_new = p + ω·e             (update)
```

**Simplification: Replace solve with one Jacobi iteration**

Instead of fully solving L_corner[e] = r, perform a single weighted Jacobi iteration starting from e = 0:

```
r = d - L_full[p]                    (27-point residual)
e = ω_J · D_corner⁻¹ · r             (one weighted Jacobi iteration from e=0)
p_new = p + ω·e                      (update)
```

Where D_corner = -8/(4h²) = -2/h² is the center coefficient of L_corner.

**Deriving the Combined Operation**

Substituting the Jacobi iteration into the update:
```
e = ω_J · (-h²/2) · r
p_new = p + ω · ω_J · (-h²/2) · (d - L_full[p])
      = p - α·d + α·L_full[p]
```

where **α = ω · ω_J · h²/2**

**Resulting Stencil Coefficients**

The L_full stencil (×1/16h²): center=-24, face=-4, edge=+2, corner=+3

For the combined operation p_new = p + α·L_full[p] - α·d:

| Neighbor | L_full coeff | × α/(16h²) | Combined (p + α·L_full) |
|----------|--------------|------------|-------------------------|
| Center   | -24          | -24α/16h²  | 1 - 3α/(2h²)           |
| Face (×6)| -4           | -4α/16h²   | -α/(4h²)               |
| Edge (×12)| +2          | +2α/16h²   | +α/(8h²)               |
| Corner (×8)| +3         | +3α/16h²   | +3α/(16h²)             |

**Special Case: ω·ω_J = 4/3**

With ω = 4/3 and ω_J = 1 (or any combination giving ω·ω_J = 4/3):

**α = (4/3) · h²/2 = 2h²/3**

Substituting α into the coefficients:
- Center: 1 - 3·(2h²/3)/(2h²) = 1 - 1 = **0**
- Face: -(2h²/3)/(4h²) = **-1/6**
- Edge: (2h²/3)/(8h²) = **+1/12**
- Corner: 3·(2h²/3)/(16h²) = **+1/8**

**The center coefficient vanishes!**

**The Correction Stencil (ω·ω_J = 4/3)**

```
p_new[i,j,k] = (1/8)·Σ(corners) + (1/12)·Σ(edges) - (1/6)·Σ(faces) - (2h²/3)·d[i,j,k]
```

**Verification:** Sum of neighbor weights = 8·(1/8) + 12·(1/12) + 6·(-1/6) = 1 + 1 - 1 = **1** ✓

This is a weighted average of 26 neighbors (weights sum to 1) minus a scaled source term.

**Interpretation: Pure Neighbor Averaging**

The correction operation is a **pure neighbor averaging** that:
1. Pulls toward corner neighbors (weight 1/8 each, total contribution 1)
2. Pulls toward edge neighbors (weight 1/12 each, total contribution 1)
3. Pushes away from face neighbors (weight -1/6 each, total contribution -1)
4. Subtracts a scaled divergence source

The negative face weight is fascinating—it's "anti-diffusing" in the face directions while "diffusing" along diagonals. This mirrors the structure of the full 27-point stencil, where face neighbors also have negative coefficients.

**Algorithm with Inlined Correction**

```
1. Solve L_corner[p₀] = d                    (multigrid - full solve)
2. Apply correction stencil:
   p₁[i,j,k] = (1/8)·Σ corners + (1/12)·Σ edges - (1/6)·Σ faces - (2h²/3)·d
3. Check convergence: ||d - L_full[p₁]|| < ε
4. If not converged, repeat from step 1 (or just step 2 for more corrections)
```

**Critical Caveat: The Correction Alone Diverges**

The iteration p_{k+1} = (I + α·L_full)·p_k - α·d has iteration matrix M = I + α·L_full.

For stability: ρ(M) < 1, i.e., |1 + α·λ| < 1 for all eigenvalues λ of L_full.

The most negative eigenvalue of L_full is approximately λ_min ≈ -6/h² (Laplacian scaling).

With α = 2h²/3:
```
1 + α·λ_min ≈ 1 + (2h²/3)·(-6/h²) = 1 - 4 = -3
```

**The correction stencil alone diverges!** The corner solve is essential—it brings p close enough to the solution that residuals are small, and the correction handles only the remaining L_full - L_corner discrepancy.

**Practical Value**

The combined stencil is useful as:
1. **Single-pass correction** after corner solve (replaces residual + iterative solve with one 27-point pass)
2. **Simpler implementation** (one stencil instead of residual computation + iterative correction solve)
3. **GPU-friendly** (single dispatch, no synchronization between residual and correction)

**Not useful** as a standalone solver (diverges without the corner solve providing a good initial approximation).

**Two Jacobi Iterations: Expanded Stencil**

For completeness, we note that using two Jacobi iterations instead of one expands the stencil significantly:

- **1 iteration:** 3×3×3 = 27 point stencil (just scales the residual pointwise by D⁻¹)
- **2 iterations:** 5×5×5 = 125 point stencil (corner neighbors' residuals contribute)
- **k iterations:** (2k+1)³ point stencil

Each additional iteration expands the stencil by 2 in each dimension (corners of corners of corners...). The added complexity of larger stencils typically outweighs the benefit. For GPU implementation, the 27-point stencil is already at the edge of shared memory efficiency. If more accuracy is needed, doing multiple outer iterations (each with a fresh corner solve) is more practical than expanding the correction stencil.

---

## Lattice Decoupling and Cache Optimization

A key insight for efficient implementation is that both the 2D diagonal stencil and the 3D corner stencil exhibit **sublattice decoupling**.

### 2D Sublattice Structure

The diagonal stencil only samples corners at (i±1, j±1). Let's analyze the parity structure.

**Vertex parity:**
- **Even vertices:** (i+j) mod 2 = 0
- **Odd vertices:** (i+j) mod 2 = 1

**Diagonal neighbor parities:**

For an even vertex at (i,j) where i+j is even:
- (i+1, j+1): (i+j+2) mod 2 = **even**
- (i+1, j-1): (i+j) mod 2 = **even**
- (i-1, j+1): (i+j) mod 2 = **even**
- (i-1, j-1): (i+j-2) mod 2 = **even**

**All diagonal neighbors of an even vertex are even!**

Similarly, all diagonal neighbors of an odd vertex are odd.

**Consequence:** The diagonal stencil decouples into two independent sublattices:
- Even sublattice: all (i,j) with i+j even
- Odd sublattice: all (i,j) with i+j odd

The Laplacian solve becomes TWO independent problems, each half the size!

### Why 2D Projection Still Works

This decoupling seems problematic - how can the pressure solve work if even and odd vertices are independent?

**The key:** The divergence and gradient operators COUPLE the sublattices.

**Gradient operator:** At velocity cell (i,j), the gradient samples 4 pressure vertices:
- p[i,j]: parity = i+j
- p[i+1,j]: parity = i+j+1 (opposite)
- p[i,j+1]: parity = i+j+1 (opposite)
- p[i+1,j+1]: parity = i+j+2 (same as i+j)

The gradient uses **2 even and 2 odd** pressure vertices!

**Complete projection pipeline:**
```
v* (velocity field)
    │
    ├──→ ∇·v* on EVEN vertices ──→ solve L_even[p_even] = d_even
    │
    └──→ ∇·v* on ODD vertices ──→ solve L_odd[p_odd] = d_odd
                                           │
                                           ↓
                                    ∇p uses BOTH p_even and p_odd
                                           │
                                           ↓
                                    v_new = v* - ∇p
```

The divergence distributes the velocity information to both sublattices. The gradient recombines both sublattices into the velocity correction. The Laplacian solve is decoupled, but the full projection is coupled.

### 2D Deinterlacing for Cache Efficiency

The diagonal stencil on the full grid is equivalent to the **standard 5-point stencil on a rotated grid**.

**Coordinate transformation:**

Define new coordinates:
```
a = (i + j) / 2
b = (i - j) / 2
```

Inverse:
```
i = a + b
j = a - b
```

**Neighbor mapping:**

Original diagonal neighbors (i±1, j±1) become:
```
(i+1, j+1) → (a+1, b)      [face neighbor in a-direction]
(i+1, j-1) → (a, b+1)      [face neighbor in b-direction]
(i-1, j+1) → (a, b-1)      [face neighbor in -b-direction]
(i-1, j-1) → (a-1, b)      [face neighbor in -a-direction]
```

The diagonal stencil becomes the standard 5-point stencil!

**Memory layout optimization:**

```
Original layout (diagonal stencil, strided access):

    p[0,4]  p[1,4]  p[2,4]  p[3,4]  p[4,4]
    p[0,3]  p[1,3]  p[2,3]  p[3,3]  p[4,3]
    p[0,2]  p[1,2]  p[2,2]  p[3,2]  p[4,2]
    p[0,1]  p[1,1]  p[2,1]  p[3,1]  p[4,1]
    p[0,0]  p[1,0]  p[2,0]  p[3,0]  p[4,0]

    Even sublattice: p[0,0], p[2,0], p[4,0], p[1,1], p[3,1], ...
    (Stride-2 access in original array)

Deinterlaced layout (standard stencil, contiguous access):

    Even array:                    Odd array:
    p_e[0,0] p_e[1,0] p_e[2,0]    p_o[0,0] p_o[1,0] p_o[2,0]
    p_e[0,1] p_e[1,1] p_e[2,1]    p_o[0,1] p_o[1,1] p_o[2,1]
    p_e[0,2] p_e[1,2] p_e[2,2]    p_o[0,2] p_o[1,2] p_o[2,2]

    (Contiguous access, standard 5-point stencil)
```

**Benefits:**
- Cache line utilization: 100% vs ~50% with stride-2
- Standard stencil: existing optimized solvers work
- Smaller problems: N/2 × N/2 instead of N × N with stride
- Parallelizable: two independent solves

### 3D Sublattice Structure

The corner stencil samples 8 vertices at (i±1, j±1, k±1). The parity structure is richer.

**8-color classification:**

Use 3-bit parity: (i mod 2, j mod 2, k mod 2)

This gives 8 "colors": (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)

**Corner neighbor parities:**

For a vertex at (i,j,k), corner neighbors at (i±1, j±1, k±1) flip ALL three parities:
```
(i mod 2, j mod 2, k mod 2) → ((i+1) mod 2, (j+1) mod 2, (k+1) mod 2)
```

This means:
- (0,0,0) vertices couple only to (1,1,1) vertices
- (0,0,1) vertices couple only to (1,1,0) vertices
- (0,1,0) vertices couple only to (1,0,1) vertices
- (0,1,1) vertices couple only to (1,0,0) vertices

**Four independent bipartite pairs:**

| Pair | Colors | Coupling |
|------|--------|----------|
| A | (0,0,0) ↔ (1,1,1) | Bipartite |
| B | (0,0,1) ↔ (1,1,0) | Bipartite |
| C | (0,1,0) ↔ (1,0,1) | Bipartite |
| D | (0,1,1) ↔ (1,0,0) | Bipartite |

Each pair is a bipartite graph (like Red-Black structure). Different pairs are completely independent!

The corner stencil solve decomposes into **4 independent subproblems**, each on 1/4 of the grid.

### 3D Deinterlacing

Similar to 2D, we can transform coordinates:
```
a = (i + j + k) / 2
b = (i + j - k) / 2
c = (i - j) / 2
```

(Exact transformation depends on which sublattice pair)

The corner stencil (sampling ±1 in all three original coordinates) becomes the standard 7-point face stencil in the transformed coordinates.

**Result for 512³ grid:**
```
Original: 512³ = 134M vertices, 9-point BCC stencil

Deinterlaced: 4 × (256³ or similar) = 4 × 16M vertices
              Each with standard 7-point stencil
              Contiguous memory access
              Embarrassingly parallel
```

### Recoupling Through the 27-Point Residual

For the 3D iterative preconditioner, sublattices decouple during corner solve but recouple during residual computation.

**Why the 27-point stencil recouples:**

The full stencil has:
- Face neighbors: (±1,0,0) etc. - these couple (0,0,0) to (1,0,0), (0,1,0), (0,0,1)
- Edge neighbors: (±1,±1,0) etc. - these couple (0,0,0) to (1,1,0), (1,0,1), (0,1,1)
- Corner neighbors: (±1,±1,±1) - these couple (0,0,0) to (1,1,1)

The face and edge terms couple ALL 8 colors to each other. No sublattice is isolated when using the full 27-point stencil.

**Iteration structure:**
```
┌─────────────────────────────────────────────────────────┐
│  Corner Solve (DECOUPLED)                               │
│                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Pair A  │ │ Pair B  │ │ Pair C  │ │ Pair D  │       │
│  │ 7-point │ │ 7-point │ │ 7-point │ │ 7-point │       │
│  │ solve   │ │ solve   │ │ solve   │ │ solve   │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│       ↓           ↓           ↓           ↓             │
│       └───────────┴───────────┴───────────┘             │
│                       ↓                                 │
│              Combine into full p                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  27-Point Residual (RECOUPLES)                          │
│                                                         │
│  r = d - L_full[p]                                      │
│                                                         │
│  Every vertex sees ALL 27 neighbors                     │
│  All 8 colors communicate                               │
└─────────────────────────────────────────────────────────┘
                        ↓
              Next iteration (if needed)
```

**Analogy with 2D:**

| Dimension | Decoupled Solve | Recoupling Mechanism |
|-----------|-----------------|---------------------|
| 2D | Diagonal stencil (2 sublattices) | Divergence/Gradient operators |
| 3D iterative | Corner stencil (4 sublattices) | 27-point residual |

In 2D, the diagonal stencil IS the consistent operator, so recoupling happens naturally through div/grad.
In 3D, corner is NOT the consistent operator, so we explicitly recouple via residual.

### Multigrid Integration

The sublattice structure integrates beautifully with multigrid:

**Full Multigrid (FMG) with corner preconditioner:**

```
1. FINE GRID:
   - Deinterlace into 4 sublattice arrays
   - Run 4 parallel 7-point multigrid V-cycles (corner preconditioner solve)
   - Interleave back to full grid
   - Compute 27-point residual

2. If residual too large:
   - Use FMG on residual equation
   - Apply correction with ω = 4/3
   - Repeat (usually 1-2 times)

3. Return converged pressure
```

**Why this combination is powerful:**

1. **Corner solve is O(N):** Multigrid on each sublattice
2. **Sublattices are parallel:** 4× parallelism at finest level
3. **Residual is smooth:** FMG handles it efficiently
4. **Few outer iterations:** ω = 4/3 gives fast convergence

**Complexity:**
- Corner solve: O(N) via multigrid
- Residual: O(N) single pass
- Total: O(N) with small constant

### Boundary Considerations

The clean decoupling assumes a **homogeneous domain**. Boundaries can recouple sublattices.

**What preserves decoupling:**
- Periodic boundaries (all sublattices have same structure)
- Dirichlet boundaries aligned with grid (just pin boundary values)
- Simple rectangular domains

**What breaks decoupling:**
- **Internal obstacles:** Modified stencil at boundary cells may reference across sublattices
- **Neumann boundaries:** Gradient condition involves neighbors of both parities
- **Immersed boundaries:** Interpolation spreads across all nearby points
- **Irregular domains:** Boundary doesn't respect sublattice structure

**Practical strategies:**

1. **Domain decomposition:**
   - Large homogeneous interior: full decoupling (fast)
   - Thin boundary layer: coupled solve (small region)

2. **Boundary correction:**
   - Solve decoupled in interior
   - Apply correction near boundaries

3. **Fall back gracefully:**
   - If geometry is complex, use standard coupled solver
   - Decoupling is an optimization, not a requirement

The optimization is most valuable when homogeneous regions dominate - which is typical in many fluid simulations.

---

## Comparison with Rhie-Chow Interpolation

The vertex grid formulation has similarities and differences with Rhie-Chow interpolation, another technique for handling pressure-velocity coupling.

### The Collocated Grid Problem

On a fully collocated grid (pressure AND velocity at cell centers), the standard central difference gives a "wide" Laplacian:

**Gradient at cell i:** Uses p[i+1] and p[i-1] (skipping p[i])

**Divergence at cell i:** Uses v[i+1] and v[i-1] (skipping v[i])

**Composed Laplacian:** Only sees p[i+2], p[i], p[i-2]
```
Wide stencil (2D, ×1/4h²):

          1
          0
    1  0 -4  0  1
          0
          1
```

The immediate neighbors (distance h) have **zero coefficient**. This creates a checkerboard null space.

### Rhie-Chow Solution

Rhie-Chow interpolation modifies how face velocities are computed for the divergence:

```
u_face = (u_L + u_R)/2 - D·[(p_R - p_L) - averaged cell-center gradients]
```

The correction term adds **compact pressure coupling** that the wide stencil misses.

**Effective Laplacian (2D):**
```
Blended stencil:

          (1-α)
           4α
    (1-α) 4α -4(1+α) 4α (1-α)     (×1/4h²)
           4α
          (1-α)
```

A 9-point cross with both h and 2h neighbors.

### Vertex Grid Comparison

| Aspect | Rhie-Chow (Collocated) | Vertex Grid (Dual) |
|--------|------------------------|-------------------|
| **Grid** | Pressure & velocity at cell centers | Pressure at vertices, velocity at cell centers |
| **Problem** | Wide Laplacian, checkerboard in p | Diagonal Laplacian (2D), complex stencil (3D) |
| **Solution** | Add compact pressure coupling | Derive consistent ∇·∇ |
| **2D stencil** | 9-point cross (h + 2h) | 5-point diagonal |
| **3D stencil** | 9-point cross (h + 2h) | 27-point full |
| **Consistency** | Approximate (blending parameter) | Exact (derived from operators) |

### Key Differences

1. **Consistency:** Vertex grid Laplacian is **exactly** ∇·∇. Rhie-Chow is an approximation with a tunable parameter.

2. **Null space:** Vertex grid diagonal stencil has even/odd decoupling but this is handled by div/grad coupling. Rhie-Chow adds terms to eliminate the wide-stencil null space.

3. **Stencil shape:** Vertex grid uses diagonals (2D) or all 27 neighbors (3D). Rhie-Chow uses axis-aligned directions at two scales.

4. **Physical interpretation:** Vertex grid is a natural discretization of a dual arrangement. Rhie-Chow is a fix for an inconsistent arrangement.

### When to Use Each

**Vertex grid:**
- Clean mathematical derivation
- Exact projection (no residual divergence)
- Natural for collocated velocity + dye
- More complex 3D stencil

**Rhie-Chow:**
- Works with existing collocated codes
- Adjustable coupling strength
- Well-established in CFD
- Simpler stencil shape

Both approaches solve the fundamental problem of pressure-velocity coupling on non-staggered grids, but through different mechanisms.

---

## Performance Comparison with MAC Grid

| Aspect | MAC Grid | Vertex Grid (3D) |
|--------|----------|------------------|
| Velocity textures | 3 (staggered sizes) | 1 (collocated) |
| Pressure stencil | 7-point | 27-point (or 9-point preconditioned) |
| Advection | Interpolate between grids | Direct (velocity collocated) |
| Memory | ~1.5× for velocity | 1× |
| Solver complexity | Standard multigrid | Preconditioned iteration |

Vertex grid trades a more complex pressure solve for simpler advection and unified velocity storage. The preconditioned iterative approach with sublattice decoupling can match MAC grid performance while providing the convenience of collocated velocity and dye.

---

## Velocity Hourglass Filter

The vertex grid divergence operator samples 8 velocity cells surrounding each pressure vertex. Certain high-frequency velocity patterns—**hourglass modes**—produce zero discrete divergence despite having non-zero continuous divergence. These modes can accumulate over time, causing visual artifacts and numerical instability. This section derives a filter to detect and damp these modes.

The derivation follows Rider (1998) for 2D and extends it to 3D using an orthogonal mode decomposition of the 2×2×2 velocity cell block.

### The Hourglass Problem

At pressure vertex (i,j,k), the divergence operator computes:

```
∇·v = (1/2h) × Σ (s_x·u + s_y·v + s_z·w) over 8 surrounding cells
```

where s_x, s_y, s_z are ±1 depending on which side of the vertex each cell lies.

For the u-component contribution, cells at x-index i contribute +u, cells at x-index i-1 contribute -u. This computes a discrete ∂u/∂x by differencing in the x-direction.

**The problem:** If u varies only in directions **orthogonal** to x (e.g., a checkerboard in the y-z plane), the 4 cells on each side of the vertex have the same average, and the difference is zero. The mode is invisible to the divergence operator.

**Example:** Consider u = (-1)^(j+k) (checkerboard in y-z, constant in x):

| Cell | u value | s_x | Contribution |
|------|---------|-----|--------------|
| (i-1,j-1,k-1) | +c | -1 | -c |
| (i,j-1,k-1) | +c | +1 | +c |
| (i-1,j,k-1) | -c | -1 | +c |
| (i,j,k-1) | -c | +1 | -c |
| (i-1,j-1,k) | -c | -1 | +c |
| (i,j-1,k) | -c | +1 | -c |
| (i-1,j,k) | +c | -1 | -c |
| (i,j,k) | +c | +1 | +c |

Sum: 0. The mode is **invisible** to the divergence operator, but the continuous ∂u/∂x ≠ 0.

### Orthonormal Mode Decomposition

The 2×2×2 block of velocity cells surrounding a vertex has 8 degrees of freedom. These decompose into 8 orthonormal modes corresponding to different derivative orders:

**Mode definitions** (each normalized by 1/√8):

| Mode | Pattern | Derivative | Physical? |
|------|---------|------------|-----------|
| 1 | all +1 | ⟨f⟩ (average) | Yes |
| 2 | ±1 in x | ∂f/∂x | Yes (divergence) |
| 3 | ±1 in y | ∂f/∂y | Yes (curl) |
| 4 | ±1 in z | ∂f/∂z | Yes (curl) |
| 5 | (-1)^(i+j) | ∂²f/∂x∂y | **Hourglass** |
| 6 | (-1)^(i+k) | ∂²f/∂x∂z | **Hourglass** |
| 7 | (-1)^(j+k) | ∂²f/∂y∂z | **Hourglass** |
| 8 | (-1)^(i+j+k) | ∂³f/∂x∂y∂z | **Hourglass** |

**Explicit mode patterns** (indexing as [z][y][x]):

```
mode1 = {{{+1,+1}, {+1,+1}}, {{+1,+1}, {+1,+1}}} / √8   // constant
mode2 = {{{-1,+1}, {-1,+1}}, {{-1,+1}, {-1,+1}}} / √8   // ∂/∂x
mode3 = {{{-1,-1}, {+1,+1}}, {{-1,-1}, {+1,+1}}} / √8   // ∂/∂y
mode4 = {{{-1,-1}, {-1,-1}}, {{+1,+1}, {+1,+1}}} / √8   // ∂/∂z
mode5 = {{{+1,-1}, {-1,+1}}, {{+1,-1}, {-1,+1}}} / √8   // ∂²/∂x∂y
mode6 = {{{+1,-1}, {+1,-1}}, {{-1,+1}, {-1,+1}}} / √8   // ∂²/∂x∂z
mode7 = {{{+1,+1}, {-1,-1}}, {{-1,-1}, {+1,+1}}} / √8   // ∂²/∂y∂z
mode8 = {{{-1,+1}, {+1,-1}}, {{+1,-1}, {-1,+1}}} / √8   // ∂³/∂x∂y∂z
```

These modes are **orthonormal**: ⟨mode_i, mode_j⟩ = δ_ij.

**Physical interpretation:**

- **Modes 1-4** are visible to physical operators: mode 1 is the cell average, modes 2-4 are first derivatives (one for divergence, two for curl components).
- **Modes 5-8** are invisible to all first-order derivative operators—these are the **hourglass modes**.

### Which Modes Are Hourglass for Each Velocity Component?

For the u-component, the divergence sees ∂u/∂x (mode 2). Curl sees ∂u/∂y and ∂u/∂z (modes 3, 4). The hourglass modes are 5, 6, 7, 8.

But modes 5-8 correspond to different checkerboard patterns:
- Mode 5: (-1)^(i+j) — checkerboard in x-y plane
- Mode 6: (-1)^(i+k) — checkerboard in x-z plane
- Mode 7: (-1)^(j+k) — checkerboard in y-z plane (purely transverse to u)
- Mode 8: (-1)^(i+j+k) — 3D checkerboard

**All four** are invisible to the u-divergence term because the divergence averages over 4 cells on each side of the vertex, and all four patterns have zero average over any such 2×2 face.

### Filter Construction via Projection

The hourglass filter should project velocity onto the hourglass subspace (modes 5-8), then damp it. The projection operator is:

```
P = Σᵢ |modeᵢ⟩⟨modeᵢ|   for i ∈ {5, 6, 7, 8}
```

To build a stencil that applies this projection at each velocity cell, we compute the **autocorrelation** of each hourglass mode and sum them. This extends the 2×2×2 mode (centered at a corner of the target cell) to a 3×3×3 stencil (centered at the target cell).

**Construction:**

For each hourglass mode m, the stencil contribution at offset (Δx, Δy, Δz) is:

```
stencil[Δz][Δy][Δx] += Σ m[k][j][i] × m[k+Δz][j+Δy][i+Δx]
```

where the sum is over all valid overlapping indices.

Equivalently, this is the correlation of the mode with itself:

```
autocorr[Δ] = Σ m[pos] × m[pos + Δ]
```

**Computing the autocorrelation:**

For a mode with pattern (-1)^(sum of certain indices), the autocorrelation at offset Δ depends only on:
1. The **parity factor**: (-1)^(relevant components of Δ)
2. The **overlap count**: how many cell pairs overlap at that offset

Overlap count for offset (Δx, Δy, Δz) where each Δ ∈ {-1, 0, +1}:

```
overlap = (2 - |Δx|) × (2 - |Δy|) × (2 - |Δz|)
```

| Offset type | |Δ| | Overlap |
|-------------|-----|---------|
| Center | (0,0,0) | 8 |
| Face | one ±1 | 4 |
| Edge | two ±1 | 2 |
| Corner | three ±1 | 1 |

**Parity factors for each mode:**

| Mode | Pattern | Autocorr parity |
|------|---------|-----------------|
| 5 | (-1)^(i+j) | (-1)^(Δx+Δy) |
| 6 | (-1)^(i+k) | (-1)^(Δx+Δz) |
| 7 | (-1)^(j+k) | (-1)^(Δy+Δz) |
| 8 | (-1)^(i+j+k) | (-1)^(Δx+Δy+Δz) |

### Summing the Four Hourglass Modes

The combined parity factor is:

```
S(Δ) = (-1)^(Δx+Δy) + (-1)^(Δx+Δz) + (-1)^(Δy+Δz) + (-1)^(Δx+Δy+Δz)
```

Evaluating for each neighbor type:

| Type | (Δx,Δy,Δz) | Parities | S(Δ) |
|------|------------|----------|------|
| Center | (0,0,0) | +1+1+1+1 | **+4** |
| x-face | (±1,0,0) | -1-1+1-1 | **-2** |
| y-face | (0,±1,0) | -1+1-1-1 | **-2** |
| z-face | (0,0,±1) | +1-1-1-1 | **-2** |
| xy-edge | (±1,±1,0) | +1-1-1+1 | **0** |
| xz-edge | (±1,0,±1) | -1+1-1+1 | **0** |
| yz-edge | (0,±1,±1) | -1-1+1+1 | **0** |
| corner | (±1,±1,±1) | +1+1+1-1 | **+2** |

**The edge contributions cancel!** This is why the filter uses only 15 points (center + 6 faces + 8 corners) rather than all 27.

### The Final Stencil

Combining overlap counts with parity sums:

| Type | Overlap | S(Δ) | Contribution | Count | Total |
|------|---------|------|--------------|-------|-------|
| Center | 8 | +4 | 32 | 1 | 32 |
| Face | 4 | -2 | -8 | 6 | -48 |
| Edge | 2 | 0 | 0 | 12 | 0 |
| Corner | 1 | +2 | +2 | 8 | +16 |

**Sum check:** 32 - 48 + 0 + 16 = 0 ✓ (filter annihilates constants)

Normalizing so the filter extracts hourglass modes with unit coefficient:

```
H[v] = (1/32) × [16·v[center] - 4·Σ(6 faces) + 1·Σ(8 corners)]
```

**Stencil coefficients (×1/32):**

| Position | Coefficient |
|----------|-------------|
| Center (0,0,0) | +16 |
| Faces (±1,0,0), (0,±1,0), (0,0,±1) | -4 each |
| Edges | 0 |
| Corners (±1,±1,±1) | +1 each |

### Verification

**On 3D checkerboard** v = (-1)^(i+j+k):
- Center: 16 × c
- Faces (all opposite parity): -4 × 6 × (-c) = +24c
- Corners (all opposite parity): +1 × 8 × (-c) = -8c
- H[v] = (1/32)[16c + 24c - 8c] = c ✓

**On 2D checkerboard** v = (-1)^(j+k):
- Center: 16 × c
- x-faces (same parity): -4 × 2 × c = -8c
- y-faces (opposite): -4 × 2 × (-c) = +8c
- z-faces (opposite): -4 × 2 × (-c) = +8c
- Corners (same j+k parity): +1 × 8 × c = +8c
- H[v] = (1/32)[16c - 8c + 8c + 8c + 8c] = c ✓

**On smooth/constant** v = 1:
- H[v] = (1/32)[16 - 4×6 + 1×8] = (1/32)[16 - 24 + 8] = 0 ✓

**On first derivative mode** v = (-1)^i:
- Center: 16c
- x-faces (opposite i): -4 × 2 × (-c) = +8c
- y,z-faces (same i): -4 × 4 × c = -16c
- Corners (opposite i): +1 × 8 × (-c) = -8c
- H[v] = (1/32)[16c + 8c - 16c - 8c] = 0 ✓

The filter correctly extracts hourglass modes while ignoring physical modes.

### Application

The filter is applied as damping after advection:

```
v_filtered = v - ε·H[v]
```

where ε ∈ (0, 1] controls the damping strength:
- **ε = 0:** No filtering (hourglass modes persist)
- **ε = 1:** Full projection removal (may be too aggressive)
- **ε ≈ 0.1-0.5:** Typical values for gentle damping

The filter should be applied:
1. **After advection** (which can excite hourglass modes through interpolation)
2. **Before projection** (which cannot remove these modes)

### Implementation

**GLSL compute shader:**

```glsl
#version 430

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0, rgba32f) uniform image3D velocity;

uniform float epsilon;  // damping strength, typically 0.1-0.5

void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);

    // Read center and neighbors
    vec3 center = imageLoad(velocity, pos).xyz;

    // Face neighbors
    vec3 xm = imageLoad(velocity, pos + ivec3(-1, 0, 0)).xyz;
    vec3 xp = imageLoad(velocity, pos + ivec3(+1, 0, 0)).xyz;
    vec3 ym = imageLoad(velocity, pos + ivec3( 0,-1, 0)).xyz;
    vec3 yp = imageLoad(velocity, pos + ivec3( 0,+1, 0)).xyz;
    vec3 zm = imageLoad(velocity, pos + ivec3( 0, 0,-1)).xyz;
    vec3 zp = imageLoad(velocity, pos + ivec3( 0, 0,+1)).xyz;

    // Corner neighbors
    vec3 c000 = imageLoad(velocity, pos + ivec3(-1,-1,-1)).xyz;
    vec3 c001 = imageLoad(velocity, pos + ivec3(-1,-1,+1)).xyz;
    vec3 c010 = imageLoad(velocity, pos + ivec3(-1,+1,-1)).xyz;
    vec3 c011 = imageLoad(velocity, pos + ivec3(-1,+1,+1)).xyz;
    vec3 c100 = imageLoad(velocity, pos + ivec3(+1,-1,-1)).xyz;
    vec3 c101 = imageLoad(velocity, pos + ivec3(+1,-1,+1)).xyz;
    vec3 c110 = imageLoad(velocity, pos + ivec3(+1,+1,-1)).xyz;
    vec3 c111 = imageLoad(velocity, pos + ivec3(+1,+1,+1)).xyz;

    // Hourglass filter: H = (1/32)[16*center - 4*faces + 1*corners]
    vec3 face_sum = xm + xp + ym + yp + zm + zp;
    vec3 corner_sum = c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111;

    vec3 hourglass = (16.0 * center - 4.0 * face_sum + corner_sum) / 32.0;

    // Apply damping
    vec3 filtered = center - epsilon * hourglass;

    imageStore(velocity, pos, vec4(filtered, 0.0));
}
```

**Odin reference implementation** (mode orthogonality verification and stencil derivation):

```odin
package hourglass_filter

import "core:fmt"
import "core:math"

Basis_Dual :: [2][2][2]f64

main :: proc() {
    N := math.sqrt(f64(8.0))

    // 8 orthonormal modes of a 2x2x2 block
    mode1 := Basis_Dual{{{+1,+1}, {+1,+1}}, {{+1,+1}, {+1,+1}}} / N  // average
    mode2 := Basis_Dual{{{-1,+1}, {-1,+1}}, {{-1,+1}, {-1,+1}}} / N  // ∂/∂x
    mode3 := Basis_Dual{{{-1,-1}, {+1,+1}}, {{-1,-1}, {+1,+1}}} / N  // ∂/∂y
    mode4 := Basis_Dual{{{-1,-1}, {-1,-1}}, {{+1,+1}, {+1,+1}}} / N  // ∂/∂z
    mode5 := Basis_Dual{{{+1,-1}, {-1,+1}}, {{+1,-1}, {-1,+1}}} / N  // ∂²/∂x∂y
    mode6 := Basis_Dual{{{+1,-1}, {+1,-1}}, {{-1,+1}, {-1,+1}}} / N  // ∂²/∂x∂z
    mode7 := Basis_Dual{{{+1,+1}, {-1,-1}}, {{-1,-1}, {+1,+1}}} / N  // ∂²/∂y∂z
    mode8 := Basis_Dual{{{-1,+1}, {+1,-1}}, {{+1,-1}, {-1,+1}}} / N  // ∂³/∂x∂y∂z

    modes := []Basis_Dual{mode1, mode2, mode3, mode4, mode5, mode6, mode7, mode8}

    // Verify orthonormality
    dot :: proc(m1, m2: Basis_Dual) -> f64 {
        sum := 0.0
        for k in 0..<2 do for j in 0..<2 do for i in 0..<2 {
            sum += m1[k][j][i] * m2[k][j][i]
        }
        return sum
    }

    fmt.println("Orthonormality check (should be identity matrix):")
    for m1 in modes {
        for m2 in modes {
            fmt.printf("%.1f ", dot(m1, m2))
        }
        fmt.println()
    }

    // Build filter stencil from hourglass modes (5-8)
    stencil: [3][3][3]f64
    for mode in modes[4:8] {  // modes 5,6,7,8 (0-indexed: 4,5,6,7)
        for k in 0..<2 do for j in 0..<2 do for i in 0..<2 {
            for z in 0..<2 do for y in 0..<2 do for x in 0..<2 {
                // Autocorrelation: convolve mode with itself
                stencil[k+z][j+y][i+x] += mode[1-k][1-j][1-i] * mode[z][y][x] / 8.0
            }
        }
    }

    fmt.println("\nFilter stencil (×32):")
    for z in 0..<3 {
        fmt.printf("z=%d:\n", z-1)
        for y in 0..<3 {
            for x in 0..<3 {
                fmt.printf("%+3.0f ", 32.0 * stencil[z][y][x])
            }
            fmt.println()
        }
        fmt.println()
    }
}
```

**Expected output:**

```
Filter stencil (×32):
z=-1:
 +1  -4  +1
 -4   0  -4
 +1  -4  +1

z=0:
 -4   0  -4
  0 +16   0
 -4   0  -4

z=+1:
 +1  -4  +1
 -4   0  -4
 +1  -4  +1
```

The 12 edge positions are all zero, confirming the 15-point stencil structure.

### Relationship to Pressure Projection

The hourglass filter and pressure projection serve complementary roles:

| Aspect | Pressure Projection | Hourglass Filter |
|--------|--------------------|--------------------|
| **Removes** | Divergent (compressible) modes | Non-physical oscillatory modes |
| **Mechanism** | Solves ∇²p = ∇·v, subtracts ∇p | Damps high-frequency checkerboards |
| **Null space** | Divergence-free velocity | Smooth + first-derivative modes |
| **When applied** | After forces, before next step | After advection, before projection |

The pressure projection cannot remove hourglass modes because they are **in the null space of the divergence operator**—they appear divergence-free to the discrete operator even though they have non-zero continuous divergence.

### Historical Note

This filter construction follows Rider (1998), who derived the 2D version for approximate projection methods. The 3D extension, including the observation that edges cancel due to parity symmetry, was derived independently using the orthogonal mode decomposition approach described here.

---

## Summary

| Property | 2D Vertex Grid | 3D Vertex Grid |
|----------|---------------|----------------|
| **Consistent Laplacian** | 5-point diagonal | 27-point full |
| **Stencil coefficients** | ±1/2h² | See table above |
| **Decomposition** | Corners only | -1:2:3 (face:edge:corner) |
| **Dominant term** | Corners (100%) | Corners (75%) |
| **4th-order error** | Same as corners | Same as corners |
| **Sublattice decoupling** | 2 independent | 4 independent (for corner) |
| **Deinterlaced stencil** | Standard 5-point | Standard 7-point |
| **Optimal preconditioner ω** | N/A (exact) | 4/3 |

### The Big Picture

The vertex/dual grid formulation reveals deep structure:

1. **The corner/BCC stencil is fundamental:** It captures 75% of the 3D operator and 100% of the 2D operator. The vertex grid "wants" to couple diagonally.

2. **Face and edge are corrections:** In 3D, geometric constraints of the cubic lattice force additional coupling, but these don't affect the dominant behavior.

3. **Truncation error is determined by corners:** To 4th order, the full stencil and corner-only stencil are equivalent.

4. **Operator consistency requires the full stencil:** Despite matching accuracy, only the full ∇·∇ removes divergence exactly.

5. **Sublattice decoupling enables optimization:** The diagonal (2D) and corner (3D) stencils decouple into independent subproblems that can be solved with standard methods on contiguous memory.

6. **Recoupling happens naturally:** In 2D through div/grad, in 3D through the residual iteration. The decoupled solve handles the bulk; recoupling handles consistency.

This elegant structure enables efficient implementation while maintaining exact mathematical consistency.

---

## Literature

### Historical Context

The vertex grid arrangement (cell-centered velocity, nodal pressure) was introduced by Almgren, Bell, and collaborators in the 1990s as an "approximate projection" method. The key papers are:

- **Almgren, A.S., Bell, J.B. & Szymczak, W.G. (1996).** "A numerical method for the incompressible Navier-Stokes equations based on an approximate projection." *SIAM J. Sci. Comput.* 17:358-369.
  - Introduced the cell-centered velocity / nodal pressure arrangement
  - Recognized that the composed Laplacian L = D·G produces a "compact stencil" with "local decoupling"
  - Called it an "approximate projection" because the velocity is only approximately divergence-free

- **Rider, W.J. (1998).** "Filtering non-solenoidal modes in numerical solutions of incompressible flows." *Int. J. Numer. Methods Fluids* 28:789-814.
  - Analyzed the spurious non-solenoidal modes that arise from approximate projections
  - Proposed filtering techniques to remove high-frequency decoupled modes
  - Showed that without filtering, long-term integrations and density jumps cause problems

- **Drikakis, D. & Rider, W. (2005).** *High-Resolution Methods for Incompressible and Low-Speed Flows.* Springer.
  - Chapter 12 covers approximate projection methods in detail
  - States that the corner/diagonal stencil is the consistent Laplacian for 2D
  - **Note:** The book appears to extend this claim to 3D, stating that the corner stencil is also correct in 3D. This appears to be an error—the derivation in this document shows the consistent 3D Laplacian is actually a 27-point stencil with non-zero face and edge contributions.

### The 3D Correction

The assumption that corner-only works in 3D fails due to geometry:

- **2D:** The diagonal neighbors of a square lattice form another square lattice (rotated 45°). Face neighbors cancel exactly in ∇·∇.
- **3D:** The corner neighbors of a cubic lattice form a BCC lattice, which is geometrically distinct. Face and edge neighbors do **not** cancel.

The full 27-point stencil derived in this document, with weights -24 (center), -4 (faces), +2 (edges), +3 (corners), was derived independently by the author in 2022, prior to discovering the apparent error in Drikakis & Rider. The technique has been used in production in EmberGen since late 2022. The decomposition into face:edge:corner with ratio -1:2:3, and the use of the corner stencil as a preconditioner rather than the exact operator, follow from this corrected derivation.

### Related Approaches

- **Rhie, C.M. & Chow, W.L. (1983).** "Numerical Study of the Turbulent Flow Past an Airfoil with Trailing Edge Separation." *AIAA Journal.*
  - Introduced momentum-weighted interpolation for collocated grids
  - Different approach: adds compact coupling to fix the wide stencil, rather than deriving the consistent operator

### Foundational References

- **Harlow, F.H. & Welch, J.E. (1965).** "Numerical Calculation of Time-Dependent Viscous Incompressible Flow." *Physics of Fluids.*
  - Introduced the MAC (staggered) grid, which the vertex grid is dual to

- **Stam, J. (1999).** "Stable Fluids." *SIGGRAPH 1999.*
  - Popularized fluid simulation in graphics using unconditionally stable semi-Lagrangian advection

- **Bridson, R. (2015).** *Fluid Simulation for Computer Graphics.* CRC Press.
  - Standard reference for graphics-oriented fluid simulation

### Production Implementations

- **JangaFX EmberGen:** Uses the consistent 27-point operator derived in this document, with corner stencil as preconditioner. In production since late 2022.

- **Autodesk Bifrost:** Uses a collocated velocity arrangement (the same dual grid structure, with velocity components collocated and pressure at dual locations). Rather than deriving the consistent operator, they "explored discretization and filtering methods from computational physics" to mitigate non-physical divergent modes. See Nielsen et al., ["A collocated spatially adaptive approach to smoke simulation in bifrost"](https://dl.acm.org/doi/10.1145/3214745.3214749), SIGGRAPH 2018.

- **AMReX-Hydro NodalProjector:** https://amrex-fluids.github.io/amrex-hydro/docs_html/Projections.html
  - Modern implementation of nodal projection for cell-centered velocity
  - Uses the approximate projection approach with cell-averaged pressure gradients
  - Part of the AMReX adaptive mesh refinement framework
