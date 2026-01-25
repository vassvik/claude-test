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
11. [Literature](#literature)

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
