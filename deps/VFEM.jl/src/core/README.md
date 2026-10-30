# Core (general-purpose FEM kernel)

This is the reusable, problem-agnostic FEM kernel. It works on any 2D
triangulation or 3D tetrahedral mesh, supports both `Float64`
(approximation) and `Interval{Float64}` (verified) element types, and
makes no assumptions about which physical problem is being solved.

Everything that depends on a specific PDE, potential, or domain
belongs in `src/applications/<name>/`, not here.

## Layout

| Directory                | Role                                                       |
| ------------------------ | ---------------------------------------------------------- |
| `singular_moments/`      | Closed-form ∫_K (1/r), (1/r²), polynomial moments over a triangle. Consumed by `quadrature_singular/` for the 3D face integrals. |
| `bernstein/`             | Bernstein polynomial kernel: multi-indices, products, Gram. |
| `quadrature_singular/`   | Vertex-singular Coulomb-style closed forms (3D faces, tets). |
| `mesh/`                  | Tetrahedral mesh layer (Mesh3D, facet/edge connectivity).  |
| `mesh2d/`                | Triangular mesh layer (Mesh2D, tri↔edge maps, h_max).      |
| `assembly2d/`            | 2D matrix builders: CR, ECR, CECR, Lagrange P1/P2.         |
| `assembly3d/`            | 3D matrix builders: CR, ECR, CECR, Lagrange, DG, Raviart-Thomas. |
| `eigensolve2d/`          | 2D Schrödinger / Laplace eigensolvers + LG sharpening.     |
| `eigensolve3d/`          | 3D eigensolvers: CECR Schrödinger, verified CR Laplace, Lehmann-Goerisch lower bounds, one-piece bubble, truncation correction. |
| `potentials/`            | Coulomb potential helpers (multi-center, generic).         |

## Conventions

- All numeric routines are generic on `T<:Real`. The same call works
  in both modes:
  ```julia
  A0_f = create_matrix_crouzeix_raviart(m)                       # Float64
  A0_i = create_matrix_crouzeix_raviart(m; T = Interval{Float64}) # interval
  ```
- 2D mesh fields are lowercase (`m.nodes`, `m.elements`); 3D fields
  are capitalised (`m.NodeList`, `m.ElementList`). This matches the
  MATLAB code base on disk for grep compatibility.
- Field naming, DOF numbering, and quadrature constants are pinned to
  match `Ver1/vfem2d/` and `Ver1/VFEM3D/`. Cross-validation fixtures
  live under `test/fixtures/`.

## When to add to core vs. applications

A function belongs in `core/` if it:

- Operates on a `Mesh2D` / `Mesh3D` plus a generic potential / form,
  with no hard-coded physical constants.
- Is reusable across more than one application.

A function belongs in `applications/<name>/` if it:

- Sets a specific potential (e.g. `V(x) = -1/|x - origin|` for
  hydrogen) or domain (e.g. ball of fixed radius).
- Bundles reference values, exact eigenvalues, or pre-baked
  parameter choices for a specific physical problem.
- Wraps the kernel into a one-call driver for a specific case.
