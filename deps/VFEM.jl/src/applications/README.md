# Applications

Concrete problem drivers that compose the general-purpose kernel in
`src/core/`. Each application is a submodule of `VFEM.Applications`
that imports kernel functions from `..VFEM` and adds:

- **Problem-specific parameters** (potential setup, exact eigenvalues
  if known, truncation radius, recommended mesh family).
- **A driver** that runs the full bound pipeline end-to-end on a
  user-supplied mesh and returns a structured result.
- **Reference data** (analytic eigenvalues, MATLAB-validated tables)
  that tests can pin against.
- **Meshes** under `applications/<name>/meshes/` if the problem ships
  with its own mesh family. Load via `mesh_load_from_folder` from the
  kernel.

Per-application directory layout:

```
src/applications/<name>/
├── <Name>.jl        # the submodule entry point
├── parameters.jl    # potential, charges, exact answers
├── driver.jl        # one-call pipeline (mesh -> bound)
└── meshes/          # optional, if the problem has a mesh family
```

## Currently planned

| Application  | Status          | MATLAB reference              |
| ------------ | --------------- | ----------------------------- |
| `Hydrogen`   | not yet ported  | `cases/hydrogen/`             |
| `H2Plus`    | not yet ported  | `cases/h2plus/`               |
| `ConstantV` | not yet ported  | `cases/constant_V/`           |

Until an application is added, the basic `Hydrogen`-style pipeline
can be assembled directly from kernel exports — see
`test/test_hydrogen_e2e.jl` for the worked example on the cube_r1
fixture.

## When to add an application

Add a new `Applications.<Name>` submodule when:

- A driver script needs to be written more than once. (One-off
  examples belong in `examples/`, not here.)
- There is reference data (analytic eigenvalues, published tables) the
  test suite should pin against.
- The problem has its own mesh family worth shipping.

Otherwise, keep the kernel exports and let user code compose them.
