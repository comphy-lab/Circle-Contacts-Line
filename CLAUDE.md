# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Initial condition generator for drop/bubble contact line simulations using Basilisk CFD. Creates smooth interface geometries for a drop/bubble touching a flat wall with a fillet transition at the contact point.

## Build Commands

Compile the Basilisk C code:
```bash
qcc -O2 -Wall InitialCondition.c -o InitialCondition -lm
```

Run the Python generator:
```bash
python generate_initial_condition.py --delta 0.01
python generate_initial_condition.py --delta 0.01 --images
python generate_initial_condition.py --delta 0.01 --images --basilisk --L0 3
```

## Architecture

### Workflow
1. **Python (CLI/Jupyter)**: Generates interface coordinates using `get_circles(delta)`
2. **Data file**: Coordinates saved to `f_Testing.dat` (space-separated x,y pairs)
3. **Basilisk C**: `InitialCondition.c` reads the data file and creates volume fraction field

### Key Parameters
- `delta`: Neck radius / minimum length scale (separation = 2*delta)

### Geometry Construction
The interface consists of three connected segments:
1. **Circle 1** (orange): Main drop, radius 1, centered at (-(1+delta), 0)
2. **Fillet circle** (red): Smooth transition at contact, computed radius Rf
3. **Vertical line** (blue): Wall contact extending upward from fillet

### Output Structure
```
DataFiles_delta{value}/
  f_init.dat           # Interface: circle + fillet + vertical line
  f-drop_init.dat      # Full drop circle (for filling)

ImageFiles_delta{value}/
  TestWithoutBasilisk.pdf    # Pure geometry visualization
  TestWithBasilisk.pdf       # Comparison with Basilisk facets

f_Testing.dat          # Working file for Basilisk input (root, gitignored)
f_facets.dat           # Basilisk facets output (root, gitignored)
```

## Python Dependencies

```python
numpy, matplotlib, pandas
```

Requires LaTeX for plot labels (`plt.rc('text', usetex=True)`).

## Basilisk Headers

```c
#include "axi.h"           // Axisymmetric coordinates
#include "fractions.h"     // Volume fraction computation
#include "distance.h"      // Signed distance function
#include "navier-stokes/centered.h"
```

## File Descriptions

- `generate_initial_condition.py`: CLI script for batch generation
- `InitialCondition.ipynb`: Interactive notebook for development
- `InitialCondition.c`: Basilisk C code for CFD validation
- `DataFiles_delta*/`: Organized data outputs by delta value
- `ImageFiles_delta*/`: Organized image outputs by delta value
