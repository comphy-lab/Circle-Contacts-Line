# Circle-Contacts-Line

Initial condition for coalescence of drops and bubbles touching a flat wall.

## Quick Start

1. Generate initial conditions:
```bash
python generate_initial_condition.py --delta 0.01 --images --basilisk
```

2. Or use the Jupyter notebook for interactive exploration:
```bash
jupyter notebook InitialCondition.ipynb
```

## Known Issues

If you get an error: `Exec format error: './InitialCondition'`

Recompile the InitialCondition.c:
```bash
qcc -O2 -Wall InitialCondition.c -o InitialCondition -lm
```

## Directory Structure

```
Circle-Contacts-Line/
├── generate_initial_condition.py  # CLI tool
├── InitialCondition.ipynb         # Interactive notebook
├── InitialCondition.c             # Basilisk C code
├── DataFiles_delta{value}/        # Data outputs
│   ├── f_init.dat                 # Interface coordinates
│   └── f-drop_init.dat            # Full drop circle coordinates
├── ImageFiles_delta{value}/       # Image outputs
│   ├── TestWithoutBasilisk.pdf    # Pure geometry
│   └── TestWithBasilisk.pdf       # Basilisk comparison
├── CLAUDE.md                      # AI documentation
└── README.md                      # This file
```

## Usage

```bash
# Basic generation
python generate_initial_condition.py --delta 0.01

# With images
python generate_initial_condition.py --delta 0.01 --images

# With Basilisk validation
python generate_initial_condition.py --delta 0.01 --images --basilisk

# Custom output folder
python generate_initial_condition.py --delta 0.01 --data-folder MyData --images
```
