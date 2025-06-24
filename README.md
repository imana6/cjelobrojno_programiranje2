# Diamond Problem Solver

This project solves a diamond placement problem on a hexagonal grid using integer linear programming (ILP) and cutting planes methods.

## Problem Description

The problem involves placing diamonds of three different orientations (yellow, green, red) on a hexagonal grid such that:
- No two diamonds share the same point
- The goal is to maximize the number of diamonds placed

## Dependencies

The project requires the following Python packages:
- `numpy` - For numerical operations
- `PuLP` - For linear programming optimization
- `matplotlib` - For visualization
- `networkx` - For graph operations

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Simply run the main script:

```bash
python dijamanti.py
```

## What the Program Does

1. **Problem Setup**: Creates a hexagonal grid and generates all possible diamond placements
2. **ILP Solution**: Solves the problem using integer linear programming to find the optimal solution
3. **Cutting Planes**: Implements a cutting planes method as an alternative approach
4. **Visualization**: Displays the solution graphically with colored diamonds on the hexagonal grid
5. **Statistics**: Shows problem statistics including total points, diamonds, and constraints

## Output

The program will:
- Print statistics about the problem setup
- Show the ILP solution results
- Show the cutting planes solution results
- Compare the two approaches
- Display a graphical visualization of the optimal solution

## Notes

- The program uses the CBC solver (included with PuLP) for optimization
- The visualization shows diamonds in different colors: gold (yellow), light green, and light coral (red)
- The hexagonal grid is limited to a diamond-shaped region for computational efficiency 