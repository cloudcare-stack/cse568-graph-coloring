# Graph Coloring with Genetic Algorithms

A Python implementation of a genetic algorithm for graph coloring, developed for ASU CSE 568: Biocomputing with AI assistance.

The goal is to assign colors to nodes so that adjacent nodes have different colors. The algorithm maximizes the number of satisfied edges for a fixed number of colors; it does not search for the chromatic number.

## Features

- Undirected graphs stored as adjacency lists
- Binary genomes with fixed-width color encoding
- Fitness based on satisfied edges, counted once
- Tournament selection, two-point crossover, and bit-flip mutation
- Invalid color repair during evolution
- Elitism and early stopping when all edges are satisfied
- Optional random seed for reproducible runs within the same environment

## Setup

Requires Python 3 and DEAP.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 student_solution.py graphs/graph_1.txt --seed 42
```

The program prints decoded node colors and the number of satisfied edges.

## Graph format

The first line is the node count, the second is the color count, and remaining lines contain zero-based undirected edge endpoints.

```text
4
3
1 3
0 2
2 1
```

## Python interface

```python
from student_solution import init_graph, decode_individual, run_ga

graph = init_graph('graphs/graph_1.txt')
best, fitness = run_ga('graphs/graph_1.txt', population_size=300, ngen=200,
                       cxpb=0.8, mutpb=0.3)
print(decode_individual(graph, best))
print(fitness)
```

`decode_individual` returns a list of colors, or `None` for invalid genomes. `eval_graph` returns a one-element fitness tuple. `run_ga` returns `(best_individual, best_fitness_tuple)`.

## Local validation

Using the supplied instructor DEAP-compatible shim, five random seeds (0 through 4) per supplied graph produced these results:

| Graph | Nodes | Colors | Satisfied edges in each run |
|---|---:|---:|---:|
| graph_1 | 4 | 3 | 3/3 |
| graph_2 | 5 | 3 | 7/7 |
| graph_3 | 11 | 4 | 20/20 |
| graph_4 | 37 | 4 | 72/72 |

Additional checks covered graph parsing, valid decoding, invalid and incomplete genomes, partial fitness, and repeated calls. These are local observations, not guarantees for other graphs or Gradescope hidden tests. Tests used the instructor shim; full DEAP may produce different seeded trajectories.

## Project notes

The graph inputs were supplied with the course materials. Instructor PDFs, screenshots, and autograder support files are excluded. The provided overview omitted the student template's exact Graph interface and run_ga return contract, so those remain compatibility assumptions pending course-template verification.
