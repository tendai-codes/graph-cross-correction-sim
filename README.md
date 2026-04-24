# graph-cross-correction-sim

A staged numerical modelling project exploring molecular cross-correction in multinucleated muscle fibres.

## Biological motivation

This project is inspired by my MSc thesis on U7 snRNA-mediated cross-correction in Duchenne muscular dystrophy. The aim is not to reproduce the experimental data, but to build a simplified computational model that asks whether local signal transport could help explain amplified rescue beyond the corrected nuclei themselves.

The core biological question is:

> Can a small number of corrected nuclei produce enough local signal to rescue neighbouring nuclei through spatial transport?

## Project structure

The project is organised as staged parts rather than calendar weeks. Each part represents a modelling decision and a refinement of the previous version.

| Part | Notebook | Purpose |
|---|---|---|
| Part 1 | `Part1.ipynb` | Build the first naive line simulation using manual neighbour averaging. |
| Part 2 | `Part2.ipynb` | Replace heuristic neighbour averaging with graph Laplacian diffusion. |

## Part 1 — Naive line simulation

Part 1 represents the fibre as a one-dimensional line of nuclei.

The first model:

- marks a small fraction of nuclei as corrected sources
- lets signal move to immediate neighbours using a hand-written averaging rule
- records rescue once signal crosses a threshold
- visualises the final signal profile and rescue pattern

This part exists to convert the biological idea into executable logic before introducing a more formal transport operator.

## Part 2 — Graph Laplacian diffusion

Part 2 keeps the same biological idea but changes the transport mechanism.

Instead of manually mixing left and right neighbours, nuclei are represented as nodes in a graph. Connectivity is stored in an adjacency matrix `A`, and signal transport is modelled using the graph Laplacian:

```text
L = D − A
```

The signal update is:

```text
u(t + 1) = u(t) + dt[-αLu(t) − βu(t) + q]
```

where:

- `u(t)` is the signal vector
- `−αLu(t)` models diffusion across connected nuclei
- `−βu(t)` models signal decay
- `q` represents continuous production from corrected nuclei

This replaces the heuristic neighbour-averaging rule from Part 1 with a standard diffusion-on-networks formulation.

## Conceptual use of Masuda, Porter and Lambiotte (2017)

Masuda, Porter and Lambiotte's review, *Random walks and diffusion on networks*, is used conceptually to justify the transition from local neighbour averaging to graph-based diffusion.

The project does not implement the paper's full random-walk methodology. Instead, the paper helps frame the modelling decision that diffusion over discrete interacting units can be represented using network structure and Laplacian operators.

In this project:

```text
corrected nuclei = source nodes
nuclei = graph nodes
local signal exchange = graph diffusion
rescue = threshold response to transported signal
```

This creates a clear bridge between the MSc thesis observation and the computational implementation.

## Current modelling assumptions

- The fibre is simplified as a one-dimensional spatial structure.
- Corrected nuclei are initially placed at the start of the line.
- Signal propagation is deterministic.
- Rescue is triggered by a fixed signal threshold.
- The current model is a proof of concept, not a calibrated biological simulation.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks:

```bash
jupyter notebook
```

## Dependencies

The current project uses:

- `numpy`
- `matplotlib`
- `jupyter`

## Next modelling steps

Planned refinements:

- introduce geometry-based graph construction
- support non-uniform nuclear spacing
- test different corrected-nuclei placement strategies
- run parameter sensitivity experiments
- replace instantaneous rescue with cumulative exposure-based rescue

## Reference

Masuda, N., Porter, M. A., & Lambiotte, R. (2017). *Random walks and diffusion on networks*. Physics Reports.
