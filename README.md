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
| Part 3 | `Part3.ipynb` | Introduce geometry-based spatial graphs where nuclear connectivity depends on distance. |
| Part 4 | `Part4.ipynb` | Run parameter sensitivity experiments to test how transport, decay, and correction patterns affect rescue. |
| Part 5 | `Part5.ipynb` | Replace instantaneous rescue thresholds with cumulative exposure dynamics to model time-integrated signal accumulation and delayed rescue effects. |

## Conceptual modelling path

The project follows this reasoning chain:

```text
MSc thesis observation
→ local cross-correction appears spatial
→ corrected nuclei can be treated as signal sources
→ signal transport can be approximated as diffusion
→ nuclei can be represented as graph nodes
→ rescue can be modelled as a threshold outcome
```

# Parts completed

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

$$
L = D - A
$$


The signal update is:

$$
u_{t+1} = u_t + dt\left[-\alpha L u_t - \beta u_t + q\right]
$$

where:

- $u_t$ is the signal vector at timestep $t$
- $-\alpha L u_t$ spreads signal across connected nuclei through graph diffusion
- $-\beta u_t$ models explicit signal decay over time
- $q$ adds continuous signal production from corrected nuclei
- $dt$ controls the size of each numerical update step


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

### Part 3 — Geometry-based spatial graph

The third model keeps the Laplacian diffusion update from Part 2 but changes how the graph is built.

Instead of using a fixed line graph, nuclear positions are generated first. Edges are then created between nuclei that are close enough to interact.

The modelling chain becomes:

```text
nuclear positions → adjacency matrix → Laplacian → diffusion simulation → rescue outcome
```

This part introduces the idea that nuclear spacing and local connectivity may influence cross-correction behaviour.

## Geometry-based diffusion animation

The animation below shows signal spreading across a spatial graph of nuclei. Corrected nuclei act as signal sources, while neighbouring nuclei receive signal through graph-based diffusion.

This demonstrates that rescue depends not only on the number of corrected nuclei, but also on where they are placed in the spatial network.

Key outcome:

- the graph is now based on spatial proximity
- irregular nuclear spacing can affect signal reach
- the same Laplacian framework can operate on more realistic geometry

### Part 4 — Parameter sensitivity experiments

The fourth model keeps the geometry-based graph from Part 3 and uses it to explore how rescue behaviour changes under different parameter assumptions.

The parameters explored are:

- diffusion strength
- signal decay rate
- corrected nuclei fraction
- rescue threshold

This part shifts the project from building the simulation structure to interrogating model behaviour.

Key outcome:

- rescue is controlled by both transport and activation assumptions
- decay and threshold can prevent rescue even when signal spreads spatially
- corrected nuclei placement may affect rescue efficiency because diffusion is local

### Part 5 — Cumulative exposure rescue model

Part 5 improves the biological interpretation of rescue by replacing the instantaneous signal threshold with a cumulative exposure threshold. Instead of asking whether a nucleus crosses a signal threshold at one timestep, this part asks whether sustained exposure to local signal is sufficient to produce rescue.

This keeps the geometry-based graph Laplacian diffusion model from Part 3 and the sensitivity-testing mindset from Part 4, but adds an exposure state variable:

$$
e_{t+1} = e_t + u_t dt
$$

A nucleus is marked as rescued when cumulative exposure exceeds a defined threshold. This makes the model better suited to exploring whether weak but persistent cross-correction signals could still produce biologically meaningful rescue.

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
- `pytest`
- `pillow`

## Reference

Masuda, N., Porter, M. A., & Lambiotte, R. (2017). Random walks and diffusion on networks. *Physics Reports*, 716-717, 1-58. https://doi.org/10.1016/j.physrep.2017.07.007

Bruusgaard, J. C., Liestøl, K., Ekmark, M., Kollstad, K., & Gundersen, K. (2003). Number and spatial distribution of nuclei in the muscle fibres of normal mice studied in vivo. *The Journal of Physiology*, 551(2), 467-478. https://doi.org/10.1113/jphysiol.2003.045328