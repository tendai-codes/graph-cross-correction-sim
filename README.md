# graph-cross-correction-sim

## Week 1 state

This is the first rough computational translation of the project idea.

### Aim this week
Get a basic proof-of-concept simulation running.

### Current model
- represent the fibre as a 1D line of nuclei
- mark a small fraction of nuclei as corrected sources
- propagate signal using a hand-written neighbour-averaging rule
- mark nuclei as rescued once signal crosses a simple threshold

### Why this version exists
At this stage the priority is not elegance. The priority is converting a biological idea into executable logic.

### Known limitations
- the update rule is ad hoc rather than derived from a formal diffusion operator
- geometry and dynamics are mixed together in one function
- rescue uses an instantaneous threshold without justification
- corrected nuclei are always placed at the start of the fibre
