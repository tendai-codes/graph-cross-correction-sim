# Biological Hypotheses and Interpretation Notes

## Biological mapping

| Model object | Biological interpretation |
| --- | --- |
| Node | Myonucleus in a multinucleated skeletal muscle fibre |
| Edge | Local molecular exchange pathway between nearby nuclei |
| Signal | Therapeutic protein, transcript, or correction-associated molecular product |
| Corrected nucleus | Successfully transduced or genetically corrected nucleus |
| `alpha` | Effective transport or diffusion strength |
| `beta` | Degradation, dilution, or turnover of the therapeutic signal |
| Source vector `q` | Production of therapeutic signal by corrected nuclei |
| Rescue threshold | Minimum signal level associated with functional rescue |
| Cumulative exposure | Integrated therapeutic dosage over time |

## Hypotheses tested

1. **Correction placement matters.** Clustered, central, random, and evenly spaced
   correction patterns are compared on the same fibre geometry.
2. **Correction efficiency matters.** Increasing the corrected fraction should often
   increase rescue, but the exact relationship is reported rather than assumed.
3. **Protein stability matters.** Lower degradation is interpreted as greater signal
   persistence and is tested through a sweep over `beta` values.
4. **Fibre size matters.** Rescue is measured as the number of nuclei changes.
5. **Random placement has variability.** Repeated trials estimate mean rescue and
   confidence intervals instead of relying on one random layout.
6. **Fiedler partitions may reveal topological bottlenecks.** Clustered and split
   placement across graph partitions are compared across repeated graph realisations.

## Summary

The simulations are best are mechanistic in nature thought experiments. If local diffusion and decay are the dominant processes, then the model will show which spatial and transport conditions make rescue more or less likely.
