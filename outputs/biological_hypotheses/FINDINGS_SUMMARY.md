# Biological hypothesis outputs — current model run

These statements describe the outputs from the current deterministic model settings.

## Clustered vs distributed correction

With 60 nuclei and 15% corrected nuclei, the left-cluster strategy rescued 20% of
nuclei. Central-cluster, evenly spaced, and random placement rescued 0% at the
chosen rescue threshold. This suggests that, under these parameters, placement can
matter, but the direction of the effect depends on graph geometry and threshold.

## Gene-correction efficiency

Evenly spaced correction at 5%, 10%, and 20% produced 0% rescued nuclei. At 40%
and 60% correction, the model produced 100% rescue. Under this threshold, rescue
behaviour is switch-like rather than gradual.

## Protein stability

At beta = 0.01, rescue reached 100%. At beta = 0.03, 0.07, and 0.12, rescue was
0% under the same rescue threshold. This supports the model-level interpretation
that signal persistence can dominate rescue outcomes.

## Fibre size scaling

For evenly spaced 15% correction, the 20-nucleus model rescued 10% of nuclei.
The 50-, 100-, and 250-nucleus models rescued 0% under the same settings.
This suggests that size/density assumptions influence rescue outcomes.

## Random placement statistics

Across 250 random placement trials, the mean rescued fraction was approximately
0.0023 with a 95% bootstrap confidence interval of approximately 0.0008 to 0.0044.
This indicates that random placement rarely crossed the selected threshold in the
current parameter regime.

## Fiedler validation

Across 100 graph realisations, Fiedler-clustered placement had a higher rescue
fraction than Fiedler-split placement in 95% of runs, was equal in 3%, and was worse
in 2%. The mean clustered-minus-split difference was approximately 0.083.
