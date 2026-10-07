# Methodological contract

## Unit of generalization

Trials are the unit of splitting. Binning turns a trial into many observations, but those bins are not independent examples for evaluation. A random bin-level split would leak trial-specific event timing and firing structure into both train and test. Every model comparison therefore uses complete held-out trials.

The current odor analysis uses a −2 to +3 s window aligned within each trial
to first odor inhalation. It does not use the session clock or trial-start time
as a substitute for that event.

## Causal feature construction

For a bin at time *t*, the design matrix includes:

- six identity-specific event inputs expanded into causal lag bases: OR/OU are
  one-bin impulses at first odor inhalation; CR/CU are impulses at context
  entry; CER/CEU are impulses at context exit. Each has 25 subsequent 50-ms
  coefficients (1.25 s), allowing its effect to evolve after the event;
- a three-kernel generic encoding (odor, context entry and context exit) is
  retained as an ablation baseline when evaluating all four odor/context trial
  types together;
- only bins before *t* from the target neuron's spike count;
- only bins before *t* from each candidate coupling neuron's spike count.

No-odor control trials are retained. Their missing odor stimulus is represented
by all-zero OR and OU covariates rather than by an artificial event time.

Current-bin and future-bin spike counts are excluded from history and coupling blocks.

The current default uses 30 history bins of 50 ms (1.5 s) for both self-history
and coupling features. This restores the history horizon used in the original
class notebook; it remains a modeling choice to validate rather than a claim
about a biological interaction timescale.

Each non-baseline model uses an L2 (ridge) penalty. Its strength is selected
only inside the outer training trials, by maximizing spike-normalized
log-likelihood over up to ten complete-trial folds. The source paper selects
its ridge strength by marginal likelihood with a Laplace approximation; the
inner-fold variant here is a transparent replacement compatible with the
current implementation and preserves an untouched outer held-out set.

## What the metrics mean

The full Poisson log-likelihood is evaluated on held-out bins. Bits per spike compares that likelihood against a constant-rate predictor fitted only on training bins:

```text
(LL_model - LL_constant) / (number of held-out spikes × ln 2)
```

Positive bits/spike means better held-out predictions than the training-derived mean rate. It is not an effect size for connectivity.

For a candidate pair, compare event-plus-self-history against
self-history-plus-coupling as alternative explanations. The latter may inherit
stimulus information through the source neuron's response, so matching the
event model would not establish a direct connection. The full nested model
(events + self-history + coupling) tests incremental predictive information.

## Scientific boundary

A model improvement after adding neuron B's history is evidence that B helps predict A under this feature set. It is not evidence that B synaptically inhibits or excites A. Unmodeled common drive, behavioral state, stimulus timing, recording artifacts and population activity remain potential explanations.

## Planned validation

1. Select regularization and bin/history hyperparameters using inner trial-level validation only.
2. Report repeated outer trial-level splits with uncertainty intervals.
3. Compare every candidate pair against label-shuffled or trial-shifted controls.
4. Separate within-condition and across-condition analyses.
5. Add data provenance and the original experimental protocol before publication.
