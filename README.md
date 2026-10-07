# Olfactory circuit GLM

An in-progress, reproducible analysis of trial-aligned spike trains from an olfactory/context-reward experiment. The project asks a deliberately limited question: how much do event timing, a neuron's own recent spiking, and another recorded neuron's recent spiking improve held-out predictions of spike counts?

It does **not** infer synaptic connectivity or causality. A coupling term captures conditional predictive association after the modeled event and self-history terms; shared unobserved inputs can still explain an apparent effect.

## Current scope

The original class notebook mixed descriptive rasters/PSTHs, Fourier exploration, correlation analysis, peak analysis and a point-process GLM. This repository promotes the GLM into a reproducible pipeline and keeps the exploratory work separate.

The first analysis reproduces the original illustrative pairing:

- target neuron: `45`
- candidate coupling neuron: `52`
- initial condition: trial type `1` (`OR + CR` in the original notebook)

Spikes and events are aligned within each trial to first odor inhalation by
default. No-odor controls cannot use that alignment; they must be modeled with
an observed alternative such as context entry.

The comparison is nested and uses a split by **complete trial**, so bins from the same trial never occur on both sides of the evaluation:

1. Constant-rate baseline.
2. Stimulus event kernels: OR/OU are one-bin impulses at first odor inhalation;
   CR/CU are impulses at context entry; CER/CEU are impulses at context exit.
   Each has a 1.25-s causal lag basis, allowing its effect to evolve over time.
   Use `--stimulus-encoding generic` to run the three-kernel ablation that
   shares odor, context-entry and context-exit responses across identities.
3. Stimuli plus the target neuron's own causal spike history.
4. Stimuli, own history and causal history of the candidate coupling neuron.

Primary metrics are held-out Poisson log-likelihood and information gain in bits per spike relative to the constant-rate baseline.

Every non-baseline model has an L2 (ridge) penalty. Rather than a fixed
coefficient, its strength is selected inside the outer training trials using
up to ten trial-level validation folds. This follows the paper's use of a ridge
prior while keeping the implementation's hyperparameter choice separate from
the final held-out evaluation.

The current default uses the preceding 30 bins (1.5 s) for spike-history and
coupling features. Stimulus durations follow the behavioral protocol rather
than an arbitrary post-event filter horizon.

## Data

The input is a headerless, eight-column CSV. It is deliberately excluded from Git. The loader assigns this schema:

```text
neuron_id, spike_time, trial_number, ctx_entry_time,
ctx_exit_time, first_odor_inhale, trial_type, trial_start_time
```

The current local default expects `../neuroscience/MatrizFinal_8col.csv`, relative to this repository. Override it explicitly when needed:

```bash
python scripts/run_evaluation.py --data /absolute/path/MatrizFinal_8col.csv
```

## Provenance and data access

This analysis is based on the mouse piriform-cortex recordings described in:

> Federman, N., Romano, S. A., Amigo-Duran, M. et al. *Acquisition of non-olfactory encoding improves odour discrimination in olfactory cortex*. Nature Communications 15, 5572 (2024). [Paper and data-availability statement](https://www.nature.com/articles/s41467-024-49897-4), doi:[10.1038/s41467-024-49897-4](https://doi.org/10.1038/s41467-024-49897-4).

The original authors' Matlab GLM code is available at [`marinburginlab/olfactionGLM`](https://github.com/marinburginlab/olfactionGLM). This repository contains the analysis code and documentation only. The original recordings are intentionally **not** included or redistributed here. Anyone who needs the data to reproduce the analysis should request access from the corresponding authors through the paper's data-availability route. The paper also links the authors' source-data record.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_evaluation.py \
  --target-neuron 45 \
  --coupling-neurons 52 \
  --trial-types 1
```

This writes one row per model and outer trial-level split to `results/held_out_metrics.csv`, plus an aggregate summary. The default uses 20 splits. It is still not a final scientific result: regularization, bin width, history horizon and condition aggregation must be selected through a documented validation protocol rather than by looking at one dataset configuration.

## Population reference

To place a particular neuron in the recorded population, run the no-coupling
reference across all 64 neurons and all trial types:

```bash
python scripts/run_population_reference.py
```

This creates the per-neuron, per-condition distribution of held-out
stimulus-plus-self-history performance. Types 1–4 are odor-aligned; no-odor
controls (5–6) are aligned to context entry. These two alignments should be
compared only with that distinction in view.

## Diagnostic rasters and PSTHs

Model scores do not by themselves identify event-locked responses. Render
odor-aligned diagnostics for selected neurons with:

```bash
python scripts/make_psth.py --neurons 13 33
```

## Repository map

```text
src/data.py        Headerless data loading and trial-metadata validation
src/features.py    Causal event, self-history and coupling design matrices
src/model.py       Trial-level split and nested Poisson GLM evaluation
src/metrics.py     Held-out log-likelihood and bits/spike
scripts/           Reproducible entry points
notebooks/         Exploratory work retained outside the main pipeline
docs/              Methodological decisions and limitations
```

## Status

The pipeline skeleton and held-out metrics are implemented. Remaining work includes data provenance, unit tests, regularization selection, repeated trial-level resampling, uncertainty estimates, condition-wide analyses and visual summaries.
