# Full MNIST digit-1 v/u initial-state sweep — completed

**Full 10 × 10 grid completed:** 100 conditions, each trained for 50 paths × 27 force steps × 80 internal neuron substeps. Ten parallel GitHub Actions jobs all passed; a separate repository-equilibrium control was executed. Reproducible workflow: https://github.com/azizqc/Munan_maskgit/actions/runs/39317059293

## Scientific scope and dataset

- Digit-1 images only, from the repository's standard MNIST archive (`mnist.npz`). Training: 512 independent train images; held-out testing: 256 official MNIST test images. Masked-token context sampling: 640 training and 192 held-out contexts.
- The exact R6B.3 Izhikevich `fast_izh_step` (E/M populations), P3 static feature operator, and MSDL 27-step readout learning are used. The other P3 weights are **frozen**. No backpropagation is used.
- The missing frozen R6B.1 SNN-VQ checkpoint is **not** simulated or fabricated: the repository's conventional MNIST KMeans patch codebook is an explicit substitute, embedded using the R6B.3 deterministic DCT recipe. **This is a completed readout-only benchmark, not an end-to-end SNN-Transform training result.**
- Every condition uses exactly the same frozen P3 weights, codebook, train/test images, masked contexts, presentation sequence, LR 0.6, and random seed 7311. Only initial v and u of the E/M readout neurons differ.
- Hold-out error is masked patch-token categorical cross-entropy (nats). 'Best overall' includes step 0 (untrained); 'best after training' excludes step 0. Initial error = ln(64) = 4.158883 nats.

## Main finding

**None of the 100 grid points learned successfully.** Every tested setting's minimum error was the untrained 4.158883 nats, observed at training path 0; no setting reached a 5% held-out CE reduction by path 50. All 100 executed without NaN, but training actually **increased** held-out cross-entropy, in many cases catastrophically. Therefore, no winning v/u combination or training time to convergence exists in the requested 0–1000 grid at this budget.

## Fixed-u summary

| Fixed initial u | v values tested | Best CE including initial | Best CE after training (lowest v) | Final CE range across v | Mean E spike fraction | Conditions converged |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 4.158883 | 17,132.7 | 17,132.7–17,138 | 0.00% | 0 |
| 111.11 | 10 | 4.158883 | 11,530,922.88 | 217,150,806.69–226,383,190.68 | 99.99% | 0 |
| 222.22 | 10 | 4.158883 | 4,874,581.77 | 214,283,606.66–223,832,748.02 | 100.00% | 0 |
| 333.33 | 10 | 4.158883 | 11,080,362.78 | 219,561,985.15–224,763,904.95 | 100.00% | 0 |
| 444.44 | 10 | 4.158883 | 12,624,213.4 | 217,079,809.17–224,302,422.68 | 100.00% | 0 |
| 555.56 | 10 | 4.158883 | 11,626,496.07 | 217,153,537.25–240,902,144.19 | 100.00% | 0 |
| 666.67 | 10 | 4.158883 | 10,750,976.13 | 218,273,110.73–224,305,153.31 | 100.00% | 0 |
| 777.78 | 10 | 4.158883 | 10,140,330.84 | 216,408,065.3–223,171,926.72 | 100.00% | 0 |
| 888.89 | 10 | 4.158883 | 12,125,866.77 | 217,036,118.7–223,895,553.34 | 100.00% | 0 |
| 1,000 | 10 | 4.158883 | 11,024,725.52 | 215,834,625.22–225,236,310.45 | 100.00% | 0 |

## Independent physical-equilibrium control

- Reference (v0, u0) = (-62.575221470, -0.013160365).
- Best and final CE = 1.136197 nats; held-out masked-token accuracy = 82.813%.
- First 5% improvement at path 5, after 135 force steps and 10800 internal substeps per neuron.
- Best result recorded at path 50. Initial CE was 4.158883 nats. This reference is **outside** the requested 0–1000 grid and was not used to choose any grid setting.

## Interpretation

At u=0, the neurons nearly never spike after their initial reset. At higher u, both E and M populations essentially spike on every internal substep. Since the MSDL readout requires informative state-driven E/M populations, the extremes induce pathological weight updates. A run may finish without numerical NaNs while still failing to learn; monitor validation CE rather than only termination status.

**Training-speed definition:** a path contains 27 force presentations; each presentation contains 80 membrane substeps. The reported paths to a 5% reduction is a reproducible convergence proxy. GitHub-hosted wall time depends on runner load, and the timed core excludes dataset loading and fixed-feature preprocessing, so raw seconds should not be interpreted as complete training wall time.

## Files and reproduction

- `conditions.csv`: all 100 grid conditions plus equilibrium reference; full-precision initial/best/trained/final CE, accuracy, spike counts, path counts, learning flags and compute timings.
- The workflow artifacts (`digit1-vu-u0` ... `digit1-vu-u9`) contain per-fixed-u `learning_curves.csv`, `summary.json`, and loss/convergence heatmaps.
- Re-run: `python experiments/sweep_digit1_vu.py --repo . --out digit1_vu_results --paths 50` from repository root with Python requirements (`numba`, `torch`, `numpy`, `scikit-learn`, `matplotlib`).

**What this does not establish:** best achievable loss for all training durations, full P3 internal-learning behavior, spike-generation image quality, or what happens with the missing original SNN-VQ encoder. The result is conditional on the defined 50-path frozen-P3 readout benchmark.

