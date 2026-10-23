# MNIST digit-one initial-state sweep (MSDL readout)

This experiment compares 10 equally spaced normalized v values at each of 10 fixed normalized u values, resulting in a 10x10 grid (100 controlled trials). It measures the minimum observed held-out masked-patch token negative log likelihood (NLL), the training path count/seconds required to reduce initial NLL by 10%, and foreground-only error.

## Scientific scope and missing dependencies

The repository's full R6B.3 P3/MaskGIT experiment requires a frozen SNN-VQ tokenizer checkpoint (external to this repository). Therefore the executable experiment here is a readout-level MSDL experiment, not an exact replay of the full SNN-Transform or its image generator. It retains the source's fast Izhikevich E/M step (including midpoint integration and spike reset), gains, matched readout, 80 microsteps, 27 force steps, zero initialized W, fixed-within-path W0 and local outer-product learning. It replaces the missing learned tokenizer with a deterministic 7x7 representation: average each 4x4 MNIST patch, quantize to 4 bins, randomly mask 25/49 positions, and predict the masked patch bin from remaining bins, mask indicators and position features.

The class label 1 is used only to select samples, not supplied as a predictor input. MNIST train and test splits remain disjoint. The script downloads the original Keras MNIST archive from the documented TensorFlow mirror and verifies SHA256.

## Initial-condition interpretation

The original model's physical E/M Izhikevich equilibrium is approximately v=-62.57522 and u=-0.01316037. Setting its physical state literally to v/u in 0..1 would take it far outside the model's baseline regime, causing initial spikes and extremely large E/M readouts. Therefore sweep values in [0,1] are explicitly normalized control values, transformed to physical Izh state by:

v_physical = V0 + (v_normalized - 0.5) * 0.0002

u_physical = U0 + (u_normalized - 0.5) * 0.0000001

Physical spans can be changed using --v-span / --u-span. No training equation is retuned.

## Reproduce

    python -m pip install numpy numba matplotlib
    python experiments/mnist_vu_sweep.py --levels 10 --paths 160 --checkpoint-every 10

Or open the repository's MNIST digit-1 v/u sweep GitHub Actions workflow, run it, then download the artifact.

Output inside results/mnist_digit1_vu/: results.csv; results.json (checkpoint histories and provenance); min_error_heatmap.png; training_speed_heatmap.png; fixed_u_curves.png.

Validation NLL is 4-class masked-patch cross entropy, not a digit classifier's error rate. "Minimum" means the smallest observed held-out metric over specified checkpoints, not a mathematical lower bound. Differences smaller than run-to-run variability should not be interpreted as robust without repeating seeds. Wall clock is machine-dependent; the number of force paths is the stable training-speed measure.

All 100 trials share the same train/validation contexts, seed-dependent sample draws, learning rate and zero starting W. A fixed u is used across all 10 v values in each row, then u changes.
