# Frozen B0-HQ: full conventional MaskGIT for MNIST {0,1,2}

This package contains the complete frozen B0-HQ model used in the project, including the tokenizer state that was omitted from the earlier single-file export.

## Exact model
- Images: MNIST 28x28, classes 0/1/2.
- Tokenizer: non-overlapping 4x4 patches; MiniBatchKMeans codebook K=64.
- Token grid: 7x7 = 49 discrete tokens.
- Tokenizer checkpoint/state: `codebook.npy`, shape 64x16. This is the learned tokenizer state; there is no neural tokenizer `.pt` checkpoint.
- MaskGIT: bidirectional Transformer, d=128, 4 layers, 4 heads, FF=512.
- Frozen MaskGIT checkpoint: `best.pt`, epoch 18, validation CE 1.1593495783.
- Trainable MaskGIT parameters: 824,896.
- Frozen decoding setting used for the final B0-HQ evaluation: 10 iterative steps, confidence/Gumbel temperature 0.25.

## Files
- `maskgit_b0_mnist012.py`: original complete training/tokenization/generation implementation.
- `best.pt`: exact frozen B0-HQ Transformer weights.
- `codebook.npy`: exact learned K=64 tokenizer codebook required to encode/decode images.
- `generate_from_frozen.py`: convenience inference script that loads both frozen artifacts.
- `metrics.json`: frozen evaluation record.
- `final_600.npz`, `final_600.png`: saved final generation outputs.
- `capacity_comparison.json`: capacity-comparison record from model selection.
- `requirements.txt`: Python dependencies.
- `MANIFEST_SHA256.json`: SHA-256 checksums for integrity.

## Generate immediately (no MNIST file required)
```bash
python generate_from_frozen.py --n-per-class 100 --steps 10 --temp 0.25 --out demo
```
This starts from all-MASK tokens and generates class-conditioned 0/1/2 images using `best.pt` + `codebook.npy`.

## Retrain the full pipeline from raw MNIST
The original training script expects the standard Keras-style `mnist.npz` containing `x_train`, `y_train`, `x_test`, `y_test`:
```bash
python maskgit_b0_mnist012.py --data /path/to/mnist.npz --out retrain --d 128 --layers 4 --heads 4 --ff 512 --drop 0.05
```
If `retrain/codebook.npy` does not exist, the script first fits a new MiniBatchKMeans tokenizer and saves it, then trains MaskGIT. A newly fitted codebook and newly trained checkpoint are a new run, not byte-identical to the frozen artifacts in this package.

## Important scope
This B0 is conventional MaskGIT only. It contains no ionic forward process, SNN, persistent dynamics, or slow-confidence mechanism. It is class-conditioned on labels 0/1/2.
