# MSDL u/v readout sweep

The source model uses an Izhikevich E/M equilibrium near v=-62.5752, u=-0.01316. A reproducible exploratory experiment tested 10 initial u values and 10 initial v values in [0,1000], with u held fixed over each 10-v group and fixed training event order.

Full R6B.3 MaskGIT training needs a frozen R6B.1 tokenizer checkpoint not in this repo. Although mnist.npz is included, the offline execution environment could not mount it. Results below therefore use scikit-learn Digits 8x8, not MNIST, and test the original MSDL E/M neuron and readout update equations rather than the full architecture.

100 pairs, 300 training paths per pair, 27 events per path, validation at every 10 paths. Best observed minimum classification error: 21%, at u=777.778 and v=888.889, path 30; final error at path 300 was 59%. No condition reached 90% accuracy. Minimum validation cross-entropy: 0.623833 nats (u=444.444, v=222.222, path 140). The canonical Izhikevich equilibrium achieved a minimum 9% held-out classification error in a separate control. These numbers are not full-model MNIST results.

The ChatGPT experiment output contains runnable msdl_uv_sweep.py, full CSV results, metadata, graphs, and a ZIP. Run that script with --mnist mnist.npz in your workspace to create MNIST-based readout results.