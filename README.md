# Neural Turing Machine (NTM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahi97/NTM/blob/master/NTM.ipynb)

A PyTorch implementation of the **Neural Turing Machine** (NTM) as described in the paper:

> Graves, A., Wayne, G., & Danihelka, I. (2014). [Neural Turing Machines](https://arxiv.org/abs/1410.5401). *arXiv:1410.5401*.

---

## Overview

Neural Turing Machines couple a neural network controller with an external memory bank, allowing the network to read from and write to memory in a differentiable manner. This enables the model to learn algorithmic tasks such as copying, sorting, and associative recall that are difficult for standard recurrent architectures.

### Key Components

| Component | Description |
|---|---|
| **Memory** | An `N × M` differentiable memory matrix with learnable bias initialization |
| **Controller** | Either an LSTM or a Feed-Forward network that processes inputs and emits read/write instructions |
| **Read Head** | Attends to memory locations via content-based and location-based addressing, returning a weighted sum |
| **Write Head** | Erases and adds content to memory at addressed locations using erase and add vectors |
| **DataPath** | Wires the controller, memory, and heads together into a unified sequence model |
| **NTM** | The top-level module combining all components for end-to-end training |

---

## Architecture

```
Input
  │
  ▼
Controller (LSTM or Feed-Forward)
  │       ▲
  │       │ read vectors
  ▼       │
Heads ──► Memory (N × M)
  │
  ▼
Output
```

### Addressing Mechanism

Each head produces a normalized attention weight over memory rows through a four-step pipeline:

1. **Content-based lookup** – cosine similarity between the key vector and each memory row, sharpened by key strength β.
2. **Interpolation** – blends the content weight with the previous time-step weight using a scalar gate *g*.
3. **Convolutional shift** – a learned 3-element shift kernel enables relative location-based addressing.
4. **Sharpening** – raises each weight to the power γ (then re-normalizes) to focus attention.

---

## Tasks

### Copy Task
The network is trained to reproduce a random binary sequence after receiving a delimiter token. This tests the model's ability to store and recall arbitrary sequences.

### Repeated Copy Task
An extension of the copy task where the network must output the sequence a given number of times. This requires the model to learn both storage and iterative recall.

---

## Project Structure

```
NTM/
├── NTM/
│   ├── __init__.py
│   └── Memory.py          # NumPy reference implementation of the memory module
├── NTM.ipynb              # Full PyTorch implementation and training notebook (runnable on Colab)
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy attrs
```

### Running in Google Colab

Click the **Open in Colab** badge at the top of this page. All dependencies are installed automatically inside the notebook.

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/mahi97/NTM.git
   cd NTM
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy attrs
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook NTM.ipynb
   ```

---

## Training

Training hyperparameters (configurable at the top of the training section in the notebook):

| Parameter | Default | Description |
|---|---|---|
| `NUM_BATCH` | 20 000 | Total number of training batches |
| `BATCH_SIZE` | 1 | Sequences per batch |
| `INTERVAL` | 500 | Reporting and checkpoint interval |
| `SEED` | 100 | Random seed for reproducibility |
| `TASK` | `recopy` | Task to train (`copy` or `recopy`) |

Checkpoints and training metrics are saved to `./res_copy/` as `.model` and `.json` files respectively.

---

## Results

After training, the notebook provides utilities to:

- **Plot training convergence** – cost per sequence (bits) vs. thousands of batches.
- **Visualize per-sequence-length convergence** – separate curves for each observed sequence length.
- **Animate training progress** – a GIF showing how the model's output evolves over the course of training.

---

## References

- Graves, A., Wayne, G., & Danihelka, I. (2014). [Neural Turing Machines](https://arxiv.org/abs/1410.5401). *arXiv:1410.5401*.
- [pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm) – another popular PyTorch NTM reference implementation.

---

## License

This project is open-source. See the repository for details.
