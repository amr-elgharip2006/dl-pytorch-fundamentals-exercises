# PyTorch Fundamentals — Exercise Solutions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amr-elgharip2006/dl-pytorch-fundamentals-exercises/blob/main/DL_00_Exercises_Solutions.ipynb)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Device](https://img.shields.io/badge/device-CUDA%20%7C%20MPS%20%7C%20CPU-555)

**Course:** Fundamentals of Deep Learning — Section (Eng. Mohamed Amin)
**Student:** Amr Ahmed Elgharip
**Notebook:** `00_pytorch_fundamentals` (2026 edition) — end-of-notebook exercises

> حل تمارين نوت بوك PyTorch Fundamentals كاملة (10 تمارين + الـ Challenge)، والكود شغال على أي device.

---

## Files

| File | What it is |
|---|---|
| [`DL_00_Exercises_Solutions.ipynb`](DL_00_Exercises_Solutions.ipynb) | **Solutions only.** A short notebook with the 10 exercises and the challenge. Start here. |
| [`DL_00_Exercises_Amr.ipynb`](DL_00_Exercises_Amr.ipynb) | The full section notebook, run from top to bottom, with the solutions added under **"Exercise Solutions — Amr"** (just before "Further study"). |

Both notebooks are saved with their outputs, so you can read them on GitHub without running anything.

---

## Exercises

| # | Exercise | Solution in short |
|---|---|---|
| 1 | Read the `torch.Tensor` and `torch.accelerator` docs | 8 key points (dtype / shape / device, creation ops, in-place `_` ops, the device-agnostic `torch.accelerator` API). Each API name is checked against the installed version. |
| 2 | Random tensor `(7, 7)` | `torch.rand(7, 7)` |
| 3 | Matmul with a second random tensor | `(7, 7) @ (1, 7).T → (7, 1)`. Also shows the error without the transpose and explains the inner-dimension rule. |
| 4 | `manual_seed(0)`, repeat 2–3 | Two seeded runs are equal (`torch.equal → True`); an unseeded third run is not. |
| 5 | `torch.manual_seed()` and accelerator RNGs | It seeds the CPU generator **and** every accelerator generator (CUDA / MPS / XPU …). The code shows that CPU and GPU give different numbers for the same seed, and includes a full reproducibility recipe. |
| 6 | Two random `(2, 3)` tensors on `device` | Seed `1234`, created on CPU then `.to(device)`. |
| 7 | Matmul after fixing shapes | `(2, 3) @ (3, 2) → (2, 2)` |
| 8 | Max / min of the result | `0.5617` / `0.3647` |
| 9 | Index of max / min | flat `3` / `0` → `(1, 1)` / `(0, 0)` with `torch.unravel_index` |
| 10 | Seed `7`, `(1, 1, 1, 10)` → `(10)` | `.squeeze()` removes every size-1 dimension. |
| ★ | **Challenge:** shape, dtype and device at each step | A 9-step trace through exercises 6–9, plus a table explaining why each value is what it is (default `float32`, `int64` indices, 0-dim results, `.to()` / `.cpu()` / `.item()`). |

---

## Device-agnostic code

```python
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
```

The same cells run on Google Colab (`cuda:0` or `cpu`) and on an Apple Silicon Mac (`mps:0`).
The saved outputs come from a Mac, so they show `mps:0`.

## Run it

- **Colab:** click the badge above → *Runtime → Run all* (a GPU runtime is optional).
- **Locally:**

  ```bash
  pip install torch jupyter
  jupyter notebook DL_00_Exercises_Solutions.ipynb
  ```

---

## Credits

The section notebook is based on Daniel Bourke's
[`pytorch-deep-learning`](https://github.com/mrdbourke/pytorch-deep-learning) (00 — PyTorch Fundamentals),
updated for 2026 for the course section. The exercise solutions are my own work.
