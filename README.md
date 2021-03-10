# NumPy-style histograms in PyTorch

The `torchist` package implements NumPy's [`histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) and [`histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) functions in PyTorch. Currently, the histogram implementations do **not** support non-uniform binning. The package also features implementations of [`ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html), [`unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html) and some useful functionals (e.g. KL divergence).

## Installation

```bash
pip install git+https://github.com/francois-rozet/torchist
```

## Getting Started

```python
import torch
import torchist

x = torch.rand(100, 3).cuda()

hist = torchist.histogramdd(x, bins=10, low=0., high=1.)

print(hist.shape)  # (10, 10, 10)
```

## Benchmark

The implementations of `torchist` are up to 2 times faster than those of `numpy` on CPU and up to 16 times faster on CUDA, especially for `histogramdd`.

```cmd
$ python torchist/__init__.py
CPU
---
np.histogram : 0.1204 s
torchist.histogram : 0.0901 s
np.histogramdd : 1.8788 s
torchist.histogramdd : 0.8561 s

CUDA
----
torchist.histogram : 0.0856 s
torchist.histogramdd : 0.1170 s
```
