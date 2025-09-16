# NumPy-style histograms in PyTorch

The `torchist` package implements NumPy's [`histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) and [`histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) functions in PyTorch with CUDA support. The package also features implementations of [`ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html), [`unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html) and some useful functionals like `entropy` or `kl_divergence`.

## Installation

The `torchist` package is available on [PyPI](https://pypi.org/project/torchist), which means it is installable with `pip`.

```
pip install torchist
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/francois-rozet/torchist
```

## Getting Started

```python
import torch
import torchist

x = torch.rand(100, 3).cuda()

hist = torchist.histogramdd(x, bins=10, low=0.0, upp=1.0)

print(hist.shape)  # (10, 10, 10)
```

## Benchmark

The implementations of `torchist` are on par or faster than those of `numpy` and `torch` on CPU. On GPU (CUDA), they are much faster.

```console
$ torchist-benchmark
CPU
---
np.histogram                      0.8917 s
np.histogram (edges)              0.5993 s
np.histogramdd                   16.8441 s
np.histogramdd (edges)           13.7680 s
torch.histogram                   0.3251 s
torch.histogram (edges)           0.4217 s
torch.histogramdd                 1.0528 s
torch.histogramdd (edges)         1.1955 s
torchist.histogram                0.4250 s
torchist.histogram (edges)        0.6372 s
torchist.histogramdd              1.6266 s
torchist.histogramdd (edges)      3.8619 s

CUDA
----
torchist.histogram                0.1045 s
torchist.histogram (edges)        0.0672 s
torchist.histogramdd              0.0906 s
torchist.histogramdd (edges)      0.1170 s
```
