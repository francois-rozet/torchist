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

The implementations of `torchist` are on par or faster than those of `numpy` on CPU and benefit greately from CUDA capabilities.

```console
$ python torchist/__init__.py
CPU
---
np.histogram : 1.2559 s
np.histogramdd : 20.7816 s
np.histogram (non-uniform) : 5.4878 s
np.histogramdd (non-uniform) : 17.3757 s
torchist.histogram : 1.3975 s
torchist.histogramdd : 9.6160 s
torchist.histogram (non-uniform) : 5.0883 s
torchist.histogramdd (non-uniform) : 17.2743 s

CUDA
----
torchist.histogram : 0.1363 s
torchist.histogramdd : 0.3754 s
torchist.histogram (non-uniform) : 0.1355 s
torchist.histogramdd (non-uniform) : 0.5137 s
```
