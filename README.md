# NumPy-style histograms in PyTorch

The `torchist` package implements NumPy's [`histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) and [`histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) functions in PyTorch with support for non-uniform binning. The package also features implementations of [`ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html), [`unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html) and some useful functionals (e.g. KL divergence).

## Installation

```bash
pip install git+https://github.com/francois-rozet/torchist
```

## Getting Started

```python
import torch
import torchist

x = torch.rand(100, 3).cuda()

hist = torchist.histogramdd(x, bins=10, low=0., upp=1.)

print(hist.shape)  # (10, 10, 10)
```

## Benchmark

The implementations of `torchist` are up to 3 times faster than those of `numpy` on CPU and benefit greately from CUDA capabilities.

```cmd
$ python torchist/__init__.py
CPU
---
np.histogram : 1.2674 s
np.histogramdd : 19.3845 s
np.histogram (non-uniform) : 5.4961 s
np.histogramdd (non-uniform) : 16.8242 s
torchist.histogram : 0.8805 s
torchist.histogramdd : 5.8721 s
torchist.histogram (non-uniform) : 2.5283 s
torchist.histogramdd (non-uniform) : 13.3900 s

CUDA
----
torchist.histogram : 0.1122 s
torchist.histogramdd : 0.2757 s
torchist.histogram (non-uniform) : 0.1253 s
torchist.histogramdd (non-uniform) : 0.4441 s
```
