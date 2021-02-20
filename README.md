# NumPy-style histograms in PyTorch

The `torchist` package implements NumPy's [`histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) and [`histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html) functions in PyTorch. Currently, the histogram implementations do **not** support non-uniform binning. The package also features implementations of [`ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html) and [`unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html).

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
