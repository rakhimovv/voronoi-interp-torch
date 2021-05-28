# voronoi-interp-torch

> If you only know a few random pixels of an image, you can fill in the rest using nearest neighbors. This can result in cool animations as you gradually add more and more pixels at random.

The [original version](https://github.com/unixpickle/voronoi-interp) of this project written in Go.

This version is differentiable using PyTorch and CUDA.

# Example

Here is an example output:

![demo](example/demo.gif)

Given that the function is differentiable you can also optimize the color values of the cells instead of using the original ones:

![opt](example/opt.gif)

# Usage
Install python module: `./setup.sh`

Example:
```python
import torch
from voronoi_interp_torch import voronoi_interpolate

coords = torch.randint(0, 100, size=(100, 2)).cuda()
values = torch.rand(100, 3).cuda()
result = voronoi_interpolate(coords, values, H=100, W=100)
```

[Demo](./example/demo.ipynb) where backprop is tested. 
