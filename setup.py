from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        'voronoi_interp_torch.cuda.voronoi_interp_cuda', [
            'voronoi_interp_torch/cuda/voronoi_interp_cuda.cpp',
            'voronoi_interp_torch/cuda/voronoi_interp_cuda_kernel.cu',
        ])
]

setup(
    version='0.1',
    author='Ruslan Rakhimov',
    author_email='ruslan.rakhimov@skoltech.ru',
    install_requires=["torch>=1.3"],
    packages=['voronoi_interp_torch', 'voronoi_interp_torch.cuda'],
    name='voronoi_interp_torch',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
