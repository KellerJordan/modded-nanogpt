import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory containing this setup.py
cuda_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='newtonschulz5_cuda',
    ext_modules=[
        CUDAExtension(
            name='newtonschulz5_cuda',
            sources=[
                os.path.join(cuda_dir, 'newtonschulz5_ops.cpp'),
                os.path.join(cuda_dir, 'newtonschulz5_kernel.cu'),
            ],
            include_dirs=[cuda_dir],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_70,code=sm_70',  # V100
                    '-gencode', 'arch=compute_75,code=sm_75',  # T4
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode', 'arch=compute_89,code=sm_89',  # RTX 4090
                    '-gencode', 'arch=compute_90,code=sm_90',  # H100
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)