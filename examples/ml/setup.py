from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

nvcc_args = ['-O3', '--expt-extended-lambda']
if 'MAX_REG' in os.environ:
    nvcc_args.append('-maxrregcount=' + os.environ['MAX_REG'])


setup(
    name='fusion_cuda',
    ext_modules=[
        CUDAExtension('fusion_cuda', [
            'wrappers/wrapper.cpp',
            'fused/MaxpoolSummaryNorm.cu',
            'fused/MaxPoolUpSample.cu',
            'fused/SummaryUpsample.cu',
            'fused/SummaryMaxpool.cu',
            'fused/SummaryNorm.cu',
            'fused/SummaryOps.cu',
            'fused/Im2ColMaxpoolNorm.cu',
            'fused/UpsampleNormalization.cu',
            'fused/Im2ColMaxPool.cu',
            'fused/Im2ColUpSample.cu',
            'fused/MaxPoolBatchNorm.cu',
            'fused/Im2ColNormalization.cu',
        ],
        extra_compile_args={'cxx': [],
                            'nvcc': nvcc_args
                                     },
        libraries = ["torch"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
