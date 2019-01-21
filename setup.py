from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
nvcc_args = [ 
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61', 
]
setup(
    name='ModulatedDeformableConvolution',
    version="1.0",
    author = "sansi",
    license = "GPL",
    ext_modules=[
        CUDAExtension('ModulatedDeformableConvolution', [
            'nn/dcnv2_cuda_kernel.cu',
            'nn/dcnv2_cuda.cpp'
			],
		library_dirs = ['/home/jiangtao.jiangtao/envoriment/local/anaconda3/envs/python3.6_pytorch0.4.1/lib/'],
		libraries = ['python3.6m'],
		extra_compile_args={	
								'cxx':[
											
										],
								'nvcc':[
											'-gencode', 'arch=compute_60,code=sm_60',
											'-gencode', 'arch=compute_61,code=sm_61',
										],
							}
		),
    ],
    include_dirs = [	
		'/home/jiangtao.jiangtao/code/libtorch/include/',	
	],
    cmdclass={
        'build_ext': BuildExtension
    })
