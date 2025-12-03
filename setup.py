from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="cs4787pa4",
  ext_modules=[
    cpp_extension.CUDAExtension(
      "cs4787pa4",
      ["wrapper.cpp","kernels.cu"]
    )],
  cmdclass={'build_ext': cpp_extension.BuildExtension}
)