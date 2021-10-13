from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
ext = Extension(
    name="scatter",
    sources=["scatter.pyx"],
    extra_compile_args=["-ffast-math", '-std=c++11', '-O3', '-fopenmp', '-lgomp'],
    extra_link_args=['-fopenmp'],
    language="c++",
)
setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])