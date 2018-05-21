import os
import Cython

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

directive_defaults = Cython.Compiler.Options.get_directive_defaults()

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

extensions = [
    Extension('core1', ['core1.pyx'],
              language='c++',
              libraries=["m"],
              define_macros=[('CYTHON_TRACE', '1')])  # for line profiling
]

setup(
    name="name",
    ext_modules=cythonize(extensions),
)
