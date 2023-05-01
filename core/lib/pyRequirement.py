# import requirments ---------------------------

# %% *NECESSARY*
# The main part of matrix environment
import numpy as np

# %% *NECESSARY*
# Illustration library
import matplotlib as mpl
import matplotlib.pyplot as plt

# %% *NECESSARY*
# Make your code faster
from numba import *
from numba.experimental import jitclass

# %% *NECESSARY*
# Is used to work with time
from datetime import datetime
from time import time

# %% *NECESSARY*
# The os module provides a way of using operating system dependent
# functionality like reading or writing to the file system.
import os

# %% *NECESSARY*
# The `copy` module is also part of the Python standard library,
# but it serves a different purpose than the `shutil and `os` modules.
# While `shutil` and `os` provide functions for interacting with
# the operating system and performing file operations, the `copy` module
# provides functions for creating copies of objects.

# The `copy` module provides two functions: `copy.copy()` and `copy.deepcopy()`.
# The `copy.copy()` function creates a shallow copy of an object, meaning that
# it creates a new object with the same contents as the original object,
# but any changes made to the new object will not affect the original object.
# The `copy.deepcopy()` function creates a deep copy of an object, meaning that it
# recursively copies all objects and data structures referenced by the original object.

# In summary, while the `shutil`, `os`, and `copy` modules are all part of the
# Python standard library, they serve different purposes. The `shutil` and `os` modules
# provide functions for interacting with the operating system and performing file operations,
# while the `copy` module provides functions for creating copies of objects.
import copy as cop

# %% *NECESSARY*
# The control package is a third-party Python library for control systems engineering
from control.matlab import *
from control import *

# %%
# The shutil module provides a number of high-level operations on files and
# collections of files, such as copying and removing files
# To remove a folder use: shutil.rmtree('logs')
# import shutil

# %%
# Module cv2 is a module of the OpenCV library, which is
# a third-party package for computer vision and image processing
from cv2 import imread, cvtColor, COLOR_BGR2GRAY

# %%
# Brian is used to simulate neuronal networks
# from brian2 import *

# %%
# To profile code, the below library is vital
# import cProfile

# %%
# Panda is applicable to work with table of large data more precisely
# import pandas as pd