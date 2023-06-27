# %% [markdown]
# # Requirements

# %% [markdown]
# ## numpy
# The main part of matrix environment.

# %%
import numpy as np

# %% [markdown]
# ## matplotlib
# Illustration libraries.

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

# %% [markdown]
# ## Time
# Facilitate struggling with time.

# %%
from datetime import datetime
from time import time, sleep

# %% [markdown]
# ## Interruptions
# The threading module in Python provides a high-level interface for creating
# and managing threads. It includes functions for starting, pausing,
# and stopping threads, as well as tools for synchronization and communication between threads.
# 
# Here are six important functions of the Python threading module:
# 
# * `Thread`: The `Thread` class is used to create a new thread.
# It takes a target function as an argument, which is the function
# that will be executed in the new thread.
# 
# * `Lock`: The `Lock` class provides a way to synchronize access to a shared resource
# between multiple threads. It is used to prevent multiple threads from accessing
# the same resource at the same time, which can lead to race conditions.
# 
# * `Event`: The `Event` class provides a way for threads to communicate with each other.
# It allows one thread to signal to another thread that a certain condition has been met,
# which can be useful for coordinating the behavior of multiple threads.
# 
# * `Semaphore`: The `Semaphore` class is similar to a lock, but it allows a limited
# number of threads to access a shared resource at the same time. This can be useful
# for controlling access to a resource that has a limited capacity.
# 
# * `Timer`: The `Timer` class is used to schedule a function to run
# after a certain amount of time has elapsed. This can be useful for creating
# timeouts or delaying the execution of a function.
# 
# * `ThreadLocal`: The `ThreadLocal` class provides a way to create thread-local data.
# This allows each thread to have its own copy of a variable, which can be useful
# for storing thread-specific state.

# %%
import threading

# %% [markdown]
# ## Logging
# Logging is a built-in module in Python that allows developers to record
# messages from their programs. It provides a flexible and configurable way
# to track the behavior of a program and diagnose issues.
# 
# Here are four useful functions of the Python logging module:
# 
# * `basicConfig()`: The `basicConfig()` function is used to configure the logging system.
# It allows you to specify the format of log messages, the logging level that
# should be used, and the output destination (such as a file or console).
# 
# * `debug()`, `info()`, `warning()`, `error()`, `critical()`: These are the built-in
# logging functions that can be used to log messages at different levels of severity.
# They allow you to provide additional information about the state of your program,
# and can be used to track down bugs or performance issues.
# 
# * `Logger`: The `Logger` class is the main interface for interacting with the logging system.
# You can create instances of this class to represent different parts of your program,
# and use its methods to log messages and control the behavior of the logging system.
# 
# * `Formatter`: The `Formatter` class is used to define the format of log messages.
# You can create instances of this class to customize the appearance of log messages,
# including the timestamp, log level, and message text.

# %%
import logging

# %% [markdown]
# ## tqdm
# `tqdm` is a Python package that allows you to easily add progress bars to your loops and iterators. It provides a simple way to visualize the progress of long-running tasks and estimate the time remaining until completion.
# 
# Functions:
# * `desc`: A string that describes the progress being made. This string is displayed next to the progress bar and can be used to provide additional information about the task being performed.
# * `total`: The total number of iterations in the loop. If this value is provided, the progress bar will display the percentage of iterations that have been completed.
# * `leave`: If set to True, the progress bar will remain on the screen after the loop has finished. This can be useful if you want to display the final state of the progress bar.
# * `bar_format`: A string that specifies the format of the progress bar. This string can include placeholders for the progress bar elements, such as the percentage completed, the elapsed time, and the estimated time remaining.
# * `disable`: If set to True, the progress bar will not be displayed. This can be useful if you want to disable the progress bar in certain situations, such as when running the code in a headless environment.
# * `dynamic_ncols`: If set to True, the width of the progress bar will be automatically adjusted to fit the screen. This can be useful if you are running the code on a terminal with a variable width.

# %%
from tqdm import tqdm

# %% [markdown]
# ## Operating System
# ### os
# The `import os` module in Python provides a way to interact with the underlying operating system.
# It allows the *creation*, *deletion*, and *renaming* of files and directories,
# as well as access to environment variables, system information, and process management.
# Some practical uses of `import os` include **automating file management tasks**,
# **executing system commands**, and **checking file permissions**.
# 
# Here are three practical functions in the os library:
# * `os.path.join()`: This function joins one or more path components intelligently
# using the correct separator for the current operating system. It's a useful way
# to create file paths that work on any platform.
# 
# * `os.listdir()`: This function returns a list of all files and directories in a given directory.
# It's useful for exploring the contents of a directory or iterating over all files in a directory.
# 
# * `os.system()`: This function allows you to execute shell commands from within Python.
# It's a powerful way to automate system tasks or run external programs.
# However, it's important to use this function with caution,
# as it can be a security risk if misused.

# %%
import os

# %% [markdown]
# ### shutil
# The `shutil` module provides a number of high-level operations on files and
# collections of files, such as *copying* and *removing* files
# * E.g., To remove a folder use: `shutil.rmtree('logs')`

# %%
# import shutil

# %% [markdown]
# ### copy
# The `copy` module is also part of the Python standard library,
# but it serves a different purpose than the `shutil` and `os` modules.
# While `shutil` and `os` provide functions for interacting with
# the operating system and performing file operations, the `copy` module
# provides functions for creating copies of objects.
# 
# The `copy` module provides two functions: `copy.copy()` and `copy.deepcopy()`.
# * The `copy.copy()` function creates a shallow copy of an object, meaning that
# it creates a new object with the same contents as the original object,
# but any changes made to the new object will not affect the original object.
# * The `copy.deepcopy()` function creates a deep copy of an object, meaning that it
# recursively copies all objects and data structures referenced by the original object.
# 
# In summary, while the `shutil`, `os`, and `copy` modules are all part of the
# Python standard library, they serve different purposes.

# %%
import copy as cop

# %% [markdown]
# ## control
# The `control` package is a third-party Python library for control systems engineering.
# 
# Installation:
# * `pip install slycot`   # optional
# * `pip install control`
# 
# Some functions:
# * `tf()`: This function creates a transfer function object from the numerator and denominator coefficients of a system. Transfer functions are a popular way of representing linear time-invariant systems.
# 
# * `step_response()`: This function calculates the step response of a system represented as a transfer function or state-space model. The step response is the system's output when the input is a step function.
# 
# * `feedback()`: This function calculates the closed-loop transfer function of a system with feedback. The closed-loop transfer function is the ratio of the output to the input of a system with feedback.
# 
# * `bode_plot()`: This function generates a Bode plot of the frequency response of a system. A Bode plot is a graph that shows the magnitude and phase of the system's frequency response as a function of frequency.

# %%
from control.matlab import *
from control import *

# %% [markdown]
# ## Accelerators
# ### numba
# `Numba` is a just-in-time (JIT) compiler for Python that can significantly speed up
# numerical computations. It generates optimized machine code for functions using
# a subset of *Python* and *NumPy* syntax. `Numba` can be used to **accelerate** code
# that relies heavily on *loops* and *mathematical operations*, making it a useful tool
# for scientific computing and data analysis.
# 
# Most striking functions:
# * `jit()`: Compiles a Python function with Numba's JIT compiler for CPU or GPU acceleration.
# * `vectorize()`: Creates a universal function that can operate on NumPy arrays, optimized by Numba's JIT compiler.
# * `guvectorize()`: An extension of vectorize() for high-performance ufuncs that operate on multiple input and output arrays, also optimized by Numba's JIT compiler.

# %%
# from numba import *
# from numba.experimental import jitclass

# %% [markdown]
# ## Image Processing
# ### OpenCV
# `cv2` (OpenCV) is a popular computer vision library for Python.
# It provides various *image* and *video processing* functions, as well as tools for
# *object detection*, *face recognition*, and other computer vision tasks.
# 
# To install: `pip install opencv-python`
# 
# If there is the `qt.qpa.plugin` error, uninstall and reinstall by:
# * `pip uninstall opencv-python`
# * `pip install opencv-python-headless`
# 
# Important functions:
# * `cv2.imread()`: This function is used to read an image file into a NumPy array.
# It supports various image formats, including JPEG, PNG, BMP, and TIFF.
# 
# * `cv2.imshow()`: This function creates a window to display an image or video stream.
# It can be used to visualize the results of image processing operations, as well as to
# interactively debug computer vision applications.
# 
# * `cv2.VideoCapture()`: This function is used to capture video frames from
# a camera or video file. It provides a simple API for accessing video streams,
# and can be used in conjunction with other OpenCV functions to perform real-time
# computer vision tasks, such as object detection and tracking.

# %%
# from cv2 import imread, cvtColor, COLOR_BGR2GRAY

# %% [markdown]
# ## Neuroscience
# ### brian
# `brian` is a Python-based simulator for spiking neural networks.
# It provides a simple and flexible way to model and simulate the behavior of
# large-scale neural networks, and includes a variety of tools for
# analyzing network activity and visualizing simulation results.
# 
# Installation: `pip install brian2`
# 
# A few functions:
# * `NeuronGroup()`: Defines a group of neurons with differential equations to govern their behavior.
# * `Synapses()`: Defines synapses between neuron groups with customizable connection types and strengths.
# * `SpikeMonitor()`: Records and analyzes spiking activity in a network to detect patterns and measure network properties.

# %%
# from brian2 import *

# %% [markdown]
# ## Profilers
# ### cProfile
# `cProfile` is a built-in Python module for profiling the **performance** of Python code.
# It provides detailed information about the *time* and *number of calls* for each function
# in a program, making it a useful tool for identifying performance bottlenecks and optimizing code.
# 
# Should you necessitate an evaluation of your project’s timing and an identification of
# the most time-intensive segments of your program, you may uncomment the contents of
# this section and scrutinize your coding performance. It is pertinent to note that
# the preceding code block should be commented.
# 
# Usage instruction:
# * `import cProfile`
# * `simulationorder = 'simulation(params, signals, models)'`
# * `profilename = 'data.prof'`
# * `cProfile.run(simulationorder, profilename)`
# 
# Illustration in jupyter:
# * `%loadext snakeviz`
# * `%snakeviz profilename`
# 
# Illustration in Command Prompt:
# * `python -m cProfile -o data.prof dyrun.py`
# * `snakeviz data.prof`

# %%
# import cProfile

# %% [markdown]
# ## pandas
# `pandas` is a popular open-source Python library for data manipulation and analysis.
# It provides powerful tools for working with structured data, including dataframes for handling tabular data
# and advanced functions for *data cleaning*, *reshaping*, and *aggregation*.
# 
# Most important functions:
# * `read_csv()`: Reads data from a CSV file into a Pandas dataframe.
# * `groupby()`: Groups data in a dataframe by one or more columns and applies
# aggregate functions to each group.
# * `merge()`: Joins two or more dataframes based on a common set of columns.
# * `pivot_table()`: Creates a spreadsheet-style pivot table from a Pandas dataframe,
# summarizing data across multiple dimensions and calculating aggregate statistics.

# %%
# import pandas as pd

# %% [markdown]
# ## pyqt5
# Use to generate plots in an independent window outside this page
# 
# Installation: `pip install pyqt5`
# 
# * `%matplotlib qt` to display Matplotlib plots in a separate window,
# * `%matplotlib inline` to display plots inline in the notebook,
# * `%matplotlib notebook` to display interactive plots in the notebook itself.

# %%
# %matplotlib qt
