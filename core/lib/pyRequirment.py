# import requirments ---------------------------
from brian2 import *
import math as math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime           # Date and time options
import cProfile                         # For profiling
from time import time
import os
# import shutil                         # To remove a folder use: shutil.rmtree('logs')
import random
from control.matlab import *            # Similar to MATLAB functions

%matplotlib qt