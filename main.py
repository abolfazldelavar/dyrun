
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
#    Updated:   8 April 2023
#               2750 code lines
#
#    Description: This package has provided to simulate easily and quickly
#    Several useful tools have been written to simulate linear and nonlinear
#    dynamic systems, dependent or independent over time. To sum up, if
#    you are a student who wants to investigate and has decided to seek
#    dynamic and control systems, this is for you. Enjoy it :_)
#
#    "Successful people are those who can build solid foundations
#     with the bricks others throw at them."
#
# \\ ---------------------------------------------------------------

# Loading requirements
from core.lib.pyRequirment import *
from initialization import *
from simulation import simulation
from finalize import finalize

# The main file (function)
def main(extopt = 0):
    # Loading all necessary functions and libraries
    lib = loadLibraries()
    # Making parameters
    params = valuation(lib, extopt)
    # Defining vectors and signals + updating the previous variables
    signals = vectors(params, lib)
    [params, signals] = signals.set(params)
    # Defining system blocks + updating the previous variables
    models = blocks(params, signals, lib)
    [params, signals, models] = models.set(params, signals)
    # Main simulation core
    [params, signals, models] = simulation(params, models, signals, lib)
    # Finalize, plotting, saving, and other after-simulated codes
    finalize(params, models, signals, lib)
    # Return the data
    return [params, signals, models]
# The end of the function

# Runnng the main function of this framework
params, signals, models = main()
print("Everything has gone right.")

# External control the main function
# If you need an external loop, Copy this piece of code after all orders
# in this file. Afterwards, uncomment it and use its options.
# params  = dict()
# signals = dict()
# models  = dict()
# for i in range(0, 1):
#     # Define changable variables
#     extopt = dict()
#     extopt['test1'] = i
#     extopt['test2'] = 'Hi!'

#     # Run main loop
#     [params[i], signals[i], models[i]] = main(extopt)
# # The end of the for
