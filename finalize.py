
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       http://abolfazldelavar.com
## // --------------------------------------------------------------

# Loading requirements
from core.lib.pyRequirment import *

def finalize(params, models, signals, lib, plotOrNot = True):
    # [Parameters, Models, Signals, Libraries] <- (Parameters, Models, Signals, Libraries, Plot or Not (1 or 0))
    # All of your plots and depiction objects shoud be coded here
    # To use this, you have initialize, main, and finalize sections.
    
    if plotOrNot == True:
        ## Initialize
        draw  = lib.draw            # Import all plotting functions
        n     = params.n
        nn    = np.arange(0, n)     # A vector from 1 to n
        tLine = signals.tLine[nn]   # A time-line vector
        
        ## Main part
        #  Insert your graphical ideas here ...

        # Illustrate all plots
        plt.show()
        
    ## Finalize ans saving data
    #  If you want to save data, put your ideas into the below function
    saving(params, models, signals, lib)
# The end of the function


def saving(params, models, signals, lib):
    # [] <- (Parameters, Models, Signals, Libraries)
    # If you want to save data, put your ideas into this function
    # All data and libraries have imported to this function, just call them and use.
    
    ## Write your codes here ...
    
    pass    
# The end of the function
