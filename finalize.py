
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
        signals.v.show(params, select=[0, 6, 50], title='Voltage', legend=1)
        signals.G.show(params, select=[0, 6, 50], title='Glutamate', legend=1)
        signals.Isum.show(params, select=[0, 6, 50], title='$I_{sum}$', legend=1)
        signals.ca.show(params, select=[0, 1, 3, 4, 5, 10], title='Ca', legend=1)

        signals.v.raster(params, title='Voltage Raster', ylabel='Neuron Index')
        signals.G.raster(params, title='Glutamate Raster', ylabel='Neuron Index')
        signals.ca.raster(params, title='Calcium Raster', ylabel='Astrocyte Index')
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
