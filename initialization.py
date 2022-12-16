
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       http://abolfazldelavar.com
## // --------------------------------------------------------------

# Import requirements
from core.lib.pyRequirment   import *
from core.lib.coreLib        import *
from functions               import *
from core.caller.dyEngine    import *
from core.caller.scopeEngine import *
from core.caller.estEngine   import *
from blocks.compNeuronBlocks import *
from blocks.chaosBlocks      import *
from blocks.industrialBlocks import *

class loadLibraries():
    def __init__(lib):
        # [lib] <- ()
        # Loading libraries
        lib.func = functionLib()
        lib.draw = plotToolLib()
        lib.mfun = ownLib()
# The end of the class


class valuation():
    def __init__(params, lib, extOpt = 0):
        # [Parameters] <- (External variables as a structure variable)
        # extOpt is used for external control on whole simulation by a
        # separate code. If you want to import data here,
        # make a dictionary data and put it into the 'validation()' in 'main.m'.
        
        # Tho below variables are time-step and simulation time, respectively.
        # You might need to modify them arbitrarily.
        params.step = 0.01 #(double)
        params.tOut = 100  #(double)
        
        # The given line below is a dependent variable. You normally
        # should NOT change it.
        params.n = int(params.tOut/params.step) #(DEPENDENT)
        
        # The two next variables carry the folders from which you can use to
        # call your files and saving the results organizely.
        params.loadPath = 'data/inputs'  #(string)
        params.savePath = 'data/outputs' #(string)
        
        # Do you want to save a diary after each simulation? So set the below logical
        # variable "True". The below directory is the place your logs are saved.
        # The third order is a string that carries the name of the file, and normally
        # is created using the current time, but you can change arbitrary.
        params.makeDiary = True   #(logical)
        params.diaryDir  = 'logs' #(string)
        params.diaryFile = lib.func.getNow(1, '-') #(string)
        
        # The amount of time (in second) between each commands printed on CW.
        params.commandIntervalSpan = 2 #(int)
        
        # The below line is related to creating names when you are saving data
        # which include the time of saving, It is a logical variable which has
        # "False" value as its default
        params.uniqueSave = False #(logical)
        
        # The below string can be set one of the following formats. It's the
        # default format whcih when you do not insert any formats, it will be
        # considered. These legall formats are "jpg", "png", "pdf"
        params.defaultImageFormat = 'png' #(string)
        
        ## Your code ...
        
        ## External properties - This is used to control 'main.m from other files
        if extOpt != 0:
            params.extOpt = extOpt
# The End of the class


class vectors():
    def __init__(signals, params, lib):
        # [Signals] <- (Parameters, Libraries)
        # This function is created to support your signals
        # Simulation time vector
        signals.tLine = np.arange(0, params.tOut, params.step)

        # Put your signals here ...
        
    def set(signals, params):
        # Put your codes to change params here ...
        params.updated = 'Signals have been created!'
        return [params, signals]
# The end of the class


class blocks():
    def __init__(models, params, signals, lib):
        # [Models] <- (Parameters, Signals, Libraries)
        # This function is created to support your systems
        models.updated = True
        
        # Insert your blocks here ...

    def set(models, params, signals):
        # Put your codes to change params here ...
        params.updated = 'Blocks have been created!'
        return [params, signals, models]
# The end of the class


