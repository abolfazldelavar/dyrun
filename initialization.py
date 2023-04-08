
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
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
        params.step = 0.0001 #(double)
        params.tOut = 6      #(double)
        
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
        
        #### Your code ------------------------------------------------------
        # Load Images options
        params.images_dir = params.loadPath + '/images - patterns/Experiment 1 - words'
        params.image_names = [
            'A.jpg',        \
            'mid_B.jpg',    \
            'P.jpg',        \
            'mid_L.jpg',    \
            'J.jpg',        \
            'I.jpg',        \
            'E.jpg',        \
            'G.jpg'         \
        ]

        # Experiment
        params.learn_start_time         = 0.2
        params.learn_impulse_duration   = 0.21
        params.learn_impulse_shift      = 0.41
        params.learn_order              = np.array([0, 1, 2, 3])
        
        params.test_start_time          = 2.3
        params.test_impulse_duration    = 0.13 # 0.15
        params.test_impulse_shift       = 0.42 # 0.4
        params.test_order               = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        
        # Applied pattern current
        params.variance_learn           = 0     # 0.05
        params.variance_test            = 0     # 0.03
        params.Iapp_learn               = 10    # 80
        params.Iapp_test                = 8     # 8

        # Movie
        params.after_sample_frames      = 200
        params.before_sample_frames     = 1

        # Poisson noise
        params.poisson_nu                = 1.5
        params.poisson_n_impulses        = 15
        params.poisson_impulse_duration  = int(0.03 / params.step)
        params.poisson_impulse_initphase = int(1.5 / params.step)
        params.poisson_amplitude         = 20

        # Network size
        # (mneuro = mastro * 3 + 1 (for az = 4))
        params.mneuro                   = 10
        params.nneuro                   = 10
        params.quantity_neurons         = params.mneuro * params.nneuro
        params.mastro                   = 5
        params.nastro                   = 5
        params.quantity_astrocytes      = params.mastro * params.nastro
        params.az                       = 2

        # Neuron model
        params.alf                      = 10   # s^-1    | Glutamate clearance constant
        params.k                        = 600  # uM.s^-1 | Efficacy of glutamate release

        # Synaptic connections
        params.N_connections            = 20   #number of synapses per neurons 
        params.quantity_connections     = params.quantity_neurons * params.N_connections
        params.lambdagain               = 1.5  #Average exponential distribution
        
        params.enter_astro              = 3    #F_astro # F_recall
        params.min_neurons_activity     = 3    #F_act   # F_memorize
        params.t_neuro                  = 0.06          # Astrocyte effect duration (second)
        params.amplitude_neuro          = 5             # Astrocyte input
        params.threshold_Ca             = 0.3  #0.15    # calcium should be higher than this to have WM
        
        window_astro_watch              = 0.01          # t(sec) # astrocyte watching this much back to neurons ...
                                                        # with bnh intervals
        shift_window_astro_watch        = 0.001         # t(sec)
        impact_astro                    = 0.26 #0.25    # t(sec)
        params.impact_astro             = int(impact_astro / params.step)
        params.window_astro_watch       = int(window_astro_watch / params.step)
        params.shift_window_astro_watch = int(shift_window_astro_watch / params.step)
        
        #### Your code ------------------------------------------------ #END#

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

        # Put your signals here ---------------------------------------------------
        #  Prepare images
        signals.images = lib.mfun.load_images(params)
        signals.Iapp, signals.T_Iapp, signals.T_Iapp_met, \
            signals.T_record_met = lib.mfun.make_experiment(signals.images, params)

        # Oscope signals
        signals.v    = scope(signals.tLine, params.quantity_neurons)  # Neuron output signal
        signals.ca   = scope(signals.tLine, params.quantity_astrocytes)  # Astrocyte Calcium
        signals.Isum = scope(signals.tLine, params.quantity_neurons)  # Neuron input signal
        signals.G    = scope(signals.tLine, params.quantity_neurons)  # Glutamate signal
        # --------------------------------------------------------------------#END#
        
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
        
        # Insert your blocks here -----------------------------------------------------
        # Neuron network, and synapses
        models.neurons = neuronGroup(Izhikevich(), params.quantity_neurons, params.step)
        models.neurons.Pre, models.neurons.Post = lib.mfun.createNeuronsConnections(params)

        # Astrocyte network, and synapses
        models.astrocytes = neuronGroup(Ullah(), params.quantity_astrocytes, params.step)
        models.astrocytes.Pre, models.astrocytes.Post = lib.mfun.createAstrocytesConnections(params)

        # Glutamate defused from presynaptic neuron to synaptic cleft
        models.G = LTISystem(tf([params.k],[1, params.alf]), params.quantity_neurons, params.step)
        # ------------------------------------------------------------------------#END#

    def set(models, params, signals):
        # Put your codes to change params here ...
        params.updated = 'Blocks have been created!'
        return [params, signals, models]
# The end of the class


