
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import necessary classes
from core.lib.pyRequirment import *
from core.caller.scopeEngine import *

def simulation(params, models, signals, lib):
    # [Parameters, Models, Signals, Libraries] <- (Parameters, Models, Signals, Libraries)
    # This file is your main core of simulation which there is a time-loop
    # Also, the model blocks and the signals must be updated here.
    # Before the main loop, you can initialize if you need, Also you can
    # finalize after that if you need, as well.
    
    ## Initial options
    func = lib.func
    st   = func.sayStart(params)
    # A trigger used to report steps in command
    trig = [st, params.commandIntervalSpan, -1]
    
    # Optimization ------------------------------------------------
    # -------------------------------------------------------------
    a = 0.1
    b = 0.2
    c = -65
    d = 2
    aep = 1.2
    gsyn = 0.05
    Esyn = 0
    ksyn = 6
    Iapp = 0
    eqs = '''
        dv/dt = (0.04*(v**2) + 5*v - u + 140 + Iapp + Isyn)/ms : 1
        du/dt = (a*(b*v - u))/ms : 1
        Isyn : 1
    '''
    reset = '''
        v  = c
        u += d
    '''
    G = NeuronGroup(params.quantity_neurons, eqs, threshold='v>30', reset=reset, method='euler')
    G.v[:] = -60
    G.u[:] = -12
    G.Isyn = 0

    synPrep = '''
        Smoother  = 1 / (1 + exp((-v_pre / ksyn)))
        Isyn_post = gsyn * Smoother * (Esyn - v_post)
    '''
    S = Synapses(G, G, on_pre=synPrep)
    S.connect(i=signals.neuronsPre, j=signals.neuronsPost)

    M = StateMonitor(G, 'v', record=True)
    run(params.tOut*second)

    signals.v = scope(signals.tLine, params.quantity_neurons, initial=M.v)  # Neuron output signal

    # -------------------------------------------------------------

    ## Finalize options
    # To report the simulation time after running has finished
    func.sayEnd(st, params)
    # Sent to output
    return [params, signals, models]
# The end of the function

