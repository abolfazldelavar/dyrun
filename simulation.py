
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import necessary classes
from core.lib.pyRequirment import *

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
    
    ## Main loop
    for k in range(0, params.n):
        # Displaying the iteration number on the command window
        trig = func.disit(k, params.n, trig, params)

        ## Write your codes here ---------------------------------------------------

        # *** NEURON PART
        # Preparing input current to apply to the network
        if signals.T_Iapp_met[k] == 0:
            Iapp = np.zeros((params.mneuro, params.nneuro), dtype=np.uint8)
        else:
            # for the timeline of applied input
            Iapp = signals.Iapp[:, :, signals.T_Iapp_met[k]] 
        pattInputCurrent = np.double(Iapp.T.flatten())
        # Keeping Isum
        signals.Isum.getdata(models.neurons.synapseCurrent + pattInputCurrent)
        # Updating neurons
        models.neurons.nextstep(pattInputCurrent)
        # Keeping voltage time series
        signals.v.getdata(models.neurons.outputs)
        # obtaining Glutamate in neuronal network
        models.G.nextstep(models.neurons.outputs == models.neurons.block.neuron_fired_thr)
        # Keeping Glutamate signal 
        signals.G.getdata(models.G.outputs)

        # *** NEURON TO ASTROCYTE PART

        # *** ASTROCYTE PART
        astroInput = np.zeros((1, params.quantity_astrocytes))
        astroInput[0, 0:3] = 3
        # Updating astrocyte network
        models.astrocytes.nextstep(astroInput)
        # Saving Calcium time series
        signals.ca.getdata(models.astrocytes.outputs)

        # *** ASTROCYTE TO NEURON PART

        # --------------------------------------------------------------------------

    ## Finalize options
    # To report the simulation time after running has finished
    func.disit(k, params.n, [st, 0, trig[2]], params)
    func.sayEnd(st, params)
    # Sent to output
    return [params, signals, models]
# The end of the function

