
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       http://abolfazldelavar.com
## // --------------------------------------------------------------

# Import initialize classes
from core.lib.pyRequirment import *

class Izhikevich():
    # Nonlinear Dynamic
    # --------------------------------------------------------------------------
    # --- INSTRUCTION -------------------------------------------------------
    # NOTE: The expressions of 'THIS' and 'THISMODEL' refer to a dynamic block like 'Izhikevich'
    # 1) Copy THIS class into your dynamic file in folder 'blocks'.
    # 2) Rename the new class, arbitrarily.
    # 3) Edit properties according to your system detail.
    # 4) Insert dynamic equations into the 'dynamics' function.
    # 5) Write your output codes into the 'measurements' function.
    # 6) If there is any state limitation, you can set them in 'limitations'.
    # 7) To use, put the below code in 'initialization.py' to set initial options
    #    models.THIS = nonlinear(THIS(), Time Line, initial=1, solver='')
    # 8) Use the piece of code showed below in 'simulation.py' to apply each step
    #    models.THIS.nextstep(Input Signal, xNoise, yNoise)
    # --------------------------------------------------------------------------
    
    # This name will be showed as its plot titles
    name          = 'Izhikevich Model'
    numStates     = 2          # Number of states
    numInputs     = 1          # Number of inputs
    numOutputs    = 1          # Number of outputs
    timeType      = 'c'        # 'c' -> Continuous, 'd' -> Discrete
    solverType    = 'Euler'    # 'Euler', 'Runge'
    initialStates = [-60, -12] # Initial value of states
    
    # Other variables
    a                = 0.1     # Time scale of the recovery variable
    b                = 0.2     # Sensitivity of the recovery variable to the sub-threshold fluctuations of the membrane potential
    c                = -65     # After-spike reset value of the membrane potential
    d                = 2       # After-spike reset value of the recovery variable
    ksyn             = 6       #
    aep              = 1.2     #
    gsyn             = 0.05    #
    Esyn             = 0       #
    timescale        = 1e3     # is used to change ms to second
    neuron_fired_thr = 30      # Maxium amount of input current for presynaptic neurons
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def dynamics(self, x, I):
        # Parameters, states, inputs
        dx      = np.zeros([2, x.shape[1]])
        dx[0,:] = self.timescale*(0.04*np.power(x[0,:],2) + 5*x[0,:] - x[1,:] + 140 + I)
        dx[1,:] = self.timescale*(self.a*(self.b*x[0,:] - x[1,:]))
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def measurements(self, x, I):
        # Parameters, states, inputs
        return x[0,:]
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def limitations(self, x, mode):
        # Obj, States, Mode
        if mode == 0:
            # before updating states
            ind    = (x[0,:] == self.neuron_fired_thr)
            x[0,:] = x[0,:]*(1 - ind) + ind*self.c
            x[1,:] = x[1,:] + ind*self.d
        elif mode == 1:
            # After updating states
            x[0,:] = np.minimum(x[0,:], self.neuron_fired_thr)
        return x
    # The end of the function

    ## Synapses between systems
    #  To have an internal static interaction between agents
    def synapses(self, x, outInput, Pre, Post):
        # Obj, States, Output input, Pre, Post
        # Neuron synaptic currents
        Smoother = 1 / (1 + np.exp((-x[0,:] / self.ksyn)))
        Isyn     = np.zeros((1, np.size(x,1)))
        gsync    = self.gsyn + outInput[:, Post]*self.aep
        Isync    = gsync * Smoother[Pre] * (self.Esyn - x[0, Post])
        for i in range(0, np.size(Pre)):
            Isyn[0, Post[i]] = Isyn[0, Post[i]] + Isync[0, i]
        return Isyn
# The end of the class


