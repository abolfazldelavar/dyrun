
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
    numStates     = 2          # The number of states
    numInputs     = 1          # The number of inputs
    numOutputs    = 1          # The number of outputs
    numSynapses   = 1          # The number of synapses signals
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
    def dynamics(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        Isum = I + iInter
        dx      = np.zeros([2, x.shape[1]])
        dx[0,:] = self.timescale*(0.04*np.power(x[0,:],2) + 5*x[0,:] - x[1,:] + 140 + Isum)
        dx[1,:] = self.timescale*(self.a*(self.b*x[0,:] - x[1,:]))
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def measurements(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
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
        # Obj, States, Foreign input, Pre, Post
        # Neuron synaptic currents
        Smoother = 1 / (1 + np.exp((-x[0,:] / self.ksyn)))
        Isyn     = np.zeros((1, np.size(x,1)))
        gsync    = self.gsyn + outInput[:, Post]*self.aep
        Isync    = gsync * Smoother[Pre] * (self.Esyn - x[0, Post])
        for i in range(0, np.size(Pre)):
            Isyn[0, Post[i]] = Isyn[0, Post[i]] + Isync[0, i]
        return Isyn
# The end of the class


class Ullah():
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
    name          = 'Ullah Model'
    numStates     = 3          # The number of states
    numInputs     = 1          # The number of inputs
    numOutputs    = 1          # The number of outputs
    numSynapses   = 2          # The number of synapses signals (diffusions: Ca and IP3)
    timeType      = 'c'        # 'c' -> Continuous, 'd' -> Discrete
    solverType    = 'Euler'    # 'Euler', 'Runge'
    ca_0          = 0.072495
    h_0           = 0.886314
    ip3_0         = 0.820204
    initialStates = [ca_0, h_0, ip3_0] # Initial value of states
    
    # Other variables
    c0    = 2.0     #
    c1    = 0.185   #
    v1    = 6.0     #
    v2    = 0.11    #
    v3    = 2.2     #
    v4    = 0.3     #
    v6    = 0.2     #
    k1    = 0.5     #
    k2    = 1.0     #
    k3    = 0.1     #
    k4    = 1.1     #
    d1    = 0.13    #
    d2    = 1.049   #
    d3    = 0.9434  #
    d5    = 0.082   #
    IP3s  = 0.16    #
    Tr    = 0.14    #
    a     = 0.8     #
    a2    = 0.14    #
    dCa   = 0.03    #
    dIP3  = 0.03    #
    timescale        = 1e3  # is used to change ms to second
    neuron_fired_thr = 30   # Maxium amount of input current for presynaptic neurons
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def dynamics(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        sum_Ca  = iInter[0,:]   # Calcium deffusion
        sum_IP3 = iInter[1,:]   # IP3 deffusion
        M       = x[2,:] / (x[2,:] + self.d1)
        NM      = x[0,:] / (x[0,:] + self.d5)
        Ier     = self.c1 * self.v1 * np.power(M,3) * np.power(NM,3) * np.power(x[1,:],3) * (((self.c0 - x[0,:]) / self.c1) - x[0,:])
        Ileak   = self.c1 * self.v2 * (((self.c0 - x[0,:]) / self.c1) - x[0,:])
        Ipump   = self.v3 * np.power(x[0,:],2) / (np.power(x[0,:],2) + np.power(self.k3,2))
        Iin     = self.v6 * (np.power(x[2,:],2) / (np.power(self.k2,2) + np.power(x[2,:],2)))
        Iout    = self.k1 * x[0,:]
        Q2      = self.d2 * ((x[2,:] + self.d1) / (x[2,:] + self.d3))
        h       = Q2 / (Q2 + x[0,:])
        Tn      = 1.0 / (self.a2 * (Q2 + x[0,:]))
        Iplc    = self.v4 * ((x[0,:] + (1.0 - self.a) * self.k4) / (x[0,:] + self.k4))
        dx      = np.zeros([3, x.shape[1]])
        dx[0,:] = Ier - Ipump + Ileak + Iin - Iout + self.dCa * sum_Ca # Calcium
        dx[1,:] = (h - x[1,:]) / Tn # H
        dx[2,:] = (self.IP3s - x[2,:]) * self.Tr + Iplc + I + self.dIP3 * sum_IP3 # IP3
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def measurements(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        return x[0,:]
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def limitations(self, x, mode):
        # Obj, States, Mode
        return x
    # The end of the function

    ## Synapses between systems
    #  To have an internal static interaction between agents
    def synapses(self, x, outInput, Pre, Post):
        # Obj, States, Foreign input, Pre, Post
        # Astrocytes synaptic currents
        Deff     = np.zeros((2, np.size(x,1)))
        for i in range(0, np.size(Pre)):
            Deff[:, Pre[i]] = Deff[:, Pre[i]] + x[[0,2], Post[i]] - x[[0,2], Pre[i]]
        return Deff
# The end of the class

