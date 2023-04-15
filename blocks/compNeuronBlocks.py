
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import initialize classes
from core.lib.pyRequirment import *
from core.caller.dyEngine import *

class izhikevich(nonlinearGroup):
    '''
    ## Izhikevich neuron model
    '''
    
    # This name will be showed as its plot titles
    name              = 'Izhikevich Neuron Model'
    numStates         = 2          # The number of states
    numInputs         = 1          # The number of inputs
    numOutputs        = 1          # The number of outputs
    numSynapsesSignal = 1          # The number of synapses signals
    timeType          = 'c'        # 'c' -> Continuous, 'd' -> Discrete
    solverType        = 'euler'    # 'Euler', 'Runge'
    initialStates     = [-60, -12] # Initial value of states
    
    # Other variables (use 'mp_' before all of them)
    mp_a                = 0.1     # Time scale of the recovery variable
    mp_b                = 0.2     # Sensitivity of the recovery variable to the sub-threshold fluctuations of the membrane potential
    mp_c                = -65     # After-spike reset value of the membrane potential
    mp_d                = 2       # After-spike reset value of the recovery variable
    mp_ksyn             = 6       #
    mp_aep              = 1.2     #
    mp_gsyn             = 0.05    #
    mp_Esyn             = 0       #
    mp_timescale        = 1e3     # is used to change ms to second
    mp_neuron_fired_thr = 30      # Maxium amount of input current for presynaptic neurons
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        Isum = I + iInter
        dx      = np.zeros([2, x.shape[1]])
        dx[0,:] = self.mp_timescale*(0.04*np.power(x[0,:],2) + 5*x[0,:] - x[1,:] + 140 + Isum)
        dx[1,:] = self.mp_timescale*(self.mp_a*(self.mp_b*x[0,:] - x[1,:]))
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def _measurements(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        return x[0,:]
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def _limitations(self, x, mode):
        # Obj, States, Mode
        if mode == 0:
            # before updating states
            ind    = (x[0,:] == self.mp_neuron_fired_thr)
            x[0,:] = x[0,:]*(1 - ind) + ind*self.mp_c
            x[1,:] = x[1,:] + ind*self.mp_d
        elif mode == 1:
            # After updating states
            x[0,:] = np.minimum(x[0,:], self.mp_neuron_fired_thr)
        return x
    # The end of the function

    ## Synapses between systems
    #  To have an internal static interaction between agents
    def _synapses(self, x, outInput, Pre, Post):
        # Obj, States, Foreign input, Pre, Post
        # Neuron synaptic currents
        Smoother = 1 / (1 + np.exp((-x[0,:] / self.mp_ksyn)))
        Isyn     = np.zeros((1, np.size(x,1)))
        gsync    = self.mp_gsyn + outInput[:, Post]*self.mp_aep
        Isync    = gsync * Smoother[Pre] * (self.mp_Esyn - x[0, Post])
        for i in range(0, np.size(Pre)):
            Isyn[0, Post[i]] = Isyn[0, Post[i]] + Isync[0, i]
        return Isyn
# The end of the class


class ullah(nonlinearGroup):
    '''
    ## Ullah astrocyte model
    '''
    
    # This name will be showed as its plot titles
    name              = 'Ullah Astrocyte Model'
    numStates         = 3          # The number of states
    numInputs         = 1          # The number of inputs
    numOutputs        = 1          # The number of outputs
    numSynapsesSignal = 2          # The number of synapses signals (diffusions: Ca and IP3)
    timeType          = 'c'        # 'c' -> Continuous, 'd' -> Discrete
    solverType        = 'euler'    # 'euler', 'rng4'
    ca_0              = 0.072495
    h_0               = 0.886314
    ip3_0             = 0.820204
    initialStates     = [ca_0, h_0, ip3_0] # Initial value of states
    
    # Other variables
    mp_c0    = 2.0     #
    mp_c1    = 0.185   #
    mp_v1    = 6.0     #
    mp_v2    = 0.11    #
    mp_v3    = 2.2     #
    mp_v4    = 0.3     #
    mp_v6    = 0.2     #
    mp_k1    = 0.5     #
    mp_k2    = 1.0     #
    mp_k3    = 0.1     #
    mp_k4    = 1.1     #
    mp_d1    = 0.13    #
    mp_d2    = 1.049   #
    mp_d3    = 0.9434  #
    mp_d5    = 0.082   #
    mp_IP3s  = 0.16    #
    mp_Tr    = 0.14    #
    mp_a     = 0.8     #
    mp_a2    = 0.14    #
    mp_dCa   = 0.03    #
    mp_dIP3  = 0.03    #
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        sum_Ca  = iInter[0,:]   # Calcium deffusion
        sum_IP3 = iInter[1,:]   # IP3 deffusion
        M       = x[2,:] / (x[2,:] + self.mp_d1)
        NM      = x[0,:] / (x[0,:] + self.mp_d5)
        Ier     = self.mp_c1 * self.mp_v1 * np.power(M,3) * np.power(NM,3) * np.power(x[1,:],3) * (((self.mp_c0 - x[0,:]) / self.mp_c1) - x[0,:])
        Ileak   = self.mp_c1 * self.mp_v2 * (((self.mp_c0 - x[0,:]) / self.mp_c1) - x[0,:])
        Ipump   = self.mp_v3 * np.power(x[0,:],2) / (np.power(x[0,:],2) + np.power(self.mp_k3,2))
        Iin     = self.mp_v6 * (np.power(x[2,:],2) / (np.power(self.mp_k2,2) + np.power(x[2,:],2)))
        Iout    = self.mp_k1 * x[0,:]
        Q2      = self.mp_d2 * ((x[2,:] + self.mp_d1) / (x[2,:] + self.mp_d3))
        h       = Q2 / (Q2 + x[0,:])
        Tn      = 1.0 / (self.mp_a2 * (Q2 + x[0,:]))
        Iplc    = self.mp_v4 * ((x[0,:] + (1.0 - self.mp_a) * self.mp_k4) / (x[0,:] + self.mp_k4))
        dx      = np.zeros([3, x.shape[1]])
        dx[0,:] = Ier - Ipump + Ileak + Iin - Iout + self.mp_dCa * sum_Ca # Calcium
        dx[1,:] = (h - x[1,:]) / Tn # H
        dx[2,:] = (self.mp_IP3s - x[2,:]) * self.mp_Tr + Iplc + I + self.mp_dIP3 * sum_IP3 # IP3
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def _measurements(self, x, I, iInter):
        # Parameters, states, inputs, Internal synapses current
        return x[0,:]
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def _limitations(self, x, mode):
        # Obj, States, Mode
        return x
    # The end of the function

    ## Synapses between systems
    #  To have an internal static interaction between agents
    def _synapses(self, x, outInput, Pre, Post):
        # Obj, States, Foreign input, Pre, Post
        # Astrocytes synaptic currents
        qua_astr = np.size(x,1)
        Deff     = np.zeros((2, qua_astr))
        for i in range(0, qua_astr):
            p = Pre == i
            Deff[:, i] = np.add.reduce(x[[0,2],:][:, Post[p]], axis=1) - np.add.reduce(p)*x[[0,2], i]
        # for i in range(0, np.size(Pre)):  # The above code is much faster than this
        #     Deff[:, Pre[i]] = Deff[:, Pre[i]] + x[[0,2], Post[i]] - x[[0,2], Pre[i]]
        return Deff
# The end of the class

