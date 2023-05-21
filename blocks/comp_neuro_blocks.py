
## // --------------------------------------------------------------
#    ***DYNAMIC RUNNER***
#    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
#    Web: https://github.com/abolfazldelavar/dyrun
## // --------------------------------------------------------------

# Import initialize classes
from core.lib.required_libraries import *
from core.caller.dynamic_engine import *

class Izhikevich(NonlinearGroup):
    '''
    ### Izhikevich neuron model
    '''
    
    # This name will be showed as its plot titles
    name = 'Izhikevich Neuron Model'
    n_states = 2 # The number of states
    n_inputs = 1 # The number of inputs
    n_outputs = 1 # The number of outputs
    n_synapses_signal = 1 # The number of synapses signals
    time_type = 'c' # 'c' -> Continuous, 'd' -> Discrete
    solver_type = 'euler' # 'euler', 'rng4'
    initial_states = [-60, -12] # Initial value of states
    
    # Other variables (use 'mp_' before all of them)
    mp_a = 0.1 # Time scale of the recovery variable
    mp_b = 0.2 # Sensitivity of the recovery variable to the sub-threshold fluctuations of the membrane potential
    mp_c = -65 # After-spike reset value of the membrane potential
    mp_d = 2 # After-spike reset value of the recovery variable
    mp_ksyn = 6 #
    mp_aep = 1.2 #
    mp_gsyn = 0.05 #
    mp_Esyn = 0 #
    mp_timescale = 1e3 # is used to change ms to second
    mp_neuron_fired_thr = 30 # Maxium amount of input current for presynaptic neurons
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, input_signal, i_internal):
        # Parameters, states, inputs, Internal synapses current
        i_sum = input_signal + i_internal
        dx      = np.zeros([2, x.shape[1]])
        dx[0,:] = self.mp_timescale*(0.04*np.power(x[0,:],2) + 5*x[0,:] - x[1,:] + 140 + i_sum)
        dx[1,:] = self.mp_timescale*(self.mp_a*(self.mp_b*x[0,:] - x[1,:]))
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def _measurements(self, x, input_signal, i_interal):
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
    # End of function

    ## Synapses between systems
    #  To have an internal static interaction between agents
    def _synapses(self, x, pre, post, ext_input):
        # Obj, States, Foreign input, Pre, Post
        # Neuron synaptic currents
        i_syn = np.zeros((1, np.size(x,1)))
        if_1 = i_syn if ext_input == False else ext_input*self.mp_aep
        smoother = 1 / (1 + np.exp((-x[0,:] / self.mp_ksyn)))
        g_sync = self.mp_gsyn + if_1
        i_sync = g_sync[:, post] * smoother[pre] * (self.mp_Esyn - x[0, post])
        # Use bincount to accumulate Isync for each unique index in Post
        i_syn = np.bincount(post, weights=i_sync[0, :], minlength=np.size(x,1))
        # # The two below lines are equal to the previous line.
        # for i in range(0, np.size(pre)):
        #     i_syn[0, post[i]] = i_syn[0, post[i]] + i_sync[0, i]
        return i_syn
# End of class


class Ullah(NonlinearGroup):
    '''
    ### Ullah astrocyte model
    '''
    
    # This name will be showed as its plot titles
    name = 'Ullah Astrocyte Model'
    n_states = 3 # The number of states
    n_inputs = 1 # The number of inputs
    n_outputs = 1 # The number of outputs
    n_synapses_signal = 2 # The number of synapses signals (diffusions: Ca and IP3)
    time_type = 'c' # 'c' -> Continuous, 'd' -> Discrete
    solver_type = 'euler' # 'euler', 'rng4'
    ca_0 = 0.072495
    h_0 = 0.886314
    ip3_0 = 0.820204
    initial_states = [ca_0, h_0, ip3_0] # Initial value of states
    
    # Other variables
    mp_c0 = 2.0 #
    mp_c1 = 0.185 #
    mp_v1 = 6.0 #
    mp_v2 = 0.11 #
    mp_v3 = 2.2 #
    mp_v4 = 0.3 #
    mp_v6 = 0.2 #
    mp_k1 = 0.5 #
    mp_k2 = 1.0 #
    mp_k3 = 0.1 #
    mp_k4 = 1.1 #
    mp_d1 = 0.13 #
    mp_d2 = 1.049 #
    mp_d3 = 0.9434 #
    mp_d5 = 0.082 #
    mp_IP3s = 0.16 #
    mp_Tr = 0.14 #
    mp_a = 0.8 #
    mp_a2 = 0.14 #
    mp_dCa = 0.03 #
    mp_dIP3 = 0.03 #
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, input_signal, i_internal):
        # Parameters, states, inputs, Internal synapses current
        sum_Ca  = i_internal[0,:] # Calcium deffusion
        sum_IP3 = i_internal[1,:] # IP3 deffusion
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
        dx[2,:] = (self.mp_IP3s - x[2,:]) * self.mp_Tr + Iplc + input_signal + self.mp_dIP3 * sum_IP3 # IP3
        return dx
    
    ## Measurement functions 
    #  ~~> y = g(x,u)
    def _measurements(self, x, input_signal, i_internal):
        # Parameters, states, inputs, Internal synapses current
        return x[0,:]
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def _limitations(self, x, mode):
        # Obj, States, Mode
        return x
    # End of function

    ## Synapses between systems
    #  To have an internal static interaction between agents
    def _synapses(self, x, pre, post, ext_input):
        # Obj, States, Foreign input, Pre, Post
        # Astrocytes synaptic currents
        qua_astr = np.size(x,1)
        deff     = np.zeros((2, qua_astr))
        for i in range(0, qua_astr):
            p = pre == i
            deff[:, i] = np.add.reduce(x[[0,2],:][:, post[p]], axis=1) - np.add.reduce(p)*x[[0,2], i]
        # for i in range(0, np.size(pre)): # The above code is much faster than this
        #     deff[:, pre[i]] = deff[:, pre[i]] + x[[0,2], post[i]] - x[[0,2], pre[i]]
        return deff
# End of class

