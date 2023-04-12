
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import initial classes
from core.lib.pyRequirment import *
from core.lib.coreLib  import solverCore
from core.caller.scopeEngine import scope

# Linear dynamics
class ltiGroup():
    def __init__(self, iSys, sampletime, **kwargs):
        '''
        This class is used to make a group of LTI systems which might have interactions, too.
        Note that linear systems or filters must be imported as a transfer function of state space form.

        ### Input variables:
        * Sysyem; e.g., `tf([1], [1,2,3])`
        * Sample time
        
        ### Options:
        * `initial` denotes the initial condition of the system
        * `replicate` is the number of blocks; default is `1`
        * `delay` cannotes the input delay in time scale; e.g., `1.3` second
        '''
        self.inputsystem = iSys
        if iSys.dt == sampletime and type(iSys) == StateSpace:
            # input is a discrete-time state-space with the same sample-time
            self.block = iSys
        elif iSys.dt == sampletime and type(iSys) == TransferFunction:
            # input is a discrete-time transfer fuction with same sample time
            self.block = minreal(tf2ss(iSys))
        elif type(iSys) == StateSpace:
            if iSys.dt != 0:
                # input is a discrete-time state-space with different sample time   *
                # MATLAB code is: self.block = d2d(iSys, sampletime);
                pass
            else:
                # input is a continuous-time state-space
                self.block = c2d(iSys, sampletime)
        elif iSys.dt != 0:
            # input is a discrete-time transfer fuction with different sample time  *
            # MATLAB code is: self.block = d2d(minreal(ss(iSys)), sampletime);
            pass
        else:
            # input is a continuous-time transfer fuction
            self.block = c2d(ss(iSys), sampletime)
        
        initialcondition = [0]
        timedelay        = 0
        nSystems         = 1
        for key, val in kwargs.items():
            # 'initial' is the initial value of states
            if key == 'initial': initialcondition = val
            # 'delay' is the input delay (in second)
            if key == 'delay': timedelay = val
            # 'replicate' is the number of blocks
            if key == 'replicate': nSystems = val

        # Setting the internal variables
        self.numberLTIs  = nSystems                       # The number of LTI systems
        self.sampleTime  = sampletime                     # Simulation sample time
        self.A           = self.block.A                   # Dynamic matrix A
        self.B           = self.block.B                   # Dynamic matrix B
        self.C           = self.block.C                   # Dynamic matrix C
        self.D           = self.block.D                   # Dynamic matrix D
        self.numStates   = self.block.A.shape[0]          # The number of states
        self.numInputs   = self.block.B.shape[1]          # The number of inputs
        self.numOutputs  = self.block.C.shape[0]          # The number of measurements
        self.currentStep = 0                              # The current step of simulation
        self.delay       = int(timedelay/self.sampleTime) # Delay steps
        self.inputs      = np.zeros([self.numInputs , self.numberLTIs, self.delay + 1])
        self.outputs     = np.zeros([self.numOutputs, self.numberLTIs])
        self.states      = np.zeros([self.numStates, self.numberLTIs])
        
        # If the initial input does not exist, set it zero
        # Else, put the initial condition in the state matrix
        initialcondition = np.array(initialcondition)
        iniSh = initialcondition.shape
        if sum(iniSh) == self.numStates or sum(iniSh) == self.numStates + 1:
            # If the imported initial value is not a column vector, do this:
            initialcondition = np.reshape(initialcondition, [np.size(initialcondition), 1])
            self.states += 1
            self.states  = initialcondition*self.states
        elif initialcondition.size != 1:
            if initialcondition.shape == (self.numStates, self.numberLTIs):
                self.states = initialcondition
            else:
                raise ValueError("The dimential of initial value that inserted is wrong. Check it please.")

    def predict(self, u, xNoise = 0, yNoise = 0):
        '''
        This function can provide an easy way to call dydnamics of the system to calculate the next sample states.

        ### Input variables:
        * Input array at step `k`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        '''
        # Making delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = u
        
        # Updating the states via dx = Ax + Bu
        x = self.A.dot(self.states) + self.B.dot(self.inputs[:,:,0])
        
        # Calculating outputs via y = Cx + Du
        y = self.C.dot(self.states) + self.D.dot(self.inputs[:,:,0])
        
        # Update internal signals which later can be used for plotting
        # and programming for other parts of the code
        self.states  = x + xNoise
        self.outputs = y + yNoise
        self.currentStep += 1
    
    def jump(self, i = 1):
        '''
        This function can make a jump in the step number variable.

        ### Input variables:
        * how many steps you would like me to jump?; default is `1`
        '''
        self.currentStep += i
        
    def reset(self):
        '''
        Reseting the block via changing the current step to zero.
        '''
        self.currentStep = 0
    # The end of the function
# The end of the class

# Nonlinear dynamic group
class nonlinearGroup():    
    def __init__(self, inputsystem, sampletime, **kwargs):
        '''
        This class provides tools to make a network of nonlinear dynamics which nodes of 
        that might have internal relations named `Synapses`.
        Note that the system imported must be a class defined in `blocks` path.

        ### Input variables:
        * Sysyem; e.g., `Izhikevich()`
        * Sample time
        
        ### Options:
        * `initial` denotes the initial condition of the system
        * `replicate` is the number of blocks; default is `1`
        * `delay` cannotes the input delay in time scale; e.g., `1.3` second
        * `Pre` indicates a vector including node IDs which are connected to `Post`s; e.g., `[1,1,2,3]`
        * `Post` represents a vector containing Posterior; e.g., `[3,2,3,2]`
        * `solver` cannotes to set the solver type; e.g., `euler`, `rng4`, etc.
        '''
        initialcondition = [0]
        timedelay        = 0
        Pre              = np.array(0)
        Post             = np.array(0)
        nNeurons         = 1
        solverType       = inputsystem.solverType
        for key, val in kwargs.items():
            # 'initial' can set the initial condition
            if key == 'initial': initialcondition = val
            # 'delay' denotes the input delay
            if key == 'delay': timedelay = val
            # 'replicate' denotes the number of blocks
            if key == 'replicate': nNeurons = val
            # 'Pre' denotes the pre numbers
            if key == 'Pre': Pre = val
            # 'Post' denotes the post numbers
            if key == 'Post': Post = val
            #  Dynamic solver type
            if key == 'solver': solverType = val

        self.delay          = timedelay              # The input signals over the time
        self.Pre            = Pre                    # Pre
        self.Post           = Post                   # Post
        self.block          = inputsystem            # Get a copy of your system class
        self.sampleTime     = sampletime             # The simulation sample-time
        self.numberNeurons  = nNeurons               # The number of neurons (Network size)
        self.numStates      = self.block.numStates   # The number of states
        self.numInputs      = self.block.numInputs   # The number of inputs
        self.numOutputs     = self.block.numOutputs  # The number of measurements
        self.numSynapses    = self.block.numSynapses # The figure for synapses
        self.solverType     = solverType             # The type of dynamic solver
        self.currentStep    = 0                      # The current step of simulation
        self.initialStates  = self.block.initialStates
        self.initialStates  = np.reshape(self.initialStates, [np.size(self.initialStates), 1])
        self.synapseCurrent = np.zeros([self.numSynapses, self.numberNeurons])
        self.inputs         = np.zeros([self.numInputs,  self.numberNeurons, self.delay + 1])
        self.outputs        = np.zeros([self.numOutputs, self.numberNeurons])
        self.states         = np.ones([self.numStates,  self.numberNeurons])
        self.states         = self.initialStates*self.states
        
        # If the initial input does not exist, set it zero
        # Else, put the initial condition in the state matrix
        initialcondition = np.array(initialcondition)
        iniSh = initialcondition.shape
        if sum(iniSh) == self.numStates or sum(iniSh) == self.numStates + 1:
            # If the imported initial value is not a column vector, do this:
            initialcondition = np.reshape(initialcondition, [np.size(initialcondition), 1])
            self.states += 1
            self.states  = initialcondition*self.states
        elif sum(iniSh) != 1 and sum(iniSh) != 2:
            if iniSh == (self.numStates, self.numberNeurons):
                self.states = initialcondition
            else:
                raise ValueError("The dimensional of initial value that inserted is wrong. Check it please.")
    
    def predict(self, u, outInput = False, xNoise = 0, yNoise = 0):
        '''
        This function can provide an easy way to call dydnamics of the system to calculate the next sample states.

        ### Input variables:
        * Input array at step `k`
        * Input used in synapses; the default value is `False`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        '''
        # Making delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = u
        systemInput = self.inputs[:,:,0]

        # Set before-state-limitations:
        # This can be used if we want to process on states before
        # calculating the next states by dynamics.
        x = self.block.limitations(self.states, 0)
        
        # The below handle function is used in the following
        handleDyn   = lambda xx: self.block.dynamics(xx, systemInput, self.synapseCurrent)
        
        # This part calculates the states and outputs using the system dynamics
        if self.block.timeType == 'c':
            # The type of solver can be under your control
            # To change your solver, do not change any code here
            # Change the solver type in 'Izhikevich.m' file or others
            
            x = solverCore.dynamicRunner(handleDyn, x, x, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, just the
            # dynamic must be solved as below
            x = handleDyn(x)
        
        # Set after-state-limitations
        x = self.block.limitations(x, 1)
        
        # The output of the system is solved by the measurement
        # dynamics of the system which are available in 'Izhikevich.m' file
        y = self.block.measurements(x, u, self.solverType)

        # Inter connections and synapses' currents are calculated here
        if self.Pre.any() and self.Post.any():
            if outInput == False: outInput = self.inputs[:,:,-1]*0
            self.synapseCurrent = self.block.synapses(x, outInput, self.Pre, self.Post)
        
        # Updating internal signals
        self.states  = x + xNoise
        self.outputs = y + yNoise
    # The end of the function
# The end of the class


class nonlinear():
    def __init__(self, inputsystem, timeline, **kwargs):
        '''
        This class provides tools to make a nonlinear system with 
        all internal vectors from the start of the simulation which can be utilized to
        have more accessibility to define a wide range of systems.
        Note that the system imported must be a class defined in `blocks` path.

        ### Input variables:
        * Sysyem; e.g., `Lorenz()`
        * Time line
        
        ### Options:
        * `initial` denotes the initial condition of the system
        * `solver` cannotes to set the solver type; e.g., `euler`, `rng4`, etc.
        '''
        self.block         = inputsystem            # Get a copy of your system class
        self.timeLine      = np.reshape(timeline, [1, np.size(timeline)])
        self.sampleTime    = np.mean(self.timeLine[0, 1:-1] - self.timeLine[0, 0:-2])
        self.numSteps      = np.size(self.timeLine) # The number of sample steps
        self.numStates     = self.block.numStates   # The number of states
        self.numInputs     = self.block.numInputs   # The number of inputs
        self.numOutputs    = self.block.numOutputs  # The number of measurements
        self.solverType    = self.block.solverType  # The type of dynamic solver
        self.currentStep   = 0                      # The current step of simulation
        self.initialStates = self.block.initialStates
        self.inputs        = np.zeros([self.numInputs,  self.numSteps])
        self.outputs       = np.zeros([self.numOutputs, self.numSteps])
        self.states        = np.zeros([self.numStates,  self.numSteps + 1])
        self.states[:, 0]  = self.initialStates.flatten()

        # Extracting the arbitraty value of properties
        for key, val in kwargs.items():
            # The initial condition
            if key == 'initial': self.states[:, 0] = np.array(val).flatten()
            # Dynamic solver type
            if key == 'solver': self.solverType = val
        
    def predict(self, u, xNoise = 0, yNoise = 0):
        '''
        This function can provide an easy way to call dydnamics of the system to calculate the next sample states.

        ### Input variables:
        * Input array at step `k`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        '''
        # The current time is calculated as below
        currentTime = self.timeLine[0, self.currentStep]
        
        # Preparing the input signal and save to the internal array
        self.inputs[:, self.currentStep] = u
        
        # Set before-state-limitations:
        # This can be used if we want to process on states before
        # calculating the next states by dynamics.
        xv = self.block.limitations(self.states, 0)
        # Getting the previous states
        xo = self.states[:, self.currentStep]
        xo = np.reshape(xo, [np.size(xo), 1])
        
        # The below handle function is used in the following
        handleDyn = lambda xx: self.block.dynamics(xx,                   \
                                                   self.inputs,          \
                                                   self.currentStep,     \
                                                   self.sampleTime,      \
                                                   currentTime)
        
        # This part calculates the states and outputs using the system dynamics
        if self.block.timeType == 'c':
            # The type of solver can be under your control
            # To change your solver type, do not change any code here
            # Change the solver type in 'chaos.m' file or others
            x = solverCore.dynamicRunner(handleDyn, xv, xo, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, just the
            # dynamic must be solved as below
            x = handleDyn(xv)
        
        # Set after-state-limitations
        x = self.block.limitations(x, 1)
        
        # The output of the system is solved by the measurement
        # dynamics of the system which are available in 'chaos.m' file
        y = self.block.measurements(self.states,        \
                                    self.inputs,        \
                                    self.currentStep,   \
                                    self.sampleTime,    \
                                    currentTime)
        # Update internal signals which later can be used for plotting
        # and programming for other parts of the code
        self.states[:, self.currentStep + 1] = x.flatten() + xNoise
        self.outputs[:, self.currentStep]    = y.flatten() + yNoise
        self.currentStep += 1
    # The end of the function

    # This function can make a jump in the step variable
    # If no arguments are available, jump 1 step
    def jump(self, i = 1):
        '''
        This function can make a jump in the step number variable.

        ### Input variables:
        * how many steps you would like me to jump?; default is `1`
        '''
        self.currentStep = self.currentStep + i
    
    # Reset Block by changing the current step to zero
    def reset(self):
        '''
        Reseting the block via changing the current step to zero.
        '''
        self.currentStep = 0
        
    # The below function is used to plot the internal signals
    def show(self, params, sel = 'x', **kwargs):
        '''
        This function makes a quick plot of internal signals.

        ### input variables:
        * `params`
        * The signal must be shown - `x`, `y`, or `u`; the default value is 'x'

        ### Options:
            * `select` is used to choose signals arbitrarily; e.g., `select=[0,2,6]`.
            * `derive` is used to get derivatives of signals, which can be used in different forms:
                * `derive=False` or `derive=True`; default is `False`,
                * `derive=[1,1,0]` is used to get derivatives of selected signals. Ones you want to get derivative must be `1` or `True`.
            * `notime` is used to remove time and illustrate timeless plots. it can be set differently:
                * `notime=[0,1]` or `notime=[0,1,2]` is utilized to depict signals 2D or 3D. Note that the numbers are signal indices,
                * `notime=[[0,1], [1,2]]` or `notime=[[0,1,2], [3,0,1]]` is utilized to depict different signal groups 2D or 3D. Note that the numbers are signal indices.
            * `save` denotes to the name of the file which the plot will be saved with. it could be `image.png/pdf/jpg` or `True`.
            * `xlabel`, `ylabel`, and `zlabel` are the x, y, and z titles of the illustration.
            * `legend` is used for legend issue:
                * `legent=True` and `legent=False`, enables and disables the legent,
                * `legent='title'` enables the legend with imported title.
            * `lineWidth` can set the line width.
            * `grid` can enables the grid of the illustration - `True` or `False`.
            * `legCol` can control the column number of the legend and must be a positive integer.
        '''
        # To illustrate states, inputs, or outputs, you might have to
        # use some varargins which are explained in 'scope' class
        if sel == 'x':
            signal   = self.states[:, 0:self.numSteps]
            nSignals = self.numStates
        elif sel == 'y':
            signal   = self.outputs[:, 0:self.numSteps]
            nSignals = self.numOutputs
        elif sel == 'u':
            signal   = self.inputs[:, 0:self.numSteps]
            nSignals = self.numInputs
        # Make a scope
        scp = scope(self.timeLine, nSignals, signal)

        scp.show(params, title=self.block.name, **kwargs)
    # The end of the function
# The end of the class

