
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       http://abolfazldelavar.com
## // --------------------------------------------------------------

# Import initial classes
from core.lib.pyRequirment import *
from core.lib.coreLib  import solverCore
from core.caller.scopeEngine import scope

class LTISystem():
    ## --------------------------------------------------------------------------------
    # Author: A. Delavar, http://abolfazldelavar.com
    #  --------------------------------------------------------------------------
    # INSTRUCTION
    # 1) Define your LTI system using 'tf', 'zpk', 'ss', ...
    # 2) use the below code in 'initialization.py' into the 'valuation' and 'blocks'
    #        models.G = LTISystem(tf([1,3],[1,3,5,2,1]), params.nG, params.step, initial=2, delay=1)
    # 3) Use below order into 'simulation.py' to apply each step
    #        models.G.nextstep(Input, xNoise, yNoise)
    # ---------------------------------------------------------------------------------
    
    def __init__(self, iSys, nSystems, sampletime, **kwargs):
        # [self] <- (self, Number of systems, Block, Sample-time, Initial Condition, Input delay)
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
        for key, val in kwargs.items():
            # 'initial' is the initial value of states
            if key == 'initial': initialcondition = val
            # 'delay' is the input delay (in second)
            if key == 'delay': timedelay = val

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

    # The 'nextstep' function can provide an easy way to call 
    # dydnamics of the system to calculate next sample states
    # To use this function, refer to the top INSTRUCTION part
    def nextstep(self, u, xNoise = 0, yNoise = 0):
        # input at time t, additive noise on states, additive noise on output

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
    
    # This function can make a jump in the step number variable
    # If no arguments are available, jump 1 step
    def goahead(self, i = 1):
        self.currentStep += i
        
    # Reset Block by changing the current step to zero
    def reset(self):
        self.currentStep = 0
    # The end of the function
# The end of the class


class neuronGroup():
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
    
    def __init__(self, inputsystem, nNeurons, sampletime, **kwargs):
        # the number of neurons, dynamic class, sample time, initial condition
        initialcondition = [0]
        timedelay        = 0
        Pre              = 0
        Post             = 0
        for key, val in kwargs.items():
            # 'initial' can set the initial condition
            if key == 'initial': initialcondition = val
            # 'delay' denotes the input delay
            if key == 'delay': timedelay = val
            # 'Pre' denotes the pre numbers
            if key == 'Pre': Pre = val
            # 'Post' denotes the post numbers
            if key == 'Post': Post = val

        self.delay          = timedelay              # The input signals over the time
        self.Pre            = Pre                    # Pre
        self.Post           = Post                   # Post
        self.block          = inputsystem            # Get a copy of your system class
        self.sampleTime     = sampletime             # The simulation sample-time
        self.numberNeurons  = nNeurons               # The number of neurons (Network size)
        self.numStates      = self.block.numStates   # The number of states
        self.numInputs      = self.block.numInputs   # The number of inputs
        self.numOutputs     = self.block.numOutputs  # The number of measurements
        self.solverType     = self.block.solverType  # The type of dynamic solver
        self.currentStep    = 0                      # The current step of simulation
        self.initialStates  = self.block.initialStates
        self.initialStates  = np.reshape(self.initialStates, [np.size(self.initialStates), 1])
        self.synapseCurrent = np.zeros([self.numInputs,  self.numberNeurons])
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
        
    # The 'nextstep' function can provide an easy way to call 
    # dydnamics of the system to calculate next sample states
    # To use this function, refer to the top INSTRUCTION part
    def nextstep(self, u, outInput = False, xNoise = 0, yNoise = 0):
        # this object, input at time t, additive noise on states, additive noise on output
        
        # Making delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = u
        systemInput = self.inputs[:,:,0] + self.synapseCurrent

        # Set before-state-limitations:
        # This can be used if we want to process on states before
        # calculating the next states by dynamics.
        x = self.block.limitations(self.states, 0)
        
        # The below handle function is used in the following
        handleDyn   = lambda xx: self.block.dynamics(xx, systemInput)
        
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
        y = self.block.measurements(x, u)

        # Inter connections and synapses' currents are calculated here
        if self.Pre != 0 and self.Post != 0:
            if outInput == False: outInput = self.inputs[:,:,-1]*0
            self.synapseCurrent = self.block.synapses(x, outInput, self.Pre, self.Post)
        
        # Updating internal signals
        self.states  = x + xNoise
        self.outputs = y + yNoise
    # The end of the function
# The end of the class


class nonlinear():
    # The runner of nonlinear dynamic systems
    # --------------------------------------------------------------------------
    # --- INSTRUCTION -------------------------------------------------------
    # NOTE: The expressions of 'THIS' and 'THISMODEL' refer to a dynamic block like 'Lorenz'
    # 1) Copy THIS class into your dynamic file in folder 'blocks'.
    # 2) Rename the new class, arbitrarily.
    # 3) Edit properties according to your system detail.
    # 4) Insert dynamic equations into the 'dynamics' function.
    # 5) Write your output codes into the 'measurements' function.
    # 6) If there is any state limitation, you can set them in 'limitations'.
    # 7) If you want to use this block for estimation purposes,
    #    edite the initial values and covariance matrices just below here, and
    #    import the Jacobian matrices into the 'jacobian' function.
    # 8) To use, put the needed below code in 'initialization.py' to set initial options
    #    8.1. To use for running a nonlinear system, use:
    #         models.THIS = nonlinear(THIS(), Time Line, initial=1, solver='')
    #    8.2. To use for estimation purposes, copy the below code:
    #         models.THISMODEL = estimator(THISMODEL(), signals.tLine, initial=1, solver='', approach='')
    # 9) Use the piece of code showed below in 'simulation.py' to apply each step
    #    9.1. To use for running a nonlinear system, use:
    #         models.THIS.nextstep(Input Signal, xNoise, yNoise)
    #    9.2. To use for estimation purposes, copy the below code:
    #         models.THIS.nextstep(u[:,k], y[:,k])
    # --------------------------------------------------------------------------

    def __init__(self, inputsystem, timeline, **kwargs):
        # system, sample time, time line, initial states condition
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
        
    # The 'nextstep' function can provide an easy way to call 
    # dydnamics of the system to calculate next sample states
    # To use this function, refer to the top INSTRUCTION part
    def nextstep(self, u, xNoise = 0, yNoise = 0):
        # input at time t, additive noise on states, additive noise on output
        
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
    def goAhead(self, i = 1):
        self.currentStep = self.currentStep + i
    
    # Reset Block by changing the current step to zero
    def goFirst(self):
        self.currentStep = 0
        
    # The below function is used to plot the internal signals
    def show(self, params, sel = 'x', **kwargs):
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

