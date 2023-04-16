
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import initial classes
from core.lib.pyRequirement import *
from core.lib.coreLib  import solverCore
from core.caller.scopeEngine import scope

# Linear dynamics
class ltiGroup():
    def __init__(self, iSys, sampletime, **kwargs):
        '''
        ### Description:
        This class is used to make a group of LTI systems which might have interactions, too.
        Note that linear systems or filters must be imported as a transfer function of state space form.

        ### Input variables:
        * Sysyem; e.g., `tf([1], [1,2,3])`
        * Sample time
        
        ### Options:
        * `initial` denotes the initial condition of the system
        * `replicate` is the number of blocks; default is `1`
        * `delay` cannotes the input delay in step scale; e.g., `18` steps
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
        self.delay       = timedelay                      # Delay steps
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

    def __call__(self, u, xNoise = 0, yNoise = 0):
        '''
        ### Description:
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
    
    def __repr__(self):
        return f"** {self.__class__.__name__} **\nNumber of elements: {self.numberLTIs}"
# The end of the class

# Nonlinear dynamic group
class nonlinearGroup(solverCore):
    def __init__(self, sampletime, **kwargs):
        '''
        ### Description:
        This class is created as a part of `Neuron Family` which provides tools to make a network of nonlinear
        dynamics which each node of them might have internal relation with others named `Synapses`.
        Note that the system imported must be a class defined in `blocks` path.

        ### Input variables:
        * Sample time
        
        ### Options:
        * `initial` denotes the initial condition of the system
        * `replicate` is the number of blocks; default is `1`
        * `delay` cannotes the input delay in step scale; e.g., `10` steps
        * `pre` indicates a vector including node IDs which are connected to `post`s; e.g., `[1,1,2,3]`
        * `post` represents a vector containing Posterior; e.g., `[3,2,3,2]`
        * `solver` cannotes to set the solver type; e.g., `euler`, `rng4`, etc.
        '''
        initialcondition = [0]
        timedelay        = 0
        self.__enSyn       = False
        pre              = np.array(0)
        post             = np.array(0)
        nNeurons         = 1
        solverType       = self.__class__.solverType
        for key, val in kwargs.items():
            # 'initial' can set the initial condition
            if key == 'initial': initialcondition = val
            # 'delay' denotes the input delay
            if key == 'delay': timedelay = val
            # 'replicate' denotes the number of blocks
            if key == 'replicate': nNeurons = val
            # 'Pre' denotes the pre numbers
            if key == 'pre':
                pre = val
                self.__enSyn = True
            # 'Post' denotes the post numbers
            if key == 'post': post = val
            #  Dynamic solver type
            if key == 'solver': solverType = val

        self.delay          = timedelay   # The input signals over the time
        self.pre            = pre         # Pre
        self.post           = post        # Post
        self.sampleTime     = sampletime  # The simulation sample-time
        self.numberNeurons  = nNeurons    # The number of neurons (Network size)
        self.solverType     = solverType  # The type of dynamic solver
        self.initialStates  = self.__class__.initialStates
        self.initialStates  = np.reshape(self.initialStates, [np.size(self.initialStates), 1])
        self.synapseCurrent = np.zeros([self.__class__.numSynapsesSignal, self.numberNeurons])
        self.inputs         = np.zeros([self.__class__.numInputs,  self.numberNeurons, self.delay + 1])
        self.outputs        = np.zeros([self.__class__.numOutputs, self.numberNeurons])
        self.states         = np.ones([self.__class__.numStates,  self.numberNeurons])
        self.states         = self.initialStates*self.states
        
        # If the initial input does not exist, set it zero
        # Else, put the initial condition in the state matrix
        initialcondition = np.array(initialcondition)
        iniSh = initialcondition.shape
        if sum(iniSh) == self.__class__.numStates or sum(iniSh) == self.__class__.numStates + 1:
            # If the imported initial value is not a column vector, do this:
            initialcondition = np.reshape(initialcondition, [np.size(initialcondition), 1])
            self.states += 1
            self.states  = initialcondition*self.states
        elif sum(iniSh) != 1 and sum(iniSh) != 2:
            if iniSh == (self.__class__.numStates, self.numberNeurons):
                self.states = initialcondition
            else:
                raise ValueError("The dimensional of initial value that inserted is wrong. Check it please.")
    
    def __call__(self, u, outInput = False, xNoise = 0, yNoise = 0):
        '''
        ### Description:
        This function can provide a `prediction` of the next step, using the current inputs.

        ### Input variables:
        * Input array at step `k`
        * Output control signal used in synapse calculations; the default value is `False`
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
        x = self.__class__._limitations(self.__class__, self.states, 0)
        
        # The below handle function is used in the following
        handleDyn = lambda xx: self.__class__._dynamics(self.__class__, xx, systemInput, self.synapseCurrent)
        
        # This part calculates the states and outputs using the system dynamics
        if self.__class__.timeType == 'c':
            # The type of solver can be under your control
            # To change your solver, do not change any code here
            # Change the solver type in 'Izhikevich.m' file or others
            
            x = super().dynamicRunner(handleDyn, x, x, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, just the
            # dynamic must be solved as below
            x = handleDyn(x)
        
        # Set after-state-limitations
        x = self.__class__._limitations(self.__class__, x, 1)
        
        # The output of the system is solved by the measurement
        # dynamics of the system which are available in 'Izhikevich.m' file
        y = self.__class__._measurements(self.__class__, x, u, self.solverType)

        # Inter connections and synapses' currents are calculated here
        if self.__enSyn == True:
            if outInput == False: outInput = self.inputs[:,:,-1]*0
            self.synapseCurrent = self.__class__._synapses(self.__class__, x, outInput, self.pre, self.post)
        
        # Updating internal signals
        self.states  = x + xNoise
        self.outputs = y + yNoise
    # The end of the function

    def __repr__(self):
        return f"** {self.__class__.__name__} **\nNumber of elements: {self.numberNeurons}\nSolver Type: '{self.solverType}'"
# The end of the class


class nonlinear(solverCore):
    def __init__(self, timeline, **kwargs):
        '''
        ### Description:
        This class provides tools to make a nonlinear system with 
        all internal vectors from the start of the simulation which can be utilized to
        have more accessibility to define a wide range of systems.
        Note that the system imported must be a class defined in `blocks` path.

        ### Input variables:
        * Time line
        
        ### Options:
        * `initial` denotes the initial condition of the system.
        * `solver` cannotes to set the solver type; e.g., `euler`, `rng4`, etc.
        * `estimator` denotes if you want this block to be an estimator. To do this, set it `True`.
        * `approach` indicates the type of estimator - `ekf` or `ukf`.
        '''
        self.timeLine      = np.reshape(timeline, [1, np.size(timeline)])
        self.sampleTime    = np.mean(self.timeLine[0, 1:-1] - self.timeLine[0, 0:-2])
        self.numSteps      = np.size(self.timeLine) # The number of sample steps
        self.solverType    = self.__class__.solverType  # The type of dynamic solver
        self.currentStep   = 0                      # The current step of simulation
        self.initialStates = self.__class__.initialStates
        self.inputs        = np.zeros([self.__class__.numInputs,  self.numSteps])
        self.outputs       = np.zeros([self.__class__.numOutputs, self.numSteps])
        self.states        = np.zeros([self.__class__.numStates,  self.numSteps + 1])
        self.states[:, 0]  = self.initialStates.flatten()
        self.estimator     = False
        estAproach         = 'ekf'

        # Extracting the arbitraty value of properties
        for key, val in kwargs.items():
            # The initial condition
            if key == 'initial': self.states[:, 0] = np.array(val).flatten()
            # The approach of estimation
            if key == 'approach': estAproach = val
            # Dynamic solver type
            if key == 'solver': self.solverType = val
            # If it is estimator
            if key == 'estimator': self.estimator = True
        
        # This part initialize the estimator by setting parameters
        if self.estimator == True:
            self.covariance = self.__class__.covariance
            self.estAproach = estAproach # The estimation approach ('ekf', 'ukf', ...)
            if self.estAproach == 'ekf':
                self.qMatrix = self.__class__.qMatrix
                self.rMatrix = self.__class__.rMatrix
            elif self.estAproach == 'ukf':
                self.qMatrix = self.__class__.qMatrix
                self.rMatrix = self.__class__.rMatrix
                self.kappa   = self.__class__.kappa
                self.alpha   = self.__class__.alpha
                # Dependent variables
                self.nUKF    = self.__class__.numStates
                self.lambd   = np.power(self.alpha,2)*(self.nUKF + self.kappa) - self.nUKF
                self.betta   = 2
                # Making weights
                self.wm      = np.ones([2*self.nUKF + 1, 1])/(2*(self.nUKF + self.lambd))
                self.wc      = self.wm
                self.wc[0,0] = self.wm[0,0] + (1 - np.power(self.alpha,2) + self.betta)

    def __call__(self, u, xNoise = 0, yNoise = 0):
        '''
        ### Description:
        This function can provide a `prediction` of the next step, using the current data.

        ### Input variables:
        * Input array at step `k`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        '''
        if self.estimator == True: return 0

        # The current time is calculated as below
        currentTime = self.timeLine[0, self.currentStep]
        
        # Preparing the input signal and save to the internal array
        self.inputs[:, self.currentStep] = u
        
        # Set before-state-limitations:
        # This can be used if we want to process on states before
        # calculating the next states by dynamics.
        xv = self.__class__._limitations(self.__class__, self.states, 0)
        # Getting the previous states
        xo = self.states[:, self.currentStep]
        xo = np.reshape(xo, [np.size(xo), 1])
        
        # The below handle function is used in the following
        handleDyn = lambda xx: self.__class__._dynamics(self.__class__,  \
                                                   xx,                   \
                                                   self.inputs,          \
                                                   self.currentStep,     \
                                                   self.sampleTime,      \
                                                   currentTime)
        
        # This part calculates the states and outputs using the system dynamics
        if self.__class__.timeType == 'c':
            # The type of solver can be under your control
            # To change your solver type, do not change any code here
            # Change the solver type in 'chaos.m' file or others
            x = super().dynamicRunner(handleDyn, xv, xo, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, just the
            # dynamic must be solved as below
            x = handleDyn(xv)
        
        # Set after-state-limitations
        x = self.__class__._limitations(self.__class__, x, 1)
        
        # The output of the system is solved by the measurement
        # dynamics of the system which are available in 'chaos.m' file
        y = self.__class__._measurements(self.__class__, \
                                    self.states,         \
                                    self.inputs,         \
                                    self.currentStep,    \
                                    self.sampleTime,     \
                                    currentTime)
        # Update internal signals which later can be used for plotting
        # and programming for other parts of the code
        self.states[:, self.currentStep + 1] = x.flatten() + xNoise
        self.outputs[:, self.currentStep]    = y.flatten() + yNoise
        self.currentStep += 1
    # The end of the function

    # The 'estimate' function can provide an easy way to call
    # dydnamics of the system to calculate next sample states
    # To use this function, refer to the top INSTRUCTION part
    def estimate(self, u, y):
        '''
        ### Description:
        This function can estimate an ahead step utilizing the current data.

        ### Input variables:
        * Input array of the real system at step `k`
        * Output array of the real system at step `k`
        '''
        if self.estimator == False: return 0

        if self.estAproach == 'ekf':
            self.__nextstepEKF(u, y)
        elif self.estAproach == 'ukf':
            self.__nextstepUKF(u, y)

    ## Next step of Extended Kalman Filter (EKF)
    def __nextstepEKF(self, u, y):
        # [Internal update] <- (Internal, Input, Output)
        
        ## Initialize parameters
        # The current time is calculated as below
        currentTime = self.timeLine[0, self.currentStep]
        # Preparing and saving inputs and outputs to internal
        self.inputs[:, self.currentStep]  = u
        self.outputs[:, self.currentStep] = y
        y = np.reshape(y, [np.size(y), 1])
        
        # Using dynamics of system to calculate Jacobians
        [A, L, H, M] = self.__class__._jacobians(self.__class__,  \
                                                self.states,      \
                                                self.inputs,      \
                                                self.currentStep, \
                                                self.sampleTime,  \
                                                currentTime)
        ## Prediction step - Update xp
        #  This part tries to obtain a prediction estimate from dynamic
        #  model of your system directly from nonlinear equations
        xm = self.states[:, self.currentStep]
        xm = np.reshape(xm, [np.size(xm), 1])

        # Set before-state-limitations
        xv  = self.__class__._limitations(self.__class__, self.states, 0)
        
        # The below handle function is used in the following
        handleDyn = lambda xx: self.__class__._dynamics(self.__class__,       \
                                                        xx,                   \
                                                        self.inputs,          \
                                                        self.currentStep,     \
                                                        self.sampleTime,      \
                                                        currentTime)
        # This part calculates the states and outputs using the system dynamics
        if self.__class__.timeType == 'c':
            # The type of solver can be under your control
            # To change your solver type, do not change any code here
            # Change the solver type in the block class or its called order in 'initialization.py'
            xp = super().dynamicRunner(handleDyn, xv, xm, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, just the
            # dynamic must be solved as below
            xp = handleDyn(xv)

        # Set after-state-limitations
        xp = self.__class__._limitations(self.__class__, xp, 1)
        
        # Prediction step - Update covariance matrix
        Pp = A.dot(self.covariance).dot(A.transpose()) + \
             L.dot(self.qMatrix).dot(L.transpose())
        
        ## Posterior step - Reciving measurements
        #  If there is not any measurement (y == NaN), posterior won't
        #  calculate and just prediction will be reported.
        if not np.any(np.isnan(y)):
            # Kalman Gain
            K  = (Pp.dot(H.transpose())).dot(                  \
                  np.linalg.inv(H.dot(Pp).dot(H.transpose()) + \
                  M.dot(self.rMatrix).dot(M.transpose())) )
            # Update the states
            xm = xp + K.dot(y - H.dot(xp))
            # Update covariance matrix
            Pm = (np.eye(self.covariance.shape[0]) - K.dot(H)).dot(Pp)
        else:
            xm = xp
            Pm = Pp
        
        ## Update internal signals
        self.states[:, self.currentStep + 1] = xm.flatten() # To save estimated states
        self.covariance                      = Pm           # To save covariance matrix
        self.currentStep += 1                               # Go to the next step
    
    ## Next step of Unscented Kalman Filter (UKF)
    def __nextstepUKF(self, u, y):
        # [Internal update] <- (Internal, Input, Output)
        # ------------------------------------------------------
        # To see how this algorithm works, refer to below source:
        #   A. Delavar and R. R. Baghbadorani, "Modeling, estimation, and
        #   model predictive control for Covid-19 pandemic with finite
        #   security duration vaccine," 2022 30th International Conference
        #   on Electrical Engineering (ICEE), 2022, pp. 78-83,
        #   doi: 10.1109/ICEE55646.2022.9827062.
        # ------------------------------------------------------
        
        ## Initialize parameters, STEP 0 & 1
        # The current time is calculated as below
        currentTime = self.timeLine[0, self.currentStep]
        # Preparing and saving inputs and outputs to internal
        self.inputs[:, self.currentStep]  = u
        self.outputs[:, self.currentStep] = y
        y = np.reshape(y, [np.size(y), 1])
        # Using dynamics of system to calculate Jacobians
        [A, L, H, M] = self.__class__._jacobians(self.__class__,  \
                                                self.states,      \
                                                self.inputs,      \
                                                self.currentStep, \
                                                self.sampleTime,  \
                                                currentTime)
        # Getting last states prior and its covariance
        xm = self.states[:, self.currentStep]
        xm = np.reshape(xm, [np.size(xm), 1])
        Pm = self.covariance
        
        ## Solving sigma points, STEP 2
        # Calculating sqrt
        dSigma = np.sqrt(self.nUKF + self.lambd)*((np.linalg.cholesky(Pm)).transpose())
        # Putting 'xm' is some column (copy)
        xmCopy = xm[:, np.int8(np.zeros([1, np.size(xm)]).flatten())]
        # Obtaining sigma points
        sp = np.concatenate((xm, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        ## Prediction states and their covariance, STEP 3
        #  This part tries to obtain a prediction estimate from dynamic
        #  model of your system directly from nonlinear equations
        nSpoints = sp.shape[1]
        xp       = np.zeros([self.__class__.numStates, 1])
        Xp       = np.zeros([self.__class__.numStates, nSpoints])
        for i in range(0, nSpoints):
            changedFullState = self.states
            changedFullState[:, self.currentStep] = sp[:, i]
            
            # Set before-state-limitations
            xv  = self.__class__._limitations(self.__class__, changedFullState, 0)
             # The below handle function is used in the following
            handleDyn = lambda xx: self.__class__._dynamics(self.__class__,   \
                                                            xx,               \
                                                            self.inputs,      \
                                                            self.currentStep, \
                                                            self.sampleTime,  \
                                                            currentTime)
            if self.__class__.timeType == 'c':
                Xp[:,i] = super().dynamicRunner(handleDyn,        \
                                                xv,               \
                                                xm,               \
                                                self.sampleTime,  \
                                                self.solverType).flatten()
            else:
                Xp[:,i] = handleDyn(xv)
            
            # Set after-state-limitations
            Xp[:,i] = self.__class__._limitations(self.__class__, Xp[:,i], 1)
            # Prediction update
            temp1 = Xp[:, i]
            xp = xp + self.wm[i,0]*(np.reshape(temp1, [np.size(temp1), 1]))
        # The end of the loop

        dPp = Xp - xp[:, np.int8(np.zeros([1, np.size(nSpoints)])).flatten()]
        # Updating the covariance of states matrix
        Pp  = dPp.dot(np.diag(self.wc.flatten())).dot(dPp.transpose()) + \
              L.dot(self.qMatrix).dot(L.transpose())
        
        ## Updating sigma points, STEP 4
        # dSigma = np.sqrt(self.nUKF + self.lambd).dot( \
        #         (np.linalg.cholesky(Pp)).transpose()) # Calculating sqrt
        # Putting 'xp' is some column (copy)
        # xmCopy = xp[:, np.int8(np.zeros([1, np.size(xp)]))]
        # sp     = np.concatenate((xp, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        if not np.any(np.isnan(y)):
            ## Solving output estimation using predicted data, STEP 5
            #  This part tries to obtain a prediction output from sigma points
            zb = np.zeros([self.__class__.numOutputs, 1])
            Zb = np.zeros([self.__class__.numOutputs, nSpoints])
            for i in range(0, nSpoints):
                changedFullState = self.states
                changedFullState[:, self.currentStep] = Xp[:, i] #Or 'Xp[:, i]' instead of 'sp[:, i]'
                Zb[:,i] = self.__class__._measurements(self.__class__,    \
                                                        changedFullState, \
                                                        self.inputs,      \
                                                        self.currentStep, \
                                                        self.sampleTime,  \
                                                        currentTime).flatten()
                # Predicted output
                temp1 = Zb[:, i]
                zb = zb + self.wm[i,0]*(np.reshape(temp1, [np.size(temp1), 1]))
            # The end of the loop

            dSt = Zb - zb[:, np.int8(np.zeros([1, np.size(nSpoints)])).flatten()]
            # Updating the covariance of output matrix
            St  = dSt.dot(np.diag(self.wc.flatten())).dot(dSt.transpose()) + \
                  M.dot(self.rMatrix).dot(M.transpose())

            ## Solving Kalman gain, STEP 6
            SiG = dPp.dot(np.diag(self.wc.flatten())).dot(dSt.transpose())
            # Kalman Gain
            K   = SiG.dot(np.linalg.inv(St))
        
        ## Solving posterior using measurement data, STEP 7
        #  If there is not any measurement (y == NaN), posterior won't
        #  calculate and just prediction will be reported.
        if not np.any(np.isnan(y)):
            # Update the states
            xm = xp + K.dot(y - zb)
            # Update covariance matrix
            Pm = Pp - K.dot(SiG.transpose())
        else:
            xm = xp
            Pm = Pp
        
        ## Update internal signals
        self.states[:, self.currentStep + 1] = xm.flatten() # To save estimated states
        self.covariance                      = Pm           # To save covariance matrix
        self.currentStep += 1                               # Go to the next step
    # The end of the function

    # This function can make a jump in the step variable
    # If no arguments are available, jump 1 step
    def __iadd__(self, i = 1):
        '''
        ### Description:
        This function can make a jump in the step number variable.

        ### Input variables:
        * how many steps you would like me to jump?; default is `1`
        '''
        self.currentStep = self.currentStep + i
    
    # Reset Block by changing the current step to zero
    def reset(self):
        '''
        ### Description:
        Reseting the block via changing the current step to zero.
        '''
        self.currentStep = 0
        
    # The below function is used to plot the internal signals
    def show(self, params, sel = 'x', **kwargs):
        '''
        ### Description:
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
            nSignals = self.__class__.numStates
        elif sel == 'y':
            signal   = self.outputs[:, 0:self.numSteps]
            nSignals = self.__class__.numOutputs
        elif sel == 'u':
            signal   = self.inputs[:, 0:self.numSteps]
            nSignals = self.__class__.numInputs
        # Make a scope
        scp = scope(self.timeLine, nSignals, initial=signal)

        scp.show(params, title=self.__class__.name, **kwargs)
    # The end of the function
    
    def __repr__(self):
        return f"** {self.__class__.__name__} **\nCurrent point: {self.currentStep}/{self.numSteps}\nSolver Type: '{self.solverType}'"
# The end of the class

