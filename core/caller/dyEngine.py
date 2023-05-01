
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
        ### Overview:
        This class facilitates the creation of a collection of LTI systems that may also have interactions.
        It is important to note that linear systems or filters must be imported in the form of
        a `tf()` or `ss()` from `control.matlab` extension.

        ### Input Parameters:
        * System; for example, `tf([1], [1,2,3])`
        * Sample Time

        ### Configuration Options:
        * `initial`: Specifies the initial state of the system
        * `replicate`: Determines the number of components; default value is `1`
        * `delay`: Defines the input delay in step scale; for example, `18` steps
        '''
        self.inputsystem = iSys
        if iSys.dt == sampletime and type(iSys) == StateSpace:
            # The input is a discrete-time state-space system with a consistent sample time.
            self.block = iSys
        elif iSys.dt == sampletime and type(iSys) == TransferFunction:
            # The input is a discrete-time transfer function with a consistent sample time.
            self.block = minreal(tf2ss(iSys))
        elif type(iSys) == StateSpace:
            if iSys.dt != 0:
                # The input is a discrete-time state-space system with a different sample time.
                # MATLAB code is: self.block = d2d(iSys, sampletime);
                pass
            else:
                # The input is a continuous-time state-space system.
                self.block = c2d(iSys, sampletime)
        elif iSys.dt != 0:
            # The input is a discrete-time transfer function with a different sample time.
            # MATLAB code is: self.block = d2d(minreal(ss(iSys)), sampletime);
            pass
        else:
            # The input is a continuous-time transfer function.
            self.block = c2d(ss(iSys), sampletime)
        
        initialcondition = [0]
        timedelay        = 0
        nSystems         = 1
        for key, val in kwargs.items():
            # 'initial' specifies the initial value of the states
            if key == 'initial': initialcondition = val
            # 'delay' specifies the input delay in seconds
            if key == 'delay': timedelay = val
            # 'replicate' specifies the number of blocks
            if key == 'replicate': nSystems = val

        self.numberLTIs = nSystems               # The number of LTI systems
        self.sampleTime = sampletime             # Simulation sample time
        self.A          = self.block.A           # Dynamic matrix A
        self.B          = self.block.B           # Dynamic matrix B
        self.C          = self.block.C           # Dynamic matrix C
        self.D          = self.block.D           # Dynamic matrix D
        self.numStates  = self.block.A.shape[0]  # The number of states
        self.numInputs  = self.block.B.shape[1]  # The number of inputs
        self.numOutputs = self.block.C.shape[0]  # The number of measurements
        self.delay      = timedelay              # Delay steps
        self.inputs     = np.zeros([self.numInputs , self.numberLTIs, self.delay + 1])
        self.outputs    = np.zeros([self.numOutputs, self.numberLTIs])
        self.states     = np.zeros([self.numStates, self.numberLTIs])
        
        # If the initial input does not exist, set it to zero. Otherwise, put the initial condition in the state matrix.
        initialcondition = np.array(initialcondition)
        iniSh = initialcondition.shape
        if sum(iniSh) == self.numStates or sum(iniSh) == self.numStates + 1:
            # If the imported initial value is not a column vector, reshape it.
            initialcondition = np.reshape(initialcondition, [np.size(initialcondition), 1])
            self.states += 1
            self.states  = initialcondition*self.states
        elif initialcondition.size != 1:
            if initialcondition.shape == (self.numStates, self.numberLTIs):
                self.states = initialcondition
            else:
                raise ValueError("The dimensions of the inserted initial value are incorrect. Please check it.")

    def __call__(self, u, xNoise = 0, yNoise = 0):
        '''
        ### Overview:
        This function can provide an easy way to call dydnamics of the system to calculate the next sample states.

        ### Input variables:
        * Input array at step `k`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        '''
        # Shifts the input signal to create a delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = u
        
        # Updates the states using the state-space equation dx = Ax + Bu
        x = self.A.dot(self.states) + self.B.dot(self.inputs[:,:,0])
        
        # Calculates the outputs using the state-space equation y = Cx + Du
        y = self.C.dot(self.states) + self.D.dot(self.inputs[:,:,0])
        
        # Updates internal signals
        self.states  = x + xNoise
        self.outputs = y + yNoise
    
    def __repr__(self):
        return f"** {self.__class__.__name__} **\nNumber of elements: {self.numberLTIs}"
# End of class

# Nonlinear dynamic group
class nonlinearGroup(solverCore):
    def __init__(self, sampletime, **kwargs):
        '''
        ### Overview:
        This class belongs to the `Neuron Family` and offers tools for constructing
        a network of nonlinear dynamics. Each component may have internal connections
        with others, known as `Synapses`. It is important to note that the imported
        system must be a class defined in the `blocks` path.

        ### Input Parameters:
        * Sample Time

        ### Configuration Options:
        * `initial`: Specifies the initial state of the system
        * `replicate`: Determines the number of components; default value is `1`
        * `delay`: Defines the input delay in step scale; for example, `10` steps
        * `pre`: A vector containing the IDs of components connected to `post`s; for example, `[1,1,2,3]`
        * `post`: A vector representing Posterior; for example, `[3,2,3,2]`
        * `solver`: Sets the type of solver to be used; for example, `euler`, `rng4`, etc.
        '''
        initialcondition = [0]
        timedelay        = 0
        self.__enSyn     = False
        pre              = np.array(0)
        post             = np.array(0)
        nNeurons         = 1
        solverType       = self.solverType
        for key, val in kwargs.items():
            # 'initial' specifies the initial condition of the system
            if key == 'initial': initialcondition = val
            # 'delay' specifies the input delay in step scale
            if key == 'delay': timedelay = val
            # 'replicate' specifies the number of blocks
            if key == 'replicate': nNeurons = val
            # 'pre' specifies the pre numbers
            if key == 'pre':
                pre = val
                self.__enSyn = True
            # 'post' specifies the post numbers
            if key == 'post': post = val
            # Specifies the dynamic solver type
            if key == 'solver': solverType = val

        self.delay          = timedelay   # The input signals over time
        self.pre            = pre         # Pre
        self.post           = post        # Post
        self.sampleTime     = sampletime  # The simulation sample time
        self.numberNeurons  = nNeurons    # The number of neurons (Network size)
        self.solverType     = solverType  # The type of dynamic solver
        self.initialStates  = np.reshape(self.initialStates, [np.size(self.initialStates), 1])
        self.synapseCurrent = np.zeros([self.numSynapsesSignal, self.numberNeurons])
        self.inputs         = np.zeros([self.numInputs,  self.numberNeurons, self.delay + 1])
        self.outputs        = np.zeros([self.numOutputs, self.numberNeurons])
        self.states         = np.ones([self.numStates,  self.numberNeurons])
        self.states         = self.initialStates*self.states
        
        # If the initial input does not exist, set it to zero. Otherwise, put the initial condition in the state matrix.
        initialcondition = np.array(initialcondition)
        iniSh = initialcondition.shape
        if sum(iniSh) == self.numStates or sum(iniSh) == self.numStates + 1:
            # If the imported initial value is not a column vector, reshape it.
            initialcondition = np.reshape(initialcondition, [np.size(initialcondition), 1])
            self.states += 1
            self.states  = initialcondition*self.states
        elif sum(iniSh) != 1 and sum(iniSh) != 2:
            if iniSh == (self.numStates, self.numberNeurons):
                self.states = initialcondition
            else:
                raise ValueError("The dimensions of the inserted initial value are incorrect. Please check it.")
    
    def __call__(self, u, outInput = False, xNoise = 0, yNoise = 0):
        '''
        ### Overview:
        This function can provide a `prediction` of the next step, using the current inputs.

        ### Input variables:
        * Input array at step `k`
        * Output control signal used in synapse calculations; the default value is `False`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        '''
        # Shifts the input signal by one time step to create a delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = u
        systemInput = self.inputs[:,:,0]

        # Applies limitations to the states before calculating the next states using the system dynamics
        x = self._limitations(self.states, 0)
        
        # Defines a handle function for calculating the system dynamics
        handleDyn = lambda xx: self._dynamics(xx, systemInput, self.synapseCurrent)
        
        # Calculates the states and outputs using the system dynamics
        if self.__class__.timeType == 'c':
            # For continuous-time systems, the solver type can be controlled by changing it in the model file or through `solver` option
            
            x = super().dynamicRunner(handleDyn, x, x, self.sampleTime, self.solverType)
        else:
            # For discrete-time systems, only the dynamic must be solved
            x = handleDyn(x)
        
        # Enforces restrictions on the states after computing the next states using the system dynamics
        x = self._limitations(x, 1)
        
        # Computes the system output using the measurement dynamics specified in the model file or other relevant files
        y = self._measurements(x, u, self.solverType)

        # Calculates interconnections and currents of synapses here
        if self.__enSyn == True:
            if outInput == False: outInput = self.inputs[:,:,-1]*0
            self.synapseCurrent = self._synapses(x, outInput, self.pre, self.post)
        
        # Updates internal signals
        self.states  = x + xNoise
        self.outputs = y + yNoise
    # End of function

    def __repr__(self):
        return f"** {self.__class__.__name__} **\nNumber of elements: {self.numberNeurons}\nSolver Type: '{self.solverType}'"
# End of class


class nonlinear(solverCore):
    def __init__(self, timeline, **kwargs):
        '''
        ### Overview:
        This class provides tools to create a nonlinear system with 
        all internal vectors from the beginning of the simulation which can be used to
        define a wide range of systems with greater accessibility.
        Note that the imported system must be a class defined in the `blocks` path.

        ### Input variables:
        * Time line
        
        ### Options:
        * `initial` specifies the initial condition of the system.
        * `solver` sets the type of solver; e.g., `euler`, `rng4`, etc.
        * `estimator` specifies whether this block should be an estimator. To do this, set it to `True`.
        * `approach` specifies the type of estimator - `ekf` or `ukf`.
        '''
        self.timeLine      = np.reshape(timeline, [1, np.size(timeline)])
        self.sampleTime    = np.mean(self.timeLine[0, 1:-1] - self.timeLine[0, 0:-2])
        self.numSteps      = np.size(self.timeLine)     # The number of sample steps
        self.solverType    = self.__class__.solverType  # The type of dynamic solver
        self.currentStep   = 0                          # The current step of simulation
        self.initialStates = self.__class__.initialStates
        self.inputs        = np.zeros([self.__class__.numInputs,  self.numSteps])
        self.outputs       = np.zeros([self.__class__.numOutputs, self.numSteps])
        self.states        = np.zeros([self.__class__.numStates,  self.numSteps + 1])
        self.states[:, 0]  = self.initialStates.flatten()
        self.estimator     = False
        estAproach         = 'ekf'

        # Retrieving the arbitrary value of properties
        for key, val in kwargs.items():
            # The initial condition
            if key == 'initial': self.states[:, 0] = np.array(val).flatten()
            # The estimation approach
            if key == 'approach': estAproach = val
            # Type of dynamic solver
            if key == 'solver': self.solverType = val
            # If it is an estimator
            if key == 'estimator': self.estimator = True
        
        # This section initializes the estimator by setting parameters
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
                # Creating weights
                self.wm      = np.ones([2*self.nUKF + 1, 1])/(2*(self.nUKF + self.lambd))
                self.wc      = self.wm
                self.wc[0,0] = self.wm[0,0] + (1 - np.power(self.alpha,2) + self.betta)

    def __call__(self, u, xNoise = 0, yNoise = 0):
        '''
        ### Overview:
        Utilizing current data, this function can furnish a `prediction` of the subsequent step.

        ### Input variables:
        * Input array at step `k`
        * Internal additive noise incorporated into the states
        * External additive noise incorporated into the measurements
        '''
        if self.estimator == True: return 0

        # The current time is calculated as follows
        currentTime = self.timeLine[0, self.currentStep]
        
        # Preparing the input signal and saving it to the internal array
        self.inputs[:, self.currentStep] = u
        
        # Establishing before-state-limitations:
        # This can be employed if we desire to process states prior to computing the subsequent states utilizing dynamics.
        xv = self.__class__._limitations(self.__class__, self.states, 0)
        
        # Retrieving the antecedent states
        xo = self.states[:, self.currentStep]
        xo = np.reshape(xo, [np.size(xo), 1])
        
        # The handle function below is utilized in the following
        handleDyn = lambda xx: self.__class__._dynamics(self.__class__,
                                                   xx,
                                                   self.inputs,
                                                   self.currentStep,
                                                   self.sampleTime,
                                                   currentTime)
        
        # This section computes the states and outputs utilizing the system dynamics
        if self.__class__.timeType == 'c':
            # The type of solver can be manipulated To alter your solver type,
            # do not modify any code here Modify the solver type in the model file
            x = super().dynamicRunner(handleDyn, xv, xo, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, only the dynamic must be solved as below
            x = handleDyn(xv)
        
        # Establishing after-state-limitations
        x = self.__class__._limitations(self.__class__, x, 1)
        
        # The system output is computed by the measurement dynamics of the system
        # which are available in the 'chaos.m' file
        y = self.__class__._measurements(self.__class__,
                                    self.states,
                                    self.inputs,
                                    self.currentStep,
                                    self.sampleTime,
                                    currentTime)
        
        # Updating internal signals
        self.states[:, self.currentStep + 1] = x.flatten() + xNoise
        self.outputs[:, self.currentStep]    = y.flatten() + yNoise
        self.currentStep += 1
    # End of function

    # The 'estimate' function can furnish an effortless method to invoke
    # the system dynamics to compute subsequent sample states.
    def estimate(self, u, y):
        '''
        ### Overview:
        Utilizing current data, this function can estimate a subsequent step.

        ### Input variables:
        * Input array of the actual system at step `k`
        * Output array of the actual system at step `k`
        '''
        if self.estimator == False: return 0

        if self.estAproach == 'ekf':
            self.__nextstepEKF(u, y)
        elif self.estAproach == 'ukf':
            self.__nextstepUKF(u, y)

    ## Subsequent step of Extended Kalman Filter (EKF)
    def __nextstepEKF(self, u, y):
        # [Internal update] <- (Internal, Input, Output)
        
        ## Initializing parameters
        # The current time is computed as follows
        currentTime = self.timeLine[0, self.currentStep]
        # Preparing and storing inputs and outputs within internal
        self.inputs[:, self.currentStep]  = u
        self.outputs[:, self.currentStep] = y
        y = np.reshape(y, [np.size(y), 1])
        
        # Employing system dynamics to compute Jacobians
        [A, L, H, M] = self.__class__._jacobians(self.__class__,
                                                self.states,
                                                self.inputs,
                                                self.currentStep,
                                                self.sampleTime,
                                                currentTime)
        ## Prediction step - Updating xp
        #  This section endeavors to obtain a prediction estimate from the dynamic
        #  model of your system directly from nonlinear equations
        xm = self.states[:, self.currentStep]
        xm = np.reshape(xm, [np.size(xm), 1])

        # Calculation before-state-limitations
        xv  = self.__class__._limitations(self.__class__, self.states, 0)
        
        # The handle function below is employed in the following
        handleDyn = lambda xx: self.__class__._dynamics(self.__class__,
                                                        xx,
                                                        self.inputs,
                                                        self.currentStep,
                                                        self.sampleTime,
                                                        currentTime)
        # Compute states and outputs using system dynamics
        if self.__class__.timeType == 'c':
            # Change the solver type in the block class instead of modifying the code here
            xp = super().dynamicRunner(handleDyn, xv, xm, self.sampleTime, self.solverType)
        else:
            # For discrete-time systems, solve the dynamic as shown below
            xp = handleDyn(xv)

        # Apply after-state limitations
        xp = self.__class__._limitations(self.__class__, xp, 1)
        
        # Prediction step - update covariance matrix
        Pp = A.dot(self.covariance).dot(A.transpose()) + \
             L.dot(self.qMatrix).dot(L.transpose())
        
        # Posterior step - receive measurements
        # If there are no measurements (y == NaN), only the prediction will be reported
        if not np.any(np.isnan(y)):
            # Calculate Kalman Gain
            K  = (Pp.dot(H.transpose())).dot(                  \
                  np.linalg.inv(H.dot(Pp).dot(H.transpose()) + \
                  M.dot(self.rMatrix).dot(M.transpose())) )
            # Update states
            xm = xp + K.dot(y - H.dot(xp))
            # Update covariance matrix
            Pm = (np.eye(self.covariance.shape[0]) - K.dot(H)).dot(Pp)
        else:
            xm = xp
            Pm = Pp
        
        ## Update internal signals
        self.states[:, self.currentStep + 1] = xm.flatten() # Save estimated states
        self.covariance                      = Pm           # Save covariance matrix
        self.currentStep += 1                               # Move to next step
    
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
        
        ## Initialize parameters - STEP 0 & 1
        # Calculate current time
        currentTime = self.timeLine[0, self.currentStep]
        # Prepare and save inputs and outputs to internal variables
        self.inputs[:, self.currentStep]  = u
        self.outputs[:, self.currentStep] = y
        y = np.reshape(y, [np.size(y), 1])
        # Calculate Jacobians using system dynamics
        [A, L, H, M] = self.__class__._jacobians(self.__class__,
                                                self.states,
                                                self.inputs,
                                                self.currentStep,
                                                self.sampleTime,
                                                currentTime)
        # Get last states prior and its covariance
        xm = self.states[:, self.currentStep]
        xm = np.reshape(xm, [np.size(xm), 1])
        Pm = self.covariance
        
        # Solve sigma points - STEP 2
        # Calculate square root
        dSigma = np.sqrt(self.nUKF + self.lambd)*((np.linalg.cholesky(Pm)).transpose())
        # Copy 'xm' to some column
        xmCopy = xm[:, np.int8(np.zeros([1, np.size(xm)]).flatten())]
        # Obtain sigma points
        sp = np.concatenate((xm, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        ## Predict states and their covariance - STEP 3
        # Obtain prediction estimate from dynamic model of system using nonlinear equations
        nSpoints = sp.shape[1]
        xp       = np.zeros([self.__class__.numStates, 1])
        Xp       = np.zeros([self.__class__.numStates, nSpoints])
        for i in range(0, nSpoints):
            changedFullState = self.states
            changedFullState[:, self.currentStep] = sp[:, i]
            
            # Apply before-state limitations
            xv  = self.__class__._limitations(self.__class__, changedFullState, 0)
            # Use handle function below to prevent redundancy
            handleDyn = lambda xx: self.__class__._dynamics(self.__class__,
                                                            xx,
                                                            self.inputs,
                                                            self.currentStep,
                                                            self.sampleTime,
                                                            currentTime)
            if self.__class__.timeType == 'c':
                Xp[:,i] = super().dynamicRunner(handleDyn,
                                                xv,
                                                xm,
                                                self.sampleTime,
                                                self.solverType).flatten()
            else:
                Xp[:,i] = handleDyn(xv)
            
            # Apply after-state limitations
            Xp[:,i] = self.__class__._limitations(self.__class__, Xp[:,i], 1)
            # Update prediction
            temp1 = Xp[:, i]
            xp = xp + self.wm[i,0]*(np.reshape(temp1, [np.size(temp1), 1]))
        # End of loop

        dPp = Xp - xp[:, np.int8(np.zeros([1, np.size(nSpoints)])).flatten()]
        # Update covariance of states matrix
        Pp  = dPp.dot(np.diag(self.wc.flatten())).dot(dPp.transpose()) + \
              L.dot(self.qMatrix).dot(L.transpose())
        
        ## Update sigma points - STEP 4
        # dSigma = np.sqrt(self.nUKF + self.lambd).dot( \
        #         (np.linalg.cholesky(Pp)).transpose()) # Calculate square root
        # Putting 'xp' is some column (copy)
        # xmCopy = xp[:, np.int8(np.zeros([1, np.size(xp)]))]
        # sp     = np.concatenate((xp, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        if not np.any(np.isnan(y)):
            ## Solve output estimation using predicted data - STEP 5
            # Obtain prediction output from sigma points
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
            # End of loop

            dSt = Zb - zb[:, np.int8(np.zeros([1, np.size(nSpoints)])).flatten()]
            # Update covariance of output matrix
            St  = dSt.dot(np.diag(self.wc.flatten())).dot(dSt.transpose()) + \
                  M.dot(self.rMatrix).dot(M.transpose())

            ## Solve Kalman gain - STEP 6
            SiG = dPp.dot(np.diag(self.wc.flatten())).dot(dSt.transpose())
            # Calculate Kalman Gain
            K   = SiG.dot(np.linalg.inv(St))
        
        ## Solve posterior using measurement data - STEP 7
        # If there are no measurements (y == NaN), only the prediction will be reported
        if not np.any(np.isnan(y)):
            # Update states
            xm = xp + K.dot(y - zb)
            # Update covariance matrix
            Pm = Pp - K.dot(SiG.transpose())
        else:
            xm = xp
            Pm = Pp
        
        ## Update internal signals
        self.states[:, self.currentStep + 1] = xm.flatten() # Save estimated states
        self.covariance                      = Pm           # Save covariance matrix
        self.currentStep += 1                               # Move to next step
    # End of function

    # Function to jump in the step variable
    # If no arguments are provided, jump 1 step
    def __iadd__(self, i = 1):
        '''
        ### Overview:
        This function can make a jump in the step number variable.

        ### Input variables:
        * how many steps you would like me to jump?; default is `1`
        '''
        self.currentStep = self.currentStep + i
    
    # Reset Block by changing current step to zero
    def reset(self):
        '''
        ### Overview:
        Reseting the block via changing the current step to zero.
        '''
        self.currentStep = 0
        
    # The below function is used to plot the internal signals
    def show(self, params, sel = 'x', **kwargs):
        '''
        ### Overview:
        This function makes a quick plot of internal signals.

        ### input variables:
        * `params`
        * The signal to be shown (`x`, `y`, or `u`); default is `x`

        ### Options:
            * `select` - choose signals arbitrarily; e.g., `select=[0,2,6]`
            * `derive` - get derivatives of signals; can be used in different ways:
                * `derive=False` or `derive=True`; default is `False`
                * `derive=[1,1,0]` - get derivatives of selected signals; set to `1` or `True` for signals you want to derive
            * `notime` - remove time and create timeless plots; can be set in different ways:
                * `notime=[0,1]` or `notime=[0,1,2]` - create 2D or 3D plots of signals; numbers are signal indices
                * `notime=[[0,1], [1,2]]` or `notime=[[0,1,2], [3,0,1]]` - create 2D or 3D plots of different signal groups; numbers are signal indices
            * `save` - name of file to save plot as; can be `image.png/pdf/jpg` or `True` to choose automatically
            * `xlabel`, `ylabel`, and `zlabel` - titles for x, y, and z axes of plot
            * `legend` - control legend display:
                * `legend=True` and `legend=False` - enable and disable legend
                * `legend='title'` - enable legend with specified title
            * `lineWidth` - set line width
            * `grid` - enable grid on plot (`True` or `False`)
            * `legCol` - control number of columns in legend (positive integer)
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
    # End of function
    
    def __repr__(self):
        return f"** {self.__class__.__name__} **\nCurrent point: {self.currentStep}/{self.numSteps}\nSolver Type: '{self.solverType}'"
# End of class

