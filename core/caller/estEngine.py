
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import initial classes
from core.lib.pyRequirment import *
from core.lib.coreLib  import solverCore
from core.caller.scopeEngine import scope

class estimator():
    # The runner of nonlinear dynamic estimators
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
    
    ## Initial function run to set default properties and make initial signals
    def __init__(self, inputModel, timeline, **kwargs):
        # [Save internal] <- (input model class, Sample time, Time line, Options)
        # Options are: 'InitialCondition', 'Approach'
        self.block         = inputModel             # Get a copy of your model class
        self.timeLine      = np.reshape(timeline, [1, np.size(timeline)])
        self.sampleTime    = np.mean(self.timeLine[0, 1:-1] - self.timeLine[0, 0:-2])
        self.numSteps      = np.size(self.timeLine) # The number of all time steps
        self.numStates     = self.block.numStates   # The number of states
        self.numInputs     = self.block.numInputs   # The number of inputs
        self.numOutputs    = self.block.numOutputs  # The number of measurements
        self.inputs        = np.zeros([self.numInputs,  self.numSteps])
        self.outputs       = np.zeros([self.numOutputs, self.numSteps])
        self.currentStep   = 0                      # The current step of simulation
        self.initialStates = self.block.initialStates
        self.states        = np.zeros([self.numStates, self.numSteps + 1])
        self.states[:, 0]  = self.initialStates.flatten()
        self.covariance    = self.block.covariance
        self.estAproach    = 'EKF';                 # The estimation approach ('EKF', 'UKF', ...)
        self.solverType    = self.block.solverType  # The type of dynamic solver
        
        # Extracting the arbitraty value of properties
        for key, val in kwargs.items():
            # 'initial' implies the initial condition of states
            if key == 'initial': self.states[:, 0] = val
            # The approach of estimation
            if key == 'approach': self.estAproach = val
            # The type of solver
            if key == 'solver': self.solverType = val
        
        # This part initialize the estimator by setting parameters
        if self.estAproach == 'EKF':
            self.qMatrix = self.block.qMatrix
            self.rMatrix = self.block.rMatrix
        elif self.estAproach == 'UKF':
            self.qMatrix = self.block.qMatrix
            self.rMatrix = self.block.rMatrix
            self.kappa   = self.block.kappa
            self.alpha   = self.block.alpha
            # Dependent variables
            self.nUKF    = self.numStates
            self.lambd   = np.power(self.alpha,2)*(self.nUKF + self.kappa) - self.nUKF
            self.betta   = 2
            # Making weights
            self.wm      = np.ones([2*self.nUKF + 1, 1])/(2*(self.nUKF + self.lambd))
            self.wc      = self.wm
            self.wc[0,0] = self.wm[0,0] + (1 - np.power(self.alpha,2) + self.betta)
    
    # The 'nextstep' function can provide an easy way to call
    # dydnamics of the system to calculate next sample states
    # To use this function, refer to the top INSTRUCTION part
    def nextstep(self, u, y):
        if self.estAproach == 'EKF':
            self._nextstepEKF(u, y)
        elif self.estAproach == 'UKF':
            self._nextstepUKF(u, y)

    ## Next step of Extended Kalman Filter (EKF)
    def _nextstepEKF(self, u, y):
        # [Internal update] <- (Internal, Input, Output)
        
        ## Initialize parameters
        # The current time is calculated as below
        currentTime = self.timeLine[0, self.currentStep]
        # Preparing and saving inputs and outputs to internal
        self.inputs[:, self.currentStep]  = u
        self.outputs[:, self.currentStep] = y
        y = np.reshape(y, [np.size(y), 1])
        
        # Using dynamics of system to calculate Jacobians
        [A, L, H, M] = self.block.jacobians(self.states,      \
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
        xv  = self.block.limitations(self.states, 0)
        
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
            # Change the solver type in the block class or its called order in 'initialization.py'
            xp = solverCore.dynamicRunner(handleDyn, xv, xm, self.sampleTime, self.solverType)
        else:
            # When the inserted system is discrete time, just the
            # dynamic must be solved as below
            xp = handleDyn(xv)

        # Set after-state-limitations
        xp = self.block.limitations(xp, 1)
        
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
    def _nextstepUKF(self, u, y):
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
        [A, L, H, M] = self.block.jacobians(self.states,      \
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
        dSigma = math.sqrt(self.nUKF + self.lambd)*((np.linalg.cholesky(Pm)).transpose())
        # Putting 'xm' is some column (copy)
        xmCopy = xm[:, np.int8(np.zeros([1, np.size(xm)]).flatten())]
        # Obtaining sigma points
        sp = np.concatenate((xm, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        ## Prediction states and their covariance, STEP 3
        #  This part tries to obtain a prediction estimate from dynamic
        #  model of your system directly from nonlinear equations
        nSpoints = sp.shape[1]
        xp       = np.zeros([self.numStates, 1])
        Xp       = np.zeros([self.numStates, nSpoints])
        for i in range(0, nSpoints):
            changedFullState = self.states
            changedFullState[:, self.currentStep] = sp[:, i]
            
            # Set before-state-limitations
            xv  = self.block.limitations(changedFullState, 0)
             # The below handle function is used in the following
            handleDyn = lambda xx: self.block.dynamics(xx,            \
                                                    self.inputs,      \
                                                    self.currentStep, \
                                                    self.sampleTime,  \
                                                    currentTime)
            if self.block.timeType == 'c':
                Xp[:,i] = solverCore.dynamicRunner(handleDyn,         \
                                                    xv,               \
                                                    xm,               \
                                                    self.sampleTime,  \
                                                    self.block.solverType).flatten()
            else:
                Xp[:,i] = handleDyn(xv)
            
            # Set after-state-limitations
            Xp[:,i] = self.block.limitations(Xp[:,i], 1)
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
            zb = np.zeros([self.numOutputs, 1])
            Zb = np.zeros([self.numOutputs, nSpoints])
            for i in range(0, nSpoints):
                changedFullState = self.states
                changedFullState[:, self.currentStep] = Xp[:, i] #Or 'Xp[:, i]' instead of 'sp[:, i]'
                Zb[:,i] = self.block.measurements(changedFullState,   \
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
