
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import initialize classes
from core.lib.pyRequirment import *

class Lorenz():
    # Nonlinear Dynamic
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
    # 7) To use, put the needed below code in 'initialization.py' to set initial options
    #    7.1. To use for running a nonlinear system, use:
    #         models.THIS = nonlinear(THIS(), Time Line, initial=1, solver='')
    #    7.2. To use for estimation purposes, copy the below code:
    #         models.THISMODEL = estimator(THISMODEL(), signals.tLine, initial=1, solver='', approach='')
    # 8) Use the piece of code showed below in 'simulation.py' to apply each step
    #    8.1. To use for running a nonlinear system, use:
    #         models.THIS.nextstep(Input Signal, xNoise, yNoise)
    #    8.2. To use for estimation purposes, copy the below code:
    #         models.THIS.nextstep(u[:,k], y[:,k])
    # --------------------------------------------------------------------------

    # This name will be showed as its plot titles
    name          = 'Lorenz System (Chaos)'
    numStates     = 3                # The number of states
    numInputs     = 1                # The number of inputs
    numOutputs    = 2                # The number of outputs
    timeType      = 'c'              # 'c' -> Continuous, 'd' -> Discrete
    solverType    = 'Euler'          # 'Euler', 'Runge'
    initialStates = np.ones([3,1])   # Initial value of states
    
    # EXTENTED KALMAN FILTER --------
    covariance = 1e+3*np.eye(3)   # Covariance of states
    qMatrix    = np.eye(3)*1e0    # Dynamic noise variance
    rMatrix    = np.eye(2)*1e0    # Measurement noise variance
    
    # UNSKENTED KALMAN FILTER -------
    # Note that 'Extended KF' parameters is also useful for 'Unscented KF', 
    # so just put your values there, as well.
    kappa = 80       # A non-negative real number
    alpha = 0.2      # a \in (0, 1]

    # Other variables
    sigma = 10
    ro    = 28
    beta  = 8/3
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def dynamics(self, x, u, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        dx      = np.zeros([3, 1])
        dx[0,0] = self.sigma*x[1, k] - self.sigma*x[0, k] + u[0, k]
        dx[1,0] = self.ro*x[0, k] - x[0, k]*x[2, k] - x[1, k]
        dx[2,0] = x[0, k]*x[1, k] - self.beta*x[2, k]
        return dx

    ## Measurement functions 
    #  ~~> y = g(x,u)
    def measurements(self, x, u, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        y      = np.zeros([2, 1])
        y[0,0] = x[0, k]
        y[1,0] = x[1, k]
        return y
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def limitations(self, x, mode):
        # Self, States, Mode
        if mode == 0:
            # before updating states
            pass
        elif mode == 1:
            # After updating states
            pass
        return x
        
    ## Jacobians
    #  ~~> d(A,B,C,D)/d(x,u)
    def jacobians(self, x, u, k, st, t):
        # [A, L, H, M] <- (Parameters, States, Inputs, Current step, Sample-time, Current time)
        # INSTRUCTION:
        #   dx = Ax + Lw,     'x' is states and 'w' denotes the process noise
        #   y  = Hx + Mv,     'x' is states and 'v' is the measurement noise
        
        # A matrix, d(q(t))/dx(t)
        A      = np.zeros([3, 3])
        A[0,0] = -self.sigma
        A[0,1] = +self.sigma
        A[0,2] = 0
        A[1,0] = self.ro - x[2, k]
        A[1,1] = -1
        A[1,2] = -x[0, k]
        A[2,0] = x[1, k]
        A[2,1] = x[0, k]
        A[2,2] = -self.beta
        # L matrix, d(q(t))/dw(t), Process noise effects
        L = np.eye(3)
        # H matrix, d(h(t))/dx(t)
        H = np.eye(2, 3)
        # M matrix, d(h(t))/dv(t), Measurement Noise effects
        M = np.eye(2)
        return A, L, H, M
    # The end of the function
# The end of the class

