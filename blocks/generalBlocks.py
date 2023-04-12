
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Import initialize classes
from core.lib.pyRequirment import *

class quadrupleTank():
    '''
    ## Quadruple tanks model
    This model is given from the below sourc:
    K. H. Johansson, "The quadruple-tank process: a multivariable laboratory
    process with an adjustable zero," in IEEE Transactions on Control Systems Technology,
    vol. 8, no. 3, pp. 456-465, May 2000, doi: 10.1109/87.845876
    '''

    # This name will be showed as its plot titles
    name          = 'Quadruple-tank process'
    numStates     = 4                # Number of states
    numInputs     = 2                # Number of inputs
    numOutputs    = 2                # Number of outputs
    timeType      = 'c'              # 'c' -> Continuous, 'd' -> Discrete
    solverType    = 'Euler'          # 'Euler', 'Runge'
    initialStates = np.ones([4,1])   # Initial value of states
    
    # EXTENTED KALMAN FILTER --------
    covariance = 1e+3*np.eye(4)   # Covariance of states
    qMatrix    = np.eye(4)*2e-1   # Dynamic noise variance
    rMatrix    = np.eye(2)*1e0    # Measurement noise variance
    
    # UNSKENTED KALMAN FILTER -------
    # Note that 'Extended KF' parameters is also useful for 'Unscented KF', 
    # so just put your values there, as well.
    kappa = 80       # A non-negative real number
    alpha = 0.2      # a \in (0, 1]

    # Other variables
    a1  = 0.071     # cm^2
    a2  = 0.057     # cm^2
    a3  = 0.071     # cm^2
    a4  = 0.057     # cm^2
    A1  = 28        # cm^2
    A2  = 32        # cm^2
    A3  = 28        # cm^2
    A4  = 32        # cm^2
    g   = 981       # cm/s^2
    k1  = 3.33      # cm^3/Vs --- (3.14) is also possible
    k2  = 3.35      # cm^3/Vs --- (3.29) is also possible
    ga1 = 0.7       # (0.43) is also possible
    ga2 = 0.6       # (0.34) is also possible
    kc  = 0.5       # V/cm

    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def dynamics(self, x, u, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        dx      = np.zeros([4, 1])
        dx[0,0] = -self.a1/self.A1*np.sqrt(2*self.g*x[0, k]) + \
                   self.a3/self.A1*np.sqrt(2*self.g*x[2, k]) + \
                   self.ga1*self.k1/self.A1*u[0, k]
        dx[1,0] = -self.a2/self.A2*np.sqrt(2*self.g*x[1, k]) + \
                   self.a4/self.A2*np.sqrt(2*self.g*x[3, k]) + \
                   self.ga2*self.k2/self.A2*u[1, k]
        dx[2,0] = -self.a3/self.A3*np.sqrt(2*self.g*x[2, k]) + \
                   (1 - self.ga2)*self.k2/self.A3*u[1, k]
        dx[3,0] = -self.a4/self.A4*np.sqrt(2*self.g*x[3, k]) + \
                   (1 - self.ga1)*self.k1/self.A4*u[0, k]
        return dx

    ## Measurement functions
    #  ~~> y = g(x,u)
    def measurements(self, x, u, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        y      = np.zeros([2, 1])
        y[0,0] = self.kc*x[0, k]
        y[1,0] = self.kc*x[1, k]
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
            x = np.maximum(x, 0)
        return x

    ## Jacobians
    #  ~~> d(A,B,C,D)/d(x,u)
    def jacobians(self, x, u, k, st, t):
        # [A, L, H, M] <- (Parameters, States, Inputs, Current step, Sample-time, Current time)
        # INSTRUCTION:
        #   dx = Ax + Lw,     'x' is states and 'w' denotes the process noise
        #   y  = Hx + Mv,     'x' is states and 'v' is the measurement noise

        # Preventing to happen zero
        epssilon = 1e-3
        x[:, k]  = np.maximum(x[:, k], epssilon)
        # A matrix, d(q(t))/dx(t)
        A      = np.zeros([4, 4])
        A[0,0] = -((self.a1/self.A1)*np.sqrt(2*self.g))/(2*np.sqrt(x[0, k]))
        A[0,1] = 0
        A[0,2] = +((self.a3/self.A1)*np.sqrt(2*self.g))/(2*np.sqrt(x[2, k]))
        A[0,3] = 0
        A[1,0] = 0
        A[1,1] = -((self.a2/self.A2)*np.sqrt(2*self.g))/(2*np.sqrt(x[0, k]))
        A[1,2] = 0
        A[1,3] = +((self.a4/self.A2)*np.sqrt(2*self.g))/(2*np.sqrt(x[3, k]))
        A[2,0] = 0
        A[2,1] = 0
        A[2,2] = -((self.a3/self.A3)*np.sqrt(2*self.g))/(2*np.sqrt(x[0, k]))
        A[2,3] = 0
        A[3,0] = 0
        A[3,1] = 0
        A[3,2] = 0
        A[3,3] = -((self.a4/self.A4)*np.sqrt(2*self.g))/(2*np.sqrt(x[0, k]))
        A      = np.real(A)
        # L matrix, d(q(t))/dw(t), Process noise effects
        L = np.eye(4)
        # H matrix, d(h(t))/dx(t)
        H      = np.zeros([2, 4])
        H[0,0] = self.kc
        H[1,1] = self.kc
        # M matrix, d(h(t))/dv(t), Measurement Noise effects
        M = np.eye(2)
        return A, L, H, M
    # The end of the function
# The end of the class


class Lorenz():
    '''
    ## Lorenz chaos model
    '''

    # This name will be showed as its plot titles
    name          = 'Lorenz Chaos'
    numStates     = 3                # The number of states
    numInputs     = 1                # The number of inputs
    numOutputs    = 2                # The number of outputs
    timeType      = 'c'              # 'c' -> Continuous, 'd' -> Discrete
    solverType    = 'euler'          # 'euler', 'rng4'
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

