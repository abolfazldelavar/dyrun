
## // --------------------------------------------------------------
#    ***DYNAMIC RUNNER***
#    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
#    Web: https://github.com/abolfazldelavar/dyrun
## // --------------------------------------------------------------

# Import initialize classes
from core.lib.required_libraries import *
from core.caller.dynamic_engine import *

class QuadrupleTank(Nonlinear):
    '''
    ### Quadruple tanks model
    This model is given from the below sourc:
    K. H. Johansson, "The quadruple-tank process: a multivariable laboratory
    process with an adjustable zero," in IEEE Transactions on Control Systems Technology,
    vol. 8, no. 3, pp. 456-465, May 2000, doi: 10.1109/87.845876
    '''

    # This name will be showed as its plot titles
    name = 'Quadruple-tank process'
    n_states = 4 # Number of states
    n_inputs = 2 # Number of inputs
    n_outputs = 2 # Number of outputs
    time_type = 'c' # 'c' -> Continuous, 'd' -> Discrete
    solver_type = 'euler' # 'euler', 'rng4'
    initial_states = [0, 0, 0, 0] # Initial value of states
    
    # EXTENTED KALMAN FILTER --------
    covariance = 1e+3*np.eye(4) # Covariance of states
    q_atrix = np.eye(4)*2e-1 # Dynamic noise variance
    r_matrix = np.eye(2)*1e0 # Measurement noise variance
    
    # UNSKENTED KALMAN FILTER -------
    # Note that 'Extended KF' parameters is also useful for 'Unscented KF', 
    # so just put your values there, as well.
    kappa = 80 # A non-negative real number
    alpha = 0.2 # a \in (0, 1]

    # Other variables
    mp_a1 = 0.071 # cm^2
    mp_a2 = 0.057 # cm^2
    mp_a3 = 0.071 # cm^2
    mp_a4 = 0.057 # cm^2
    mp_A1 = 28 # cm^2
    mp_A2 = 32 # cm^2
    mp_A3 = 28 # cm^2
    mp_A4 = 32 # cm^2
    mp_g = 981 # cm/s^2
    mp_k1 = 3.33 # cm^3/Vs --- (3.14) is also possible
    mp_k2 = 3.35 # cm^3/Vs --- (3.29) is also possible
    mp_ga1 = 0.43 # (0.43) is also possible
    mp_ga2 = 0.34 # (0.34) is also possible
    mp_kc = 0.5 # V/cm

    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, input_signal, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        dx      = np.zeros([4, 1])
        dx[0,0] = -self.mp_a1/self.mp_A1*np.sqrt(2*self.mp_g*x[0, k]) + \
                   self.mp_a3/self.mp_A1*np.sqrt(2*self.mp_g*x[2, k]) + \
                   self.mp_ga1*self.mp_k1/self.mp_A1*input_signal[0, k]
        dx[1,0] = -self.mp_a2/self.mp_A2*np.sqrt(2*self.mp_g*x[1, k]) + \
                   self.mp_a4/self.mp_A2*np.sqrt(2*self.mp_g*x[3, k]) + \
                   self.mp_ga2*self.mp_k2/self.mp_A2*input_signal[1, k]
        dx[2,0] = -self.mp_a3/self.mp_A3*np.sqrt(2*self.mp_g*x[2, k]) + \
                   (1 - self.mp_ga2)*self.mp_k2/self.mp_A3*input_signal[1, k]
        dx[3,0] = -self.mp_a4/self.mp_A4*np.sqrt(2*self.mp_g*x[3, k]) + \
                   (1 - self.mp_ga1)*self.mp_k1/self.mp_A4*input_signal[0, k]
        return dx

    ## Measurement functions
    #  ~~> y = g(x,u)
    def _measurements(self, x, input_signal, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        y      = np.zeros([2, 1])
        y[0,0] = self.mp_kc*x[0, k]
        y[1,0] = self.mp_kc*x[1, k]
        return y
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def _limitations(self, x, mode):
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
    def _jacobians(self, x, input_signal, k, st, t):
        # [A, L, H, M] <- (Parameters, States, Inputs, Current step, Sample-time, Current time)
        # INSTRUCTION:
        #   dx = Ax + Lw,     'x' is states and 'w' denotes the process noise
        #   y  = Hx + Mv,     'x' is states and 'v' is the measurement noise

        # Preventing to happen zero
        epssilon = 1e-3
        x[:, k]  = np.maximum(x[:, k], epssilon)
        # Matrix A, d(q(t))/dx(t)
        A = np.zeros([4, 4])
        A[0,0] = -((self.mp_a1/self.mp_A1)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[0, k]))
        A[0,1] = 0
        A[0,2] = +((self.mp_a3/self.mp_A1)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[2, k]))
        A[0,3] = 0
        A[1,0] = 0
        A[1,1] = -((self.mp_a2/self.mp_A2)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[1, k]))
        A[1,2] = 0
        A[1,3] = +((self.mp_a4/self.mp_A2)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[3, k]))
        A[2,0] = 0
        A[2,1] = 0
        A[2,2] = -((self.mp_a3/self.mp_A3)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[2, k]))
        A[2,3] = 0
        A[3,0] = 0
        A[3,1] = 0
        A[3,2] = 0
        A[3,3] = -((self.mp_a4/self.mp_A4)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[3, k]))
        A      = np.real(A)
        # Matix B, input to state
        B = np.zeros([4, 2])
        B[0,0] = self.mp_ga1*self.mp_k1/self.mp_A1
        B[0,1] = 0
        B[1,0] = 0
        B[1,1] = self.mp_ga2*self.mp_k2/self.mp_A2
        B[2,0] = 0
        B[2,1] = (1 - self.mp_ga2)*self.mp_k2/self.mp_A3
        B[3,0] = (1 - self.mp_ga1)*self.mp_k1/self.mp_A4
        B[3,1] = 0
        B = np.real(B)
        # H matrix, d(h(t))/dx(t)
        H = np.zeros([2, 4])
        H[0,0] = self.mp_kc
        H[1,1] = self.mp_kc
        # Matrix D, input to output
        D = np.zeros([2, 2])
        # Matrix L, d(q(t))/dw(t), Process noise effects
        L = np.eye(4)
        # Matrix M, d(h(t))/dv(t), Measurement Noise effects
        M = np.eye(2)
        return A, B, H, D, L, M
    # End of function
# End of class

class TwoTanks(Nonlinear):
    '''
    ### Two tanks model
    '''

    # This name will be showed as its plot titles
    name = 'Two-tank process'
    n_states = 2 # Number of states
    n_inputs = 2 # Number of inputs
    n_outputs = 1 # Number of outputs
    time_type = 'c' # 'c' -> Continuous, 'd' -> Discrete
    solver_type = 'euler' # 'euler', 'rng4'
    initial_states = [0, 0] # Initial value of states
    
    # EXTENTED KALMAN FILTER --------
    covariance = 1e+3*np.eye(2) # Covariance of states
    q_atrix = np.eye(2)*2e-1 # Dynamic noise variance
    r_matrix = np.eye(2)*1e0 # Measurement noise variance
    
    # UNSKENTED KALMAN FILTER -------
    # Note that 'Extended KF' parameters is also useful for 'Unscented KF', 
    # so just put your values there, as well.
    kappa = 80 # A non-negative real number
    alpha = 0.2 # a \in (0, 1]

    # Other variables
    mp_a1 = 0.071 # cm^2
    mp_a2 = 0.057 # cm^2
    mp_a21 = 0.06 # cm^2
    mp_a12 = 0.04 # cm^2
    mp_A1 = 28 # cm^2
    mp_A2 = 32 # cm^2
    mp_g = 981 # cm/s^2
    mp_ga1 = 0.43 # (0.43) is also possible
    mp_ga2 = 0.34 # (0.34) is also possible
    mp_C = np.array([[1, 0], [2, 1]])

    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, input_signal, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        dx      = np.zeros([2, 1])
        dx[0,0] = -self.mp_a1/self.mp_A1*np.sqrt(2*self.mp_g*x[0, k]) + \
                   self.mp_a21/self.mp_A1*np.sqrt(2*self.mp_g*x[1, k]) + \
                   self.mp_ga1/self.mp_A1*input_signal[0, k]
        dx[1,0] = -self.mp_a2/self.mp_A2*np.sqrt(2*self.mp_g*x[1, k]) + \
                   self.mp_a12/self.mp_A2*np.sqrt(2*self.mp_g*x[0, k]) + \
                   self.mp_ga2/self.mp_A2*input_signal[1, k]
        return dx

    ## Measurement functions
    #  ~~> y = g(x,u)
    def _measurements(self, x, input_signal, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        x_k = x[:, k].reshape(-1,1)
        return self.mp_C @ x_k
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def _limitations(self, x, mode):
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
    def _jacobians(self, x, input_signal, k, st, t):
        # [A, L, H, M] <- (Parameters, States, Inputs, Current step, Sample-time, Current time)
        # INSTRUCTION:
        #   dx = Ax + Lw,     'x' is states and 'w' denotes the process noise
        #   y  = Hx + Mv,     'x' is states and 'v' is the measurement noise

        # Preventing to happen zero
        epssilon = 1e-3
        x[:, k]  = np.maximum(x[:, k], epssilon)
        # Matrix A, d(q(t))/dx(t)
        A = np.zeros([2, 2])
        A[0,0] = -((self.mp_a1/self.mp_A1)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[0, k]))
        A[0,1] = +((self.mp_a21/self.mp_A1)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[1, k]))
        A[1,0] = +((self.mp_a12/self.mp_A2)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[0, k]))
        A[1,1] = -((self.mp_a2/self.mp_A2)*np.sqrt(2*self.mp_g))/(2*np.sqrt(x[1, k]))
        A      = np.real(A)
        # Matix B, input to state
        B = np.zeros([2, 2])
        B[0,0] = self.mp_ga1/self.mp_A1
        B[0,1] = 0
        B[1,0] = 0
        B[1,1] = self.mp_ga2/self.mp_A2
        B = np.real(B)
        # H matrix, d(h(t))/dx(t)
        H = self.mp_C
        # Matrix D, input to output
        D = np.zeros([2, 2])
        # Matrix L, d(q(t))/dw(t), Process noise effects
        L = np.eye(2)
        # Matrix M, d(h(t))/dv(t), Measurement Noise effects
        M = np.eye(2)
        return A, B, H, D, L, M
    # End of function
# End of class


class Lorenz(Nonlinear):
    '''
    ### Lorenz chaos model
    '''

    # This name will be showed as its plot titles
    name = 'Lorenz Chaos'
    n_states = 3 # The number of states
    n_inputs = 1 # The number of inputs
    n_outputs = 2 # The number of outputs
    time_type = 'c' # 'c' -> Continuous, 'd' -> Discrete
    solver_type = 'euler' # 'euler', 'rng4'
    initial_states = [1, 1, 1] # Initial value of states
    
    # EXTENTED KALMAN FILTER --------
    covariance = 1e+3*np.eye(3) # Covariance of states
    q_matrix = np.eye(3)*1e0 # Dynamic noise variance
    r_matrix = np.eye(2)*1e0 # Measurement noise variance
    
    # UNSKENTED KALMAN FILTER -------
    # Note that 'Extended KF' parameters is also useful for 'Unscented KF', 
    # so just put your values there, as well.
    kappa = 80       # A non-negative real number
    alpha = 0.2      # a \in (0, 1]

    # Other variables
    mp_sigma = 10
    mp_ro    = 28
    mp_beta  = 8/3
    
    ## This part is internal dynamic functions that represents
    #  internal relations between states and inputs
    #  ~~> dx = f(x,u)
    def _dynamics(self, x, input_signal, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        dx      = np.zeros([3, 1])
        dx[0,0] = self.mp_sigma*x[1, k] - self.mp_sigma*x[0, k] + input_signal[0, k]
        dx[1,0] = self.mp_ro*x[0, k] - x[0, k]*x[2, k] - x[1, k]
        dx[2,0] = x[0, k]*x[1, k] - self.mp_beta*x[2, k]
        return dx

    ## Measurement functions 
    #  ~~> y = g(x,u)
    def _measurements(self, x, input_signal, k, st, t):
        # Parameters, States, Inputs, Current step, Sample-time, Current time
        y      = np.zeros([2, 1])
        y[0,0] = x[0, k]
        y[1,0] = x[1, k]
        return y
    
    ## All limitations before and after the state updating
    #  It can be useful for systems which have rules
    def _limitations(self, x, mode):
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
    def _jacobians(self, x, input_signal, k, st, t):
        # [A, L, H, M] <- (Parameters, States, Inputs, Current step, Sample-time, Current time)
        # INSTRUCTION:
        #   dx = Ax + Lw,     'x' is states and 'w' denotes the process noise
        #   y  = Hx + Mv,     'x' is states and 'v' is the measurement noise
        
        # A matrix, d(q(t))/dx(t)
        A      = np.zeros([3, 3])
        A[0,0] = -self.mp_sigma
        A[0,1] = +self.mp_sigma
        A[0,2] = 0
        A[1,0] = self.mp_ro - x[2, k]
        A[1,1] = -1
        A[1,2] = -x[0, k]
        A[2,0] = x[1, k]
        A[2,1] = x[0, k]
        A[2,2] = -self.mp_beta
        # L matrix, d(q(t))/dw(t), Process noise effects
        L = np.eye(3)
        # H matrix, d(h(t))/dx(t)
        H = np.eye(2, 3)
        # M matrix, d(h(t))/dv(t), Measurement Noise effects
        M = np.eye(2)
        return A, B, H, D, L, M
    # End of function
# End of class

