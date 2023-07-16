
## // --------------------------------------------------------------
#    ***DYNAMIC RUNNER***
#    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
#    Web: https://github.com/abolfazldelavar/dyrun
## // --------------------------------------------------------------

# Import initial classes
from core.lib.required_libraries import *
from core.lib.core_library import SolverCore, Clib, Structure
from core.caller.scope_engine import Scope

# Linear dynamics
class LtiGroup():
    def __init__(self, inserted_system, **kwargs):
        '''
        ### Overview:
        This class facilitates the creation of a collection of LTI systems that may also have interactions.
        It is important to note that linear systems or filters must be imported in the form of
        a `tf()` or `ss()` from `control.matlab` extension, or directly like `(A, B, C, D)` in discrete form.

        ### Input Parameters:
        * System; for example, `tf([1], [1,2,3])` or `(A, B, C, D)`

        ### Configuration Options:
        * `sample_time`: Defines the sample time of the simulation; default is `1`
        * `initial`: Specifies the initial state of the system
        * `replicate`: Determines the number of components; default value is `1`
        * `delay`: Defines the input delay in step scale; for example, `18` steps
        * `newest`: The outputs are calculate from the predicted states or the current ones; default
        is `False` which means the current states
        * `name`: Defines a name for the object
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        self.inserted_system = inserted_system
        initial_condition = [0]
        time_delay = 0
        n_systems = 1
        name = 'LTI system'
        sample_time = 1
        self.newest = False
        for key, val in kwargs.items():
            if key == 'initial': initial_condition = val
            if key == 'delay': time_delay = val
            if key == 'replicate': n_systems = val
            if key == 'name': name = val
            if key == 'sample_time': sample_time = val
            if key == 'newest': self.newest = val
            
        if isinstance(inserted_system, tuple):
            self.system = Structure()
            self.system.A = inserted_system[0]
            self.system.B = inserted_system[1]
            self.system.C = inserted_system[2]
            self.system.D = inserted_system[3]
            if np.array(inserted_system[3]).all() == 0:
                self.system.D = np.zeros((np.size(self.system.C, 0), np.size(self.system.B, 1)))
        else:
            if inserted_system.dt == sample_time and type(inserted_system) == LinearIOSystem:
                # The input is a discrete-time state-space system with a consistent sample time.
                self.system = inserted_system
            elif inserted_system.dt == sample_time and type(inserted_system) == TransferFunction:
                # The input is a discrete-time transfer function with a consistent sample time.
                self.system = minreal(tf2ss(inserted_system))
            elif type(inserted_system) == LinearIOSystem:
                if inserted_system.dt != 0:
                    # The input is a discrete-time state-space system with a different sample time.
                    # MATLAB code is: self.system = d2d(inserted_system, sample_time);
                    pass
                else:
                    # The input is a continuous-time state-space system.
                    self.system = c2d(inserted_system, sample_time)
            elif inserted_system.dt != 0:
                # The input is a discrete-time transfer function with a different sample time.
                # MATLAB code is: self.system = d2d(minreal(ss(inserted_system)), sample_time);
                pass
            else:
                # The input is a continuous-time transfer function.
                self.system = c2d(ss(inserted_system), sample_time)

        self.name = name
        self.number_ltis = n_systems # The number of LTI systems
        self.sample_time = sample_time # Simulation sample time
        self.A = np.array(self.system.A) # Dynamic matrix A
        self.B = np.array(self.system.B) # Dynamic matrix B
        self.C = np.array(self.system.C) # Dynamic matrix C
        self.D = np.array(self.system.D) # Dynamic matrix D
        self.n_states = self.system.A.shape[0] # The number of states
        self.n_inputs = self.system.B.shape[1] # The number of inputs
        self.n_outputs = self.system.C.shape[0] # The number of measurements
        self.delay = time_delay # Delay steps
        self.inputs = np.zeros([self.n_inputs , self.number_ltis, self.delay + 1])
        self.outputs = np.zeros([self.n_outputs, self.number_ltis])
        self.states = np.zeros([self.n_states, self.number_ltis])
        
        # If the initial input does not exist, set it to zero.
        # Otherwise, put the initial condition in the state matrix.
        initial_condition = np.array(initial_condition)
        inish = initial_condition.shape
        if sum(inish) == self.n_states or sum(inish) == self.n_states + 1:
            # If the imported initial value is not a column vector, reshape it.
            initial_condition = np.reshape(initial_condition, (-1, 1))
            self.states += 1
            self.states  = initial_condition*self.states
        elif initial_condition.size != 1:
            if initial_condition.shape == (self.n_states, self.number_ltis):
                self.states = initial_condition
            else:
                err_text = "The dimensions of the inserted initial value are incorrect. Please check it."
                logging.error(err_text)
                raise ValueError(err_text)
        # Comment and diary
        Clib.diary(f'{self.name} has been created.')

    def __repr__(self):
        return f"** {self.name} **\nNumber of elements: {self.number_ltis}"
    
    # Representation LaTeX form
    def _repr_latex_(self):
        ss_form  = r'\begin{align}'
        ss_form += r'x_{k' + ('' if self.newest == True else '+1') + '} &= '
        ss_form += Clib.latex_matrix(self.A, (6,6))
        ss_form += r'x_{k' + ('-1' if self.newest == True else '') + '} + '
        ss_form += Clib.latex_matrix(self.B, (6,6))
        ss_form += r'u_{k' + ('-1' if self.newest == True else '') + '} \\nonumber \\\\ '
        ss_form += r'y_k & = ' + Clib.latex_matrix(self.C, (6,6))
        ss_form += r'x_{k} + ' + Clib.latex_matrix(self.D, (6,6)) + r'u_{k} \nonumber'
        ss_form += r'\end{align}'
        
        properties  = r'\begin{aligned}'
        properties += r'&\text{System poles: } &&= ' + Clib.latex_matrix(np.linalg.eig(self.A)[0].reshape((1,-1))) + ' \\\\ '
        properties += r'\end{aligned}'
        
        table_c  = r'\begin{aligned}'
        table_c += f'&number\_ltis &&= {self.number_ltis} \\\\ '
        table_c += f'&sample\_time &&= {self.sample_time} \\\\ '
        table_c += f'&n\_states &&= {self.n_states} \\\\ '
        table_c += f'&n\_outputs &&= {self.n_outputs} \\\\ '
        table_c += f'&n\_inputs &&= {self.n_inputs} \\\\ '
        table_c += f'&delay &&= {self.delay} \\\\ '
        table_c += f'&newest &&= {self.newest} \\\\ '
        table_c += f'&states &&= {Clib.latex_matrix(self.states)} \\\\ '
        table_c += f'&outputs &&= {Clib.latex_matrix(self.outputs)} '
        table_c += r'\end{aligned}'
        
        text  = f"### {self.name}\n\n "
        text += f"#### Dynamic matrices: \n\n ${ss_form}$ \n\n "
        text += f"#### Dynamic properties: \n\n ${properties}$ \n\n "
        text += f"#### Variables: \n\n ${table_c}$"
        return text
    
    def predict(self, input_signal, x_noise = 0, y_noise = 0):
        '''
        ### Overview:
        This function can provide an easy way to call dydnamics of the system to calculate the next sample states.
        
        ### Input variables:
        * Input array at step `k`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Shifts the input signal to create a delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = input_signal
        
        # Updates the states using the state-space equation dx = Ax + Bu
        x = np.dot(self.A, self.states) + np.dot(self.B, self.inputs[:,:,0])
        
        # Calculates the outputs using the state-space equation y = Cx + Du
        y = np.dot(self.C, x if self.newest == 1 else self.states) + np.dot(self.D, self.inputs[:,:,0])
        
        # Updates internal signals
        self.states = x + x_noise
        self.outputs = y + y_noise
# End of class

# Linear Kalman Filter
class LinearKalmanFilter():
    def __init__(self, inserted_system, **kwargs):
        '''
        ### Overview:
        Linear Kalman filter to observe the states of an LIT system.
        It is important to note that linear systems or filters must be imported in the form of
        a `LTIgroup` class, or directly like `(A, B, C, D)` in discrete form.

        ### Input Parameters:
        * System; for example, `(A, B, C, D)`

        ### Configuration Options:
        * `name`: Defines a name for the object
        * `x_0`: Specifies the initial state
        * `P_0`: Denotes the initial covariance
        * `process_noise`: Sets the process noise covariacne `Q`
        * `measurement_noise`: Defines the measurement noise covariance `R`
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        self.inserted_system = inserted_system
        initial_condition = [0]
        initial_cov = [0]
        Q = [0]
        R = [0]
        name = 'LTI system'
        for key, val in kwargs.items():
            if key == 'x_0': initial_condition = val
            if key == 'P_0': initial_cov = val
            if key == 'process_noise': Q = val
            if key == 'measurement_noise': R = val
            if key == 'name': name = val
        
        self.system = Structure()
        if isinstance(inserted_system, tuple):
            self.system.A = inserted_system[0]
            self.system.B = inserted_system[1]
            self.system.C = inserted_system[2]
            self.system.D = inserted_system[3]
        else:
            self.system.A = inserted_system.A
            self.system.B = inserted_system.B
            self.system.C = inserted_system.C
            self.system.D = inserted_system.D
            
        if np.array(self.system.D).all() == 0:
            self.system.D = np.zeros((np.size(self.system.C, 0), np.size(self.system.B, 1)))

        self.name = name
        self.A = np.array(self.system.A) # Dynamic matrix A
        self.B = np.array(self.system.B) # Dynamic matrix B
        self.C = np.array(self.system.C) # Dynamic matrix C
        self.D = np.array(self.system.D) # Dynamic matrix D
        self.n_states = self.system.A.shape[0] # The number of states
        self.n_inputs = self.system.B.shape[1] # The number of inputs
        self.n_outputs = self.system.C.shape[0] # The number of measurements
        self.x_pr = np.zeros([self.n_states, 1]) # Priori
        self.x_ps = np.zeros([self.n_states, 1]) # Posteriori
        self.res = np.zeros([self.n_outputs, 1]) # Residual
        self.K = np.zeros([self.n_states, self.n_outputs]) # Kalman Gain
        self.P_pr = np.eye(self.n_states, self.n_states) # Priori covariance
        self.P_ps = np.eye(self.n_states, self.n_states)*1e3 # Posteriori covariance
        self.Q = np.eye(self.n_states, self.n_states)*1e-4 # Process noise covariance
        self.R = np.eye(self.n_outputs, self.n_outputs)*1e-4 # Measurement noise covariance
        
        # Adjusting the initial condition
        initial_condition = np.array(initial_condition).ravel()
        if initial_condition.size == self.n_states:
            self.x_ps = initial_condition.reshape((-1,1))
        # Setting the initial covariance
        initial_cov = np.array(initial_cov)
        if sum(initial_cov.shape) == 2*self.n_states:
            self.P_ps = initial_cov
        # Setting Q
        Q = np.array(Q)
        if sum(Q.shape) == 2*self.n_states:
            self.Q = Q
        # Setting R
        R = np.array(R)
        if sum(R.shape) == 2*self.n_outputs:
            self.R = R
        # Comment and diary
        Clib.diary(f'{self.name} has been created.')
        
    # Representation LaTeX form
    def _repr_latex_(self):            
        table_c  = r'\begin{aligned}'
        table_c += f'&n\_states &&= {self.n_states} \\\\ '
        table_c += f'&n\_outputs &&= {self.n_outputs} \\\\ '
        table_c += f'&n\_inputs &&= {self.n_inputs} \\\\ '
        table_c += f'&A &&= {Clib.latex_matrix(self.A, (6,6))} \\\\ '
        table_c += f'&B &&= {Clib.latex_matrix(self.B, (6,6))} \\\\ '
        table_c += f'&C &&= {Clib.latex_matrix(self.C, (6,6))} \\\\ '
        table_c += f'&D &&= {Clib.latex_matrix(self.D, (6,6))} \\\\ '
        table_c += f'&x\_pr &&= {Clib.latex_matrix(self.x_pr)} \\\\ '
        table_c += f'&x\_ps &&= {Clib.latex_matrix(self.x_ps)} \\\\ '
        table_c += f'&res &&= {Clib.latex_matrix(self.res)} \\\\ '
        table_c += f'&K &&= {Clib.latex_matrix(self.K)} \\\\ '
        table_c += f'&P\_pr &&= {Clib.latex_matrix(self.P_pr)} \\\\ '
        table_c += f'&P\_ps &&= {Clib.latex_matrix(self.P_ps)} \\\\ '
        table_c += f'&Q &&= {Clib.latex_matrix(self.Q)} \\\\ '
        table_c += f'&R &&= {Clib.latex_matrix(self.R)} '
        table_c += r'\end{aligned}'
        
        text  = f"### {self.name}\n\n "
        text += f"#### Variables: \n\n ${table_c}$"
        return text
    
    def predict(self, inputs, outputs):
        '''
        ### Overview:
        Making a prediction of the next step using the current data

        ### Input variables:
        * Input array at step `k`
        * Output array at step `k`
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        
        # Priori step
        self.x_pr = np.dot(self.A, self.x_ps) + np.dot(self.B, np.array(inputs).reshape((-1,1)))
        self.P_pr = self.A @ self.P_ps @ self.A.T + self.Q
        
        # Kalman gain
        self.K = self.P_pr @ self.C.T @ np.linalg.inv(self.C @ self.P_pr @ self.C.T + self.R)
        
        # Residual
        self.res = np.array(outputs).reshape((-1,1)) - np.dot(self.C, self.x_pr)
        
        # Posteriori step
        self.x_ps = self.x_pr + np.dot(self.K, self.res)
        self.P_ps = (np.eye(self.n_states) - self.K @ self.C) @ self.P_pr
    
    
# End of class

# Nonlinear dynamic group
class NonlinearGroup(SolverCore):
    def __init__(self, sample_time, **kwargs):
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
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        initial_condition = [0]
        time_delay = 0
        self.enable_synapses = False
        pre = np.array(0)
        post = np.array(0)
        n_neurons = 1
        solver_type = self.solver_type
        for key, val in kwargs.items():
            # 'initial' specifies the initial condition of the system
            if key == 'initial': initial_condition = val
            # 'delay' specifies the input delay in step scale
            if key == 'delay': time_delay = val
            # 'replicate' specifies the number of blocks
            if key == 'replicate': n_neurons = val
            # 'pre' specifies the pre numbers
            if key == 'pre':
                pre = val
                self.enable_synapses = True
            # 'post' specifies the post numbers
            if key == 'post': post = val
            # Specifies the dynamic solver type
            if key == 'solver': solver_type = val

        self.delay = time_delay # The input signals over time
        self.pre = pre # Pre
        self.post = post # Post
        self.sample_time = sample_time # The simulation sample time
        self.n_neurons = n_neurons # The number of neurons (Network size)
        self.solver_type = solver_type  # The type of dynamic solver
        self.initial_states = np.array(self.initial_states).reshape(-1, 1)
        self.synapses_current = np.zeros([self.n_synapses_signal, self.n_neurons])
        self.inputs = np.zeros([self.n_inputs,  self.n_neurons, self.delay + 1])
        self.outputs = np.zeros([self.n_outputs, self.n_neurons])
        self.states = np.ones([self.n_states,  self.n_neurons])
        self.states = self.initial_states*self.states
        
        # If the initial input does not exist, set it to zero. Otherwise, put the initial condition in the state matrix.
        initial_condition = np.array(initial_condition)
        inish = initial_condition.shape
        if sum(inish) == self.n_states or sum(inish) == self.n_states + 1:
            # If the imported initial value is not a column vector, reshape it.
            initial_condition = np.reshape(initial_condition, (-1, 1))
            self.states += 1
            self.states  = initial_condition*self.states
        elif sum(inish) != 1 and sum(inish) != 2:
            if inish == (self.n_states, self.n_neurons):
                self.states = initial_condition
            else:
                err_text = "The dimensions of the inserted initial value are incorrect. Please check it."
                logging.error(err_text)
                raise ValueError(err_text)
        # Comment & diary
        Clib.diary('"' + self.name + '" has been created.')
    
    def __repr__(self):
        return f"** {self.__name__} **\nNumber of elements: {self.n_neurons}\nSolver Type: '{self.solver_type}'"
    
    def predict(self, input_signal, **kwargs):
        '''
        ### Overview:
        This function can provide a `prediction` of the next step, using the current inputs.

        ### Input variables:
        * Input array at step `k`
        * Output control signal used in synapse calculations; the default value is `False`
        * Internal additive noise which is added to the states
        * External additive noise which is added to the measurements
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        ext_input = False
        x_noise = 0
        y_noise = 0
        for key, val in kwargs.items():
            if key == 'external_input': ext_input = val
            if key == 'x_noise': x_noise = val
            if key == 'y_noise': y_noise = val
            
        # Shifts the input signal by one time step to create a delayed input signal
        self.inputs = np.roll(self.inputs, -1, axis=2)
        self.inputs[:,:,-1] = input_signal
        system_input = self.inputs[:,:,0]

        # Applies limitations to the states before calculating the next states using the system dynamics
        x = self._limitations(self.states, 0)
        
        # Defines a handle function for calculating the system dynamics
        handle_dyn = lambda xx: self._dynamics(xx, system_input, self.synapses_current)
        
        # Calculates the states and outputs using the system dynamics
        if self.time_type == 'c':
            # For continuous-time systems, the solver type can be controlled by changing it in the model file or through `solver` option
            
            x = super().dynamic_runner(handle_dyn, x, x, self.sample_time, self.solver_type)
        else:
            # For discrete-time systems, only the dynamic must be solved
            x = handle_dyn(x)
        
        # Enforces restrictions on the states after computing the next states using the system dynamics
        x = self._limitations(x, 1)
        
        # Computes the system output using the measurement dynamics specified in the model file or other relevant files
        y = self._measurements(x, system_input, self.synapses_current)

        # Calculates interconnections and currents of synapses here
        if self.enable_synapses == True:
            self.synapses_current = self._synapses(x, self.pre, self.post, ext_input)
        
        # Updates internal signals
        self.states  = x + x_noise
        self.outputs = y + y_noise
    # End of function

# End of class


class Nonlinear(SolverCore):
    def __init__(self, time_line, **kwargs):
        '''
        ### Overview:
        This class provides tools to create a nonlinear system with 
        all internal vectors from the beginning of the simulation which can be used to
        define a wide range of systems with greater accessibility.
        Note that the imported system must be a class defined in the `blocks` path.

        ### Input variables:
        * Time line
        
        ### Configuration Options:
        * `initial` specifies the initial condition of the system.
        * `solver` sets the type of solver; e.g., `euler`, `rng4`, etc.
        * `estimator` specifies whether this block should be an estimator. To do this, set it to `True`.
        * `approach` specifies the type of estimator - `ekf` or `ukf`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        self.time_line = np.reshape(time_line, (1, -1))
        self.sample_time = np.mean(self.time_line[0, 1:-1] - self.time_line[0, 0:-2])
        self.n_steps = np.size(self.time_line) # The number of sample steps
        self.solver_type = self.solver_type # The type of dynamic solver
        self.current_step = 0 # The current step of simulation
        self.inputs = np.zeros([self.n_inputs,  self.n_steps])
        self.outputs = np.zeros([self.n_outputs, self.n_steps])
        self.states = np.zeros([self.n_states,  self.n_steps + 1])
        self.states[:, 0] = np.array(self.initial_states).flatten()
        self.estimator = False
        
        est_approach = 'ekf'
        # Retrieving the arbitrary value of properties
        for key, val in kwargs.items():
            # The initial condition
            if key == 'initial': self.states[:, 0] = np.array(val).flatten()
            # The estimation approach
            if key == 'approach': est_approach = val
            # Type of dynamic solver
            if key == 'solver': self.solver_type = val
            # If it is an estimator
            if key == 'estimator': self.estimator = True
        
        # This section initializes the estimator by setting parameters
        if self.estimator == True:
            self.est_approach = est_approach # The estimation approach ('ekf', 'ukf', ...)
            if self.est_approach == 'ekf':
                pass
            elif self.est_approach == 'ukf':
                # Dependent variables
                self.n_ukf = self.num_states
                self.lambd = np.power(self.alpha,2)*(self.n_ukf + self.kappa) - self.n_ukf
                self.betta = 2
                # Creating weights
                self.wm = np.ones([2*self.n_ukf + 1, 1])/(2*(self.n_ukf + self.lambd))
                self.wc = self.wm
                self.wc[0,0] = self.wm[0,0] + (1 - np.power(self.alpha,2) + self.betta)
        # Comment & diary
        Clib.diary('"' + self.name + '" has been created.')

    def __call__(self, input_signal, x_noise = np.array([0]), y_noise = np.array([0])):
        '''
        ### Overview:
        Utilizing current data, this function can furnish a `prediction` of the subsequent step.

        ### Input variables:
        * Input array at step `k`
        * Internal additive noise incorporated into the states
        * External additive noise incorporated into the measurements
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        if self.estimator == True: return 0

        # The current time is calculated as follows
        current_time = self.time_line[0, self.current_step]
        
        # Preparing the input signal and saving it to the internal array
        self.inputs[:, self.current_step] = input_signal.flatten()
        
        # Establishing before-state-limitations:
        # This can be employed if we desire to process states prior to computing the subsequent states utilizing dynamics.
        xv = self._limitations(self.states, 0)
        
        # Retrieving the antecedent states
        xo = self.states[:, self.current_step]
        xo = np.reshape(xo, (-1, 1))
        
        # The handle function below is utilized in the following
        handle_dyn = lambda xx: self._dynamics(xx,
                                            self.inputs,
                                            self.current_step,
                                            self.sample_time,
                                            current_time)
        
        # This section computes the states and outputs utilizing the system dynamics
        if self.time_type == 'c':
            # The type of solver can be manipulated To alter your solver type,
            # do not modify any code here Modify the solver type in the model file
            x = super().dynamic_runner(handle_dyn, xv, xo, self.sample_time, self.solver_type)
        else:
            # When the inserted system is discrete time, only the dynamic must be solved as below
            x = handle_dyn(xv)
        
        # Establishing after-state-limitations
        x = self._limitations(x, 1)
        
        # The system output is computed by the measurement dynamics of the system
        # which are available in the 'chaos.m' file
        y = self._measurements(self.states,
                            self.inputs,
                            self.current_step,
                            self.sample_time,
                            current_time)
        
        # Updating internal signals
        self.states[:, self.current_step + 1] = x.flatten() + x_noise.flatten()
        self.outputs[:, self.current_step] = y.flatten() + y_noise.flatten()
        self.current_step += 1
    # End of function

    # The 'estimate' function can furnish an effortless method to invoke
    # the system dynamics to compute subsequent sample states.
    def estimate(self, input_signal, output_signal):
        '''
        ### Overview:
        Utilizing current data, this function can estimate a subsequent step.

        ### Input variables:
        * Input array of the actual system at step `k`
        * Output array of the actual system at step `k`
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        if self.estimator == False: return 0

        if self.est_approach == 'ekf':
            self.__nextstepEKF(input_signal, output_signal)
        elif self.est_approach == 'ukf':
            self.__nextstepUKF(input_signal, output_signal)

    ## Subsequent step of Extended Kalman Filter (EKF)
    def __nextstepEKF(self, input_signal, output_signal):
        # [Internal update] <- (Internal, Input, Output)
        
        ## Initializing parameters
        # The current time is computed as follows
        current_time = self.time_line[0, self.current_step]
        # Preparing and storing inputs and outputs within internal
        self.inputs[:, self.current_step] = input_signal
        self.outputs[:, self.current_step] = output_signal
        output_signal = np.reshape(output_signal, (-1, 1))
        
        # Employing system dynamics to compute Jacobians
        [A, _, H, _, L, M] = self._jacobians(self.states,
                                            self.inputs,
                                            self.current_step,
                                            self.sample_time,
                                            current_time)
        ## Prediction step - Updating xp
        #  This section endeavors to obtain a prediction estimate from the dynamic
        #  model of your system directly from nonlinear equations
        xm = self.states[:, self.current_step]
        xm = np.reshape(xm, (-1, 1))

        # Calculation before-state-limitations
        xv  = self._limitations(self.states, 0)
        
        # The handle function below is employed in the following
        handle_dyn = lambda xx: self._dynamics(xx,
                                            self.inputs,
                                            self.current_step,
                                            self.sample_time,
                                            current_time)
        # Compute states and outputs using system dynamics
        if self.time_type == 'c':
            # Change the solver type in the block class instead of modifying the code here
            xp = super().dynamic_runner(handle_dyn, xv, xm, self.sample_time, self.solver_type)
        else:
            # For discrete-time systems, solve the dynamic as shown below
            xp = handle_dyn(xv)

        # Apply after-state limitations
        xp = self._limitations(xp, 1)
        
        # Prediction step - update covariance matrix
        Pp = A.dot(self.covariance).dot(A.transpose()) + \
             L.dot(self.q_matrix).dot(L.transpose())
        
        # Posterior step - receive measurements
        # If there are no measurements (y == NaN), only the prediction will be reported
        if not np.any(np.isnan(output_signal)):
            # Calculate Kalman Gain
            K  = (Pp.dot(H.transpose())).dot(                  \
                  np.linalg.inv(H.dot(Pp).dot(H.transpose()) + \
                  M.dot(self.r_matrix).dot(M.transpose())) )
            # Update states
            xm = xp + K.dot(output_signal - H.dot(xp))
            # Update covariance matrix
            Pm = (np.eye(self.covariance.shape[0]) - K.dot(H)).dot(Pp)
        else:
            xm = xp
            Pm = Pp
        
        ## Update internal signals
        self.states[:, self.current_step + 1] = xm.flatten() # Save estimated states
        self.covariance = Pm # Save covariance matrix
        self.current_step += 1 # Move to next step
    
    ## Next step of Unscented Kalman Filter (UKF)
    def __nextstepUKF(self, input_signal, output_signal):
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
        current_time = self.time_line[0, self.current_step]
        # Prepare and save inputs and outputs to internal variables
        self.inputs[:, self.current_step] = input_signal
        self.outputs[:, self.current_step] = output_signal
        output_signal = np.reshape(output_signal, (-1, 1))
        # Calculate Jacobians using system dynamics
        [_, _, _, _, L, M] = self._jacobians(self.states,
                                            self.inputs,
                                            self.current_step,
                                            self.sample_time,
                                            current_time)
        # Get last states prior and its covariance
        xm = self.states[:, self.current_step]
        xm = np.reshape(xm, (-1, 1))
        Pm = self.covariance
        
        # Solve sigma points - STEP 2
        # Calculate square root
        dSigma = np.sqrt(self.n_ukf + self.lambd)*((np.linalg.cholesky(Pm)).transpose())
        # Copy 'xm' to some column
        xmCopy = xm[:, np.int8(np.zeros([1, np.size(xm)]).flatten())]
        # Obtain sigma points
        sp = np.concatenate((xm, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        ## Predict states and their covariance - STEP 3
        # Obtain prediction estimate from dynamic model of system using nonlinear equations
        nSpoints = sp.shape[1]
        xp = np.zeros([self.n_states, 1])
        Xp = np.zeros([self.n_states, nSpoints])
        for i in range(0, nSpoints):
            changed_full_state = self.states
            changed_full_state[:, self.current_step] = sp[:, i]
            
            # Apply before-state limitations
            xv  = self._limitations(changed_full_state, 0)
            # Use handle function below to prevent redundancy
            handle_dyn = lambda xx: self._dynamics(xx,
                                                self.inputs,
                                                self.current_step,
                                                self.sample_time,
                                                current_time)
            if self.time_type == 'c':
                Xp[:,i] = super().dynamic_runner(handle_dyn,
                                                xv,
                                                xm,
                                                self.sample_time,
                                                self.solver_type).flatten()
            else:
                Xp[:,i] = handle_dyn(xv)
            
            # Apply after-state limitations
            Xp[:,i] = self._limitations(Xp[:,i], 1)
            # Update prediction
            temp1 = Xp[:, i]
            xp = xp + self.wm[i,0]*(np.reshape(temp1, (-1, 1)))
        # End of loop

        dPp = Xp - xp[:, np.int8(np.zeros([1, np.size(nSpoints)])).flatten()]
        # Update covariance of states matrix
        Pp  = dPp.dot(np.diag(self.wc.flatten())).dot(dPp.transpose()) + \
              L.dot(self.q_matrix).dot(L.transpose())
        
        ## Update sigma points - STEP 4
        # dSigma = np.sqrt(self.n_ukf + self.lambd).dot( \
        #         (np.linalg.cholesky(Pp)).transpose()) # Calculate square root
        # Putting 'xp' is some column (copy)
        # xmCopy = xp[:, np.int8(np.zeros([1, np.size(xp)]))]
        # sp     = np.concatenate((xp, xmCopy + dSigma, xmCopy - dSigma), axis=1)
        
        if not np.any(np.isnan(output_signal)):
            ## Solve output estimation using predicted data - STEP 5
            # Obtain prediction output from sigma points
            zb = np.zeros([self.n_outputs, 1])
            Zb = np.zeros([self.n_outputs, nSpoints])
            for i in range(0, nSpoints):
                changed_full_state = self.states
                changed_full_state[:, self.current_step] = Xp[:, i] #Or 'Xp[:, i]' instead of 'sp[:, i]'
                Zb[:,i] = self._measurements(changed_full_state,
                                            self.inputs,
                                            self.current_step,
                                            self.sample_time,
                                            current_time).flatten()
                # Predicted output
                temp1 = Zb[:, i]
                zb = zb + self.wm[i,0]*(np.reshape(temp1, (-1, 1)))
            # End of loop

            dSt = Zb - zb[:, np.int8(np.zeros([1, np.size(nSpoints)])).flatten()]
            # Update covariance of output matrix
            St = dSt.dot(np.diag(self.wc.flatten())).dot(dSt.transpose()) + \
                  M.dot(self.r_matrix).dot(M.transpose())

            ## Solve Kalman gain - STEP 6
            SiG = dPp.dot(np.diag(self.wc.flatten())).dot(dSt.transpose())
            # Calculate Kalman Gain
            K = SiG.dot(np.linalg.inv(St))
        
        ## Solve posterior using measurement data - STEP 7
        # If there are no measurements (output_signal == NaN), only the prediction will be reported
        if not np.any(np.isnan(output_signal)):
            # Update states
            xm = xp + K.dot(output_signal - zb)
            # Update covariance matrix
            Pm = Pp - K.dot(SiG.transpose())
        else:
            xm = xp
            Pm = Pp
        
        ## Update internal signals
        self.states[:, self.current_step + 1] = xm.flatten() # Save estimated states
        self.covariance = Pm # Save covariance matrix
        self.current_step += 1 # Move to next step
    # End of function

    # Function to jump in the step variable
    # If no arguments are provided, jump 1 step
    def __iadd__(self, i = 1):
        '''
        ### Overview:
        This function can make a jump in the step number variable.

        ### Input variables:
        * how many steps you would like me to jump?; default is `1`
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        self.current_step = self.current_step + i
    
    # Reset Block by changing current step to zero
    def reset(self):
        '''
        ### Overview:
        Reseting the block via changing the current step to zero.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        self.current_step = 0
        
    # The below function is used to plot the internal signals
    def show(self, params, what = 'x', **kwargs):
        '''
        ### Overview:
        This function makes a quick plot of internal signals.

        ### input variables:
        * `params`
        * The signal to be shown (`x`, `y`, or `u`); default is `x`

        ### Configuration Options:
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
            
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # To illustrate states, inputs, or outputs, you might have to
        # use some varargins which are explained in 'scope' class
        if what == 'x':
            signal = self.states[:, 0:self.n_steps]
            n_signals = self.n_states
        elif what == 'y':
            signal = self.outputs[:, 0:self.n_steps]
            n_signals = self.n_outputs
        elif what == 'u':
            signal = self.inputs[:, 0:self.n_steps]
            n_signals = self.n_inputs
        # Make a scope
        scp = Scope(time_line = self.time_line, n_signals = n_signals, initial=signal)

        # If title is set
        for key, val in kwargs.items():
            if key == 'title':
                scp.show(params, **kwargs)
                return
        # Else plot the name of the model as the plot title
        scp.show(params, title=self.name, **kwargs)
    # End of function
    
    def __repr__(self):
        return f"** {self.__name__} **\nCurrent point: {self.current_step}/{self.n_steps}\nSolver Type: '{self.solver_type}'"
# End of class

