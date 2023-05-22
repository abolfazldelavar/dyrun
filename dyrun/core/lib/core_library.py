
## // --------------------------------------------------------------
#    ***DYNAMIC RUNNER***
#    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
#    Web: https://github.com/abolfazldelavar/dyrun
## // --------------------------------------------------------------

# Loading the requirements
from core.lib.required_libraries import *

# %% Structure class
class Structure(dict):
    '''
    ## C/C++ or MATLAB-like structures
    A simple and easy-to-use MATLAB-like structure type for Python.

    ## Sample Usage
    The structure is defined by `struct` class.
        ```
        p = struct()
        p.x = 3
        p.y = 4
        p.A = p.x * p.y
        print(p)
        ```
    The output will be:
        ```
        struct({'x': 3, 'y': 4, 'A': 12})
        ```
    Here, an instance of `struct` is created and then fields `x` and `y` are defined. Then, a new field `A` is added to the structure. Finally, the string representation of the `struct` is printed.

    In the previous code, you can simply use the following method to create and initialize the structure.
        ```
        p = struct(x=3, y=4)
        p.A = p.x * p.y
        ```

    ## Dictionary Vs Structure
    Actually `struct` is a subclass of built-in dictionary type `dict`, and it can be converted to or created from `dict` objects.

    You can convert `dict` objects to `struct` as follows:
        ```
        my_dict = {'x':3, 'y':4}
        p = struct(my_dict)
        ```
    Also, `struct` object can be converted to `dict` as well:
        ```
        p = struct(x=3, y=4)
        p.z = 12
        my_dict = dict(p)
        print(my_dict)
        ```
    The output will be this:
        ```
        {'x': 3, 'y': 4, 'z': 12}
        ```

    ## Merging Structures
    It is possible to merge two `struct` objects. For example let's define two structures as:
        ```
        a = struct(x=1, y=2, z=3)
        b = struct(y=5, t=20, u=30)
        ```
    We can merge these structure using `+` operator:
        ```
        c = a + b
        print(c)
        ```
    The output will be:
        ```
        struct({'x': 1, 'y': 5, 'z': 3, 't': 20, 'u': 30})
        ```

    ## List of the Methods
    In this part, we are going to discuss the methods implemented and available in
    `struct` class. You can use these methods with any instance of `struct` class.
    Additionally, because the `struct` class is a subclass of `dict`, all of the
    methods defined in the `dict` class are available too.

    ### Fields Method
    The `fields()` method returns a `list` of fields defined in structure. An example usage follows:
        ```
        p = struct(x=3, y=4)
        print(p.fields())
        ```
    The output will be:
        ```
        ['x', 'y']
        ```

    ### Add Field Method
    A new field can be added using `add_field()` method. This method accepts two input arguments:
    field name and its value. The value is optional and if it is ignored,
    then value is assumed to be `None`. A sample code follows:
        ```
        p = struct(x=3, y=4)
        p.add_field('z', 12)
        p.add_field('L')
        print(p)
        ```
    The output of this code will be:
        ```
        struct({'x': 3, 'y': 4, 'z': 12, 'L': None})
        ```
    Instead of using the `add_field()` method, it is possible to use `.` and `=` operators.
    For example, the above-mentioned code can be simplified as this:
        ```
        p = struct(x=3, y=4)
        p.z = 12
        p.L = None
        print(p)
        ```
    The result will be the same.

    ### Remove Field Method
    A field can be removed from a `struct` object using `remove_field()` method.
    This method gets a field name and it removes (deletes) the specified field.
    An example is given below:
        ```
        p = struct(x=3, y=4, z=12)
        print('Before remove_field: {}'.format(p))
        p.remove_field('z')
        print('After remove_field: {}'.format(p))
        ```
    The output will be this:
        ```
        Before remove_field: struct({'x': 3, 'y': 4, 'z': 12})
        After remove_field: struct({'x': 3, 'y': 4})
        ```

    ### Repeat Method
    Sometimes we need to repeat/replicate a structure. For example, assume that
    we are going to implement an Evolutionary Algorithm and we defined the
    individualsas `struct` objects. First we need to create a template:
        ```
        empty_individual = struct(pos=None, fval=None)
        ```
    Then we can initialize the population array using following code:
        ```
        pop_size = 10
        pop = empty_individual.repeat(pop_size)
        ```
    This code uses the `repeat()` method to initialize a list of distinct
    `struct` objects with the same data fields. Instead of using `repeat()` method,
    simply we can use `*` operator to perform replication:
        ```
        pop = empty_individual * pop_size
        ```

    ### Copy and Deep-copy Methods
    The `struct` is a reference type. To have a copy of a `struct` object,
    you cannot simply use assignment operator. To create copies of structure objects,
    two methods are implemented in `struct` class: `copy()` and `deepcopy()`.
    The first one gives us a shallow copy of the `struct` object.But using `deepcopy()`,
    as the name of the method says, we can create deep copies of structure objects.

    ## Copyright
    Copyright 2018-2020, Mostapha Kalami Heris / Yarpiz Team
    Web page: https://github.com/smkalami/ypstruct
    '''
    def __repr__(self):
        """
        String representation of the struct
        """
        return "struct({})".format(super().__repr__())

    def __getattr__(self, field):
        """
        Gets value of a field
        """
        if field not in dir(self):
            if field in self.keys():
                return self[field]
            else:
                return None
        else:
            return None
    
    def __setattr__(self, field, value):
        """
        Sets value of a field
        """
        if field not in dir(self):
            self[field] = value
        else:
            return super().__setattr__(field, value)
    
    def fields(self):
        """
        Gets the list of defined fields of the struct
        """
        return list(self.keys())

    def remove_field(self, field):
        """
        Removes a field from the struct
        """
        if field in self.keys():
            del self[field]
    
    def add_field(self, field, value = None):
        """
        Adds a new field to the struct
        """
        if field not in self.keys():
            self[field] = value

    def copy(self):
        """
        Creates a shallow copy of the struct
        """
        self_copy = Structure()
        for field in self.keys():
            if isinstance(self[field], Structure):
                self_copy[field] = self[field].copy()
            else:
                self_copy[field] = cop.copy(self[field])
        
        return self_copy

    def deepcopy(self):
        """
        Creates a deep copy of the struct
        """
        self_copy = Structure()
        for field in self.keys():
            if isinstance(self[field], Structure):
                self_copy[field] = self[field].deepcopy()
            else:
                self_copy[field] = cop.deepcopy(self[field])
        
        return self_copy

    def repeat(self, n):
        """
        Repeats/replicates the struct to create an array of structs (e.g. for initialization)
        """
        return [self.deepcopy() for i in range(n)]
    
    def __mul__(self, n):
        """
        Overload * operator (multiplication) to repeat/replicate the struct
        """
        if not isinstance(n, int) and not isinstance(n, float):
            raise TypeError("Only integers are allowed.")
        return self.repeat(n)
    
    def __add__(self, other):
        """
        Overload + operator (addition) to merge two struct objects
        """
        if not isinstance(other, dict):
            raise TypeError("Only structure and dict objects are allowed.")
        result = self.deepcopy()
        result.update(other)
        return result
# End of class

# %% internal effective functions
class Clib():
    '''
    ### Description:
    This library provides you several practical functions that some of them are essential for the framework, 
    and others might be useful for other purposes.
    
    ### Copyright:
    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
    Web page: https://github.com/abolfazldelavar/dyrun
    '''

    # Saying a hello at the start of the simulation
    @staticmethod
    def set_timer(ID):
        '''
        ### Description:
        This function sets the timer and initializes the simulation.
        
        ### Input variables:
            * `ID` - The id number of simulation; it is usually unique.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # [Start time] <- (Parameters)
        # Setting up the logging module
        if not os.path.isdir('logs'): os.makedirs('logs')
        logging.basicConfig(filename= 'logs/' + ID + '.log',
                            level=logging.INFO,
                            format='%(asctime)s %(filename)s, %(lineno)d, %(levelname)s: %(message)s')
        Clib.diary('--- DYNAMIC RUNNER PACKAGE ---')
        Clib.diary('The simulation has kicked off! (' + Clib.get_now(5,'/') + ', ' + Clib.get_now(6,':') + ')')
        sleep(0.5)
        # Send the current time to the output
        return time()

    # Report the simulation time when it finishes
    @staticmethod
    def end_report(start_time):
        '''
        ### Description:
        This function reports the end of the simulation and the time it took to be run.
        
        ### Input variables:
            * The time simulation was started.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Stopping the timer which started before, and saving the time.
        ntime = time() - start_time
        tmin = min(int(ntime/60), 1e5) # How minutes
        tsec = round(ntime%60, 4)      # Remained seconds
        txt = 'The simulation has been completed. (' + str(tmin) +   \
                 ' minutes and ' + str(tsec) + ' seconds)'
        Clib.diary(txt)
        
    ## Returning a text contained date and time
    @staticmethod
    def get_now(report_type = 0, splitter_char = '_'):
        '''
        ### Description:
        To make a delay in a descrete function is used.
        
        ### Input variables:
            * The type of report
                * default: `YMDHMS`
                * 1: `YMD_HMS`
                * 2: `Y_M_D_H_M_S`
                * 3: `YMD_HM`
                * 4: `YMD_HM`
                * 5: `Y_M_D`
                * 6: `H_M_S`
            * Splitter; like `_` or `*`
            
        ### Output variable:
            * Output string
            
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''        
        fullDate = datetime.now() # Getting full information of time
        year     = str(fullDate.year)
        month    = str(fullDate.month)
        day      = str(fullDate.day)
        hour     = str(fullDate.hour)
        minute   = str(fullDate.minute)
        second   = str(fullDate.second)
        
        # If the numbers are less than 10, add a zero before them
        if len(month)<2:  month  = '0' + month
        if len(day)<2:    day    = '0' + day
        if len(hour)<2:   hour   = '0' + hour
        if len(minute)<2: minute = '0' + minute
        if len(second)<2: second = '0' + second
        
        # What style do you want? You can change arbitrary
        if report_type == 1:
            txt = year + month + day + splitter_char + hour + minute + second
        elif report_type == 2:
            txt = year + splitter_char + month + splitter_char + day + splitter_char + \
                  hour + splitter_char + minute + splitter_char + second
        elif report_type == 3:
            txt = year + month + day + splitter_char + hour + minute
        elif report_type == 4:
            txt = year + month + day + splitter_char + hour + minute
        elif report_type == 5:
            txt = year + splitter_char + month + splitter_char + day
        elif report_type == 6:
            txt = hour + splitter_char + minute + splitter_char + second
        else:
            txt = year + month + day + hour + minute + second
        # Returning the output
        return txt

    ## Diary
    @staticmethod
    def diary(message, no_print=False):
        '''
        ### Description:
        This function is used to put the message in the diary file as well as to print it.
        
        ### Input variables:
            * `message` - The message you want to show or put in diary.
            * `no_print` - To turn printing off, set this `True`; The default value is `False`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        if no_print == False: print(message)
        logging.info(message)
    
    ## Saving matrices - .csv
    @staticmethod
    def save_csv(tensor, name, **kwargs):
        '''
        ### Overview:
        Saving the given data in a `.csv` file.

        ### Input variables:
        * `tensor` - Data, e.g., matrices or numerical vectors.
        * `name` - The path including the name of the file. e.g. `data/outputs/TensorFileName`.
        
        ### Configuration Options:
        * `zip` - is used to save as a `.zip` file; default = `False`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Requirements
        import csv

        file_format = '.csv'
        zipp = False
        for key, val in kwargs.items():
            # 'zip' specifies whether a zip file must be created
            if key == 'zip': zipp = val
        # Extracting the name and directory seperately
        if isinstance(name, str):
            # Split folders
            slash_splitted = name.split('/')
            save_path = ''
            if len(slash_splitted) > 1:
                # Directory maker
                save_path = '/'.join(slash_splitted[0:-1])
                file_name    = slash_splitted[-1]
                # Create the directory if it does not exist
                if not os.path.isdir(save_path): os.makedirs(save_path)
            else:
                # If directory is not adjusted:
                file_name = slash_splitted[0]
        else:
            raise ValueError("Please enter the name of the file correctly.")
        dim = np.array(tensor.shape)
        # The number of elements
        total = np.prod(dim)
        # How many elements in a row
        eachRow = 100
        # The figure for completed rows
        rowCounts = int(total/eachRow)
        # The number of elements in the whole completed matrix
        cutNum = rowCounts*eachRow
        # Remained numbers
        rem = total - cutNum
        # The default value of order is 'C', which stands for row-major order.
        # You can also use 'F' to specify column-major order.
        daTa = tensor.flatten(order='F')
        # Main data as a matrix
        bodyDaTa = np.reshape(daTa[0:cutNum], (rowCounts, eachRow), order='F')
        # Overflow elements
        lastDaTa = daTa[cutNum:]
        # Open the file in write mode and automatically close it after the block of code inside it is executed
        # The newline='' argument clears the file when opened in write mode.
        with open(name + file_format, 'w', newline='') as f:
            # Create a writer object that can write data to the file
            writer = csv.writer(f)
            # Write the file dimention
            writer.writerow(dim)
            # Iterate over each row in the matrix list (Body)
            for row in bodyDaTa:
                # Write the row to the file
                writer.writerow(row)
            # Last row
            writer.writerow(lastDaTa)
        # Zipper
        if zipp == True:
            import zipfile
            # Create a new ZIP archive
            with zipfile.ZipFile(name + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zip:
                # Add one or more files to the archive
                zip.write(name + file_format, arcname= file_name + file_format)
            os.remove(name + file_format)
            Clib.diary('The file named "' + file_name + '.zip" has been saved into "' + save_path + '".')
            return
        # Print the result of saving
        Clib.diary('The file named "' + file_name + file_format + '" has been saved into "' + save_path + '".')

    ## Loading matrices - .csv
    @staticmethod
    def load_csv(name, **kwargs):
        '''
        ### Overview:
        Loading the content of the given `.csv` file.

        ### Input variables:
        * `name` - The path including the name of the file. e.g. `data/outputs/TensorFileName`.

        ### Options:
        * `maxsize` - Providing the size of the file is bigger than usual (default is `1000 MB`),
        set this value more than your file size (in `MB`). Note that the bigger value needs more RAM space.
        Do not waste your RAM carelessly.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Requirements
        import csv
        import ast
        # Extracting the name and directory seperately
        if not isinstance(name, str): raise ValueError("Please enter the name of the file correctly.")
        # Default size limit is 1000 MB
        maxFileSizeInMB = 1000
        # Extract the given file size restriction
        for key, val in kwargs.items():
            if key == 'maxsize': maxFileSizeInMB = val
        csv.field_size_limit(maxFileSizeInMB * 1024 * 1024)
        # Open the file in read mode and automatically close it after the block of code inside it is executed
        with open(name + '.csv', 'r') as file:
            # Create a reader object that can read the contents of the file
            reader = csv.reader(file)
            # Create empty lists to store the data and its dimention
            data = []
            dim  = []
            # Iterate over each row in the reader object
            for i, row in enumerate(tqdm(reader, desc='Loading "' + name + '.csv"', leave=False, dynamic_ncols=True)):
                # Check if the current row is the first row
                if i ==0:
                    dim = [float(x) for x in row]  # Save the first row in `dd`
                else:
                    # Append a new list of floats to the data list byconverting
                    # each element in the row from a string to a float.
                    data.append([float(x) for x in row])
        # Reshaping the matrix
        daTa = np.array(data[:-1]).flatten(order='F')
        # Adding the last part of data
        daTa = np.concatenate((daTa, np.array(data[-1])), axis=0)
        # Reshape to original
        daTa = daTa.reshape(np.int32(dim), order='F')
        # Print the result of loading
        Clib.diary('The file named "' + name + '.csv" has been loaded.')
        return daTa

    ## Saving tensors - .npy
    @staticmethod
    def save_npy(tensor, name, **kwargs):
        '''
        ### Overview:
        Saving the given tensor in a `.npy` file.

        ### Input variables:
        * `tensor` - Data, e.g., matrices or numerical vectors.
        * `name` - The path including the name of the file. e.g. `data/outputs/TensorFileName`.
        
        ### Configuration Options:
        * `zip` - is used to save as a `.zip` file; default = `False`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        
        file_format = '.npy'
        zipp = False
        for key, val in kwargs.items():
            # 'zip' specifies whether a zip file must be created
            if key == 'zip': zipp = val
        # Extracting the name and directory seperately
        if isinstance(name, str):
            # Split folders
            slash_splitted = name.split('/')
            save_path = ''
            if len(slash_splitted) > 1:
                # Directory maker
                save_path = '/'.join(slash_splitted[0:-1])
                file_name    = slash_splitted[-1]
                # Create the directory if it does not exist
                if not os.path.isdir(save_path): os.makedirs(save_path)
            else:
                # If directory is not adjusted:
                file_name = slash_splitted[0]
        else:
            raise ValueError("Please enter the name of the file correctly.")
        tensor = np.array(tensor)        
        # Save the tensor as a .npy file
        np.save(name + file_format, tensor)
        # Zipper
        if zipp == True:
            import zipfile
            # Create a new ZIP archive
            with zipfile.ZipFile(name + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zip:
                # Add one or more files to the archive
                zip.write(name + file_format, arcname= file_name + file_format)
            os.remove(name + file_format)
            Clib.diary('The file named "' + file_name + '.zip" has been saved into "' + save_path + '".')
            return
        # Print the result of saving
        Clib.diary('The file named "' + file_name + file_format + '" has been saved into "' + save_path + '".')

    ## Loading tensors - .npy
    @staticmethod
    def load_npy(name):
        '''
        ### Overview:
        Loading the content of the given `.npy` file.

        ### Input variables:
        * `name` - The path including the name of the file. e.g. `data/outputs/TensorFileName`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Extracting the name and directory seperately
        if not isinstance(name, str): raise ValueError("Please enter the name of the file correctly.")
        # Load the tensor from the .npy file
        tensorData = np.load(name + '.npy')
        # Print the result of loading
        Clib.diary('The file named "' + name + '.npy" has been loaded.')
        return tensorData

    ## Delayed in a signal
    @staticmethod
    def delayed(signal, k, delay_steps):
        '''
        ### Description:
        To make a delay in a descrete function is used.
        
        ### Input variables:
            * Full signal
            * Current point `k`
            * Delay amount (in integer)
            
        ### Output variable:
            * delayed value
            
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        if k - delay_steps >= 0:
            y = signal[:, k - delay_steps]
        else:
            y = signal[:, k]*0
        return y

    ## Making a signal of an exponential inverse
    @staticmethod
    def sigmoid(time_line, bias, alph, area):
        '''
        ### Description:
        Sigmoid generator function.

        ### Input variables:
        * Time line
        * Time delay bias
        * Smoother
        * Domain in form of `[a, b]`
        
        ### Output variable:
        * The output signal
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Note that signal domain is a two component vector [a(1), a(2)]
        output = 1/(1 + np.exp(-alph*(time_line - bias)))
        return (area[1] - area[0])*output + area[0]
    
    ## Making an exponential signal
    @staticmethod
    def exp_f(time_line, decay_rate, area):
        '''
        ### Description:
        Exposential function.

        ### Input variables:
        * Time line
        * decay_rate
        * Domain in form of `[a, b]`
        
        ### Output variable:
        * The output signal
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        Ss = area(1); # Start point
        Sf = area(2); # Final value
        return (Ss - Sf)*np.exp(-decay_rate*time_line) + Sf

    ## Linear mapping a number
    @staticmethod
    def linear_mapping(x, current_area, target_area):
        '''
        ### Description:
        Linear Mapping the point (or array) `x` from `[a1, b1]` domain to `y` in domain `[a2, b2]`

        ### Input variables:
        * array
        * `x` domain: `[a1, b1]`
        * `y` domain: `[a2, b2]`
        
        ### Output variable:
        * The output array
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Map 'x' from band [w1, v1] to band [w2, v2]
        w1        = np.array(current_area[0])
        v1        = np.array(current_area[1])
        w2        = np.array(target_area[0])
        v2        = np.array(target_area[1])
        x         = np.array(x)
        output    = 2*((x - w1)/(v1 - w1)) - 1
        output    = (output + 1)*(v2 - w2)/2 + w2
        return output
# End of class

# %% Numerical calculations
class SolverCore():
    '''
    ### Description:
    The numerical core of these framework are considered as this class which includes `optimizations` and other `numerical solvers`.
    
    ### Copyright:
    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
    Web page: https://github.com/abolfazldelavar/dyrun
    '''
    # Dynamic Solver: This funtion contains some numerical methods
    # like 'euler', 'rng4', etc., which can be used in your design.
    @staticmethod
    def dynamic_runner(handle_dyn, xv, xo, sample_time, solver_type):
        '''
        ### Description:
        To calculate a prediction using a `handler` of the function, this function could be utilized.

        ### Input variables:
        * handler (make one by `lambda x: x**2 + 5`)
        * Full time vector of states
        * current states
        * Sample time
        * Solver (`euler`, `rng4`)
        
        ### Output variable:
        * Predicted states
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        if solver_type == 'euler':
            # Euler method properties is given below (T is sample time):
            #   x(t+1) = x(t) + T*f(x(t))
            xn = xo + sample_time*handle_dyn(xv)
        elif solver_type == 'rng4':
            # 4th oder of 'Runge Kutta' is described below (T is sample time):
            #   K1     = T*f(x(t))
            #   K2     = T*f(x(t) + K1/2)
            #   K3     = T*f(x(t) + K2/2)
            #   K4     = T*f(x(t) + K3)
            #   x(t+1) = x(t) + 1/6*(K1 + 2*K2 + 2*K3 + K4)
            k_1 = sample_time*handle_dyn(xv)
            k_2 = sample_time*handle_dyn(xv + k_1/2)
            k_3 = sample_time*handle_dyn(xv + k_2/2)
            k_4 = sample_time*handle_dyn(xv + k_3)
            xn = xo + 1/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
        else:
            err_text = 'The solver name is not correct, please change the word "' + solver_type + '"'
            logging.error(err_text)
            raise ValueError(err_text)
        return xn

# %% Illustration functions
class Plib():
    '''
    ### Description:
    This library provides you several practical functions related to `illustration` purposes.
    
    ### Copyright:
    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
    Web page: https://github.com/abolfazldelavar/dyrun
    '''
    @staticmethod
    def initialize():
        '''
        ### Description:
        This function initializes the environment and sets the `font.family`, `font.size`, and `text.usetex` to any further draws.
        Furthermore, the size of plot windows and several other essential properties are adjusted.        
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Setting the font of LaTeX
        plt.rcParams.update({
            "text.usetex": True,
            "font.size": 17,
            "font.family": ""       # Times New Roman, Helvetica
        })
        # To set a size for all figures
        plt.rcParams["figure.figsize"] = [8.5, 5.07]
        # Set all font properties
        # plt.rc('font', size = 17)
        # Set all plotted lines properties
        plt.rc('lines', linewidth = 0.5)
        
        # Change the color of all axes, ticks and grid
        AXES_COLOR = '#111'
        mpl.rc('axes', edgecolor = AXES_COLOR, linewidth=0.5, labelcolor=AXES_COLOR)
        mpl.rc('xtick', color = AXES_COLOR, direction='in')
        mpl.rc('ytick', color = AXES_COLOR, direction='in')
        mpl.rc('grid', color = '#eee')
        
        #! /usr/bin/env python

        # import matplotlib
        # import matplotlib.pyplot as plt
        # import matplotlib.font_manager as fm

        # # size
        # plt.rcParams['figure.figsize'] = (1.62 * 2, 1 * 2)
        # # font
        # matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['ps.fonttype'] = 42
        # plt.rcParams['font.family'] = 'serif'
        # plt.rcParams['font.serif'] = ['Times New Roman']
        # plt.rcParams['mathtext.fontset'] = 'cm'
        # plt.rcParams['font.size'] = 10
        # plt.rcParams['ytick.labelsize'] = 10
        # plt.rcParams['ytick.labelsize'] = 10
        # # axis
        # plt.rcParams['xtick.direction'] = 'in'
        # plt.rcParams['ytick.direction'] = 'in'
        # plt.rcParams['axes.formatter.use_mathtext'] = True
        # plt.rcParams['axes.linewidth'] = 1.0
        # plt.rcParams['xtick.major.width'] = 1.0
        # plt.rcParams['ytick.major.width'] = 1.0
        # # legend
        # plt.rcParams['legend.fancybox'] = False
        # plt.rcParams['legend.framealpha'] = 1.0
        # plt.rcParams['legend.edgecolor'] = '#000000'
        # plt.rcParams['legend.handlelength'] = 1.0
        # # grid
        # plt.rcParams['axes.grid'] = False


    @staticmethod
    def isi(params, fig = 0, save = False, width = 8.5, hw_ratio = 0.65):
        '''
        ### Description:
        Making plots prettier and ready to use in academic purposes.

        ### Input variables:
        * `params`
        * Figure handler; use `h = plt.figure(tight_layout=True)` to make one.
        * Saving as a file - If you want the illustration is saved, enter the name of that,
          like `image.png/pdf/jpg`, or just insert `True`
        * Width; default is `8.5 inch`.
        * Height to width ratio between 0 and 1; default is `0.65`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # This function make the graph pretty.
        
        if fig == 0: raise ValueError("Please enter the figure handler.")
        # Changing size of the figure as you ordered
        fig.set_figwidth(width)
        fig.set_figheight(width*hw_ratio)

        # The number of axis existed in fig
        n_axs = len(fig.axes)
        for i in range(0, n_axs):
            # Extract the name of subplot

            ax = fig.axes[i]
            if ax.name == 'rectilinear':
                # For 2D graph lines

                # Set all ticks width 0.5
                ax.tick_params(width=0.5)
                
                # Hide the right and left
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)
                
                # Set the pad and color of thicks and labels 
                ax.xaxis.set_tick_params(pad=10)
                ax.yaxis.set_tick_params(pad=10)

                # How many ticks do you want to have in each axis
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

            elif ax.name == '3d':
                # 3D figure settings

                # Hide the right and left
                ax.spines.right.set_visible(False)
                ax.spines.top.set_visible(False)

                # Set all ticks width 0.5
                ax.tick_params(width=0.5)

                # Set the distance between axis and numbers and color of thicks and labels 
                ax.xaxis.set_tick_params(pad=6)
                ax.yaxis.set_tick_params(pad=6)
                ax.zaxis.set_tick_params(pad=6)
                
                # How many ticks do you want to have in each axis
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

                # Set the color of thicks and labels 
                ax.w_xaxis.set_tick_params(color='none')
                ax.w_yaxis.set_tick_params(color='none')
                ax.w_zaxis.set_tick_params(color='none')

                # Turning off the background panes
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # The distance between labels and their axis
                ax.xaxis.labelpad = 15
                ax.yaxis.labelpad = 15
                ax.zaxis.labelpad = 15

                # The view of camera (degree)
                ax.view_init(20, -40)
        # End of loop

        # Saving the graph, if it is under demand.
        # User must import the figure name as 'save' var
        if not save==False:
            Plib.save_figure(params, save, fig)
    # End of function

    @staticmethod
    def save_figure(params, save = True, fig = plt.gcf(), dpi = 300):
        '''
        ### Description:
        Use this function to save an illustration.

        ### Input variables:
        * `params`
        * Saving as a file - enter the name, like `image.png/pdf/jpg`, insert `True`, or just let it go.
        * Figure handler; use `h = plt.figure(tight_layout=True)` to make one.
        * Dots per inch; default is `300` pixel.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''

        # To get current PC time to use as a prefix in the name of file
        save_path = params.save_path + '/figs'
        # Default saving format
        file_format = params.default_image_format
        all_formats = ['jpg', 'png', 'pdf']
        is_set_dir = False
        need_to_set_unique = True

        # Extracting the name and the directory which inported
        if (not isinstance(save, str)) and (isinstance(save, bool) or isinstance(save, int)):
            # Set the time as its name, if there is no input in the arguments
            save = 'FSF_' + params.id
            need_to_set_unique = False
        elif isinstance(save, str):
            # Split folders
            slash_splitted = save.split('/')
            if len(slash_splitted) > 1:
                # Directory maker
                save_path = '/'.join(slash_splitted[0:-1])
                dot_splitted = slash_splitted[-1].split('.')
                is_set_dir = True
            else:
                # If directory is not adjusted:
                dot_splitted = slash_splitted[0].split('.')
            
            # Name and format
            if len(dot_splitted) > 1:
                # The name is also adjusted:
                if len(dot_splitted[0]) == 0 or len(dot_splitted[-1]) == 0:
                    err_text = 'Please enter the correct notation for the file name.'
                    logging.error(err_text)
                    raise ValueError(err_text)
                elif not (dot_splitted[-1] in all_formats):
                    err_text = 'You must input one of these formats: png/jpg/pdf'
                    logging.error(err_text)
                    raise ValueError(err_text)

                save = ''.join(dot_splitted[0:-1])
                file_format = dot_splitted[-1]
            else:
                # One of name or format just is inserted
                # There is just name or format. It must be checked
                if dot_splitted[-1] in all_formats:
                    file_format = dot_splitted[0]
                    # Set the time as its name, if there is no input in the arguments
                    save = 'FSF_' + params.id
                    need_to_set_unique = False
                else:
                    # Just a name is imported, without directory
                    # and any formats
                    save = dot_splitted[0]
        else:
            err_text = "Please enter the name of file correctly, or the expression of 'True'."
            logging.error(err_text)
            raise ValueError(err_text)

        save_path = str(save_path)
        file_format = str(file_format)
        save = str(save)
        
        # Changing the file name
        if params.unique == 1 and need_to_set_unique:
            file_name = save + '_' + params.id
        else:
            file_name = save
        
        # Prepare direct
        if is_set_dir == 0:
            file_dir = save_path + '/' + file_format
        else:
            file_dir = save_path
        
        # Check the folders existance and make them if do not exist
        if not os.path.isdir(file_dir): os.makedirs(file_dir)
        
        # Saving part
        full_name = file_dir + '/' + file_name + '.' + file_format
        fig.savefig(full_name, transparent=True, dpi=dpi)
        
        # Print the result of saving
        Clib.diary('The graph named "' + file_name + '.' + file_format + '" has been saved into "' + file_dir + '".')
    # End of function

    @staticmethod
    def linear_gradient(colors, locs, n=256, show=False):
        '''
        ### Description:
        To make a linear gradient from one color to another, use this option.

        ### Input variables:
        * Colors; e.g., `[[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]`.
        * Splitting points; e.g., `[0, 0.1, 1]`.
        * the number of colors in output; default is `256`.
        * If you want to have a pre-shown of the gradient, set this `True`.

        ### Output variables:
        * An array comprising all colors
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''

        locs = np.reshape(np.double(locs), (1, -1))
        colors = np.array(colors)

        n_gradients = colors.shape[0] - 1
        n_signals = colors.shape[1]
        cols = np.zeros([1, n_gradients])
        area = np.array([locs[0, 0], locs[0, -1]])

        for i in range(0, n_gradients+1):
            locs[0, i] = Clib.linear_mapping(locs[0, i], area, np.array([0, 1]))

        for i in range(0, n_gradients):
            cols[0, i] = np.ceil(n * (locs[0, i + 1] - locs[0, i]) )
        cols[0,-1] = n - (int(np.sum(cols[0,0:-1])) - n_gradients) 

        init_colors = np.zeros([int(np.sum(cols)), n_signals])
        shift_var = 0

        # Gradient maker
        for i in range(0, n_gradients):
            color_1 = colors[i, :]
            color_2 = colors[i + 1, :]
            gradian = np.zeros([int(cols[0, i]), n_signals])
            for j in range(0,n_signals):
                gradian[:,j] = np.interp(np.linspace(0, 1, int(cols[0, i])), [0, 1], [color_1[j], color_2[j]] )
            temp_var = (np.arange(0, int(cols[0,i])) + shift_var + int(i<1) - 1).flatten()
            init_colors[np.int16(temp_var), :] = gradian
            shift_var = shift_var + cols[0,i] - 1

        export_colors = init_colors[:n, :]

        # Plot gradient
        if show == 1:
            plt.figure()
            img = np.ones([256, 256])
            u = np.linspace(0, 1, 256)
            img = u*img
            plt.imshow(img)
            plt.colorbar()
            plt.show()
        # Send to the output
        return export_colors

    @staticmethod
    def cmap_maker(name, colors, n=256):
        '''
        ### Description:
        This function is used to make a Linear Segmented Color Map (LSCM).

        ### Input variables:
        * Name
        * Colors and their location in the graph line; e.g., `[(0, '#ffff00'), (0.25, '#002266'), (1, '#002266')]`.
        * The number of colors in the output.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors, n)
        return cmap
# End of class
