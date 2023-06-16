
## // --------------------------------------------------------------
#    ***DYNAMIC RUNNER***
#    Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
#    Web: https://github.com/abolfazldelavar/dyrun
## // --------------------------------------------------------------

# Import initial classes
from core.lib.required_libraries import *
from core.lib.core_library import Plib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Scope():
    # 1) Use the below code in 'initialization.py', into the 'vectors', to set initial options
    #    signals.x = scope(signals.time_line, The number of signals, initial=1)
    # 2) Use the below piece of code in 'simulation.py' to save each step
    #    signals.x.getdata(Input signal at step k, noise=0)
    
    def __init__(self, **kwargs):
        '''
        This class is a kind of scope that can save your signals, and can be used for after running purposes.
        
        ### Configuration Options:
        * `time_line` - time line
        * `n_signals` - number of signals
        * `initial` - denotes the initial condition of the estimator
        * `load` - to load a `.csv` file which has been saved before. To import
        your saved files, use `load=(params, name)` structure, which `params` denotes
        the parameters and `name` is the name of file you would like the file save with it.
        Bear in mind that your files must be moved into `../outputs/scope/` directory,
        although the directory must not inserted as a part of `name`. e.g., `scope(load=(params, 'Voltage'))`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Requirements
        from core.lib.core_library import Clib
        # Initial variable values
        initial_condition = np.array([0])
        load_flag = False
        n_signals = 1
        auto_create_time_line = True
        time_line = 0
        # Extracting the arbitraty value of properties
        for key, val in kwargs.items():
            if key == 'initial':
                initial_condition = np.array(val)
            if key == 'load':
                load_flag = True
                params = val[0]
                file_name = val[1]
            if key == 'n_signals': n_signals = val
            if key == 'time_line':
                auto_create_time_line = False
                time_line = val
        
        self.time_line = np.reshape(time_line, (1, -1)) # Time line vector
        self.time_line = np.array(self.time_line)
        self.n_signals = n_signals
        self.signals = np.zeros([self.n_signals, np.size(self.time_line)]) # Signal matrix
        
        if load_flag == True:
            # Load the file and extract the content
            data = Clib.load_npy(params.save_path + '/scope/' + file_name)
            # Set the dependent vectors
            self.time_line = data[0,:].reshape((1, -1))
            self.signals = data[1:,:]
            self.n_signals = np.size(self.signals, 0)

        # If the initial input does not exist, set it zero
        # Else, put the initial condition in the state matrix
        inish = initial_condition.shape
        if sum(inish) == self.n_signals or sum(inish) == self.n_signals + 1:
            # If the imported initial value is not a column vector, do this:
            initial_condition = np.reshape(initial_condition, (-1, 1))
            self.signals += 1
            self.signals  = initial_condition*self.signals
        elif sum(inish) != 1 and sum(inish) != 2:
            if inish == (self.n_signals, np.size(self.time_line)) \
              or np.size(self.time_line) == 1 \
              or np.size(self.time_line) == inish[1]:
                self.signals = initial_condition
                self.n_signals = inish[0]
                if auto_create_time_line == True:
                    self.time_line = np.arange(0, np.size(initial_condition, 1)).reshape(1,-1)
            else:
                err_text = "The dimensional of initial value that inserted is wrong. Check it please."
                logging.error(err_text)
                raise ValueError(err_text)
        
        self.current_step = 0 # The current step of simulation
        self.n = np.size(self.time_line) # The number of time steps
        
    # The 'get' function can receive value and keep it.
    def get(self, data, **kwargs):
        '''
        ### Overview:
        Getting data given step-by-step.

        ### Input variables:
        * Getting data at step `k`

        ### Configuration Options:
        * `noise` is used to add a noise as the measurement noise
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Inserted data, additive noise Variance
        
        add_noise_var = 0
        # Extracting the arbitraty value of properties
        for key, val in kwargs.items():
            # The noise variance
            if key == 'noise': add_noise_var = val

        # If the noise signals do not exist, consider them zero.
        noise_signal = np.random.normal(0, add_noise_var, [self.n_signals, 1])
        
        # Update internal signals which later can be used for plotting
        # and programming for other parts of the code
        self.signals[:, self.current_step] = np.array(data).flatten() + noise_signal.flatten()
        self.current_step += 1
        
    # The 'save' function saves the internal data.
    def store(self, params, name = 'Unknown', **kwargs):
        '''
        ### Overview:
        Used to save data as a `.npy` file.

        ### Input variables:
        * `params`
        * `name` - The name of the file (is not necessary)

        ### Configuration Options:
        * `zip` - is used to save as a `.zip` file; default = `False`.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Requirements
        from core.lib.core_library import Clib
        # The directory
        save_path = params.save_path + '/scope/'
        # Changing the file name
        if params.unique == True:
            name = name + '_' + params.id
        # Preparing data
        data = np.concatenate((self.time_line, self.signals), axis=0)
        Clib.save_npy(data, save_path + name, **kwargs)

    # This function can make a jump in the step number variable
    # If no arguments are available, jump 1 step
    def __iadd__(self, i = 1):
        self.current_step += i

    # Adding two similar Scope
    def __add__(self, other):
        if isinstance(other, Scope):
            signals = self.signals + other.signals
            n_signals = self.n_signals
            return Scope(time_line = self.time_line, n_signals = n_signals, initial = signals)
        else:
            return False
    
    # Subtracting two similar Scope
    def __sub__(self, other):
        if isinstance(other, Scope):
            signals = self.signals - other.signals
            n_signals = self.n_signals
            return Scope(time_line = self.time_line, n_signals = n_signals, initial = signals)
        else:
            return False
    
    # Multiply two similar Scope
    def __mul__(self, other):
        if isinstance(other, Scope):
            signals = self.signals * other.signals
            n_signals = self.n_signals
            return Scope(time_line = self.time_line, n_signals = n_signals, initial = signals)
        else:
            return False
        
    # Division two similar Scope
    def __truediv__(self, other):
        if isinstance(other, Scope):
            signals = self.signals / other.signals
            n_signals = self.n_signals
            return Scope(time_line = self.time_line, n_signals = n_signals, initial = signals)
        else:
            return False
    
    # Reset Block by changing the current step to zero
    def reset(self):
        '''
        ### Overview:
        Reseting the block via changing the current step to zero.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        self.current_step = 0

    # Appending
    def append(self, other):
        """
        ### Overview:
        Appending signals with similar `time_line`.
        
        ### Input variables:
        `other` - the second Scope object that is supposed to be appended to the first one.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        """
        if isinstance(other, Scope):
            obj = Scope(time_line = self.time_line,
                        n_signals = self.n_signals + other.n_signals,
                        initial = np.concatenate((self.signals, other.signals), axis=0))
            return obj
        else:
            return False
    
    # Removing signals
    def remove(self, rows_to_remove):
        """
        ### Overview:
        Eliminating signals.
        
        ### Input variables:
        `rows_to_remove` - the rows is supposed to be removed.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        """
        # Remove the second row (index 1)
        signals = np.delete(self.signals, rows_to_remove, axis=0)
        n_signals = self.n_signals - np.size(rows_to_remove)
        return Scope(time_line = self.time_line, n_signals = n_signals, initial = signals)
    
    # Selecting signals
    def select(self, rows_to_select):
        """
        ### Overview:
        Selecting arbitrary signals.
        
        ### Input variables:
        `rows_to_select` - the rows is supposed to be selected.
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        """
        # Remove the second row (index 1)
        signals = self.signals[rows_to_select, :]
        n_signals = np.size(rows_to_select)
        return Scope(time_line = self.time_line, n_signals = n_signals, initial = signals)
    
    # The below function is used to plot the internal signals
    def show(self, params, **kwargs):
        '''
        ### Overview:
        This function makes a quick plot of internal signals.

        ### input variables:
        * `params`

        ### Configuration Options:
            * `select` - choose signals arbitrarily; e.g., `select=[0,2,6]`
            * `derive` - get derivatives of signals; can be used in different ways:
                * `derive=False` or `derive=True`; default is `False`
                * `derive=[1,1,0]` - get derivatives of selected signals; set to `1` or `True` for signals you want to derive
            * `no_time` - remove time and create timeless plots; can be set in different ways:
                * `no_time=[0,1]` or `no_time=[0,1,2]` - create 2D or 3D plots of signals; numbers are signal indices
                * `no_time=[[0,1], [1,2]]` or `no_time=[[0,1,2], [3,0,1]]` - create 2D or 3D plots of different signal groups; numbers are signal indices
            * `save` - name of file to save plot as; can be `image.png/pdf/jpg` or `True` to choose automatically
            * `x_label`, `y_label`, and `z_label` - titles for x, y, and z axes of plot
            * `x_lim`, `y_lim`, and `z_lim` - titles for x, y, and z axes limitation
            * `x_line` and `y_line` - is used to plot x or y lines; the adjusted data must be a vector of x and y values
            * `legend` - control legend display:
                * `legend=True` and `legend=False` - enable and disable legend
                * `legend=('day', 'night')` - enable legend with specified labels
            * `line_width` - set line width
            * `grid` - enable grid on plot (`True` or `False`)
            * `leg_col` - control number of columns in legend (positive integer)
            * `leg_loc` - the location which the legend must be illustrated
            
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Get the input arguments
        select = -1
        derive = [[0]]
        no_time = [[0]]
        save = False
        x_label = '$x$'
        y_label = '$y$'
        z_label = '$z$'
        title = ''
        legend = -1
        line_width = 0.5
        grid = 0
        n_col = 1
        leg_loc = 'best'
        x_lim = 0
        y_lim = 0
        z_lim = 0
        x_line = []
        y_line = []
        for key, val in kwargs.items():
            if key == 'select': select = val
            if key == 'derive': derive = val
            if key == 'no_time': no_time = val
            if key == 'save': save = val
            if key == 'x_label': x_label = val
            if key == 'y_label': y_label = val
            if key == 'x_lim': x_lim = val
            if key == 'y_lim': y_lim = val
            if key == 'z_lim': z_lim = val
            if key == 'x_line': x_line = val
            if key == 'y_line': y_line = val
            if key == 'z_label': z_label = val
            if key == 'title': title = val
            if key == 'legend': legend = val
            if key == 'line_width': line_width = val
            if key == 'grid': grid  = val
            if key == 'leg_col': n_col = val
            if key == 'leg_loc': leg_loc = val

        # Extracting the arbitraty values of properties
        if select == -1: select = range(0, self.n_signals)
        no_time = np.array(no_time)
        if no_time.shape == (len(no_time),):
            no_time = np.reshape(no_time, (1, -1))
        derive = np.array(derive)

        if np.size(no_time, 1) == 1:
            # Plot time axis signals

            # Starting to create and plot
            h  = plt.figure(tight_layout=True)
            ax = h.subplots()

            # Pre-processing on data
            processed_signals = self.signals[select,0:-2]
            # If all derivatives are requested, make them here
            if derive.any():
                if derive.all():
                    processed_signals = processed_signals - np.roll(processed_signals, +1, axis=1)
                else:
                    indi = (derive>0).flatten()
                    processed_signals = processed_signals - np.diag(indi).dot(np.roll(processed_signals, +1, axis=1))
                processed_signals[:,0] = processed_signals[:,1]
                processed_signals[:,-1] = processed_signals[:,-2]
            
            # General time line plot
            for i in range(0, processed_signals.shape[0]):
                label_text = '[' + str(select[i]) + ']'
                ax.plot(self.time_line[0, 0:-2].flatten(), processed_signals[i,:], lw = line_width, label = label_text)
            ax.set_xlabel(r'Time (s)')
            if x_label != '$x$': plt.xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim([self.time_line[0, 0], self.time_line[0, -1]])
        
        elif np.size(no_time, 1) == 2:
            # 2D signal line plot

            # Starting to create and plot
            h  = plt.figure(tight_layout=True)
            ax = h.subplots()

            for i in range(0, no_time.shape[0]):
                # Pre-processing on data
                processed_signals = self.signals[no_time[i,:], 0:-2]
                x_label_2 = x_label
                y_label_2 = y_label
                # Affect derivatives
                if derive.any():
                    # If all derivatives are requested, make them here
                    if derive.all():
                        processed_signals = processed_signals - np.roll(processed_signals, +1, axis=1)
                        x_label_2 = 'd$x/$d$t$'
                        y_label_2 = 'd$y/$d$t$'
                    else:
                        indi = (derive>0).flatten()
                        processed_signals = processed_signals - np.diag(indi).dot(np.roll(processed_signals, +1, axis=1))
                        if derive[0] != 0: x_label_2 = 'd$x/$d$t$'
                        if derive[1] != 0: y_label_2 = 'd$y/$d$t$'
                    processed_signals[:,0] = processed_signals[:,1]
                    processed_signals[:,-1] = processed_signals[:,-2]
                # Plotting
                label_text = '[' + str(no_time[i, 0]) + ', ' + str(no_time[i, 1]) + ']'
                ax.plot(processed_signals[0,:], processed_signals[1,:], lw = line_width, label =  label_text)
            # The end of the for loop
            ax.set_xlabel(x_label_2)
            ax.set_ylabel(y_label_2)

            # Legend options
            # 'frameon' can hide the background and the border of the legend
            # 'fontsize' changes the font. The value must be True or False
            # 'bbox_to_anchor' can change the position of the legend arbitrary
            # 'ncol' indicated the number of columns that components are arranged.
            # 'mode' implies the mode of legend whcih can be 'expand' or ?
            ax.legend(loc='upper right', frameon=False, fontsize=14, bbox_to_anchor=(0,1,1,0), ncol=n_col)

        elif np.size(no_time, 1) == 3:
            # 3D signal line plot

            # Starting to create and plot
            h  = plt.figure(tight_layout=True)
            # This order can prepare the are for 3D curve plotting
            ax = h.add_subplot(projection='3d')

            for i in range(0, no_time.shape[0]):
                # Pre-processing on data
                processed_signals = self.signals[no_time[i,:], 0:-2]
                x_label_2 = x_label
                y_label_2 = y_label
                z_label_2 = z_label
                # If all derivatives are requested, make them here
                if derive.any():
                    if derive.all():
                        processed_signals = processed_signals - np.roll(processed_signals, +1, axis=1)
                        x_label_2 = 'd$x/$d$t$'
                        y_label_2 = 'd$y/$d$t$'
                        z_label_2 = 'd$z/$d$t$'
                    else:
                        indi = (derive>0).flatten()
                        processed_signals = processed_signals - np.diag(indi).dot(np.roll(processed_signals, +1, axis=1))
                        if derive[0] != 0: x_label_2 = 'd$x/$d$t$'
                        if derive[1] != 0: y_label_2 = 'd$y/$d$t$'
                        if derive[2] != 0: z_label_2 = 'd$z/$d$t$'
                    processed_signals[:,0]  = processed_signals[:,1]
                    processed_signals[:,-1] = processed_signals[:,-2]
                # Plotting
                label_text = '[' + str(no_time[i, 0]) + ', ' + str(no_time[i, 1]) + ', ' + str(no_time[i, 2]) + ']'
                ax.plot(processed_signals[0, 1:-2], processed_signals[1, 1:-2], processed_signals[2, 1:-2], lw = line_width, label = label_text)
            # The end of the for loop
            ax.set_xlabel(x_label_2)
            ax.set_ylabel(y_label_2)
            ax.set_zlabel(z_label_2)

            ax.legend(loc='upper right', frameon=False, fontsize=14, bbox_to_anchor=(0,1,1,0), ncol=n_col, mode='expand')
        # The end of the if and elif
        
        if x_label != '$x$': ax.set_xlabel(x_label)
        if y_label != '$y$': ax.set_ylabel(y_label)
        if z_label != '$z$': ax.set_zlabel(z_label)
        if title != '':    ax.set_title(title)
        
        # Set the legend if it is required
        if legend == -1:
            pass
        elif legend == 0:
            ax.legend().remove()
        elif legend == 1:
            ax.legend(loc=leg_loc, frameon=False, bbox_to_anchor=(0,1,1,0), ncol=n_col)
        else:
            ax.legend(legend, loc=leg_loc, frameon=False, bbox_to_anchor=(0,1,1,0), ncol=n_col)
        
        if x_lim != 0: ax.set_xlim(x_lim)
        if y_lim != 0: ax.set_ylim(y_lim)
        if z_lim != 0: ax.set_zlim(z_lim)
        
        if np.size(y_line) != 0:
            for y_l in y_line:
                ax.axhline(y=y_l, color='#888', linestyle='-.')
        
        if np.size(x_line) != 0:
            for x_l in x_line:
                ax.axvline(x=x_l, color='#888', linestyle='-.')
                
        # Draw a grid net
        if grid != 0: ax.grid(True)

        # Make it pretty + saving engine
        Plib.isi(params, h, save=save)
    # End of function

    # The below function is used to plot a raster plot
    def raster(self, params, **kwargs):
        '''
        ### Overview:
        This function creates a raster plot of internal signals.

        ### Input variables:
        * `params`

        ### Configuration Options:
            * `select` - choose signals arbitrarily; e.g., `select=[0,2,6]`
            * `derive` - get derivatives of signals; can be used in different ways:
                * `derive=False` or `derive=True`; default is `False`
                * `derive=[1,1,0]` - get derivatives of selected signals; set to `1` or `True` for signals you want to derive
            * `save` - name of file to save plot as; can be `image.png/pdf/jpg` or `True`
            * `x_label` and `y_label` - titles for x and y axes of plot
            * `title` - title of figure
            * `color_bar` - enable or disable color bar (`True` or `False`)
            * `color_limit` - restrict color bar values; e.g., `[0, 1]`
            * `bar_ticks` - change ticks of color bar
            * `cmap` - change colors arbitrarily; use pre-made ones like `mpl.cm.RdGy` and `mpl.cm.RdYlGn` or create your own with `Plib.linear_gradient()` or `Plib.cmap_maker()`
            * `hw_ratio` - height to width ratio; value between 0 and 1; default is `0.68`
            * `interpolation` - blurization; can be set to `none`, `nearest`, `bilinear`, or `bicubic`
            * `rasterized` - change vector graph elements into images (`True` or `False`) to reduce file size significantly
        
        ### Copyright:
        Copyright (c) 2023, Abolfazl Delavar, all rights reserved.
        Web page: https://github.com/abolfazldelavar/dyrun
        '''
        # Get the input arguments
        select = np.arange(0, self.n_signals)
        derive = False
        save = False
        x_label = 'Time (s)'
        y_label = 'Signal index'
        title = False
        color_bar = True
        color_limit = False
        hw_ratio = 0.68
        interp = 'none'
        rasterize = True
        switcher_1 = 1
        switcher_2 = 1
        color_limit = [0, 0]
        gradient = mpl.cm.RdGy

        for key, val in kwargs.items():
            if key == 'select': select = val
            if key == 'derive': derive = val
            if key == 'save': save = val
            if key == 'x_label': x_label = val
            if key == 'y_label': y_label = val
            if key == 'title': title = val
            if key == 'color_bar': color_bar = val
            if key == 'color_limit':
                color_limit = val
                switcher_1 = 0
            if key == 'bar_ticks':
                bar_ticks = val
                switcher_2 = 0
            if key == 'cmap': gradient = val
            if key == 'hw_ratio': hw_ratio = val
            if key == 'interpolation': interp = val
            if key == 'rasterized': rasterize = val

        # Get data
        processed_signals = self.signals[select,0:-2]

        # If the derivatives are requested, make them here
        if derive == True:
            processed_signals = processed_signals - np.roll(processed_signals, +1, axis=1)
            processed_signals[:,0] = processed_signals[:,1]
            processed_signals[:,-1] = processed_signals[:,-2]

        ## Plot part        
        #  Starting to create and plot
        h  = plt.figure(tight_layout=True)
        ax = h.subplots()

        # Plot the images
        im_width = self.time_line[0,-1]
        im_height = hw_ratio*im_width
        v_min = processed_signals.min()*switcher_1 + (1-switcher_1)*color_limit[0]
        v_max = processed_signals.max()*switcher_1 + (1-switcher_1)*color_limit[1]

        im = ax.imshow(processed_signals,
                interpolation = interp, # nearest, bilinear, bicubic
                cmap = gradient, # RdYlGn
                origin = 'lower', # This can reverse the Y axis
                extent = [0, im_width, 0, im_height],
                vmax = v_max, # The maximum value
                vmin = v_min, # The mimimum value
                rasterized = rasterize)

        axins = inset_axes(
            ax,
            width = "3%", # width: 5% of parent_bbox width
            height = "100%", # height: 50%
            loc = "lower left",
            bbox_to_anchor = (1.02, 0., 1, 1),
            bbox_transform = ax.transAxes,
            borderpad = 0,
        )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title != False: ax.set_title(title)

        # Change the appearance to LaTeX form
        Plib.isi(params, h, save=save)

        # Color Bar
        if switcher_2 == 1:
            bar_ticks = [v_min, v_max]
            
        if color_bar == True:
            h.colorbar(im, cax=axins, ticks=bar_ticks)
        # Modify the Y ticks
        ax.set_yticks([0, im_height], [str(select[0]), str(select[-1])])
# End of class
