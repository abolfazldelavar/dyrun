
## // --------------------------------------------------------------
#    ***FARYADELL SIMULATION FRAMEWORK***
#    Creator:   Abolfazl Delavar
#    Web:       https://github.com/abolfazldelavar
## // --------------------------------------------------------------

# Loading the requirements
from core.lib.pyRequirment import *

class clib():
    # Default functions, neccessary and essential functions that have been
    # provided to be used in projects. To use, you can call 'lib.func' in
    # anywhere you need.

    # The below function print a comment periodically
    @staticmethod
    def disit(k, n, trig, params):
        # Trig: [Started time, Trigger, Previous k]
        nowt = time() - trig[0]
        if int(nowt) >= trig[1]:
            trig[1] = trig[1] + params['commandIntervalSpan']   # To make a horizon
            mea     = k - trig[2]                               # Calculatation of the mean
            trig[2] = k
            trem    = (n - k)*params['commandIntervalSpan']/mea # Remained time in second
            tremmin = min(int(trem/60),1e5)                     # How minutes
            tremsec = round(trem%60)                            # Remained seconds
            comPerc = round(k/n*100)                            # Completed percentage
            
            # Making a graphical text
            txt     = ''
            eachBox = 16
            # 1st box
            nn   = str(k)
            reem = ' '*(eachBox - len(nn))
            txt  = txt + '  ' + nn + reem + '|'
            # 2nd box
            nn   = str(n)
            reem = ' '*(eachBox - len(nn))
            txt  = txt + '  ' + nn + reem + '|'
            # 3rd box
            nn   = str(comPerc)
            reem = ' '*(eachBox - len(nn))
            txt  = txt + '  ' + nn + reem + '|'
            # 4th box
            nn   = str(tremmin) + ':' + str(tremsec)
            reem = ' '*(eachBox - len(nn) + 4)
            txt  = txt + '  ' + nn + reem
            # Releasing the provided text
            print(txt)
            # Starting to append command context into the diary file
            if params['makeDiary'] == True:
                with open(params['diaryDir'] + '/' + params['diaryFile'] + '.txt', 'a', encoding='utf-8') as f:
                    f.write('\n' + txt)
        # Sending data for the next step
        return trig

    # Saying a hello at the start of the simulation
    @staticmethod
    def sayStart(params):
        # [Start time] <- (Parameters)
        # Print the header
        txt = list(['', '', '', '', '', ''])
        txt[1] = 'Faryadell Simulation Framework (FSF) - Version 1.0.0'
        txt[2] = 'The simulation has kicked off! (' + clib.getNow(5,'/') + ', ' + clib.getNow(6,':') + ')'
        # Below codes make the header table
        txt[3] = '-'*79
        txt[4] = '  Current step    |  All steps       |  Progress (%)    |  Remained time (m:s) '
        txt[5] = '-'*79
        fullTx = '\n'.join(txt)
        print(fullTx)
        # Check the directory and start a diary to save command window
        if params['makeDiary'] == True:
            if not os.path.isdir(params['diaryDir']): os.makedirs(params['diaryDir'])
            with open(params['diaryDir'] + '/' + params['diaryFile'] + '.txt', 'a', encoding='utf-8') as f:
                f.write(fullTx)
        # Send the current time to the output
        return time()

    # Report the simulation time when it finishes
    @staticmethod
    def sayEnd(starttime, params):
        # Stopping the timer which started before, and saving the time.
        ntime   = time() - starttime
        tmin    = min(int(ntime/60), 1e5) # How minutes
        tsec    = round(ntime%60, 4)      # Remained seconds
        txt     = list(['', ''])
        txt[0]  = '__' + chr(7424) + chr(7427) + chr(7439) + chr(7436) + \
                   chr(4325) + chr(7424) + chr(4301) + chr(7436) + ' ' + \
                   chr(7429) + chr(7431) + chr(7436) + chr(7424) +       \
                   chr(7456) + chr(7424) + chr(7450) + '_'*61 
        txt[1]  = 'The simulation has been completed. (' + str(tmin) +   \
                   ' minutes and ' + str(tsec) + ' seconds)'
        fullTx  = '\n'.join(txt)
        print(fullTx)
        # Starting to append command context into the diary file
        if params['makeDiary'] == True:
            with open(params['diaryDir'] + '/' + params['diaryFile'] + '.txt', 'a', encoding='utf-8') as f:
                f.write('\n' + fullTx)

    ## Returning a text contained date and time
    @staticmethod
    def getNow(typeReport = 0, splitchar = '_'):
        '''
        To make a delay in a descrete function is used.
        
        Input variables:
            * The type of report
                * default: `YMDHMS`
                * 1: `YMD_HMS`
                * 2: `Y_M_D_H_M_S`
                * 3: `YMD_HM`
                * 4: `YMD_HM`
                * 5: `Y_M_D`
                * 6: `H_M_S`
            * Splitter; like `_` or `*`
            
        Output variable:
            * Output string
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
        if typeReport == 1:
            txt = year + month + day + splitchar + hour + minute + second
        elif typeReport == 2:
            txt = year + splitchar + month + splitchar + day + splitchar + \
                  hour + splitchar + minute + splitchar + second
        elif typeReport == 3:
            txt = year + month + day + splitchar + hour + minute
        elif typeReport == 4:
            txt = year + month + day + splitchar + hour + minute
        elif typeReport == 5:
            txt = year + splitchar + month + splitchar + day
        elif typeReport == 6:
            txt = hour + splitchar + minute + splitchar + second
        else:
            txt = year + month + day + hour + minute + second
        # Returning the output
        return txt

        
    ## Delayed in a signal
    @staticmethod
    def delayed(u, k, pdelay):
        '''
        To make a delay in a descrete function is used.
        
        Input variables:
            * Full signal
            * Current point `k`
            * Delay amount (in integer)
            
        Output variable:
            * delayed value
        '''
        if k - pdelay >= 0:
            y = u[:, k - pdelay]
        else:
            y = u[:, k]*0
        return y

    ## Signal Generator
    @staticmethod
    def signalMaker(params, tLine):
        # SETUP -------------------------------------------------------------------------
        # 1) Insert the below code in 'setParameters.m' to use signal generator:
        #        ## Signal Generator parameters
        #        params.referAddType  = 'onoff';    % 'none', 'square', 'sin', 'onoff', ...                        
        #        params.referAddAmp   = 30;         % Amplitude of additive signal
        #        params.referAddFreq  = 2;          % Signal period at simulation time
        #
        # 2) Use the below code to call signal generation in 'modelMaker.m':
        #        signal = func.signalMaker(params, models.Tline);
        # -------------------------------------------------------------------------------
        tLine = tLine[0:params.n]
        if params.referaddtype == 'none':
            return tLine*0
        elif params.referaddtype == 'square':
            freq    = params.referaddfreq/params.tout
            return np.sign(np.sin(2*math.pi*freq*tLine + 1e-20)) * params.referaddamp
        elif params.referaddtype == 'onoff':
            freq    = params.referaddfreq/params.tout
            return (2 - np.sign(np.sin(2*math.pi*freq*tLine + 1e-20)) - 1)/2 * params.referaddamp
        elif params.referaddtype == 'sin':
            freq    = params.referaddfreq/params.tout
            return  np.sin(2*math.pi*freq*tLine) * params.referaddamp

           
    ## Making a signal of an exponential inverse
    @staticmethod
    def expInverse(tLine, bias, alph, areaa):
        '''
        Sigmoid generator function.

        Input variables:
        * Time line
        * Time delay bias
        * Smoother
        * Domain in form of [a, b]
        
        Output variable:
        * The output signal
        '''
        # Note that signal domain is a two component vector [a(1), a(2)]
        output = 1/(1 + np.exp(-alph*(tLine - bias)))
        return (areaa[1] - areaa[0])*output + areaa[0]
    
    ## Making an exponential signal
    @staticmethod
    def exponensh(tLine, Sr, para):
        '''
        Exposential function.

        Input variables:
        * Time line
        * Smoother
        * Domain in form of [a, b]
        
        Output variable:
        * The output signal
        '''
        Ss = para(1);        # Start point
        Sf = para(2);        # Final value
        return (Ss - Sf)*np.exp(-Sr*tLine) + Sf

    
    ## Linear mapping a number
    @staticmethod
    def lineMap(x, fro, to):
        '''
        Linear Mapping the point (or array) `x` from `[a1, b1]` domain to `y` in domain `[a2, b2]`

        Input variables:
        * array
        * `x` domain: `[a1, b1]`
        * `y` domain: `[a2, b2]`
        
        Output variable:
        * The output array
        '''
        # Map 'x' from band [w1, v1] to band [w2, v2]
        w1        = np.array(fro[0])
        v1        = np.array(fro[1])
        w2        = np.array(to[0])
        v2        = np.array(to[1])
        x         = np.array(x)
        output    = 2*((x - w1)/(v1 - w1)) - 1
        output    = (output + 1)*(v2 - w2)/2 + w2
        return output
# The end of the class

class solverCore():
    # Dynamic Solver: This funtion contains some numerical methods
    # like 'euler', 'rng4', etc., which can be used in your design.
    @staticmethod
    def dynamicRunner(handleDyn, xv, xo, sTime, solverType):
        '''
        To calculate a prediction using a `handler` of the function, this function could be utilized.

        Input variables:
        * handler (make one by `lambda x: x**2 + 5`)
        * Full time vector of states
        * current states
        * Sample time
        * Solver (`euler`, `rng4`)
        
        Output variable:
        * Predicted states
        '''
        if solverType == 'euler':
            # Euler method properties is given below (T is sample time):
            #   x(t+1) = x(t) + T*f(x(t))
            xn = xo + sTime*handleDyn(xv)
        elif solverType == 'rng4':
            # 4th oder of 'Runge Kutta' is described below (T is sample time):
            #   K1     = T*f(x(t))
            #   K2     = T*f(x(t) + K1/2)
            #   K3     = T*f(x(t) + K2/2)
            #   K4     = T*f(x(t) + K3)
            #   x(t+1) = x(t) + 1/6*(K1 + 2*K2 + 2*K3 + K4)
            K1 = sTime*handleDyn(xv)
            K2 = sTime*handleDyn(xv + K1/2)
            K3 = sTime*handleDyn(xv + K2/2)
            K4 = sTime*handleDyn(xv + K3)
            xn = xo + 1/6*(K1 + 2*K2 + 2*K3 + K4)
        else:
            raise ValueError('The solver name is not correct, please change the word "' + solverType + '"')
        return xn


class plib():
    # In this library, all functions are related to plotting and depiction are
    # provided which can be used in 'depiction.py' file.

    def __init__(self):
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

    @staticmethod
    def isi(params, fig = 0, save = False, width = 8.5, hwRatio = 0.65):
        '''
        Making plots prettier and ready to use in academic purposes.

        Input variables:
        * `params`
        * Figure handler (Use `h = plt.figure(tight_layout=True)` to make one)
        * Saving as a file - If you want the illustration is saved, enter the name of that,
          like `image.png/pdf/jpg`, or just insert `True`
        * Width; default is `8.5 inch`
        * Height to width ratio between 0 and 1; default is `0.65`
        
        '''
        # This function make the graph pretty.
        
        if fig == 0: raise ValueError("Please enter the figure handler.")
        # Changing size of the figure as you ordered
        fig.set_figwidth(width)
        fig.set_figheight(width*hwRatio)

        # The number of axis existed in fig
        nAx = len(fig.axes)
        for i in range(0, nAx):
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
        # The end of the loop

        # Saving the graph, if it is under demand.
        # User must import the figure name as 'save' var
        if isinstance(save, str) or isinstance(save, int):
            plib.figureSaveCore(params, save, fig)
    # The end of the function

    @staticmethod
    def figureSaveCore(params, save = True, fig = plt.gcf(), dpi = 300):
        '''
        Use this function to save an illustration.

        Input variables:
        * `params`
        * Saving as a file - enter the name, like `image.png/pdf/jpg`, insert `True`, or just let it go
        * Figure handler (Use `h = plt.figure(tight_layout=True)` to make one)
        * Dots per inch; default is `300 pixel`
        '''

        # To get current PC time to use as a prefix in the name of file
        savePath   = params['savePath'] + '/figs'
        # Default saving format
        fFormat    = params['defaultImageFormat']
        AllFormats = ['jpg', 'png', 'pdf']
        isSetDirec = False
        needToSetUniqAgain = True

        # Extracting the name and the directory which inported
        if (not isinstance(save, str)) and (isinstance(save, bool) or isinstance(save, int)):
            # Set the time as its name, if there is no input in the arguments
            save = 'Faryad-' + clib.getNow(3,'-')
            needToSetUniqAgain = False
        elif isinstance(save, str):
            # Split folders
            tparts = save.split('/')
            if len(tparts) > 1:
                # Directory maker
                savePath   = '/'.join(tparts[0:-1])
                fparts     = tparts[-1].split('.')
                isSetDirec = True
            else:
                # If directory is not adjusted:
                fparts = tparts[0].split('.')
            
            # Name and format
            if len(fparts) > 1:
                # The name is also adjusted:
                if len(fparts[0]) == 0 or len(fparts[-1]) == 0:
                    raise ValueError('Please enter the correct notation for the file name.')
                elif not (fparts[-1] in AllFormats):
                    raise ValueError('You must input one of these formats: png/jpg/pdf')

                save    = ''.join(fparts[0:-1])
                fFormat = fparts[-1]
            else:
                # One of name or format just is inserted
                # There is just name or format. It must be checked
                if fparts[-1] in AllFormats:
                    fFormat = fparts[0]
                    # Set the time as its name, if there is no input in the arguments
                    save = 'Faryad-' + clib.getNow(3,'-')
                    needToSetUniqAgain = False
                else:
                    # Just a name is imported, without directory
                    # and any formats
                    save = fparts[0]
        else:
            raise ValueError("Please enter the name of file correctly, or the expression of 'True'.")

        savePath = str(savePath)
        fFormat  = str(fFormat)
        save     = str(save)
        
        # Changing the file name 
        if params['uniqueSave'] == 1 and needToSetUniqAgain:
            fName = save + '_' + clib.getNow()
        else:
            fName = save
        
        # Prepare direct
        if isSetDirec == 0:
            fDir = savePath + '/' + fFormat
        else:
            fDir = savePath
        
        # Check the folders existance and make them if do not exist
        if not os.path.isdir(fDir): os.makedirs(fDir)
        
        # Saving part
        fullName = fDir + '/' + fName + '.' + fFormat
        if fFormat == 'png':
            fig.savefig(fullName, transparent=True, dpi=dpi)
        elif fFormat == 'pdf':
            fig.savefig(fullName, transparent=True, dpi=dpi)
        elif fFormat == 'jpg':
            fig.savefig(fullName, transparent=True, dpi=dpi)
        # Print the result of saving
        print('The graph named "' + fName + '.' + fFormat + '" has been saved into "' + fDir + '".')
    # The end of the function

    @staticmethod
    def linGradient(colors, locs, num = 256, showIt=False):
        '''
        To make a linear gradient from one color to another, use this option.

        Input variables:
        * Colors; e.g., `[[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]`
        * Splitting points; e.g., `[0, 0.1, 1]`
        * the number of colors in output; default is `256`
        * If you want to have a pre-shown of the gradient, set this `true`

        Output variables:
        * An array comprising all colors
        '''

        locs   = np.reshape(np.double(locs), [1, np.size(locs)])
        colors = np.array(colors)

        numofGrads = colors.shape[0] - 1
        numofSigns = colors.shape[1]
        cols       = np.zeros([1, numofGrads])
        areaa      = np.array([locs[0, 0], locs[0, -1]])
        
        func = clib()

        for i in range(0, numofGrads+1):
            locs[0, i] = func.lineMap(locs[0, i], areaa, np.array([0, 1]))

        for i in range(0, numofGrads):
            cols[0, i] = np.ceil(num * (locs[0, i + 1] - locs[0, i]) )
        cols[0,-1] = num - (int(np.sum(cols[0,0:-1])) - numofGrads) 

        initcolors = np.zeros([int(np.sum(cols)), numofSigns])
        shifft     = 0

        # Gradient maker
        for i in range(0, numofGrads):
            color1  = colors[i, :]
            color2  = colors[i + 1, :]
            gradian = np.zeros([int(cols[0, i]), numofSigns])
            for j in range(0,numofSigns):
                gradian[:,j] = np.interp(np.linspace(0, 1, int(cols[0, i])), [0, 1], [color1[j], color2[j]] )
            tem1 = (np.arange(0, int(cols[0,i])) + shifft + int(i<1) - 1).flatten()
            initcolors[np.int16(tem1), :] = gradian
            shifft  = shifft + cols[0,i] - 1

        outputColors = initcolors[:num, :]

        # Plot gradient
        if showIt == 1:
            plt.figure()
            img = np.ones([256, 256])
            u = np.linspace(0, 1, 256)
            img = u*img
            plt.imshow(img)
            plt.colorbar()
            plt.show()
        # Send to the output
        return outputColors

    def cmapMaker(Name, Colors, N=256):
        '''
        This function is used to make a Linear Segmented Color Map (LSCM).

        Input variables:
        * Name
        * Colors and their location in the graph line; e.g., `[(0, '#ffff00'), (0.25, '#002266'), (1, '#002266')]`
        * The number of colors in the output
        '''
        # Instruction
        # Colors = [(0, '#ffff00'), (0.25, '#002266'), (1, '#002266')]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(Name, Colors, N)
        return cmap
# The end of the class
        
        
#         function dark(obj, varargin)
#             % The function generates from a Matlab plot figure a version that can be
#             % copied to a dark mode theme presentation or website.
#             % The function replaces the default texts and box colors to
#             % a user input color (default is white), and make the plot area transparent
#             % to accept the dark background below it. The function also transform the
#             % graphic colors that are not appropriate (low contrast) for a dark mode
#             % theme to a version that is dark theme legible using a desaturation and
#             % brightness approach.
#             %
#             % preparing this function I was inspired by https://material.io/design/color/dark-theme.html
#             %
#             % The function is a work in progess and may not support all figure objects


#             %  Inputs:
#             %  varargin(1)- The text color to modify (default is white)
#             %  varargin(2)- The threshold from which to apply the cotrast correction (default is 4.5)
#             %  varargin(3)- The dark background  (default is gray of value 0.16)
#             %
#             %
#             %  How to the function:
#             %  generate or import a Matlab figure and run the function:
#             %
#             %       plot(bsxfun(@times,[1:4],[2:5]'));xlabel('X');ylabel('Y');
#             %       plot_darkmode
#             %
#             %  next copy the figure from the clipboard using Edit>Copy Figure and
#             %  paste it on top of the dark background theme, for example in
#             %  PowerPoint. Make sure that in the Copy Option, the  Transparent
#             %  Background is enabled


#             %   Ver 1.02 (2021-09-28)
#             %   Adi Natan (natan@stanford.edu)

#             %% defaults and initialize
#             switch nargin-1
#                 case 3
#                     textcolor           = varargin{1 + 1};
#                     contrast_ratio      = varargin{2 + 1};
#                     dark_bkg_assumption = varargin{3 + 1};
#                 case 2
#                     textcolor           = varargin{1 + 1};
#                     contrast_ratio      = varargin{2 + 1};
#                     dark_bkg_assumption = ones(1,3)*39/255; % 0.16
#                 case 1
#                     textcolor           = varargin{1 + 1};
#                     contrast_ratio      = 4.5;
#                     dark_bkg_assumption = ones(1,3)*39/255; % 0.16
#                 otherwise
#                     textcolor           = [1,1,1]*0.95;
#                     contrast_ratio      = 4.5;
#                     dark_bkg_assumption = ones(1,3)*39/255; % 0.16
#             end

#             tcd = [{textcolor} , {contrast_ratio} , {dark_bkg_assumption}];


#             % Getting the plot type
#             g = get(get(gcf, 'children'), 'type');
#             if ~strcmp(g, 'tiledlayout')
#                 % If it is not a kind of 'tiledlayout' do this

#                 % Getting figure children as a var named 'h'
#                 h             = get(gcf, 'children');
#                 % Providing a handle of each sections
#                 axes_ind      = findobj(h, 'type', 'Axes');
#                 legend_ind    = findobj(h, 'type', 'Legend');
#                 colorbar_ind  = findobj(h, 'type', 'Colorbar');
#             else
#                 % If it is a kind of 'tiledlayout' do this

#                 % Getting figure children as a var named 'h'
#                 h0                = get(gcf, 'children');
#                 h0.Title.Color    = tcd{1};
#                 h0.Subtitle.Color = tcd{1};
#                 % Getting all children of main figure (like subsections of a subplot)
#                 h = get(get(gcf, 'children'), 'children');
#                 % Providing a handle of each sections
#                 axes_ind      = findobj(h, 'type', 'Axes');
#                 legend_ind    = findobj(h, 'type', 'Legend');
#                 colorbar_ind  = findobj(h, 'type', 'Colorbar');
#             end

#             %% modify Axes
#             % Below loop repeats for each subplot of a mother figure
#             for n = 1:numel(axes_ind)

#                 % % Edit x-ticks color
#                 % for m = 1:numel(axes_ind(n).XTickLabel)
#                 %     axes_ind(n).XTickLabel{m} = ['\color[rgb]', sprintf('{%f,%f,%f}%s', textcolor), axes_ind(n).XTickLabel{m}];
#                 % end

#                 % % Edit y-ticks color
#                 % for m = 1:numel(axes_ind(n).YTickLabel)
#                 %     axes_ind(n).YTickLabel{m} = ['\color[rgb]', sprintf('{%f,%f,%f}%s', textcolor), axes_ind(n).YTickLabel{m}];
#                 % end

#                 axes_ind(n).Color           = 'none';      % 'none' or tcd{3}; % make white area transparent
#                 axes_ind(n).XColor          = tcd{1};      % edit x axis color
#                 axes_ind(n).YColor          = tcd{1};      % edit y axis color
#                 axes_ind(n).ZColor          = tcd{1};      % edit z axis color

#                 axes_ind(n).XLabel.Color    = tcd{1};      % edit x label color
#                 axes_ind(n).YLabel.Color    = tcd{1};      % edit y label color
#                 axes_ind(n).ZLabel.Color    = tcd{1};      % edit z label color

#                 axes_ind(n).Title.Color     = tcd{1};      % edit title text color
                
#                 adjust_color                = @(x, y) obj.adjust_color(x, y);
#                 axes_ind(n).GridColor       = adjust_color(axes_ind(n).GridColor,      tcd);
#                 axes_ind(n).MinorGridColor  = adjust_color(axes_ind(n).MinorGridColor, tcd);
#                 % axes_ind(n).Subtitle.Color = textcolor;

#                 % take care of other axes children:
#                 h2              = get(axes_ind(n),'Children');
#                 g2              = get(axes_ind(n).Children,'type');
#                 text_ind        = find(strcmp(g2,'text'));
#                 patch_ind       = find(strcmp(g2,'patch'));
#                 line_ind        = find(strcmp(g2,'line'));
#                 errorbar_ind    = find(strcmp(g2,'errorbar'));
#                 area_ind        = find(strcmp(g2,'area'));
#                 bar_ind         = find(strcmp(g2,'bar'));
#                 hist_ind        = find(strcmp(g2,'histogram'));
#                 % contour_ind  = find(strcmp(g2,'contour'));
#                 % surface_ind = find(strcmp(g2,'surface'));

#                 % edit texts color
#                 for m = 1:numel(text_ind)
#                     h2(text_ind(m)).Color = adjust_color( h2(text_ind(m)).Color ,tcd);
#                     if ~strcmp( h2(text_ind(m)).BackgroundColor,'none')
#                         %if text has some background color switch to dark bkg theme
#                         h2(text_ind(m)).BackgroundColor = tcd{3};
#                     end
#                 end

#                 % brighten patch colors if dim (use for the case of arrows etc)
#                 % this might not work well for all patch types so consider to comment
#                 for m = 1:numel(patch_ind)
#                     h2(patch_ind(m)).FaceColor = adjust_color(h2(patch_ind(m)).FaceColor,tcd);
#                     h2(patch_ind(m)).EdgeColor = adjust_color(h2(patch_ind(m)).EdgeColor,tcd);
#                 end

#                 for m = 1:numel(line_ind)
#                     h2(line_ind(m)).Color = adjust_color(h2(line_ind(m)).Color,tcd);
#                 end


#                 for m = 1:numel(errorbar_ind)
#                     h2(errorbar_ind(m)).Color = adjust_color(h2(errorbar_ind(m)).Color,tcd);
#                     h2(errorbar_ind(m)).MarkerEdgeColor = adjust_color(h2(errorbar_ind(m)).MarkerEdgeColor,tcd);
#                     h2(errorbar_ind(m)).MarkerFaceColor = adjust_color(h2(errorbar_ind(m)).MarkerFaceColor,tcd);
#                 end

#                 for m = 1:numel(area_ind)
#                     h2(area_ind(m)).FaceColor = adjust_color(h2(area_ind(m)).FaceColor,tcd);
#                     h2(area_ind(m)).EdgeColor = adjust_color(h2(area_ind(m)).EdgeColor,tcd);
#                 end

#                 for m = 1:numel(bar_ind)
#                     h2(bar_ind(m)).FaceColor = adjust_color(h2(bar_ind(m)).FaceColor,tcd);
#                     h2(bar_ind(m)).EdgeColor = adjust_color(h2(bar_ind(m)).EdgeColor,tcd);
#                 end


#                 for m = 1:numel(hist_ind)
#                     h2(hist_ind(m)).FaceColor = adjust_color(h2(hist_ind(m)).FaceColor,tcd);
#                     h2(hist_ind(m)).EdgeColor = adjust_color(h2(hist_ind(m)).EdgeColor,tcd);
#                 end

#                 %       for m=1:numel(contour_ind)
#                 %         h2(contour_ind(m)).FaceColor = adjust_color(h2(contour_ind(m)).FaceColor,tcd);
#                 %         h2(contour_ind(m)).EdgeColor = adjust_color(h2(contour_ind(m)).EdgeColor,tcd);
#                 %     end


#             end
#             %% modify Colorbars:
#             for n = 1:numel(colorbar_ind)
#                 colorbar_ind(n).Color        =  textcolor;
#                 colorbar_ind(n).Label.Color  =  textcolor;
#             end

#             %% modify Legends:
#             for n = 1:numel(legend_ind)
#                 legend_ind(n).Color     = 'none';     % make white area transparent
#                 legend_ind(n).TextColor = textcolor;  % edit text color
#                 legend_ind(n).Box       = 'off';      % delete box
#             end


#             %% modify annotations:
#             ha = findall(gcf,'Tag','scribeOverlay');
#             % get its children handles
#             if ~isempty(ha)
#                 for n = 1:numel(ha)
#                     hAnnotChildren = get(ha(n),'Children');
#                     try
#                         hAnnotChildrenType = get(hAnnotChildren, 'type');
#                     catch
#                         disp('annotation not available')
#                         return
#                     end

#                     % edit lineType and shapeType colors
#                     textboxshape_ind        = find(strcmp(hAnnotChildrenType,'textboxshape'));
#                     ellipseshape_ind        = find(strcmp(hAnnotChildrenType,'ellipseshape'));
#                     rectangleshape_ind      = find(strcmp(hAnnotChildrenType,'rectangleshape'));
#                     textarrowshape_ind      = find(strcmp(hAnnotChildrenType,'textarrowshape'));
#                     doubleendarrowshape_ind = find(strcmp(hAnnotChildrenType,'doubleendarrowshape'));
#                     arrowshape_ind          = find(strcmp(hAnnotChildrenType,'arrowshape'));
#                     arrow_ind               = find(strcmp(hAnnotChildrenType,'Arrow')); % older Matlab ver
#                     lineshape_ind           = find(strcmp(hAnnotChildrenType,'lineshape'));


#                     for m = 1:numel(textboxshape_ind)
#                         hAnnotChildren(textboxshape_ind(m)).Color      =  textcolor;
#                         hAnnotChildren(textboxshape_ind(m)).EdgeColor  =  adjust_color(hAnnotChildren(textboxshape_ind(m)).EdgeColor);
#                     end

#                     for m = 1:numel(ellipseshape_ind)
#                         hAnnotChildren(ellipseshape_ind(m)).Color      =  adjust_color(hAnnotChildren(ellipseshape_ind(m)).Color,tcd);
#                         hAnnotChildren(ellipseshape_ind(m)).FaceColor  =  adjust_color(hAnnotChildren(ellipseshape_ind(m)).FaceColor,tcd);
#                     end

#                     for m = 1:numel(rectangleshape_ind)
#                         hAnnotChildren(rectangleshape_ind(m)).Color      =  adjust_color(hAnnotChildren(rectangleshape_ind(m)).Color,tcd);
#                         hAnnotChildren(rectangleshape_ind(m)).FaceColor  =  adjust_color(hAnnotChildren(rectangleshape_ind(m)).FaceColor,tcd);
#                     end

#                     for m = 1:numel(textarrowshape_ind)
#                         hAnnotChildren(textarrowshape_ind(m)).Color      =  adjust_color(hAnnotChildren(textarrowshape_ind(m)).Color,tcd);
#                         hAnnotChildren(textarrowshape_ind(m)).TextColor  =  textcolor;
#                         hAnnotChildren(textarrowshape_ind(m)).TextEdgeColor = adjust_color(hAnnotChildren(textarrowshape_ind(m)).TextEdgeColor,tcd);
#                     end

#                     for m = 1:numel(doubleendarrowshape_ind)
#                         hAnnotChildren(doubleendarrowshape_ind(m)).Color = adjust_color(hAnnotChildren(doubleendarrowshape_ind(m)).Color,tcd);
#                     end

#                     for m = 1:numel(arrowshape_ind)
#                         hAnnotChildren(arrowshape_ind(m)).Color = adjust_color(hAnnotChildren(arrowshape_ind(m)).Color,tcd);
#                     end

#                     for m = 1:numel(arrow_ind)
#                         hAnnotChildren(arrow_ind(m)).Color = adjust_color(hAnnotChildren(arrow_ind(m)).Color,tcd);
#                     end

#                     for m = 1:numel(lineshape_ind)
#                         hAnnotChildren(lineshape_ind(m)).Color = adjust_color(hAnnotChildren(lineshape_ind(m)).Color,tcd);
#                     end
#                 end
#             end
#         end
        
        
#         % End of methods
#     end
#     % End of class
    
#     methods (Hidden=true)
#         function out = adjust_color(~, in, tcd)
#             % This function modifies an input color to fit a dark theme background.
#             % For that a color needs to have sufficient contrast (WCAG's AA standard of at least 4.5:1)
#             % The contrast ratio is calculate via :  cr = (L1 + 0.05) / (L2 + 0.05),
#             % where L1 is the relative luminance of the input color and L2 is the
#             % relative luminance of the dark mode background.
#             % For this case we will assume a dark mode theme background of...
#             % If a color is not passing this ratio, it will be modified to meet it
#             % via desaturation and brightness to be more legible.
#             % the function uses fminbnd, if you dont have the toolbox to use it you can
#             % replace it with fmintx (avaiable in Matlab's file exchange)

#             % if color is 'none' return as is
#             if strcmp(in,'none')
#                 out=in;
#                 return
#             end

#             if isa(in,'char') % for inputs such as 'flat' etc...
#                 out=in;
#                 return
#             end

#             dark_bkg_assumption=tcd{3};

#             % find the perceived lightness which is measured by some vision models
#             % such as CIELAB to approximate the human vision non-linear response curve.
#             % 1. linearize the RGB values (sRGB2Lin)
#             % 2. find Luminance (Y)
#             % 3. calc the perceived lightness (Lstar)
#             % Lstar is in the range 0 to 1 where 0.5 is the perceptual "middle gray".
#             % see https://en.wikipedia.org/wiki/SRGB ,

#             sRGB2Lin=@(in) (in./12.92).*(in<= 0.04045) +  ( ((in+0.055)./1.055).^2.4 ).*(in> 0.04045);
#             %Y = @(in) sum(sRGB2Lin(in).*[0.2126,  0.7152,  0.0722 ]);
#             Y = @(in) sum(bsxfun(@times,sRGB2Lin( in ),[0.2126,  0.7152,  0.0722 ]),2 );
#             Lstar = @(in)  0.01.*( (Y(in).*903.3).*(Y(in)<= 0.008856) + (Y(in).^(1/3).*116-16).*(Y(in)>0.008856));

#             Ybkg = sum(sRGB2Lin(dark_bkg_assumption).*[0.2126,  0.7152,  0.0722 ]);

#             cr = @(in)   (Y(in)' + 0.05) ./ (Ybkg + 0.05); % contrast ratio

#             % rgb following desaturation of factor x
#             ds=@(in,x) hsv2rgb( bsxfun(@times,rgb2hsv(in),[ones(numel(x),1) x(:) ones(numel(x),1)] ));

#             % rgb following brightness change of factor x
#             br=@(in,x) hsv2rgb( bsxfun(@power,rgb2hsv(in),[ones(numel(x),1) ones(numel(x),1) x(:)] ));


#             if cr(in)<tcd{2} % default is 4.5

#                 %check if color is just black and replace with perceptual "middle gray"
#                 if ~sum(in)
#                     fun0 = @(x) abs(Lstar( (ones(1,3)*x-dark_bkg_assumption ))-0.5);
#                     L_factor=fminbnd(fun0,0.3,1);

#                     out = ones(1,3)*L_factor;
#                     return

#                 end


#                 % if saturation is what reduce contrast then desaturate
#                 in_hsv = rgb2hsv(in);
#                 if in_hsv(2) > 0.5
#                     fun1 = @(x) abs(cr(ds(in,x)) - tcd{2});
#                     [ds_factor, val] = fminbnd(fun1, 0, in_hsv(2));
#                     if val < 1e-2
#                         out = ds(in, ds_factor);
#                         return
#                     end
#                 end
#                 % desaturation alone didn't solve it, try to increase brightness
#                 fun2 = @(x) abs(cr(br(in,x)) - tcd{2});
#                 [br_factor, val] = fminbnd(fun2, 0, 1);

#                 if val<1e-2 && Lstar(br(in,br_factor))>0.5
#                     out = br(in,br_factor);
#                     return
#                 end
#                 % if niether worked then brightening + desaturation:
#                 fun3 = @(x) abs(cr(ds(br(in,br_factor),x))-tcd{2});
#                 [brds_factor, val]=fminbnd(fun3,0,1);

#                 if val<1e-2 && Lstar(ds(br(in,br_factor),brds_factor))>0.5
#                     out = ds(br(in,br_factor),brds_factor);
#                     return

#                 end
#                 % if all fails treat the color as black as above:
#                 fun0 = @(x) abs(Lstar( (ones(1,3)*x-dark_bkg_assumption ))-0.5);
#                 L_factor=fminbnd(fun0,0.3,1);
#                 out = ones(1,3)*L_factor;
#             else
#                 out = in ;
#             end
#         end
        
#         % End of the methods
#     end
# end

