# %% [markdown]
# # FARYADELL SIMULATION FRAMEWORK
# This package has been provided to simulate easily and quickly several useful tools that have been written to simulate `linear` and `nonlinear dynamic` systems, dependent or independent over time. If you are a student who whishes to investigate and has decided to seek dynamic and control systems, this is for you. Enjoy it.
# 
# * Creator:   Abolfazl Delavar
# * Web:       https://github.com/abolfazldelavar
# * Updated:   12 April 2023
# 
# "*Successful people are those who can build solid foundations with the bricks others throw at them.*"

# %% [markdown]
# ## Requirements
# All external **extensions** utilized in the project must be added in this section. To have comprehensive accessibility to libraries, if there is some you would like to add, you can put them as an object to have better control and management of your libraries. Furthermore, There are two libraries name `clib` and `plib` placed in `coreLib.py` in which there are several useful functions that could be used for drawing or other purposes.

# %%
# To enable plotting outside of this page in a separate window
# %matplotlib qt

# Requirements
from core.lib.pyRequirement  import *
from core.lib.coreLib        import *

# %% [markdown]
# ## Specialized functions
# As long as there is any specific function that you have to define, here is the best place to reach this aim. This section is considered as a pre-defined function step that could be available to use in the whole project.

# %%
def test():
    '''
    DocString `is` a text to help using the difined funtions.
    '''
    pass

# %% [markdown]
# ## Valuation
# This section provides the privilege to define **constant** values, which are usually used with a unique value in the whole project. Note that, they must be defined as a part of the `params` variable, which is a *list*.

# %%
params = clib.struct()
# The below variables are time-step and simulation time, respectively.
# You might need to alter them arbitrarily.
params.step = 0.0001 #(double)
params.tOut = 6      #(double)

# The given line below is a dependent variable. You normally
# should NOT change it.
params.n = int(params.tOut/params.step) #(DEPENDENT)

# The two next variables carry the folders from which you can use to
# call your files and saving the results organizely.
params.dataPath = 'data'
params.loadPath = params.dataPath + '/inputs'  #(string)
params.savePath = params.dataPath + '/outputs' #(string)

# Do you want to save a diary after each simulation? So set the below logical
# variable "True". The below directory is the place your logs are saved.
# The third order is a string that carries the name of the file, and normally
# is created using the current time, but you can change it arbitrary.
params.makeDiary = True   #(logical)
params.diaryDir  = 'logs' #(string)
params.diaryFile = clib.getNow(1, '-') #(string)

# The amount of time (in second) between each commands printed on CP.
params.commandIntervalSpan = 2 #(int)

# The below line is related to creating names when you are saving data
# which include the time of saving, It is a logical variable.
params.uniqueSave = False #(logical)

# The below string can be set like one of the following formats. It's the
# default format that when you do not adjust any formats, it will be
# considered. These allowed formats are "jpg", "png", "pdf"
params.defaultImageFormat = 'png' #(string)
# Put your params here ~~~>

# %% [markdown]
# ## Signals
# Any signals and array variables should be defined in this section. Also, scope objects which are effective to observe a signal, should be defined here. All your signals must be a part of `signals` variable.

# %%
signals = clib.struct()
# Simulation time vector
signals.tLine = np.arange(0, params.tOut, params.step)
# Put your signals here ~~~>

# %% [markdown]
# ## Models
# Any dynamic objects and those which are not as simple as an array must be defined as a part of `models` variable; Objects such as estimators and controllers.

# %%
models = clib.struct()
# Put your models here ~~~>

# %% [markdown]
# ## Simulation
# The key function of the project, undoubtedly, can be mentioned is `simulation` sunction which is given below. In this part, regarding the availability of all variables (`params`, `signals`, `models`, and `lib`) you are able to code your main purpose of this project here. Note that there is a loop named `Main loop` which you can utilize to use as time step loop, although you might not need to use that in many projects.

# %%
def simulation(params, signals, models):
    # [Parameters, Models, Signals, Libraries] <- (Parameters, Signals, Models, Libraries)
    # This function is your main code which there is a time-loop in
    # Also, the model blocks and the signals must be updated here.
    # Before the main loop, you can initialize if you need, Also you can
    # finalize after that if you need, as well.
    
    ## Initial options
    func = clib()
    st   = clib.sayStart(params)
    # A trigger used to report steps in command
    trig = [st, params.commandIntervalSpan, -1]
    
    ## Main loop
    for k in range(0, params.n):
        # Displaying the iteration number on the command window
        trig = clib.disit(k, params.n, trig, params)

        # Put your codes here ~~~>

    ## Finalize options
    # To report the simulation time after running has finished
    clib.disit(k, params.n, [st, 0, trig[2]], params)
    clib.sayEnd(st, params)
    # Sent to output
    return [params, signals, models]

# %% [markdown]
# ## Running
# To run the project, the below piece of code is provided. In simple projects, you might need to run once, while there are numerous ones that you will have to run the simulation for several times; to do that, you can use loops and other arbitrary techniques to call `simulation` which might be given changed inputs such as `params`, etc.

# %%
[params, signals, models] = simulation(params, signals, models)

# %% [markdown]
# ### Profiling
# If you need to check the timing of your project and know which part of your program is the most time-consuming, uncomment the content in this section and observe your performance of coding. It is worth mentioning that the previous code block should be commented.
# 
# If you want to use it in `Command Prompt`, run the below codes to create and show the results.
# * `python -m cProfile -o logs\profiler\cprofiler.prof main.py`
# * `snakeviz logs\profiler\cprofiler.prof`

# %%
# # Running the code and save the results into a file
# order = 'simulation(params, signals, models, lib)'
# profileName = params.diaryDir + '/profiler/' + params.diaryFile + '.prof'
# cProfile.run(order, profileName)
# # Depiction of the results 
# %load_ext snakeviz
# %snakeviz profileName

# %% [markdown]
# ## Illustration
# If you need to demonsrate the results obtained, this section can be utilized to have better organization.

# %%
## Initialize
n     = params.n
nn    = np.arange(0, n)   # A vector from 1 to n
tLine = signals.tLine[nn] # A time-line vector
plib.initialize()
## Write your codes here ~~~>

# %% [markdown]
# ## Saving
# Use the below part, providing to save data.

# %%
## Write your codes here ~~~>


