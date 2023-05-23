<img src="https://github.com/abolfazldelavar/dyrun/blob/main/logo.png?raw=true" align="right" width="300" alt="header pic"/>

# Dynamic Runner (DYRUN)

This Python-based basement environment is designed for simulating dynamical systems, including differential equations, and visualizing their results, which can be particularly useful for academic research and paper presentations. `dyrun` is a simple tool that facilitates simulations expeditiously and effortlessly, using invaluable instruments for modeling linear and nonlinear dynamic systems. This formidable instrument is capable of executing **differential mathematical equations** [1] and is advantageous for

* **Control engineering**,
* **Estimation** [2, 3],
* **Encryption** [4],
* and **Neuroscience** [5, 6].

Researchers who wish to delve into the realm of dynamic and control systems will find this package to be an invaluable resource.

# License

MIT license

# Author

Abolfazl Delavar
- Email: faryadell [at] gmail [dot] com
- Webpage: https://github.com/abolfazldelavar

"*Successful people are those who can build solid foundations with the bricks others throw at them.*"

# Instructions

## Installation

To install the full package, run the below code:

```
pip install dyrun
```

To build a new project, just run the following order:

```
dyrun DIR_NAME
```

If the package has been installed, you can update it by:

```
pip install --upgrade dyrun
```

where the **DIR_NAME** denotes the folder you wish to create the project.

## First trying

The main file of this package is `dyrun.py`, which can be run easily using:

```
python3 dyrun.py
```

However, the jupyter format of that is also provided in order to have better control on each section of that.
It is recommended you to modify `dyrun.ipynb` file to make the most of jupyter tools and facilities.
All sections will be introduced in the following.

## The structure of `dyrun.ipynb`

The notebook is organized into the following sections: **Introduction:**, **Requirements:**, **Custom Functions:**, **Parameters:**, **Signals:**, **Models:**, **Main:**, **Execution:**, **Illustration:**, and **Preservation:**.

The notebook is designed to be self-contained and easy to follow, even for users who are new to dynamic simulations and the `dyrun` library.
The only effort you must make is to read the instruction presented in `dyrun.ipynb` and this guideline.

<details open>
    <summary><h3> Introduction </h3></summary>
    A brief overview of the *dyrun* library and its main features.
</details>

<details open>
    <summary><h3> Requirements </h3></summary>
    The external dependencies utilized in the project are documented in this section.
</details>

<details open>
    <summary><h3> Custom Functions </h3></summary>
    Any custom functions for the project are defined in this section.
</details>

<details open>
    <summary><h3> Parameters </h3></summary>
    This section establishes the static quantities such as model parameters used throughout the entire project.
</details>

<details open>
    <summary><h3> Signals </h3></summary>
    This section is designated for the definition of any signals and array variables.
</details>

<details open>
    <summary><h3> Models </h3></summary>
    Dynamic objects and those that are not as elementary as an array must be included in this section.
</details>

<details open>
    <summary><h3> Main </h3></summary>
    This section contains the simulation function that is used to run the project.
</details>

<details open>
    <summary><h3> Execution </h3></summary>
    This section contains the code snippet to run the project.
</details>

<details open>
    <summary><h3> Illustration </h3></summary>
    This section is used to display the results of the simulation.
</details>

<details open>
    <summary><h3> Preservation </h3></summary>
    This section is used to store data.
</details>


## Requirements

The file named `core/lib/required_libraries.py` is where the crucial requirements are loaded and used througout the project. You will not need it normally, unless you wish to use a specific library in the whole project. In other words, all addition requirements should be called in the requirements section of `dyrun` file. For instance, as long as the project needs to use a specific model file (This will be discussed in the following parts of this instrustion), they should be called by `from blocks.general_blocks import *` in this important section.

## Libraries

In this section, all available libraries that can be utilized for better managing your projects are given.

<details open>
    <summary><h2> `Clib` </h2></summary>
  
This pre-imported library which you can find it in the **requirements** section of the `dyrun.ipynb`, brings you several useful functions, which will be elaborated now:

1. `set_timer(ID)`: This function sets the timer and initializes the simulation. It also sets up the logging module, creates a log file in the `logs` directory with the name `<ID>.log`, and writes a message to the log file indicating that the simulation has started.
    * `ID` (input) - The id number of the simulation; it is usually unique.
    * Output: The current time at the moment that the function is called.

2. `end_report(start_time)`: This function reports the end of the simulation and the time it took to run. It calculates the elapsed time since the simulation started, formats it as a string, and writes a message to the log file indicating that the simulation has ended and the elapsed time.
    * `start_time` (input) - The time the simulation was started (returned by `set_timer()`).
    * Output: None.

3. `get_now(report_type=0, splitter_char='_')`: This function returns a string containing the current date and time in the specified format. The format of the output string depends on the value of `report_type` and `splitter_char`. The function gets the current date and time using the `datetime.now()` method, formats it according to the specified format, and returns the formatted string.
    * `report_type` (input, optional) - The type of report to generate. The default value is `0`.
        * 0: YMDHMS
        * 1: YMD_HMS
        * 2: Y_M_D_H_M_S
        * 3: YMD_HM
        * 4: YMD_HM
        * 5: Y_M_D
        * 6: H_M_S
    * `splitter_char` (input, optional) - The character used to separate the different parts of the date and time. The default value is `'_'`.
    * Output: A string containing the current date and time in the specified format.

Certainly! Here are the revised instructions for the `Clib` class, which provides several useful functions related to file I/O. For each function, I've provided a brief description of its operation, along with the input variables and output variables:

4. `diary(message, no_print=False)`: This function is used to put the message in the diary file as well as to print it to the console. It writes the message to the log file using the `logging` module and also prints the message to the console if `no_print` is `False`.
    * `message` (input) - The message that you want to show or put in the diary.
    * `no_print` (input, optional) - A boolean value that specifies whether to disable printing to the console. The default value is `False`.
    * Output: None.

5. `save_csv(tensor, name, zip=False)`: This function saves the given data in a `.csv` file. If `zip` is `True`, it creates a `.zip` file with the same name as the `.csv` file and saves it in the same directory.
    * `tensor` (input) - The data that you want to save, e.g., matrices or numerical vectors.
    * `name` (input) - The path including the name of the file where you want to save the data. For example, `data/outputs/TensorFileName`.
    * `zip` (input, optional) - A boolean value that specifies whether to save the data as a `.zip` file. The default value is `False`.
    * Output: None.

6. `load_csv(name, maxsize=1000)`: This function loads the content of the given `.csv` file saved by `save_cvs()`. If the data file is too large, it can be restricted by specifying the maximum file size in MB using the `maxsize` option.
    * `name` (input) - The path including the name of the `.csv` file from which you want to load the data. For example, `data/outputs/TensorFileName`.
    * `maxsize` (input, optional) - An integer value that specifies the maximum size of the file to be loaded in MB. The default value is `1000`.
    * Output: The loaded data in the form of a NumPy array.

7. `save_npy(tensor, name, zip=False)`: This function saves the given `tensor` as a `.npy` file with the specified `name`. If `zip=True`, the function saves the tensor as a `.zip` file instead of a `.npy` file.
    * `tensor` (input) - The data to be saved, such as a matrix or numerical vector.
    * `name` (input) - The path and name of the file to be saved, e.g. `data/outputs/TensorFileName`.
    * `zip` (input, optional) - A boolean value that specifies whether to save the data as a `.zip` file. The default value is `False`.
    * Output: None.

8. `load_npy(name)`: This function loads the content of the given `.npy` file with the specified `name`.
    * `name` (input) - The path and name of the file to be loaded, e.g. `data/outputs/TensorFileName`.
    * Output: The loaded tensor data.

9. `delayed(signal, k, delay_steps)`: This function creates a delayed version of a discrete signal.
    * `signal` (input): The full signal.
    * `k` (input): The current point in the signal.
    * `delay_steps` (input): The amount of delay in integer steps.
    * Output: The delayed signal value.

10. `sigmoid(time_line, bias, alph, area)`: This method generates a sigmoid function signal.
    * `time_line` (input): The time line.
    * `bias` (input): The time shift bias.
    * `alph` (input): The smoother.
    * `area` (input): The domain in the form of a list `[a, b]`.
    * Output: The output signal.

11. `exp_f(time_line, decay_rate, area)`: This method generates an exponential function signal.
    * `time_line` (input): The time line.
    * `decay_rate` (input): The decay rate.
    * `area` (input): The domain in the form of a list `[a, b]`.
    * Output: The output signal.

12. `linear_mapping(x, current_area, target_area)`: This method linearly maps the input `x` from the domain `[a1, b1]` to the range `[a2, b2]`.
    * `x` (input): The input array.
    * `current_area` (input): The domain `[a1, b1]`.
    * `target_area` (input): Therange `[a2, b2]`.
    * Output: The output array.

To use these functions, you can call them as static methods of the `Clib` class. For example, to save a tensor, you can call `Clib.save_npy(tensor, name)`, where `tensor` is the data to be saved, and `name` is the path and name of the file to be saved.
</details>

<details open>
    <summary> <h2> `Plib` </h2></summary>
  
This pre-imported library which you can find it in the **requirements** section of the `dyrun.ipynb`, provides several practical functions related to illustration purposes. The following methods are available:


The `Plib` class provides several practical functions related to illustration purposes. The following methods are available:

1. `initialize()`: This function initializes the environment and sets the `font.family`, `font.size`, and `text.usetex` to any further draws. Furthermore, the size of plot windows and several other essential properties are adjusted. There is no input to this function. Note that it is recommended to make sure the last `latex` package (like `TexLive`) has been installed in your systems. you can download from <a target="_blank" href="https://www.tug.org/texlive/">here</a>. It is used to draw for academic purposes. If you do not have `ltex`, you can temporarily ignore this function, although it is suggested to use it.
    * Output: None.

2. `isi(params, fig = 0, save = False, width = 8.5, hw_ratio = 0.65)`: This function makes plots prettier and ready to use in academic purposes. The inputs to this function are:
    * `params` (input): a string that describes the plot.
    * `fig` (input, optional): the figure handler; use `h = plt.figure(tight_layout=True)` to create one.
    * `save` (input, optional): a string that specifies the name of the file to save the illustration, or just insert `True`.
    * `width` (input, optional): a float that represents the width of the figure in inches; default is `8.5`.
    * `hw_ratio` (input, optional): a float that represents the height to width ratio between `0` and `1`; default is `0.65`.
    * Output: None.

3. `save_figure(params, save=True, fig=plt.gcf(), dpi=300)`: This function can be used to save a figure/plot as an image file. The inputs to this function are:
    * `params` (input): An object that contains various parameters related to the current run of the program.
    * `save` (input, optional): A boolean or string value that specifies whether to save the figure or not. If a string value is provided, it is used as the **filename** for the saved image.
    * `fig` (input, optional): A matplotlib figure object that represents the current figure/plot to be saved.
    * `dpi` (input, optional): An integer value that specifies the resolution of the saved image file in dots per inch.
    * Output: None.

4. `linear_gradient(colors, locs, n=256, show=False)`: This function can be used to generate a linear gradient of colors. The inputs to this function are:
    * `colors` (input): A list of color values in RGBA format. For example, `[[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]` represents a gradient from red to white to black.
    * `locs` (input): A list of values between 0 and 1 that specify the location of each color in the gradient.
    * `n` (input, optional): An integer value that specifies the number of colors in the output gradient. Default value is `256`.
    * `show` (input, optional): A boolean value that, if set to True, shows a plot of the generated gradient.
    * Output: The output of this function is an array of colors that represents the generated gradient.

5. `cmap_maker(name, colors, n=256)`: This function can be used to create a Linear Segmented Color Map (LSCM) using the given colors and their locations. The inputs to this function are:
    * `name` (input): A string that specifies the name of the color map.
    * `colors` (input): A list of tuples where each tuple contains a value between 0 and 1 (specifying the location of the color on the graph line) and a color value in any valid format (e.g., `#FF0000` for red).
    * `n` (input, optional): An integer value that specifies the number of colors in the output color map. Default value is `256`.
    * Output: The output of this function is a matplotlib colormap object that can be used for visualizations.

To use these functions, you can call them as static methods of the `Plib` class, similar to `Clib`.
</details>


<details open>
    <summary> <h2> `eclib` </h2></summary>
    This is a Python library for numerical simulation of encrypted control.
    You have to import the library you need in the requirements section. For instance, to utilize `Paillier`, insert `import core.lib.eclib.paillier as pai` and use `pai` in the whole project.
    Visit <a target="_blank" href="https://github.com/KaoruTeranishi/EncryptedControl">EncryptedControl</a> [4] for more information.
</details>

## A survey on `Scope` class

...

## Questions

<details open>
    <summary> <h4> How to model an LTI system?</h4></summary>
</details>

<details open>
    <summary> <h4> How to model a nonlinear system?</h4></summary>
</details>

<details open>
    <summary> <h4> How to model a network including nonlinear agents?</h4></summary>
</details>

# References

[1]. Butcher, John C. (2003). Numerical Methods for Ordinary Differential Equations. New York: John Wiley & Sons. ISBN 978-0-471-96758-3.

[2]. Simon, Dan (2006). Optimal State Estimation. Hoboken, NJ: John Wiley & Sons. ISBN 978-0-471-70858-2.

[3]. E. A. Wan and R. Van Der Merwe, "The unscented Kalman filter for nonlinear estimation," Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium (Cat. No.00EX373), Lake Louise, AB, Canada, 2000, pp. 153-158, doi: 10.1109/ASSPCC.2000.882463.

[4]. Teranishi K, Encrypted control library, github. https://github.com/KaoruTeranishi/EncryptedControl

[5]. Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569-1572.

[6]. Ullah, M., Fletcher, A. E. C., & Hake, J. (2006). The Role of Calcium Release-Activated Calcium Currents in Bursting Pacemaker Activity of Nucleus Ambiguus Neurons. Journal of Computational Neuroscience, 21(3), 271-282.
