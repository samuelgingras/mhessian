### mhessian

A Matlab toolbox implementing the HESSIAN method (Highly Efficient Simulation Smoothing, In A Nutshell).

#### Installation notes

You will need Matlab and a way to compile Matlab mex files. See Matlab documentation for details, including a list of the C compilers that are supported for your system.

1. Clone the Github repo https://github.com/samuelgingras/mhessian or download a zip file and unzip.

1. Add the highest level directory, mhessian, to your Matlab path, with subdirectories.

1. Compile the required mex files using the command ```compile_mhessian```. The output should be "MEX completed successfully." five times, possibly with additional information on the build process.

#### Getting started

1. There are five mex functions (i.e. C-coded Matlab functions). They are ```drawObs```, ```drawState```, ```evalObs```, ```evalState``` and ```hessianMethod```. Use, for example, ```help hessianMethod``` for information on how to call them.

1. The ```examples``` folder has Matlab code illustrating how to use the toolbox.

1. The document ```documentation/models.pdf``` describes the various state space models available.

#### Directory structure

At the highest level directory, mhessian, there are two files, the file ```README.md``` that you are reading, and a Matlab script ```compile_mhessian```. Other files are organized by directory as follows:

1. The ```documentation``` directory has the document ```models.pdf``` describing the various state space models available.

1. The ```examples``` directory has Matlab code illustrating how to use the Matlab version of the HESSIAN method.

1. The ```gir``` directory contains obsolete Matlab code illustrating how to test code for correctness using the "Getting it Right" paper, Geweke (2004).
See instead the Matlab script ```getting_it_right_example.m``` in the ```examples``` directory and the ```getting_model_right()``` function in ```matlab/getting_model_right.m```.

1. The ```matlab``` directory has Matlab code that supports posterior simulation.

1. The ```mex``` directory has C code implementing the Matlab functions ```drawObs```, ```drawState```, ```evalObs```, ```evalState``` and ```hessianMethod```.

1. The ```model``` directory has C code implementing the various models.
Each model is in a separate file.

1. The ```src``` directory has C code called by the mex functions.

1. The ```test``` directory has Matlab code for testing code.

#### ESBOE Slides 

[Slides from ESOBE 2023](/McCausland_ESOBE23.pdf)
