# Variable Sensitivity E-Skins
This code accompanies the paper Variable Sensitivity Multimaterial Robotic E-Skins using Electrical Impedance Tomography.

Written using MATLAB 2021b (Deep Learning Toolbox).

## Data Availability
The 2,500 responses and random press locations used for each of the patterns are stored in _Data/Concentric Circles_, _Data/Full CB_, _Data/No Pattern_, _Data/Parallel Lines_, & _Data/Radial_, as 'input' and 'output' CSV files respectively.

Trained neural networks and their accompanying predictions/errors are also stored in these folders, with each filename indicating the size of the dataset used for training.


## Functions
**sensorTrain.m** trains a feedforward neural network and provides feedback about its performance by calling the **calculateErrors.m** function.