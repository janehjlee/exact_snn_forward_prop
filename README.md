# README

This repository contains the code used to run the experiments in "Exact Gradient Computation for Spiking Neural Networks via Forward Propagation" found here: https://proceedings.mlr.press/v206/lee23b/lee23b.pdf. The organization of the code is detailed as follows:

- SNN Simulation.ipynb: In this notebook, you can explore how our continuous-time kernel equations result in simulating the neuron dynamics in continuous time. In particular, you can visualize and explore the effect of tuning the membrane and synaptic time constants, while also adjust the input firing times and see their effect on the membrane potential.

- xor_snn_np.ipynb: In this notebook, you can explore the XOR task and train the model. You can also plot the membrane potentials of the output neurons (and hidden neurons too, with some modifications).

- iris_snn_np.ipynb: In this notebook, you can explore the Iris dataset and train the model. You can also plot the histogram of output neuron firing times.

- yin_yang_snn.py: A Python runnable which trains SNN for the Yin-Yang dataset classification task. Some multiprocessing has been added to speed up computation, and should print out information about the hyperparameters used, test accuracy, training loss, epochs, etc.

- py/* and Yin Yang Exp.ipynb: The py folder just contains the code from yin_yang_snn.py so that it can be imported into the notebook Yin Yang Exp.ipynb to visualize the dataset and model predictions. This is mainly a notebook version of yin_yang_snn.py.

- surrogate: This folder contains code and data to compare exact gradient method to surrogate gradient methods.