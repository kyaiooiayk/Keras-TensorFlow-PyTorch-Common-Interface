# Keras-TensorFlow-PyTorch-Common-Interface

This tool offers a **common interface** to create a KERAS and PyTorch model by providing a **single** input architecture. The motivation for this work stems from the fact that the two frameworks use a different syntax. This common interface is then used to automatically build two models, one in Keras and one in PyTorch. These models are then run and compared given a function. This allows to test the common statement that ANNs are *universal function approximator*.

## Universality Theorem of NNs
[This](http://neuralnetworksanddeeplearning.com/chap4.html#other_techniques_for_regularization) reference provides a concise intuition of what it is meant by universality. No matter what the function, there is guaranteed to be a neural network so that for every possible input, `x`, the value `f(x)` (or some close approximation) is output from the network. This universality theorem holds even if we restrict our networks to have just a single layer intermediate between the input and the output neurons - a so-called single hidden layer. So even very simple network architectures can be extremely powerful.
Any inaccuracies between the target function and the NN prediction can be reduced to:
- Lack of a deterministic relation between input and outputs
- Insufficient number of hidden units
- Inadequate training (stopping the training too soon)
- Poor choice of the optimisation algorithm.

## Repository Structure
The order in which the files are described follows the step-by-step procedure used to create the project
- [`Regression with Neural Networks implemented in PyTorch and Keras.ipynb`](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/blob/master/Regression%20with%20Neural%20Networks%20implemented%20in%20PyTorch%20and%20Keras.ipynb) This notebook describes how the models were originally created. This was used to quickly see what I wanted to automate.
- [`Testing the common interface code.ipynb`](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/blob/master/_Testing%20the%20common%20interface%20code.ipynb) This is the notebook I've used to test the code. Essentially code and plot
- [`Testing several functions`](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/blob/master/Testing%20several%20functions.ipynb) A notebook where serveral functions, from simple to complex are being retrofitted. This is to test the hypothesis that ANNs are universal approximators.
- [`main.py`](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/blob/master/main.py) Just an example of how to call the code. No plotting is called here.
- [`KPT`](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/tree/master/KPT) It stands for KerasPyTorch and contains the source code. This is made of 5 scripts:
    - `Modelling.py`
    - `Modules.py`
    - `PostProcessing.py`
    - `PreProcessing.py`
    - `PyTorchTools.py`

## Other Codes
`PyTorchTools.py` is a call method used to implement early stopping
while training with PyTorch. [References](https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py)

## Special Notes
To be able to see the markdown comments in the notebooks, please download the files locally. GitHib does not render these markdowns very well. Alternatively, you can see them in the GitHub notebook rendering in the folder: `GitHub_MD_rendering`.


## Some Results
Below are some screenshots of the results obtained retrofitting a simple sine function:

![ScreenShot](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/blob/master/img/result.png)

![ScreenShot](https://github.com/kyaiooiayk/Keras-TensorFlow-PyTorch-Common-Interface/blob/master/img/learningCurves.png)
