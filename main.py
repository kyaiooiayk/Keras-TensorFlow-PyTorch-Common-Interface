"""
The purpouses of this script is to show how to
call the KPT mdodule from a python script. It is
however suggested to use a jupyter notebook to use
the plotting routintes.
"""

# Import modules
import os
import numpy as np
from KPT.Modelling import Modelling
from KPT.PreProcessing import PreProcessing
from KPT.PostProcessing import PostProcessing
import KPT
from KPT.Modules import Modules
M = Modules()


def main():

    # Creating test function
    """
    We provide a function and a range of value.
    As simple as this
    """
    x = np.linspace(0, 10, 500)
    y = np.sin(x)

    # Split the data
    """
    Split the data according to methds:
    =2 train and test
    =3 train, val and test
    testSize control the size of the test set
    """
    PP_K = PreProcessing(x, y, log_file_name="Log_Keras")
    PP_PT = PreProcessing(x, y, log_file_name="Log_PyTorch")
    splitSet_K = PP_K.split_dataset(method=3, test_size=0.2)
    splitSet_PT = PP_PT.split_dataset(method=3, test_size=0.2)

    # Prepare the data
    """
    This where the data are prepared and ready to be
    ingested by the framework. Keras is a little bit less
    picky in term on preparation than Keras.
    """
    SetK = PP_K.prepare_input("Keras", batch_size=64)
    SetPT = PP_PT.prepare_input("PyTorch", batch_size=64)

    # Define the ANN architecture
    """
    A simple ANNs fully connected common architecture is 
    here built. We'd like to build an architecture for
    regression problem.
    data structure is  list of list
    The first entry is the type of later. Dense means fully connected and the only one supported.
    The second entry is the number of neurons

    """
    architecture = [
        ["Dense", 200, "ReLu"],
        ["Dense", 100, "ReLu"],
        ["Dense", 100, "ReLu"],
        ["Dense", 1],
    ]

    # Instantiate the architecture
    """
    The high level architecture is equal for both model.
    to instantiate the model we just need to pass the set.
    The set was already prepared in the pre-processing phase.
    """
    ModelK = Modelling(PP_K)
    ModelPT = Modelling(PP_PT)

    # Building the model
    """
    This is where the model is built step-by=step.
    No_feature is the number of input column. Since we
    have a simple regession problem with just one feature (x)
    we the use =1.
    """
    modelPT_build_OK = ModelPT.build_model(
        "PyTorch", architecture, No_feature=1)
    modelK_build_OK = ModelK.build_model("keras", architecture, No_feature=1)

    # Printing the model summary
    """
    This outpput the summary for each  model.
    It is useful because it allows to quickly compare the total number
    of learnt parameters.
    """
    ModelPT.summary()
    ModelK.summary()

    # Train the model
    """
    This is were the model get trained.
    """
    #ModelPT.train(verbose=True, lr=0.01, patience=20, epoch=200)
    ModelK.train(verbose=False, lr=0.01, patience=20, epoch=200)

    # Create post-processing objectss
    PP_PT = PostProcessing(ModelPT)
    PP_K = PostProcessing(ModelK)

    # Get some metrics
    PP_PT.get_metrics()
    PP_K.get_metrics()


if __name__ == '__main__':

    # Deleting previous logging files
    if os.path.exists("./Log_Keras.log"):
        os.remove("./Log_Keras.log")
    if os.path.exists("./Log_PyTorch.log"):
        os.remove("./Log_PyTorch.log")

    main()