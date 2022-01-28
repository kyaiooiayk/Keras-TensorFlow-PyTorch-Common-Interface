"""
Import modules
M.O  Module.Others  = access Other imports
M.K  Module.Keras   = access Keras imports
K.PT Module.PyTorch = access PyTorch imports
"""
from KPT.Modules import Modules
from KPT.Decorators import timer, logger
M = Modules()


class Modelling():
    """Modelling class.

    Contains all the actions performed while the model is built 
    and run. It also contains all the diagnostics for checking 
    the soundness of the model.
    """

    def __init__(self, PP):
        """Initialise the object
        Some object attributes are defined later one while still
        providing a structure at the instantiation time.
        """
        self.seed = int(1)
        self.architecture = None
        self.No_feature = None
        self.first_layer = True
        self.last_layer = False
        self.previous_layer_neurons = None
        self.model = None
        self.framework = None
        self.loss_func = None
        self.early_stopping = None
        self.optimiser = None
        self.history = None
        # Pre-Processing insstance attributess
        self.set_ = PP.Set
        self.logger = PP.logger
        self.log_file_name = PP.log_file_name
        self.log_file_dir = PP.log_file_dir

    @logger
    @timer
    def build_model(self, framework, architecture, No_feature):
        """Build model.

        Build a Keras or PyTorch model depending on the chosen 
        framework.

        Parameters
        ----------
        framework : string
            Either PyTorch or Keras.
        architecture : list of list
            System architecture describing the layer type, No of
            layers and activation functions. An example could be:
            architecture = [["Dense", 200, "ReLu"],
            ["Dense", 100, "ReLu"]].            
        No_feature : int
            No of features in the dataset.

        Returns
        -------
        String = "Succcess"
            This is used for unittesting purpouses only
        """

        # Validate architecture
        issue, test = self._validate_architecture(architecture)
        if test == "Failed":
            self.logger.error("")
            self.logger.error("Layers: " + str(issue) +
                              " is NOT a valid layer!")
            self.logger.error("")
            # Exit the code!
            M.sys.exit()

        # Updating some instance/object attributes
        self.architecture = architecture
        self.No_feature = No_feature
        self.framework = framework

        if framework.lower() in ["pytorch", "pt"]:
            self._build_PT()
        elif framework.lower() in ["keras", "k"]:
            self._build_K()
        else:
            self.logger.error(" ")
            self.logger.error(
                "Framework" + str(self.framework) + " NOT recognised!")
            self.logger.error(" ")
            # Exit program
            M.sys.exit()

        return "Success"

    @logger
    def _validate_architecture(self, architecture):
        """Validate architecture.

        Check if the architecture is in the format of
        list of list. There could be two types of entries:
        [string, int, string] or [string, int]. Anything else
        should raise an exception or throw and error.

        Parameters
        ----------
        architecture - list of list
            This is the layer-by-layer architecture

        Returns
        -------        
        validation - string
            Successful or Failed
        """

        """
        There two types of check:
        Check the length of each list
        Check the structure of each list entry
        """
        validation = "Successful"
        issue = "None"
        for index, current_block in enumerate(architecture):
            if len(current_block) == 3:
                if isinstance(current_block[0], str) == False:
                    issue = current_block
                    validation = "Failed"
                    return issue, validation
                if isinstance(current_block[1], int) == False:
                    issue = current_block
                    validation = "Failed"
                    return issue, validation
                if isinstance(current_block[2], str) == False:
                    issue = current_block
                    validation = "Failed"
                    return issue, validation
            elif len(current_block) == 2:
                if isinstance(current_block[0], str) == False:
                    issue = current_block
                    validation = "Failed"
                    return issue, validation
                if isinstance(current_block[1], int) == False:
                    issue = current_block
                    validation = "Failed"
                    return issue, validation
            else:
                issue = current_block
                validation = "Failed"
                return issue, validation

        return issue, validation

    def _print_input_architecture(self):
        """Print input architecture.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.logger.info(" ")
        self.logger.info("Provided architecture")
        for i, layer in enumerate(self.architecture):
            self.logger.info("Layer No. " + str(i+1) +
                             " is defined as: " + str(layer))
        self.logger.info(" ")

    def _build_PT(self):
        """Build PyTorch model.

        Build a MLP PyTorch-based architecture in a sequential manner.

        Parameters
        ----------
        None

        Returns
        -------
        model - instance
            PyTorch model instance
        """

        # Fix the seeder for reproduciblity
        M.torch.manual_seed(self.seed)
        # Print on screen the provided architecture
        self._print_input_architecture()

        layers = []
        for index, current_block in enumerate(self.architecture):
            self.logger.info(" ")
            self.logger.info("-> Adding layer: " + str(current_block))
            if index == len(self.architecture) - 1:
                self.last_layer = True

            # Adding linear/dense layer
            if current_block[0].lower() in ["d", "dense"]:
                layers.append(
                    self._bbPT("linear", current_block[1]))

                if self.last_layer == False:
                    if current_block[2].lower() == "relu":
                        layers.append(self._bbPT("relu"))
                # Update attribute
                self.first_layer = False

            else:
                self.logger.error(" ")
                self.logger.error(
                    "Layer " + str(current_block[0]) + " NOT ecognised!")
                self.logger.error(" ")
                # Exit the code!
                M.sys.exit()

        self.logger.info("Model was successfully built!")

        model = M.nn.Sequential(*layers)

        # Resetting to the default value
        self.last_layer = False
        self.model = model
        return model

    def _build_K(self):
        """Build Keras model.

        Build a MLP Keras-based architecture in a sequential manner.

        Parameters
        ----------
        None

        Returns
        -------
        None        
        """

        M.seed(self.seed)
        self._print_input_architecture()

        model = M.keras.Sequential()
        for index, current_block in enumerate(self.architecture):

            self.logger.info("")
            self.logger.info("-> Adding layer: " + str(current_block))
            if index == len(self.architecture) - 1:
                self.last_layer = True

            if current_block[0].lower() in ["d", "dense"] and self.last_layer == False:
                model.add(
                    self._bbK("linear", current_block[1], current_block[2]))

            elif self.last_layer == True:
                model.add(self._bbK("last"))

            else:
                self.logger.error(" ")
                self.logger.error("Layer NOT recognised!")
                self.logger.error(" ")
                # Exit the code!
                M.sys.exit()

        self.logger.info("Model was successfully built!")

        # Resetting to the default value
        self.last_layer = False
        self.model = model
        return model

    @logger
    @timer
    def summary(self, input_size=None):
        """Summary of the mode architecture.

        Print the summary of the model. This method was 
        inspireed by Keras model.summary(). It is mainly 
        used to get a bird-eye view of the model and in 
        particular of the Total No of parameters.

        Parameters
        ----------
        input_size : int
            This is needed for the PyTorch model summary
            only. Unlike Keras, PyTorch has a dynamic computational
            graph which can adapt to any compatible input shape 
            across multiple calls e.g. any sufficiently large image
            size (for a fully convolutional network). As such, it 
            cannot present an inherent set of input/output shapes for
            each layer, as these are input-dependent, and why in the 
            above package you must specify the input dimensions.

        Returns
        -------
        String
            This is used for unittesting purpouses only
        """

        # Keras model summary
        if self.framework.lower() in ["k", "keras"]:
            # The output cannot be push to logging, so it is only print on the console!s
            self.logger.warning("Printing is available on console ONLY!")
            print(self.model.summary())

        # PyTorch model summary
        elif self.framework.lower() in ["pt", "pytorch"]:
            if input_size == None:
                self.logger.warning(
                    "Printing is available on console ONLY for rendering reasons!")
                print(M.summary(self.model))
            else:
                self.logger.warning(
                    "Printing is available on console ONLY for rendering reasons!")
                print(M.summary(self.model, input_size=input_size))
        else:
            self.logger.error("")
            self.logger.error("Framework: ", self.framework, " NOT known!")
            self.logger.error("")
            # Exit the code!
            M.sys.exit()

        return "Success"

    def _bbPT(self, which_block, No_neurons=None):
        """PyTorch building block.

        Provide an interface to the building block of a 
        PyTorch model.Keep in mind that Linear is the 
        equivalent of Dense in Keras.

        Parameters
        ----------
        which_block : string
            block type as in linear which is the only
            one supported now. There exist more types.
        No_neurons : int Default is None 
            No of neurons for each layer.

        Returns
        -------
        Torch Layer - instance
        """

        if which_block.lower() == "linear" and self.first_layer == True:
            self.logger.info("-> Adding FIRST linear layer inputs: " +
                             str(self.No_feature) + " " + str(No_neurons))
            self.previous_layer_neurons = No_neurons
            return M.torch.nn.Linear(self.No_feature, No_neurons)

        elif which_block.lower() == "linear" and self.first_layer == False and self.last_layer == False:
            self.logger.info("-> Adding OTHER linear layer inputs: " +
                             str(self.previous_layer_neurons) + " " + str(No_neurons))
            output = M.torch.nn.Linear(self.previous_layer_neurons, No_neurons)
            # The attribute update must come after we defined the output
            self.previous_layer_neurons = No_neurons
            return output

        elif which_block.lower() == "linear" and self.first_layer == False and self.last_layer == True:
            self.logger.info("Adding LAST linear layer inputs:" +
                             str(self.previous_layer_neurons) + " " + str(self.No_feature))
            output = M.torch.nn.Linear(
                self.previous_layer_neurons, self.No_feature)
            # The attribute update must come after we defined the output
            self.previous_layer_neurons = No_neurons
            return output

        elif which_block.lower() == "relu":
            self.logger.info("-> Adding activation RELU")
            return M.torch.nn.ReLU()

    def _bbK(self, which_block, No_neurons=None, activation=None):
        """Keras building block.

        Keep in mind that Linear is the equivalent
        of Dense in Keras.

        Parameters
        ----------
        which_block : string
            block type as in linear which is the only
            one supported now. There exist more types.
        No_neurons : int Default is None 
            No of neurons for each layer.
        activation : string
            Activation function. As example is "ReLu"

        Returns
        -------
        Keras Layer : instnace

        """
        if which_block.lower() == "linear":
            self.logger.info(str(No_neurons) + " " +
                             str(activation) + " " + str(self.No_feature))
            self.logger.info(
                "-> Adding linear layer with neurons No: " + str(No_neurons))
            return M.Dense(No_neurons, activation=activation.lower(), input_dim=self.No_feature)

        elif which_block.lower() == "last":
            self.logger.info(
                "Adding LAST linear layer with neurons No: " + str(self.No_feature))
            return M.Dense(self.No_feature)

    @logger
    @timer
    def train(self, verbose=False, lr=0.01, patience=20, epoch=200):
        """Train the model.

        This function calls the right training function depending
        on the selected framework.

        Attributes
        ----------
        verbose : bool (Default=False)
            Whether you to run in verbose mode or silent mode.
            Essentially the silent mode will not print trining
            histories.
        lr : float (Default=0.01)
            learning rate
        patience : int (Default=220)
            No of steps before the traininig is stopped.
        epoch : int (Default=200)
            No of epoches

        Returns
        -------
        None
        """

        self.logger.warning(
            "Console iteration output is ommitted, but can be acccessed via Post-Processing")

        if self.framework.lower() in ["py", "pytorch"]:

            # Define the optimiser
            optimiser = M.torch.optim.Adam(
                self.model.parameters(), lr=lr)
            self.optimiser = optimiser

            # MSE is the mean squared loss
            loss_func = M.torch.nn.MSELoss()
            self.loss_func = loss_func

            # Early stopping
            early_stopping = M.EarlyStopping(patience=patience, delta=0.001)
            self.early_stopping = early_stopping

            # Call PT training loop
            self._training_loop_PT(verbose, epoch)

        elif self.framework.lower() in ["k", "keras"]:

            # Call Keras training loop
            self._training_loop_K(verbose, epoch, patience, lr)

        else:
            self.logger.error(" ")
            self.logger.error("The framework:" + " " +
                              self.framework + " was NOT recognised!")
            self.logger.error(" ")
            # Terminate the code!
            M.sys.exit()

    def _training_loop_PT(self, verbose, epoch):
        """PyTorch training loop.

        This is where the trainig loop is controlled:
        step, update and batches are all defined here.

        Parameters
        ----------
        verbose : bool
            If true prints the history at each epoch

        Returns
        -------
        None
        """

        train_lossHistory_PT = []
        valid_lossHistory_PT = []
        train_losses = []
        valid_losses = []

        # Start training
        for epoch in range(epoch):
            if verbose == True:
                print("Epoch No: " + str(epoch))
            # ----Training loop----
            # model_PT.train()
            for step, (batch_x, batch_y) in enumerate(self.set_["PT_loader_train"]):

                b_x = M.Variable(batch_x)
                b_y = M.Variable(batch_y)

                # Input x and predict based on x
                prediction = self.model(b_x)
                # Must be (1. nn output, 2. target)
                loss = self.loss_func(prediction, b_y)

                # Clear gradients for next train
                self.optimiser.zero_grad()
                # Backpropagation, compute gradients
                loss.backward()
                # Apply gradients
                self.optimiser.step()
                # Record training loss
                currentBatchLoss = loss.item()
                train_losses.append(currentBatchLoss)

            # ----Validation loop----
            # model_PT.eval()
            for batch_x, batch_y in self.set_["PT_loader_val"]:

                b_x = M.Variable(batch_x)
                b_y = M.Variable(batch_y)

                # Input x and predict based on x
                prediction = self.model(b_x)
                # Must be (1. nn output, 2. target)
                loss = self.loss_func(prediction, b_y)

                # Record training loss
                currentBatchLoss = loss.item()
                valid_losses.append(currentBatchLoss)

            train_loss_avr = M.np.average(train_losses)
            train_lossHistory_PT.append(train_loss_avr)

            valid_loss_avr = M.np.average(valid_losses)
            valid_lossHistory_PT.append(valid_loss_avr)
            if verbose == True:
                # We are not ussing logging here otherwise it takes too much space in the .log file
                print("Current #", epoch, " loss", train_loss_avr,
                      " valid_loss", valid_loss_avr)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            # initialize the early_stopping object
            self.early_stopping(valid_loss_avr, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                print("Current #" + str(epoch) + " loss " + str(train_loss_avr) +
                      " valid_loss" + str(valid_loss_avr))

                history = {}
                history["train"] = train_lossHistory_PT
                history["val"] = valid_lossHistory_PT
                self.history = history
                break

    def _training_loop_K(self, verbose, epoch, patience, lr):
        """Keras training loop.

        This is where the trainig loop is controlled:
        step, update and batches are all defined here.

        Parameters
        ----------
        verbose : bool
            If true prints the history at each epoch

        Returns
        -------
        None
        """

        optimizer = M.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(
            optimizer=optimizer,
            loss='mae'
        )

        early_stopping = M.callbacks.EarlyStopping(
            # Minimium amount of change to count as an improvement
            min_delta=0.001,
            # How many epochs to wait before stopping.
            # Meaning, when there is no improvement for more than 50 epoches, the program stop
            patience=patience,
            restore_best_weights=True,
        )

        # Keras uses 0,1 switches but we use booleans
        verbose_ = 0
        if verbose == True:
            verbose_ = 1

        history_ = self.model.fit(
            self.set_["K_X_train"], self.set_["K_y_train"],
            validation_data=(self.set_["K_X_val"], self.set_["K_y_val"]),
            batch_size=64,
            epochs=epoch,
            callbacks=[early_stopping],
            verbose=verbose_)

        history = {}
        history["train"] = history_.history["loss"]
        history["val"] = history_.history["val_loss"]
        self.history = history
