"""
Import modules
M.O  Module.Others  = access Other imports
M.K  Module.Keras   = access Keras imports
K.PT Module.PyTorch = access PyTorch imports
"""
from .log import get_logger
from KPT.Modules import Modules
from KPT.Decorators import timer, logger
M = Modules()


class PreProcessing():
    """Pre processing class.

    Contains all the actions perform BEFORE the model is trained.
    A timer decorator is applied only to public methods.
    A timer and a logger dectorator is applied to all methods.
    """

    def __init__(self, x, y, with_noise=True, log_file_name="logFile", log_file_dir="./"):
        """Initialise the object
        """
        self.x = x
        self.y = y
        self.with_noise = with_noise
        self.yNoise = self._construct_function(self.x, self.y)
        self.Set = None
        #self.Set_Keras = None
        #self.Set_PyTorch = None
        self.split_method = None
        # Initializing logger object to write custom logs
        self.log_file_name = log_file_name
        self.log_file_dir = log_file_dir
        self.logger = get_logger(
            log_file_name=self.log_file_name, log_dir=self.log_file_dir)

    def _construct_function(self, x, y):
        """Construct testing function.

        Essentially add noise to the testing function. This is
        generally done to spicy up the training and make it a 
        bit more difficult. NNs are known to be very good at
        not learning the noise. Obviously in extreme cases this is
        no longer valid and if overfitted then the ANN has learnt
        the noise!

        Parameters
        ----------
        x : array-like of shape (len(x))
            Independent variable
        y : array-like of shape (len(y))
            Dependent variable with no noise added

        Returns
        -------
        y : array-like of shape (len(y))
            With and without noise
        """

        if self.with_noise == True:
            y = self.y + M.np.random.normal(0, 0.01, len(x))
        else:
            y = self.y

        return y

    @logger
    @timer
    def plot_test_function(self):
        """Plot test function.

        Quickly visualise the test function. Useful to see the level of
        noise added in the constructionFunction method.

        Parameters
        ----------
        x : array-like of shape (len(x))
            Independent variable
        y : array-like of shape (len(y))
            Dependent variable with no noise added

        Returns
        -------
        None
        """

        M.rcParams['font.size'] = 20
        M.rcParams['figure.figsize'] = 15, 6

        fig = M.plt.figure()
        ax = fig.add_subplot(111)

        M.plt.plot(self.x, self.y, "k-", lw=5, label="No noise")
        if self.with_noise == True:
            M.plt.plot(self.x, self.yNoise, "r-", lw=3, label="With noise")

        ax, legenfObejct = self._fancy_plot(ax)

    @logger
    @timer
    def plot_split_set(self):
        """Plot split set.

        Plot train, set and validation tests.

        Parameters:
        ----------
        None

        Returns:
        --------
        None
        """

        M.rcParams['font.size'] = 20
        M.rcParams['figure.figsize'] = 15, 6

        fig = M.plt.figure()
        ax = fig.add_subplot(111)

        # Plot the train and set which are always present
        if self.split_method in [2, 3]:
            M.plt.plot(self.Set["X_train"], self.Set["y_train"],
                       "ks", lw=5, label="Train set")
            M.plt.plot(self.Set["X_test"], self.Set["y_test"],
                       "rs", lw=5, label="Test set")
        # Plot the validation set if present
        if self.split_method in [3]:
            M.plt.plot(self.Set["X_val"], self.Set["y_val"],
                       "ys", lw=5, label="Val set")

        ax, legenfObejct = self._fancy_plot(ax, col_No=3)

        return None

    @logger
    def _fancy_plot(self, ax, col_No=3):
        """Fancy plot.

        Just add some fancy grid and label. Essentially some
        boiler plate code bmoved into a function.

        Parameters
        ----------
        ax : axis instance
        col_No : int
            No of columns in the legend

        Returns
        -------
        ax : axis instance
            Updated fancy instance of the axis
        legend_object : legend axis
            Updated fancy instance of the legend
        """

        ax.tick_params(which='major', direction='in', length=10, width=2)
        ax.tick_params(which='minor', direction='in', length=6, width=2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.grid()
        ax.minorticks_on()

        legend_object = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
                                  ncol=col_No, fontsize=25, fancybox=True, shadow=False,
                                  facecolor="w", framealpha=1)
        legend_object.get_frame().set_linewidth(2.0)
        legend_object.get_frame().set_edgecolor("k")

        return ax, legend_object

    @logger
    @timer
    def split_dataset(self, method=2, test_size=0.2):
        """Split dataset.

        Decide how to spit the dataset.
        method = 2 splits dataset in train + test
        method = 3 splits datset in train + validation + test

        Parameters
        ----------
        method : int
            2 split the set into training and test
            3 split the set into training, validation and test

        test_size : float
            Provides in percentage the test size

        Returns:
        --------
        Set : dict[str, float]
            Dictionary containing the split dataset
        """

        self.split_method = method
        self.yNoise = self._construct_function(self.x, self.y)

        def _minimumSplit():
            Set = {}
            X_, X_test, y_, y_test = M.train_test_split(self.x,
                                                        self.yNoise,
                                                        test_size=test_size,
                                                        random_state=7,
                                                        shuffle=True)
            Set["X_test"] = X_test
            Set["X_train"] = X_
            Set["y_test"] = y_test
            Set["y_train"] = y_

            self.logger.info("Checking ORIGINAL dimensions: " +
                             str(len(self.x)) + ", " + str(len(self.yNoise)))
            return Set

        def _threeSetSplit():
            Set = {}
            Set_min = _minimumSplit()
            X_train, X_val, y_train, y_val = M.train_test_split(Set_min["X_train"],
                                                                Set_min["y_train"],
                                                                test_size=test_size,
                                                                random_state=7,
                                                                shuffle=True)
            Set["X_test"] = Set_min["X_test"]
            Set["X_train"] = X_train
            Set["X_val"] = X_val
            Set["y_test"] = Set_min["y_test"]
            Set["y_train"] = y_train
            Set["y_val"] = y_val

            self.logger.info("Checking VALIDATION set dimensions: " +
                             str(len(Set["X_val"])) + ", " + str(len(Set["y_val"])))
            self.logger.info("Checking TEST set dimensions: " + str(len(
                Set_min["X_test"])) + ", " + str(len(Set_min["y_test"])))
            self.logger.info("Checking TRAIN set dimensions: " + str(len(
                Set["X_train"])) + ", " + str(len(Set["y_train"])))
            return Set

        if int(method) == 3:
            Set = _threeSetSplit()
        else:
            Set = _minimumSplit()

        # Update the object
        self.split_method = method
        self.Set = Set

        return Set

    @logger
    @timer
    def prepare_input(self, framework, batch_size=16):
        """Prepare Inputs

        Each framework prepare the input in different way.
        Keras (_K) accepts numpy.ndarray
        PyTorch (_PT) is a bit more picky and you have to use Variable() which returns a tensor. 
        Make sure you pass a 2D numpy.ndarray to Variable, you can do this by using reshape()

        Parameters
        ----------
        framework : string
            The name of the framework used

        batch_size : int
            The size of the batch size

        Returns
        -------
        SetFramework : dict [str, float]
            For Keras a dictionary with X, y for each split
            For PyTorch a dictionary with a loader, X, y for each split
        """

        #print("Current set splitting is: ", self.Set.keys())

        def prepareKerasInputs(X, y):
            X = X.reshape(-1, 1)
            y = y.reshape(-1, 1)

            return X, y

        def preparePyTorchInputs(X, y):
            """Create PyTorch data loader

            X np.ndarray
                Feature vector/matrix
            y np.ndarray
                target(s)
            """

            # From numpy to torch tensor
            x = M.torch.from_numpy(X)
            y = M.torch.from_numpy(y)
            #print(x.shape, y.shape)

            # 2D dimensinal input
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            #print(x.shape, y.shape)

            # PyTorch can only train on Variable
            x = M.Variable(x.float())
            y = M.Variable(y.float())
            #print(x.shape, y.shape)
            #print(type(x), type(y))

            torch_dataset = M.Data.TensorDataset(x, y)

            # Split data in batches
            loader = M.Data.DataLoader(
                dataset=torch_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2)

            return loader, x, y

        if framework.lower() in ["pytorch", "pt"]:
            Set_PyTorch = {}
            loader_test, X_test, y_test = preparePyTorchInputs(
                self.Set["X_test"], self.Set["y_test"])
            Set_PyTorch["PT_loader_test"] = loader_test
            Set_PyTorch["PT_X_test"] = X_test
            Set_PyTorch["PT_y_test"] = y_test

            loader_train, X_train, y_train = preparePyTorchInputs(
                self.Set["X_train"], self.Set["y_train"])
            Set_PyTorch["PT_loader_train"] = loader_train
            Set_PyTorch["PT_X_train"] = X_train
            Set_PyTorch["PT_y_train"] = y_train

            if self.split_method == 3:
                loader_val, X_val, y_val = preparePyTorchInputs(
                    self.Set["X_val"], self.Set["y_val"])
                Set_PyTorch["PT_loader_val"] = loader_val
                Set_PyTorch["PT_X_val"] = X_val
                Set_PyTorch["PT_y_val"] = y_val

            else:
                self.logger.error(" ")
                self.logger.error("Splitting method:" +
                                  str(self.split_method) + " NOT known!")
                self.logger.error(" ")
                M.sys.exit

            SetFramework = Set_PyTorch

        elif framework.lower() in ["keras", "k"]:
            Set_Keras = {}
            X_test, y_test = prepareKerasInputs(
                self.Set["X_test"], self.Set["y_test"])
            Set_Keras["K_X_test"] = X_test
            Set_Keras["K_y_test"] = y_test

            X_train, y_train = prepareKerasInputs(
                self.Set["X_train"], self.Set["y_train"])
            Set_Keras["K_X_train"] = X_train
            Set_Keras["K_y_train"] = y_train

            if self.split_method == 3:
                X_val, y_val = prepareKerasInputs(
                    self.Set["X_val"], self.Set["y_val"])
                Set_Keras["K_X_val"] = X_val
                Set_Keras["K_y_val"] = y_val

            else:
                self.logger.error("")
                self.logger.error("Splitting method:" +
                                  str(self.split_method) + " NOT known!")
                self.logger.error("")
                M.sys.exit

            SetFramework = Set_Keras

        self.Set = SetFramework
        return SetFramework
