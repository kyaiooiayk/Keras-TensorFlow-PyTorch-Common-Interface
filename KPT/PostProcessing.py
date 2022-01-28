"""
Import modules
M.O  Module.Others  = access Other imports
M.K  Module.Keras   = access Keras imports
K.PT Module.PyTorch = access PyTorch imports
"""
from KPT.Modules import Modules
from KPT.Decorators import timer, logger
M = Modules()


class PostProcessing():
    """Post processing class.

    Contains all the actions perform AFTER the model is trained.
    """

    def __init__(self, model_obj):
        """Initialise the object
        """
        self.model = model_obj.model
        self.set = model_obj.set_
        self.framework = model_obj.framework
        self.history = model_obj.history
        self.logger = model_obj.logger
        self.log_file_name = model_obj.log_file_name
        self.log_file_dir = model_obj.log_file_dir

    @logger
    @timer
    def get_metrics(self):
        """Get metrics.

        Get metrics for regression problems. These are:
        MSE  : Mean  Square Error
        RMSE : RootMean  Square Error
        MAE  : Mean Absolute Error
        R2   : coefficient of determination. This is a value between 
               0 and 1 for no-fit and perfect fit respectively. 

        Parameters
        ----------
        None

        Returns
        -------
        String = "Succcess"
            This is used for unittesting purpouses only
        """

        if self.framework.lower() in ["keras",  "k"]:
            key = "K"
            preds = self.model.predict(self.set[key + "_X_test"])
        elif self.framework.lower() in ["pt", "pytorch"]:
            key = "PT"
            preds = self.model(self.set[key + "_X_test"]).data.numpy()

        else:
            self.logger.error(" ")
            self.logger.error("Model name:" + str(self.framework) + " NOT known!")
            self.logger.error(" ")
            # Exit the code!
            M.sys.exit()

        r2s = M.r2_score(self.set[key + "_y_test"], preds)
        mse = M.mean_squared_error(self.set[key + "_y_test"], preds)
        rmse = M.np.sqrt(mse)
        mae = M.mean_absolute_error(self.set[key + "_y_test"], preds)

        self.logger.info("[MSE]_" + self.framework + ": %.4f" % mse)
        self.logger.info("[RMSE]_" + self.framework + ": %.4f" % rmse)
        self.logger.info("[MAE]_" + self.framework + ": %.4f" % mae)
        self.logger.info("[R2]_" + self.framework + ": %.4f" % r2s)

        return "Success"

    @logger
    @timer
    def plot_learning_curve(self,  history_extra=None, framework_extra=None):
        """Plot learning curve.

        Compare training vs. valdation losses. If history_extra parameter
        is provided it then compares two plots.

        Parameters
        ----------
        history_extra : dict, default=None
            Extra dictionary containing the training history of
            another model
        framework_extra : string, default=None
            Name of the other framework

        Returns
        -------
        None
        """

        # High level plotting setting
        M.rcParams['font.size'] = 20
        M.rcParams['figure.figsize'] = 15, 6

        fig = M.plt.figure()
        ax = fig.add_subplot(111)

        M.plt.plot(range(len(self.history["train"])), self.history["train"],
                   "k-", lw=3, label=self.framework + " train")
        M.plt.plot(range(len(self.history["val"])), self.history["val"],
                   "k--", lw=3, label=self.framework + " val")

        if history_extra != None:
            M.plt.plot(range(len(
                history_extra["train"])), history_extra["train"], "r-",  lw=3, label=framework_extra + " train")
            M.plt.plot(range(len(
                history_extra["val"])), history_extra["val"], "r--",  lw=3, label=framework_extra + " val")

        M.plt.ylabel("loss")
        M.plt.xlabel("epoch")
        M.plt.yscale('log')
        ax, legenfObejct = self._fancy_plot(ax, col_No=2)

    @logger
    def _fancy_plot(self, ax, col_No=3):
        """Fancy plot.

        Just add some fancy grid and label. Essentially some
        boiler plate code moved into a function.

        Parameters
        ----------
        ax : axis instance
            Axis instance
        col_No : int
            No of columns

        Returns
        -------
        ax : axis instance
            Modified axis instance
        legendObejct : legend instance
            Modified legend instance
        """

        ax.tick_params(which='major', direction='in', length=10, width=2)
        ax.tick_params(which='minor', direction='in', length=6, width=2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.grid()
        ax.minorticks_on()

        legend_object = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
                                  ncol=col_No, fontsize=20, fancybox=True, shadow=False,
                                  facecolor="w", framealpha=1)
        legend_object.get_frame().set_linewidth(2.0)
        legend_object.get_frame().set_edgecolor("k")

        return ax, legend_object

    @logger
    @timer
    def plot_final_result(self, which_set):
        """Plot final results.

        Plot the given sets against the predicted values.

        Parameters
        ----------
        which_set : string
            List of string with the name of the set.
            For instnace: ["train", "test", "val"]

        Returns
        -------
        None
        """

        M.rcParams['font.size'] = 20
        M.rcParams['figure.figsize'] = 15, 6

        fig = M.plt.figure()
        ax = fig.add_subplot(111)

        colourList = ["k", "r", "g", "m", "b"]

        # Keras framework
        if self.framework.lower() in ["k", "keras"]:
            for tmp in which_set:
                clr = M.random.choice(colourList)
                try:
                    self.set["K_X_" + tmp.lower()]
                    ax.scatter(self.set["K_X_" + tmp.lower()], self.set["K_y_" + tmp.lower()],
                               color=clr, marker='^', label=tmp.upper() + " set")
                    ax.scatter(self.set["K_X_" + tmp.lower()], self.model.predict(self.set["K_X_" + tmp.lower()]),
                               color=clr, marker='o', label="Keras " + tmp.upper() + " set prediction")
                except:
                    print("Key ", tmp, " does NOT exist!")

        # PyTorch framework
        elif self.framework.lower() in ["pt", "pytorch"]:
            for tmp in which_set:
                clr = M.random.choice(colourList)
                try:
                    self.set["PT_X_" + tmp.lower()]
                    prediction = self.model(self.set["PT_X_" + tmp.lower()])
                    ax.scatter(self.set["PT_X_" + tmp.lower()].data.numpy(), self.set["PT_y_" + tmp.lower()].data.numpy(),
                               color=clr, marker='^', label=tmp.upper() + " set")
                    ax.scatter(self.set["PT_X_" + tmp.lower()].data.numpy(), prediction.data.numpy(),
                               color=clr, marker='o', label="PyTorch " + tmp.upper() + " set prediction")
                except:
                    print("Key ", tmp, " does NOT exist!")
        else:
            self.logger.error(" ")
            self.logger.error("Framework: " + self.framework + " NOT known!")
            self.logger.error(" ")
            # Exit the code
            M.sys, exit()

        ax.set_xlabel('Independent variable', fontsize=24)
        ax.set_ylabel('Dependent variable', fontsize=24)
        # Apply fancy axis modififcation
        ax, legenfObejct = self._fancy_plot(ax, col_No=2)
