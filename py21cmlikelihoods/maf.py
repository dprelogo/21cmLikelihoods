import tensorflow as tf
import tensorflow_probability as tfp
import cloudpickle
import pickle

tfd = tfp.distributions
tfb = tfp.bijectors

# TODO: saving and loading of the model
class MaskedAutoregressiveFlow:
    """Masked Autoregressive Flow, series of MADE bijectors.

    Args:
        n_dim (int): dimensionality of the pdf.
        hidden_units_dim (int): dimensionality of fully-connected layers in every MADE.
        activation: nonlinear activation function.
        optimizer: optimizer to use.
        input_order (str): order of degrees to the input units, one of
            `["random", "left-to-right", "right-to-left"]`, applied to every MADE.
        n_MADEs (int): number of MADE transformations.
        kernel_initializer: initializer for weights, see `tfb.AutoregressiveNetwork`.
        bias_initializer: initializer for biases, see `tfb.AutoregressiveNetwork`.
        kernel_regularizer: weights' regularizer, see `tfb.AutoregressiveNetwork`.
        bias_regularizer: biases' regularizer, see `tfb.AutoregressiveNetwork`.
    """

    def __init__(
        self,
        n_dim,
        hidden_units_dim=50,
        activation=tf.keras.layers.LeakyReLU(0.01),
        optimizer=tf.optimizers.Adam(1e-3),
        input_order="random",
        n_MADEs=10,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        last_bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
    ):
        self.n_dim = n_dim
        self.hidden_units_dim = hidden_units_dim
        self.activation = activation
        self.optimizer = optimizer
        self.input_order = input_order
        self.n_MADEs = n_MADEs

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        if last_bias_initializer is None:
            self.last_bias_initializer = bias_initializer
        else:
            self.last_bias_initializer = last_bias_initializer

        self._configure()

    def compile(self, new_optimizer=None):
        """Compiling keras MAF model. Pass `new_optimizer` for re-compilation
        of the model with the new model, keeping the model's weights.
        """
        if new_optimizer is None:
            self.model.compile(
                optimizer=self.optimizer, loss=lambda _, log_prob: -log_prob
            )
        else:
            self.model.compile(
                optimizer=new_optimizer, loss=lambda _, log_prob: -log_prob
            )

    def _configure_model(self):
        x_ = tf.keras.layers.Input(shape=(self.n_dim,), dtype=tf.float32)
        log_prob_ = tf.reduce_mean(self.log_prob(x_))
        self.model = tf.keras.Model(x_, log_prob_)

    def _configure(self):
        mades = [
            tfb.MaskedAutoregressiveFlow(
                tfb.AutoregressiveNetwork(
                    params=2,
                    event_shape=(self.n_dim,),
                    hidden_units=[self.hidden_units_dim] * self.n_dim,
                    activation=self.activation,
                    input_order=self.input_order,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer if i < self.n_MADEs - 1 else self.last_bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                ),
                name=f"MADE_{i}",
            )
            for i in range(self.n_MADEs)
        ]
        #         mades = [i for j in mades for i in (j, tfb.BatchNormalization())][:-1]
        bijector = tfb.Chain(mades)

        self.distribution = tfd.TransformedDistribution(
            distribution=tfd.Sample(
                tfd.Normal(loc=0.0, scale=1.0), sample_shape=(self.n_dim,)
            ),
            bijector=bijector,
        )

        self._configure_model()
        self.compile()

    def train(
        self,
        epochs,
        dataset,
        dataset_val=None,
        verbose=1,
        save=True,
        save_history=True,
        filename="model",
        callbacks=None,
        **kwargs,
    ):
        history = self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=dataset_val,
            verbose=verbose,
            callbacks=callbacks,
            **kwargs,
        )
        if save:
            self.save(filename)
        if save_history:
            with open(f"{filename}_history.pkl", "wb") as f:
                cloudpickle.dump(history.history, f)

    def save(self, filename="model"):
        # self.model.save_weights(filename + ".h5")
        with open(filename + ".pkl", "wb") as f:
            cloudpickle.dump(self.distribution.trainable_variables, f)

    def load(self, filename="model"):
        # self.model.load_weights(filename + ".h5")
        with open(filename + ".pkl", "rb") as f:
            variables = pickle.load(f)
        for c, v in zip(self.distribution.trainable_variables, variables):
            c.assign(v)
        self._configure_model()
        self.compile()

    def sample(self, s):
        return self.distribution.sample(s)

    def prob(self, tensor):
        return self.distribution.prob(tensor)

    def log_prob(self, tensor):
        return self.distribution.log_prob(tensor)


class ConditionalMaskedAutoregressiveFlow(MaskedAutoregressiveFlow):
    """Conditional Masked Autoregressive Flow, series of MADE bijectors, with
    conditional and variable pdf dimensions.

    Args:
        n_dim (int): dimensionality of the non-conditional part of the pdf.
        cond_n_dim (int): dimensionality of the conditional part of the pdf.
        cond_input_layers (str): either "all_layers", when the conditional input
            will be combined with the network at every layer, or "first_layer",
            when the conditional input is combined only at first layer.
        hidden_units_dim (int): dimensionality of fully-connected layers in every MADE.
        activation: nonlinear activation function.
        optimizer: optimizer to use.
        input_order (str): order of degrees to the input units, one of
            `["random", "left-to-right", "right-to-left"]`, applied to every MADE.
        n_MADEs (int): number of MADE transformations.
        kernel_initializer: initializer for weights, see `tfb.AutoregressiveNetwork`.
        bias_initializer: initializer for biases, see `tfb.AutoregressiveNetwork`.
        kernel_regularizer: weights' regularizer, see `tfb.AutoregressiveNetwork`.
        bias_regularizer: biases' regularizer, see `tfb.AutoregressiveNetwork`.
    """

    def __init__(
        self,
        n_dim,
        cond_n_dim,
        cond_input_layers="all_layers",
        hidden_units_dim=50,
        activation=tf.keras.layers.LeakyReLU(0.01),
        optimizer=tf.optimizers.Adam(1e-3),
        input_order="random",
        n_MADEs=10,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        last_bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
    ):
        self.n_dim = n_dim
        self.cond_n_dim = cond_n_dim
        self.cond_input_layers = cond_input_layers
        self.hidden_units_dim = hidden_units_dim
        self.activation = activation
        self.optimizer = optimizer
        self.input_order = input_order
        self.n_MADEs = n_MADEs

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        if last_bias_initializer is None:
            self.last_bias_initializer = bias_initializer
        else:
            self.last_bias_initializer = last_bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self._configure()

    def _conditional_input(self, conditional):
        return dict(
            zip(
                [f"MADE_{i}" for i in range(self.n_MADEs)],
                [{"conditional_input": conditional} for i in range(self.n_MADEs)],
            )
        )

    def _configure_model(self):
        x_ = tf.keras.layers.Input(shape=(self.n_dim,), dtype=tf.float32)
        c_ = tf.keras.layers.Input(shape=(self.cond_n_dim,), dtype=tf.float32)
        log_prob_ = tf.reduce_mean(self.log_prob(x_, c_))
        self.model = tf.keras.Model([x_, c_], log_prob_)

    def _configure(self):
        mades = [
            tfb.MaskedAutoregressiveFlow(
                tfb.AutoregressiveNetwork(
                    params=2,
                    conditional=True,
                    conditional_input_layers=self.cond_input_layers,
                    event_shape=(self.n_dim,),
                    conditional_event_shape=(self.cond_n_dim,),
                    hidden_units=[self.hidden_units_dim]
                    * (self.n_dim + self.cond_n_dim),
                    activation=self.activation,
                    input_order=self.input_order,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer if i < self.n_MADEs - 1 else self.last_bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                ),
                name=f"MADE_{i}",
            )
            for i in range(self.n_MADEs)
        ]
        #         mades = [i for j in mades for i in (j, tfb.BatchNormalization())][:-1]
        bijector = tfb.Chain(mades)

        self.distribution = tfd.TransformedDistribution(
            distribution=tfd.Sample(
                tfd.Normal(loc=0.0, scale=1.0), sample_shape=(self.n_dim,)
            ),
            bijector=bijector,
        )

        self._configure_model()
        self.compile()

    def sample(self, s, conditional):
        return self.distribution.sample(
            s, bijector_kwargs=self._conditional_input(conditional)
        )

    def prob(self, tensor, conditional):
        return self.distribution.prob(
            tensor, bijector_kwargs=self._conditional_input(conditional)
        )

    def log_prob(self, tensor, conditional):
        return self.distribution.log_prob(
            tensor, bijector_kwargs=self._conditional_input(conditional)
        )
