from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import cloudpickle

tfd = tfp.distributions
tfb = tfp.bijectors

# TODO: make bias initializer option
class ConditionalGaussian(tfd.Distribution):
    """Conditional Gaussian NDE distribution.

    Args:
        n_parameters (int): dimensionality of the parameter space.
        n_data (int): dimensionality of the data space.
        covariance (array): the covariance to use.
            It can be either 1D or 2D array, depending if full covariance or just
            diagonal is keeping fixed. If not specified, the covariance is fitted across
            the parameter space.
        exponentiate (bool): In the case `covariance` is fixed, it controls if the final output
            of the NN should be exponentiated in order to match the space
            in which the covariance has been calculated. If `z` is the vector outputted
            by the NN and `exponentiate is True`, then final output is `exp(z)`.
        mean (array): should the final output of the NN be shifted by some mean vector?
            This allows for the non-zero mean data to be passed as a training data to the network,
            while keeping nice properties of better training when the data is zero-mean.
        std (array): should the final output of the NN be scaled by some standard deviation?
            This allows for not normalized data to be passed as a training data to the network,
            while keeping nice properties of better training when the data is normalized.
            Furthermore, it will give correctly normalized log-likelihoods.
            With `mean` and `std` specified and `z` the output vector of the NN,
            the final output will be `std * z + mean` or `std * exp(z) + mean`
            if `exponentiate is True`.
        diagonal_covariance (bool): to fit diagonal covariance or full covariance.
            It is used only if the `covariance` is not specified, i.e. it has to be fitted.
        n_hidden (list): specify the number of hidden layers and number of neurons
            in the fully-connected network going from parameter space to data space.
        activation (tf.keras.activations, tf.keras.layers): `keras` activation function to use.
        optimizer (tf.optimizers.Optimizer): `keras` optimizer to use.
        kernel_initializer (callable): function to initialize kernels.
        final_bias_initializer (str): gives a fine control over what bias initializer
            should be used in the final layer. Common choices are "zeros" or "ones".
        dtype: see `tfd.Distribution`.
        reparametrization_type: see `tfd.Distribution`.
        validate_args: see `tfd.Disitrbution`.
        allow_nan_stats: see `tfd.Distribution`.

    Attributes:
        train: training the NDE
        compile: compiling the keras model
        save: saving the model
        load: loading the model
        log_prob: log-probability of the likelihood for samples
        prob: probability of the likelihood for samples
        sample: sampling the likelihood for the fixed conditional
    """
    def __init__(
        self,
        n_parameters,
        n_data,
        covariance=None,
        exponentiate=False,
        mean=None,
        std=None,
        diagonal_covariance=False,
        n_hidden=[50, 50],
        activation=tf.keras.layers.LeakyReLU(0.01),
        optimizer=tf.optimizers.Adam(1e-3),
        kernel_initializer=partial(
            tf.keras.initializers.RandomNormal, mean=0.0, stddev=1e-5, seed=None
        ),
        final_bias_initializer="zeros",
        dtype=tf.float32,
        reparameterization_type=None,
        validate_args=False,
        allow_nan_stats=True,

    ):
        super().__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
        )

        # dimension of data and parameter spaces
        self.n_parameters = n_parameters
        self.n_data = n_data

        if covariance is not None and covariance.shape not in (
            (n_data,),
            (n_data, n_data),
        ):
            raise ValueError(
                f"Covariance should be of shape {n_data} or ({n_data}, {n_data}) "
                f"but is {covariance.shape}."
            )
        self.covariance = covariance
        self.diagonal_covariance = diagonal_covariance

        self.n_hidden = n_hidden
        self.exponentiate = exponentiate
        if self.exponentiate and self.covariance is None:
            raise ValueError(
                "In the most general model there's no reason to exponentiate. "
                "The purpose of it is to correct for the training vs. covariance dimensions."
            )
        self.mean = mean
        self.std = std
        self.activation = activation
        self.optimizer = optimizer
        self.architecture = [self.n_parameters] + self.n_hidden

        self.model = self._build_network(kernel_initializer, final_bias_initializer)
        # self.compile()
        self.conditional_log_prob = self.log_prob
        self.conditional_prob = self.prob
        self.conditional_sample = self.sample

    def train(
        self,
        epochs,
        dataset,
        dataset_val=None,
        verbose=1,
        pretrain=True,
        pretrain_epochs=None,
        pretrain_optimizer=None,
        save=True,
        save_history=True,
        filename="model",
        callbacks=None,
        pretrain_callbacks=None,
    ):
        """Training the likelihood.

        It allows for two-phase traininig, in which pre-training fits only
        the Gaussian mean, and train jointly fits mean and covariance
        (ambiguous for the case of fixed covariance case).

        Args:
            epochs (int): number of epochs to train.
            dataset (tf.dataset.Dataset): batched training dataset of 
                `(parameters, data)` pairs.
            dataset_val (tf.dataset.Dataset): batched validation dataset of
                `(parameters, data)` pairs.
            verbose (int): verbosity of the output during the training.
            pretrain (bool): if `True`, runs pretraining of the mean.
            pretrain_epochs (int): number of pretraining epochs.
            pretrain_optimizer (tf.optimizers.Optimizer): if specified, a separate optimizer will
                be used for pretraining.
            save (bool): either to save model or not.
            save_history (bool): either to save training history or not.
            filename (str): base filename.
            callbacks (list): list of keras callbacks called during training
            pretrain_callbacks (list): list of keras callbacks called during pre-training.
        
        """
        if pretrain:
            self.compile(new_optimizer=pretrain_optimizer, pretrain_phase=True)
            pretrain_epochs = epochs if pretrain_epochs is None else pretrain_epochs
            self.model.fit(
                dataset,
                epochs=pretrain_epochs,
                validation_data=dataset_val,
                verbose=verbose,
                callbacks=pretrain_callbacks,
            )

        self.compile(pretrain_phase=False)
        history = self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=dataset_val,
            verbose=verbose,
            callbacks=callbacks,
        )
        if save:
            self.save(filename)
        if save_history:
            with open(f"{filename}_history.pkl", "wb") as f:
                cloudpickle.dump(history.history, f)

    def _loss(self, x, distribution):
        if self.last_transformation is not None:
            x = self.last_transformation(x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return -distribution.log_prob(x)

    def _loss_pretrain(self, x, distribution):
        mean = distribution.mean()
        if self.last_inverse_activation is not None:
            mean = self.last_inverse_activation(mean)
        if self.last_inverse_transformation is not None:
            mean = self.last_inverse_transformation(mean)
        squared_difference = 0.5 * tf.square(x - mean)
        return tf.reduce_sum(squared_difference, axis=-1)

    def compile(self, new_optimizer=None, pretrain_phase=True):
        """Compiling the model.

        Args:
            new_optimizer (tf.optimizers.Optimizer): if specified, the model is
                compiled with that particular optimizer, otherwise it is compiled
                with the one specified with the particular class instance.
            pretrain_phase (bool): if the compilation is done for the pre-training
                phase or not. If yes, it uses pre-training loss function, otherwise
                it uses the main loss.
        """
        loss = self._loss_pretrain if pretrain_phase else self._loss
        optimizer = self.optimizer if new_optimizer is None else new_optimizer
        self.model.compile(optimizer=optimizer, loss=loss)

    def _build_network(self, kernel_initializer, final_bias_initializer):
        if self.exponentiate:
            self.last_activation = tf.math.exp
            self.last_inverse_activation = tf.math.log
        else:
            self.last_activation = None
            self.last_inverse_activation = None

        if self.mean is None or self.std is None:
            self.last_transformation = None
            self.last_inverse_transformation = None
        else:
            self.mean = tf.convert_to_tensor(self.mean, dtype=self._dtype)
            self.std = tf.convert_to_tensor(self.std, dtype=self._dtype)
            self.last_transformation = lambda x: self.std * x + self.mean
            self.last_inverse_transformation = lambda x: (x - self.mean) / self.std
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    self.architecture[layer + 1],
                    input_shape=(size,),
                    activation=self.activation,
                    kernel_initializer=kernel_initializer(),
                )
                for layer, size in enumerate(self.architecture[:-1])
            ]
        )

        if self.covariance is None:
            if self.diagonal_covariance:
                model.add(
                    tf.keras.layers.Dense(
                        2 * self.n_data,
                        kernel_initializer=kernel_initializer(),
                        bias_initializer=final_bias_initializer,
                    )
                )
                model.add(
                    tf.keras.layers.Lambda(
                        lambda x: tf.concat(
                            [x[..., : self.n_data], x[..., self.n_data :] ** 2],
                            -1,
                        )
                    )
                )
                model.add(
                    tfp.layers.DistributionLambda(
                        lambda x: tfd.MultivariateNormalDiag(
                            loc=x[..., : self.n_data],
                            scale_diag=x[..., self.n_data :],
                            validate_args=self._validate_args,
                        )
                    )
                )
            else:
                model.add(
                    tf.keras.layers.Dense(
                        tfp.layers.MultivariateNormalTriL.params_size(self.n_data),
                        kernel_initializer=kernel_initializer(),
                        bias_initializer=final_bias_initializer,
                    )
                )
                model.add(
                    tfp.layers.MultivariateNormalTriL(
                        self.n_data,
                        validate_args=self._validate_args,
                    )
                )
        else:
            self.covariance = tf.convert_to_tensor(
                self.covariance.astype(self._dtype.as_numpy_dtype)
            )
            if tf.rank(self.covariance) == 1:
                L = tf.linalg.diag(tf.math.sqrt(self.covariance))
            else:
                L = tf.linalg.cholesky(self.covariance)

            model.add(
                tf.keras.layers.Dense(
                    self.n_data,
                    kernel_initializer=kernel_initializer(),
                )
            )
            if self.last_transformation is not None:
                model.add(tf.keras.layers.Lambda(self.last_transformation))
            if self.last_activation is not None:
                model.add(tf.keras.layers.Activation(self.last_activation))
            model.add(
                tfp.layers.DistributionLambda(
                    lambda x: tfd.MultivariateNormalTriL(
                        loc=x,
                        scale_tril=L,
                        validate_args=self._validate_args,
                    )
                )
            )

        return model

    def save(self, filename="model"):
        """Saving the model."""
        self.model.save_weights(filename + ".h5")

    def load(self, filename="model"):
        """Loading the model."""
        self.model.load_weights(filename + ".h5")
        self.compile(pretrain_phase=False)

    def log_prob(self, x, conditional, x_in_final_space=False):
        """Log-probability of the likelihood for a set of samples.
        
        Args:
            x (array): of shape `(N, n_data)`, data samples for which to
                compute log-probability.
            conditional (array): of shape `(N, n_parameters)`, parameters of the
                data samples for which to compute log-probability.
            x_in_final_space (bool): are data samples in the "final space" or not.
                This is valid only if `exponentiate` and/or (`mean` and `std`)
                have been specified. For instance, if z is the output of the NN
                and the final output is `std * exp(z) + mean`, this flag specifies
                if the data samples are in z-space or final space.

        Returns:
            Array of `(N,)` log-likelihoods.
        """
        x = tf.cast(x, self._dtype)
        if not x_in_final_space:
            if self.last_transformation is not None:
                x = self.last_transformation(x)
            if self.last_activation is not None:
                x = self.last_activation(x)

        conditional = tf.cast(conditional, self._dtype)
        # squeeze = True if tf.rank(x) == 1 else False
        x = tf.expand_dims(x, 0) if tf.rank(x) == 1 else x
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        prob = self.model(conditional).log_prob(x)
        # return tf.squeeze(prob, 0) if squeeze else prob
        return prob

    def prob(self, x, conditional, x_in_final_space=False):
        """Probability of the likelihood for a set of samples.
        
        Args:
            x (array): of shape `(N, n_data)`, data samples for which to
                compute log-probability.
            conditional (array): of shape `(N, n_parameters)`, parameters of the
                data samples for which to compute log-probability.
            x_in_final_space (bool): are data samples in the "final space" or not.
                This is valid only if `exponentiate` and/or (`mean` and `std`)
                have been specified. For instance, if z is the output of the NN
                and the final output is `std * exp(z) + mean`, this flag specifies
                if the data samples are in z-space or final space.

        Returns:
            Array of `(N,)` likelihoods.
        """
        x = tf.cast(x, self._dtype)
        if not x_in_final_space:
            if self.last_transformation is not None:
                x = self.last_transformation(x)
            if self.last_activation is not None:
                x = self.last_activation(x)
        conditional = tf.cast(conditional, self._dtype)
        # squeeze = True if tf.rank(x) == 1 else False
        x = tf.expand_dims(x, 0) if tf.rank(x) == 1 else x
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        prob = self.model(conditional).prob(x)
        # return tf.squeeze(prob, 0) if squeeze else prob
        return prob

    def sample(self, s, conditional, result_in_final_space=True):
        """Sampling the likelihood.

        Args:
            s (int): number of samples.
            conditional (array): parameters on which to condition the samples.
            result_in_final_space (bool): should samples be returned in "final space"
                or not. This is valid only if `exponentiate` and/or (`mean` and `std`)
                have been specified. For instance, if z is the output of the NN
                and the final output is `std * exp(z) + mean`, this flag specifies
                if the samples are returned in z-space or final space.

        Returns:
            Array of samples of shape `(N, n_data)`.
        """
        conditional = tf.cast(conditional, self._dtype)
        squeeze = True if tf.rank(conditional) == 1 else False
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        samples = self.model(conditional).sample(s)
        if not result_in_final_space:
            if self.last_inverse_activation is not None:
                samples = self.last_inverse_activation(samples)
            if self.last_inverse_transformation is not None:
                samples = self.last_inverse_transformation(samples)
        return tf.squeeze(samples, 1) if squeeze else samples


class ConditionalGaussianMixture(tfd.Distribution):
    """Conditional Gaussian Mixture NDE distribution.

    Args:
        n_parameters (int): dimensionality of the parameter space.
        n_data (int): dimensionality of the data space.
        n_components (int): number of Gaussian Mixture components.
        n_hidden (list): specify the number of hidden layers and number of neurons
            in the fully-connected network going from parameter space to data space.
        activation (tf.keras.activations.Activation, tf.keras.layers.Layer): `keras` activation function to use.
        optimizer (tf.optimizers.Optimizer): `keras` optimizer to use.
        kernel_initializer (callable): function to initialize kernels.
        final_bias_initializer (str): gives a fine control over what bias initializer
            should be used in the final layer. Common choices are "zeros" or "ones".
        dtype: see `tfd.Distribution`.
        reparametrization_type: see `tfd.Distribution`.
        validate_args: see `tfd.Disitrbution`.
        allow_nan_stats: see `tfd.Distribution`.

    Attributes:
        train: training the NDE
        compile: compiling the keras model
        save: saving the model
        load: loading the model
        log_prob: log-probability of the likelihood for samples
        prob: probability of the likelihood for samples
        sample: sampling the likelihood for the fixed conditional
    """
    def __init__(
        self,
        n_parameters,
        n_data,
        n_components,
        n_hidden=[50, 50],
        activation=tf.keras.layers.LeakyReLU(0.01),
        optimizer=tf.optimizers.Adam(1e-3),
        kernel_initializer=partial(
            tf.keras.initializers.RandomNormal, mean=0.0, stddev=1e-5, seed=None
        ),
        final_bias_initializer="zeros",
        dtype=tf.float32,
        reparameterization_type=None,
        validate_args=False,
        allow_nan_stats=True,
    ):
        super().__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
        )

        # dimension of data and parameter spaces
        self.n_parameters = n_parameters
        self.n_data = n_data
        self.n_components = n_components

        self.n_hidden = n_hidden
        self.activation = activation
        self.optimizer = optimizer
        self.architecture = [self.n_parameters] + self.n_hidden

        self.model = self._build_network(kernel_initializer, final_bias_initializer)
        # self.compile()
        self.conditional_log_prob = self.log_prob
        self.conditional_prob = self.prob
        self.conditional_sample = self.sample

    def train(
        self,
        epochs,
        dataset,
        dataset_val=None,
        verbose=1,
        pretrain=True,
        pretrain_epochs=None,
        pretrain_optimizer=None,
        save=True,
        save_history=True,
        filename="model",
        callbacks=None,
        pretrain_callbacks=None,
    ):
        """Training the likelihood.

        It allows for two-phase traininig, in which pre-training fits only
        the Gaussian mean, and train jointly fits mean and covariance
        (ambiguous for the case of fixed covariance case).

        Args:
            epochs (int): number of epochs to train.
            dataset (tf.dataset.Dataset): batched training dataset of 
                `(parameters, data)` pairs.
            dataset_val (tf.dataset.Dataset): batched validation dataset of
                `(parameters, data)` pairs.
            verbose (int): verbosity of the output during the training.
            pretrain (bool): if `True`, runs pretraining of the mean.
            pretrain_epochs (int): number of pretraining epochs.
            pretrain_optimizer (tf.optimizers.Optimizer): if specified, a separate optimizer will
                be used for pretraining.
            save (bool): either to save model or not.
            save_history (bool): either to save training history or not.
            filename (str): base filename.
            callbacks (list): list of keras callbacks called during training
            pretrain_callbacks (list): list of keras callbacks called during pre-training.
        """
        if pretrain:
            self.compile(new_optimizer=pretrain_optimizer, pretrain_phase=True)
            pretrain_epochs = epochs if pretrain_epochs is None else pretrain_epochs
            self.model.fit(
                dataset,
                epochs=pretrain_epochs,
                validation_data=dataset_val,
                verbose=verbose,
                callbacks=pretrain_callbacks,
            )

        self.compile(pretrain_phase=False)
        history = self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=dataset_val,
            verbose=verbose,
            callbacks=callbacks,
        )
        if save:
            self.save(filename)
        if save_history:
            with open(f"{filename}_history.pkl", "wb") as f:
                cloudpickle.dump(history.history, f)

    def _loss(self, x, distribution):
        return -distribution.log_prob(x)

    def _loss_pretrain(self, x, distribution):
        mean = distribution.mean()
        squared_difference = 0.5 * tf.square(x - mean)
        return tf.reduce_sum(squared_difference, axis=-1)

    def compile(self, new_optimizer=None, pretrain_phase=True):
        """Compiling the model.

        Args:
            new_optimizer (tf.optimizers.Optimizer): if specified, the model is
                compiled with that particular optimizer, otherwise it is compiled
                with the one specified with the particular class instance.
            pretrain_phase (bool): if the compilation is done for the pre-training
                phase or not. If yes, it uses pre-training loss function, otherwise
                it uses the main loss.
        """
        loss = self._loss_pretrain if pretrain_phase else self._loss
        optimizer = self.optimizer if new_optimizer is None else new_optimizer
        self.model.compile(optimizer=optimizer, loss=loss)

    def _build_network(self, kernel_initializer, final_bias_initializer):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    self.architecture[layer + 1],
                    input_shape=(size,),
                    activation=self.activation,
                    kernel_initializer=kernel_initializer(),
                )
                for layer, size in enumerate(self.architecture[:-1])
            ]
        )

        final_dense_size = tfp.layers.MixtureSameFamily.params_size(
            self.n_components,
            component_params_size=tfp.layers.MultivariateNormalTriL.params_size(
                self.n_data
            ),
        )

        model.add(
            tf.keras.layers.Dense(
                final_dense_size,
                kernel_initializer=kernel_initializer(),
                bias_initializer=final_bias_initializer,
            )
        )

        model.add(
            tfp.layers.MixtureSameFamily(
                self.n_components,
                tfp.layers.MultivariateNormalTriL(self.n_data),
                validate_args=self._validate_args,
            )
        )
        return model

    def save(self, filename="model"):
        """Saving the model."""
        self.model.save_weights(filename + ".h5")

    def load(self, filename="model"):
        """Loading the model."""
        self.model.load_weights(filename + ".h5")
        self.compile()

    def log_prob(self, x, conditional):
        """Log-probability of the likelihood for a set of samples.
        
        Args:
            x (array): of shape `(N, n_data)`, data samples for which to
                compute log-probability.
            conditional (array): of shape `(N, n_parameters)`, parameters of the
                data samples for which to compute log-probability.

        Returns:
            Array of `(N,)` log-likelihoods.
        """
        x = tf.cast(x, self._dtype)

        conditional = tf.cast(conditional, self._dtype)
        # squeeze = True if tf.rank(x) == 1 else False
        x = tf.expand_dims(x, 0) if tf.rank(x) == 1 else x
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        prob = self.model(conditional).log_prob(x)
        # return tf.squeeze(prob, 0) if squeeze else prob
        return prob

    def prob(self, x, conditional):
        """Probability of the likelihood for a set of samples.
        
        Args:
            x (array): of shape `(N, n_data)`, data samples for which to
                compute log-probability.
            conditional (array): of shape `(N, n_parameters)`, parameters of the
                data samples for which to compute log-probability.

        Returns:
            Array of `(N,)` likelihoods.
        """
        x = tf.cast(x, self._dtype)
        conditional = tf.cast(conditional, self._dtype)
        # squeeze = True if tf.rank(x) == 1 else False
        x = tf.expand_dims(x, 0) if tf.rank(x) == 1 else x
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        prob = self.model(conditional).prob(x)
        # return tf.squeeze(prob, 0) if squeeze else prob
        return prob

    def sample(self, s, conditional):
        """Sampling the likelihood.

        Args:
            s (int): number of samples.
            conditional (array): parameters on which to condition the samples.

        Returns:
            Array of samples of shape `(N, n_data)`.
        """
        conditional = tf.cast(conditional, self._dtype)
        squeeze = True if tf.rank(conditional) == 1 else False
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        samples = self.model(conditional).sample(s)
        return tf.squeeze(samples, 1) if squeeze else samples
