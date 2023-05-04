from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import cloudpickle

tfd = tfp.distributions
tfb = tfp.bijectors

# TODO: make bias initializer option
class ConditionalGaussian(tfd.Distribution):
    def __init__(
        self,
        n_parameters,
        n_data,
        covariance=None,
        diagonal_covariance=False,
        n_hidden=[50, 50],
        exponentiate=False,
        mean=None,
        std=None,
        activation=tf.keras.layers.LeakyReLU(0.01),
        optimizer=tf.optimizers.Adam(1e-3),
        dtype=tf.float32,
        reparameterization_type=None,
        validate_args=False,
        allow_nan_stats=True,
        kernel_initializer=partial(
            tf.keras.initializers.RandomNormal, mean=0.0, stddev=1e-5, seed=None
        ),
        final_bias_initializer="zeros",
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
                        # kernel_initializer=tf.keras.initializers.zeros(),
                        # bias_initializer=tf.keras.initializers.ones(),
                    )
                )
                model.add(
                    tf.keras.layers.Lambda(
                        # lambda x: tf.concat(
                        #     [
                        #         x[..., : self.n_data],
                        #         tf.keras.activations.relu(
                        #             x[..., self.n_data :],
                        #             max_value=1e1,
                        #             threshold=1e-6,
                        #             alpha=1e-3,
                        #         ),
                        #     ],
                        #     -1,
                        # )
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
                        # kernel_initializer=tf.keras.initializers.zeros(),
                        # bias_initializer=tf.keras.initializers.ones(),
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
                    # activation=last_activation,
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
        self.model.save_weights(filename + ".h5")

    def load(self, filename="model"):
        self.model.load_weights(filename + ".h5")
        self.compile(pretrain_phase=False)

    def log_prob(self, x, conditional, x_in_final_space=False):
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
    def __init__(
        self,
        n_parameters,
        n_data,
        n_components,
        n_hidden=[50, 50],
        activation=tf.keras.layers.LeakyReLU(0.01),
        optimizer=tf.optimizers.Adam(1e-3),
        dtype=tf.float32,
        reparameterization_type=None,
        validate_args=False,
        allow_nan_stats=True,
        kernel_initializer=partial(
            tf.keras.initializers.RandomNormal, mean=0.0, stddev=1e-5, seed=None
        ),
        final_bias_initializer="zeros",
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
        self.model.save_weights(filename + ".h5")

    def load(self, filename="model"):
        self.model.load_weights(filename + ".h5")
        self.compile()

    def log_prob(self, x, conditional):
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
        conditional = tf.cast(conditional, self._dtype)
        squeeze = True if tf.rank(conditional) == 1 else False
        conditional = (
            tf.expand_dims(conditional, 0) if tf.rank(conditional) == 1 else conditional
        )

        samples = self.model(conditional).sample(s)
        return tf.squeeze(samples, 1) if squeeze else samples
