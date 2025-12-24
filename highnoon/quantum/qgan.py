# src/quantum/qgan.py
#
# Quantum-inspired GAN module for augmenting the memory hierarchy.

from __future__ import annotations

import tensorflow as tf

from highnoon.models.utils.control_vars import ControlVarMixin

from .layers import QuantumEnergyLayer


class QuantumMemoryAugmentor(ControlVarMixin, tf.keras.layers.Layer):
    """
    Generates synthetic memory embeddings using a quantum-informed generator.
    """

    def __init__(
        self,
        compressed_dim: int,
        latent_dim: int = 8,
        samples_per_batch: int = 2,
        energy_scale: float = 1.5,
        num_qubits: int | None = None,
        **kwargs,
    ):
        super().__init__(dtype="float32", **kwargs)
        if compressed_dim <= 0:
            raise ValueError("compressed_dim must be positive.")

        self.compressed_dim = compressed_dim
        self.latent_dim = latent_dim
        self.samples_per_batch = samples_per_batch
        self._requested_qubits = num_qubits

        self.generator_energy = QuantumEnergyLayer(
            energy_scale=energy_scale,
            num_qubits=self._resolve_num_qubits(),
            name=f"{self.name or 'qgan'}_energy",
        )
        self.latent_proj = tf.keras.layers.Dense(
            latent_dim,
            activation=tf.nn.gelu,
            dtype="float32",
            name=f"{self.name or 'qgan'}_latent_proj",
        )
        self.output_proj = tf.keras.layers.Dense(
            compressed_dim,
            activation=None,
            dtype="float32",
            name=f"{self.name or 'qgan'}_output_proj",
        )
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(shape=(compressed_dim,)),
                tf.keras.layers.Dense(latent_dim, activation=tf.nn.relu, dtype="float32"),
                tf.keras.layers.Dense(1, activation=None, dtype="float32"),
            ],
            name=f"{self.name or 'qgan'}_discriminator",
        )
        self._disc_loss_metric = None
        self._gen_loss_metric = None

    def _resolve_num_qubits(self) -> int:
        if self._requested_qubits is not None:
            return max(1, int(self._requested_qubits))
        return 2

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        if input_dim != self.compressed_dim:
            raise ValueError(
                f"QuantumMemoryAugmentor expects compressed_dim={self.compressed_dim}, "
                f"but received input with dimension {input_dim}."
            )
        if not self.latent_proj.built:
            self.latent_proj.build(tf.TensorShape([None, self.latent_dim]))
        if not self.output_proj.built:
            self.output_proj.build(tf.TensorShape([None, self.latent_dim]))
        if self._disc_loss_metric is None:
            metric_prefix = self.name or "qgan"
            self._disc_loss_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}_disc_loss")
            self._gen_loss_metric = tf.keras.metrics.Mean(name=f"{metric_prefix}_gen_loss")
        super().build(input_shape)

    @property
    def metrics(self):
        base_metrics = super().metrics
        trackers = []
        if self._disc_loss_metric is not None:
            trackers.append(self._disc_loss_metric)
        if self._gen_loss_metric is not None:
            trackers.append(self._gen_loss_metric)
        return base_metrics + trackers

    def _sample_latent(self, batch_size: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal([batch_size * self.samples_per_batch, self.latent_dim])
        latent = self.latent_proj(noise)
        latent = tf.nn.gelu(latent)
        projected = self.output_proj(latent)
        projected = tf.reshape(
            projected,
            [batch_size, self.samples_per_batch, self.compressed_dim],
        )
        return projected

    def call(self, memory_level: tf.Tensor, training: bool = False, return_losses: bool = False):
        """
        Generate synthetic memory embeddings.

        Args:
            memory_level: Input memory tensor [B, N, D]
            training: Whether in training mode
            return_losses: If True, return losses instead of calling add_loss.
                          Use this when calling from within a tf.while_loop.

        Returns:
            synthetic: Generated synthetic samples
            aux_metrics: Dictionary of auxiliary metrics (includes losses if return_losses=True)
        """
        batch_size = tf.shape(memory_level)[0]
        stats_mean = tf.reduce_mean(memory_level, axis=1)  # [B, D]
        stats_std = tf.math.reduce_std(memory_level, axis=1)  # [B, D]
        stats = tf.concat([stats_mean, stats_std], axis=-1)

        scaling = self.generator_energy(stats, training=training)  # [B, 1]
        latent_samples = self._sample_latent(batch_size)

        scaling_expanded = tf.expand_dims(scaling, axis=1)  # [B, 1, 1]
        synthetic = stats_mean[:, tf.newaxis, :] + scaling_expanded * latent_samples

        synthetic = tf.clip_by_value(synthetic, -5.0, 5.0)

        aux_metrics = {
            "synthetic_scale": tf.reduce_mean(scaling),
        }

        if training:
            real_samples = tf.reshape(memory_level, [-1, self.compressed_dim])
            fake_samples = tf.reshape(synthetic, [-1, self.compressed_dim])
            real_logits = self.discriminator(real_samples)
            fake_logits = self.discriminator(fake_samples)

            disc_loss = tf.reduce_mean(tf.nn.softplus(-real_logits)) + tf.reduce_mean(
                tf.nn.softplus(fake_logits)
            )
            gen_loss = tf.reduce_mean(tf.nn.softplus(-fake_logits))
            total_loss = disc_loss + gen_loss

            # Update metrics
            if self._disc_loss_metric is not None:
                self._disc_loss_metric.update_state(disc_loss)
            if self._gen_loss_metric is not None:
                self._gen_loss_metric.update_state(gen_loss)

            # Either add loss directly or return it for external handling
            if return_losses:
                # When inside a while loop, return losses to be accumulated externally
                aux_metrics["qgan_total_loss"] = total_loss
                aux_metrics["qgan_disc_loss"] = disc_loss
                aux_metrics["qgan_gen_loss"] = gen_loss
            else:
                # Normal case: add loss directly
                self.add_loss(total_loss)

        return synthetic, aux_metrics
