# src/models/utils/control_vars.py
import tensorflow as tf


class ControlVarMixin:
    """
    A mixin class to provide a standardized way for Keras layers to register
    their tunable control variables (e.g., 'evolution_time', 'epsilon_param').

    This enables robust, annotation-based discovery of these variables, rather
    than relying on fragile name-matching schemes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a dictionary to store lists of variables, keyed by a tag.
        self._control_vars = {}

    def register_control_var(self, tag: str, var: tf.Variable):
        self._control_vars.setdefault(tag, []).append(var)

    @property
    def control_vars(self):
        return self._control_vars
