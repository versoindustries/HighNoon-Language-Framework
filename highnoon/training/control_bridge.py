# highnoon/training/control_bridge.py
# Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge for applying evolution_time control updates from Meta-Controller.

This module provides the interface between the Hamiltonian Meta-Controller
(C++ MIMO PID) and the model's evolution_time variables. It:
1. Discovers all TimeCrystal blocks with evolution_time parameters
2. Maps block names to their tf.Variables
3. Applies evolution_time updates from the controller

This is critical for numerical stability - the Meta-Controller monitors
energy_drift metrics and adaptively reduces evolution_time when the
Hamiltonian dynamics are becoming unstable.
"""

import logging
from typing import Optional

import tensorflow as tf


logger = logging.getLogger(__name__)


class EvolutionTimeControlBridge:
    """Bridge for applying Meta-Controller evolution_time updates to model.
    
    The HamiltonianMetaController (C++) outputs:
    - block_names: tf.Tensor[string] with TimeCrystal block names
    - evolution_times: tf.Tensor[float] with new evolution_time values
    
    This bridge applies those updates to the corresponding tf.Variables.
    
    Note:
        This class uses lazy discovery of evolution_time variables. The
        _discover_evolution_time_vars() method is NOT called in __init__
        because TimeCrystal blocks create their evolution_time_bias weights
        in build(), which is only called on first forward pass. Call
        ensure_discovery() after the model has been built (i.e., after
        the first forward pass in training).
    """
    
    def __init__(self, model: tf.keras.Model):
        """Initialize bridge with model reference.
        
        Args:
            model: The HSMN model containing TimeCrystal blocks.
        
        Note:
            Does NOT discover evolution_time variables immediately. Call
            ensure_discovery() after the model has been built.
        """
        self.model = model
        self._evolution_time_vars: dict[str, tf.Variable] = {}
        self._discovery_attempted = False
        # DON'T call _discover_evolution_time_vars here - model not built yet
    
    def ensure_discovery(self) -> None:
        """Lazy discovery - call after first forward pass.
        
        This method should be called after the model has been built (i.e.,
        after the first forward pass) to discover all evolution_time variables.
        It is safe to call multiple times; subsequent calls are no-ops.
        """
        if not self._discovery_attempted:
            self._discover_evolution_time_vars()
            self._discovery_attempted = True
        
    def _discover_evolution_time_vars(self) -> None:
        """Discover all evolution_time variables in the model.
        
        Uses multiple discovery strategies in priority order:
        1. ControlVarMixin.control_vars property (preferred - used by TimeCrystal)
        2. Direct attribute access (evolution_time_bias)
        3. Cell attribute access (for RNN-style wrappers)
        4. Non-trainable variables search (fallback)
        """
        self._evolution_time_vars.clear()
        
        def _search_layer(layer: tf.keras.layers.Layer, prefix: str = "") -> None:
            """Recursively search for evolution_time variables."""
            layer_name = prefix + layer.name if prefix else layer.name
            
            # STRATEGY 1: Check ControlVarMixin.control_vars (PREFERRED)
            # TimeCrystalBlock registers evolution_time via this API
            if hasattr(layer, 'control_vars') and isinstance(layer.control_vars, dict):
                if 'evolution_time' in layer.control_vars:
                    vars_list = layer.control_vars['evolution_time']
                    if vars_list and len(vars_list) > 0:
                        var = vars_list[0]  # Take first evolution_time var
                        if isinstance(var, tf.Variable):
                            self._evolution_time_vars[layer_name] = var
                            logger.info(f"Found via control_vars: {layer_name} -> {var.name}")
            
            # STRATEGY 2: Direct attribute access (evolution_time_bias)
            if layer_name not in self._evolution_time_vars:
                if hasattr(layer, 'evolution_time_bias'):
                    var = layer.evolution_time_bias
                    if isinstance(var, tf.Variable):
                        self._evolution_time_vars[layer_name] = var
                        logger.info(f"Found via attribute: {layer_name} -> {var.name}")
                    
            # STRATEGY 3: Cell attribute access (for RNN-style wrappers)
            if layer_name not in self._evolution_time_vars:
                if hasattr(layer, 'cell'):
                    cell = layer.cell
                    # Check cell's control_vars first
                    if hasattr(cell, 'control_vars') and isinstance(cell.control_vars, dict):
                        if 'evolution_time' in cell.control_vars:
                            vars_list = cell.control_vars['evolution_time']
                            if vars_list and len(vars_list) > 0:
                                var = vars_list[0]
                                if isinstance(var, tf.Variable):
                                    key = f"{layer_name}/{layer.name}_cell"
                                    self._evolution_time_vars[key] = var
                                    logger.info(f"Found via cell control_vars: {key} -> {var.name}")
                    # Fallback: check cell's evolution_time_bias directly
                    elif hasattr(cell, 'evolution_time_bias'):
                        var = cell.evolution_time_bias
                        if isinstance(var, tf.Variable):
                            key = f"{layer_name}/{layer.name}_cell"
                            self._evolution_time_vars[key] = var
                            logger.info(f"Found via cell attribute: {key} -> {var.name}")
            
            # STRATEGY 4: Search all non-trainable variables (fallback)
            if layer_name not in self._evolution_time_vars:
                for var in layer.non_trainable_variables:
                    if "evolution_time" in var.name and "evolution_time_" not in var.name:
                        self._evolution_time_vars[layer_name] = var
                        logger.debug(f"Found via non_trainable: {layer_name} -> {var.name}")
                        break  # Take first match
            
            # Recurse into sub-layers
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    _search_layer(sub_layer, f"{layer_name}/")
            if hasattr(layer, '_layers'):
                for sub_layer in layer._layers:
                    _search_layer(sub_layer, f"{layer_name}/")
                    
        # Search all model layers
        for layer in self.model.layers:
            _search_layer(layer)
            
        logger.info(f"EvolutionTimeControlBridge: Found {len(self._evolution_time_vars)} evolution_time variables")
        for name in self._evolution_time_vars:
            logger.debug(f"  - {name}")
            
    def get_evolution_time_var_names(self) -> list[str]:
        """Get names of all discovered evolution_time variables.
        
        This method triggers lazy discovery if not already attempted,
        ensuring variables are found even if ensure_discovery() was not
        explicitly called. However, for best results, call ensure_discovery()
        after the model's first forward pass.
        
        Returns:
            List of block names that have evolution_time variables.
        """
        self.ensure_discovery()  # Lazy init if not done yet
        return list(self._evolution_time_vars.keys())
    
    def apply_evolution_times(
        self,
        block_names: tf.Tensor,
        evolution_times: tf.Tensor,
    ) -> int:
        """Apply evolution time updates from Meta-Controller.
        
        Args:
            block_names: Tensor of block name strings.
            evolution_times: Tensor of new evolution_time values.
            
        Returns:
            Number of successful updates applied.
        """
        if tf.size(block_names) == 0:
            return 0
            
        # Convert tensors to Python lists
        try:
            names = [n.numpy().decode('utf-8') for n in block_names]
            times = [float(t.numpy()) for t in evolution_times]
        except Exception as e:
            logger.warning(f"Failed to decode Meta-Controller output: {e}")
            return 0
            
        applied = 0
        for name, new_time in zip(names, times):
            # Find matching variable (partial match for nested blocks)
            # C++ may return names with /evolution_time suffix that we added
            matched_key = None
            name_without_suffix = name.removesuffix('/evolution_time')
            for key in self._evolution_time_vars:
                # Match either with or without the suffix
                if name in key or key in name or name_without_suffix in key or key in name_without_suffix:
                    matched_key = key
                    break
                    
            if matched_key is None:
                logger.debug(f"No evolution_time var found for block: {name}")
                continue
                
            var = self._evolution_time_vars[matched_key]
            
            # Validate new time is reasonable
            # Also reject non-finite values (NaN, Inf)
            import math
            if new_time <= 0 or not math.isfinite(new_time):
                logger.warning(f"Invalid evolution_time {new_time} for {name}, skipping")
                continue
                
            old_time = float(var.numpy())
            var.assign(new_time)
            applied += 1
            logger.debug(f"Updated {name}: {old_time:.6f} -> {new_time:.6f}")
            
        if applied > 0:
            logger.info(f"Meta-Controller applied {applied} evolution_time updates")
            
        return applied


__all__ = ["EvolutionTimeControlBridge"]
