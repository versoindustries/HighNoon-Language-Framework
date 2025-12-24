#!/usr/bin/env python3
"""
src/ops/mps_contract.py

Matrix Product State (MPS) contraction operation for efficient quantum state representation.

Provides TensorFlow interface to native CPU kernel implementing tensor network contractions
with SVD truncation, canonical forms, and entanglement entropy computation.
"""

import tensorflow as tf

from highnoon._native import get_op

# Load native library
_highnoon_core = get_op("highnoon_core")
if _highnoon_core is None:
    # Fallback for individual builds or legacy structure
    _highnoon_core = get_op("mps_contract")

if _highnoon_core is None:
    raise ImportError("Could not load highnoon_core or mps_contract native op library.")

_mps_contract_module = _highnoon_core
_HAS_MPS_EXPECT = hasattr(_mps_contract_module, "mps_expect")
_HAS_MPS_CANONICALIZE = hasattr(_mps_contract_module, "mps_canonicalize")


@tf.custom_gradient
def _mps_contract_with_gradient(
    mps_tensors,
    physical_dims,
    bond_dims,
    max_bond_dim,
    compute_entropy_bool,
    truncation_threshold_float,
    uniform_bool,
    use_tdvp_bool,
):
    """Inner function with custom gradient that always receives all 6 arguments explicitly."""
    # Forward pass
    # CRITICAL: Pass compute_entropy and truncation_threshold as KEYWORD ARGUMENTS (op attributes),
    # not positional arguments. Positional arguments become input tensors; keyword arguments
    # become op attributes which must be Python literals (bool/float).
    contracted_state, entanglement_entropies, log_norm = _mps_contract_module.mps_contract(
        mps_tensors,
        physical_dims,
        bond_dims,
        max_bond_dim,
        compute_entropy=compute_entropy_bool,  # Op attribute (keyword)
        truncation_threshold=truncation_threshold_float,  # Op attribute (keyword)
        uniform=uniform_bool,  # Op attribute (keyword)
    )

    def grad_fn(grad_state, grad_entropies, grad_log_norm, variables=None):
        """
        Custom gradient for MPS contraction using DMRG adjoint algorithm.

        Args:
            grad_state: Gradient w.r.t. contracted state
            grad_entropies: Gradient w.r.t. entanglement entropies (unused)
            grad_log_norm: Gradient w.r.t. log-norm (unused)
            variables: TensorFlow variables (auto-provided when function uses variables)

        Returns:
            Tuple of (input_grads, variable_grads) if variables provided, else input_grads
        """
        # Compute gradients using DMRG adjoint algorithm in C++ kernel
        grad_mps_list = _mps_contract_module.mps_contract_grad(
            grad_state,
            mps_tensors,
            physical_dims,
            bond_dims,
            max_bond_dim,
            uniform=uniform_bool,
            use_tdvp=use_tdvp_bool,
        )

        # Convert to list to ensure consistent handling
        grad_list = list(grad_mps_list)

        # When variables is None, return just input gradients
        if variables is None:
            return grad_list + [None, None, None, None, None, None, None]

        # When variables is provided, return (input_grads, variable_grads)
        # Input grads: None for MPS tensors (since they're tracked as variables)
        #              None for other inputs (not differentiable)
        input_grads = [None] * len(grad_list) + [None, None, None, None, None, None, None]
        variable_grads = grad_list  # Gradients for the variables

        return input_grads, variable_grads

    return (contracted_state, entanglement_entropies, log_norm), grad_fn


def mps_contract(
    mps_tensors,
    physical_dims,
    bond_dims,
    max_bond_dim,
    compute_entropy=True,
    truncation_threshold=1e-10,
    uniform=False,
    use_tdvp=False,
):
    """
    Contract a Matrix Product State (MPS) tensor network into a full wavefunction.

    Args:
        mps_tensors: List[tf.Tensor] - MPS core tensors, each with shape
            [bond_left, physical_dim, bond_right]. Bond dimensions must match:
            bond_dims[i] = mps_tensors[i].shape[0], bond_dims[i+1] = mps_tensors[i].shape[2]
        physical_dims: tf.Tensor[int32] - Physical dimension at each site, shape [N]
        bond_dims: tf.Tensor[int32] - Bond dimensions (N+1 values, first and last = 1), shape [N+1]
        max_bond_dim: int - Maximum bond dimension for SVD truncation
        compute_entropy: bool - If True, compute entanglement entropy at each bond
        truncation_threshold: float - Truncation error threshold for SVD (default 1e-10)
        uniform: bool - If True, assume uniform MPS (shared parameters)
        use_tdvp: bool - If True, project gradients onto tangent space

    Returns:
        contracted_state: tf.Tensor - Full wavefunction vector, shape [product(physical_dims)]
        entanglement_entropies: tf.Tensor - Entanglement entropy at each bond, shape [N-1]
        log_norm: tf.Tensor - Scalar log-norm of the state

    Example:
        >>> # 4-site MPS with physical dim=2, bond dim=4
        >>> mps = [tf.random.normal([1, 2, 4], dtype=tf.float32),
        ...        tf.random.normal([4, 2, 4], dtype=tf.float32),
        ...        tf.random.normal([4, 2, 4], dtype=tf.float32),
        ...        tf.random.normal([4, 2, 1], dtype=tf.float32)]
        >>> phys_dims = tf.constant([2, 2, 2, 2], dtype=tf.int32)
        >>> bond_dims = tf.constant([1, 4, 4, 4, 1], dtype=tf.int32)
        >>> state, entropies = mps_contract(mps, phys_dims, bond_dims, max_bond_dim=8)
        >>> state.shape  # TensorShape([16])  # 2^4 = 16
        >>> entropies.shape  # TensorShape([3])  # 3 internal bonds
    """
    # Input validation
    if not isinstance(mps_tensors, (list, tuple)):
        raise TypeError("mps_tensors must be a list or tuple of tensors")

    N = len(mps_tensors)
    if N < 1:
        raise ValueError("mps_tensors must contain at least 1 tensor")

    # Convert inputs to tensors
    physical_dims = tf.convert_to_tensor(physical_dims, dtype=tf.int32)
    bond_dims = tf.convert_to_tensor(bond_dims, dtype=tf.int32)
    max_bond_dim = tf.convert_to_tensor(max_bond_dim, dtype=tf.int32)

    # Shape validation
    tf.debugging.assert_equal(
        tf.size(physical_dims), N, message=f"physical_dims must have {N} elements"
    )
    tf.debugging.assert_equal(
        tf.size(bond_dims), N + 1, message=f"bond_dims must have {N+1} elements"
    )

    # Ensure all MPS tensors are float32
    mps_tensors = [tf.cast(t, tf.float32) for t in mps_tensors]

    # Convert compute_entropy to Python bool for op attribute
    # Must extract static value for op attribute - cannot be a tensor
    # CRITICAL: Use tf.autograph.experimental.do_not_convert to prevent tracing
    @tf.autograph.experimental.do_not_convert
    def _extract_bool(val):
        """Extract Python bool from value, preventing AutoGraph tracing."""
        if isinstance(val, bool):
            return val
        if isinstance(val, tf.Tensor):
            # Try to extract constant value
            try:
                static_val = tf.get_static_value(val)
                if static_val is not None:
                    return bool(static_val)
            except (TypeError, ValueError, tf.errors.InvalidArgumentError):
                pass
            # If tensor is a constant, evaluate it in numpy
            try:
                return bool(val.numpy())
            except (TypeError, ValueError, RuntimeError):
                pass
        try:
            return bool(val)
        except (TypeError, ValueError):
            return True  # Safe default

    @tf.autograph.experimental.do_not_convert
    def _extract_float(val):
        """Extract Python float from value, preventing AutoGraph tracing."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, tf.Tensor):
            # Try to extract constant value
            try:
                static_val = tf.get_static_value(val)
                if static_val is not None:
                    return float(static_val)
            except (TypeError, ValueError, tf.errors.InvalidArgumentError):
                pass
            # If tensor is a constant, evaluate it in numpy
            try:
                return float(val.numpy())
            except (TypeError, ValueError, RuntimeError):
                pass
        try:
            return float(val)
        except (TypeError, ValueError):
            return 1e-10  # Safe default

    compute_entropy_bool = _extract_bool(compute_entropy)
    truncation_threshold_float = _extract_float(truncation_threshold)
    uniform_bool = _extract_bool(uniform)
    use_tdvp_bool = _extract_bool(use_tdvp)

    # Call inner function with all arguments explicitly
    return _mps_contract_with_gradient(
        mps_tensors,
        physical_dims,
        bond_dims,
        max_bond_dim,
        compute_entropy_bool,
        truncation_threshold_float,
        uniform_bool,
        use_tdvp_bool,
    )


@tf.custom_gradient
def _mps_contract_no_entropy(mps_tensors, physical_dims, bond_dims, max_bond_dim):
    """
    Internal: Contract MPS without computing entropy (for expectation value calculations).

    This version hardcodes compute_entropy=False and truncation_threshold=1e-10 to avoid
    TensorFlow graph mode issues with Python literals being converted to tensor constants.

    Implements custom gradient inline to avoid passing constants as function arguments.

    Args:
        mps_tensors: List[tf.Tensor] - MPS core tensors
        physical_dims: tf.Tensor[int32] - Physical dimensions
        bond_dims: tf.Tensor[int32] - Bond dimensions
        max_bond_dim: int - Maximum bond dimension

    Returns:
        Tuple of (contracted_state, entanglement_entropies)
    """
    # Forward pass - call C++ op with hardcoded attributes (not arguments!)
    # CRITICAL: compute_entropy and truncation_threshold are passed as KEYWORD ARGUMENTS,
    # which makes them op attributes (compile-time constants) rather than input tensors
    contracted_state, entanglement_entropies = _mps_contract_module.mps_contract(
        mps_tensors,
        physical_dims,
        bond_dims,
        max_bond_dim,
        compute_entropy=False,  # Hardcoded op attribute
        truncation_threshold=1e-10,  # Hardcoded op attribute
    )

    def grad_fn(grad_state, grad_entropies, variables=None):
        """Custom gradient using DMRG adjoint algorithm."""
        # Compute gradients using DMRG adjoint algorithm in C++ kernel
        grad_mps_list = _mps_contract_module.mps_contract_grad(
            grad_state, mps_tensors, physical_dims, bond_dims, max_bond_dim
        )

        # Convert to list
        grad_list = list(grad_mps_list)

        # When variables is None, return just input gradients
        if variables is None:
            return grad_list + [None, None, None]

        # When variables is provided, return (input_grads, variable_grads)
        input_grads = [None] * len(grad_list) + [None, None, None]
        variable_grads = grad_list

        return input_grads, variable_grads

    return (contracted_state, entanglement_entropies), grad_fn


def mps_expect(mps_tensors, operator, physical_dims, bond_dims, max_bond_dim):
    """
    Compute normalized expectation value <MPS|operator|MPS> / <MPS|MPS> for a local or global operator.

    CRITICAL: Returns normalized expectation value to prevent energy divergence during optimization.
    Without normalization, unnormalized MPS states lead to exponential divergence.

    The normalization ensures that:
    E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩

    TensorFlow's automatic differentiation correctly handles the quotient rule for gradients.

    Args:
        mps_tensors: List[tf.Tensor] - MPS core tensors
        operator: tf.Tensor - Operator matrix, shape [phys_dim_total, phys_dim_total]
            or list of local operators
        physical_dims: tf.Tensor[int32] - Physical dimensions
        bond_dims: tf.Tensor[int32] - Bond dimensions
        max_bond_dim: int - Maximum bond dimension

    Returns:
        expectation: tf.Tensor - Scalar normalized expectation value
    """
    if _HAS_MPS_EXPECT:
        return _mps_expect_with_gradient(
            mps_tensors, operator, physical_dims, bond_dims, max_bond_dim
        )

    return _mps_expect_python(mps_tensors, operator, physical_dims, bond_dims, max_bond_dim)


def _mps_expect_python(mps_tensors, operator, physical_dims, bond_dims, max_bond_dim):
    """Python fallback for expectation value computation."""
    # Contract MPS to full state using specialized function that avoids graph mode issues
    state, _ = _mps_contract_no_entropy(mps_tensors, physical_dims, bond_dims, max_bond_dim)

    # Compute norm: <psi|psi>
    state_conj = tf.math.conj(state)
    norm_squared = tf.reduce_sum(state_conj * state)
    norm_squared = tf.cast(tf.math.real(norm_squared), tf.float32)

    # Apply operator: <psi|O|psi> = psi^† O psi
    op_state = tf.linalg.matvec(operator, state)
    expectation_unnorm = tf.reduce_sum(state_conj * op_state)
    expectation_unnorm = tf.cast(tf.math.real(expectation_unnorm), tf.float32)

    # Normalize: <psi|O|psi> / <psi|psi>
    # Add small epsilon for numerical stability (prevents division by zero)
    # TensorFlow autodiff will correctly compute the quotient rule gradient
    expectation_normalized = expectation_unnorm / (norm_squared + 1e-10)

    return expectation_normalized


@tf.custom_gradient
def _mps_expect_with_gradient(
    mps_tensors,
    operator,
    physical_dims,
    bond_dims,
    max_bond_dim,
):
    """Native expectation op with custom gradient."""
    expectation = _mps_contract_module.mps_expect(
        mps_tensors,
        operator,
        physical_dims,
        bond_dims,
        max_bond_dim,
    )

    def grad_fn(grad_out, variables=None):
        # Recompute state to build gradients via the native contract gradient.
        state, _ = _mps_contract_no_entropy(mps_tensors, physical_dims, bond_dims, max_bond_dim)
        state = tf.cast(state, tf.float32)
        op_state = tf.linalg.matvec(operator, state)
        op_state_t = tf.linalg.matvec(tf.transpose(operator), state)

        norm_squared = tf.reduce_sum(state * state)
        denom = norm_squared + 1e-10
        expectation_unnorm = tf.reduce_sum(state * op_state)

        grad_state = ((op_state + op_state_t) * denom - 2.0 * expectation_unnorm * state) / (
            denom * denom
        )
        grad_state = tf.cast(grad_state, tf.float32) * tf.cast(grad_out, tf.float32)

        grad_mps_list = _mps_contract_module.mps_contract_grad(
            grad_state,
            mps_tensors,
            physical_dims,
            bond_dims,
            max_bond_dim,
        )
        grad_list = list(grad_mps_list)

        grad_operator = tf.einsum("i,j->ij", state, state) * (
            tf.cast(grad_out, tf.float32) / tf.cast(denom, tf.float32)
        )

        if variables is None:
            return grad_list + [grad_operator, None, None, None]

        input_grads = [None] * len(grad_list) + [grad_operator, None, None, None]
        variable_grads = grad_list
        return input_grads, variable_grads

    return expectation, grad_fn


def canonical_mps(mps_tensors, physical_dims, bond_dims, center_site=None):
    """
    Convert MPS to canonical form (left-orthogonal or right-orthogonal).

    Canonical forms are useful for:
    - Numerical stability in long contractions
    - Efficient computation of reduced density matrices
    - Extracting dominant Schmidt coefficients

    Args:
        mps_tensors: List[tf.Tensor] - MPS core tensors (will be modified in-place)
        physical_dims: tf.Tensor[int32] - Physical dimensions
        bond_dims: tf.Tensor[int32] - Bond dimensions
        center_site: Optional[int] - Site index for mixed canonical form (default: N//2)

    Returns:
        canonical_tensors: List[tf.Tensor] - MPS in canonical form
        center_matrix: tf.Tensor - Central orthogonality center (Schmidt coefficients)
    """
    N = len(mps_tensors)
    if center_site is None:
        center_site = N // 2

    if _HAS_MPS_CANONICALIZE:
        canonical_tensors = _mps_contract_module.mps_canonicalize(
            mps_tensors,
            physical_dims,
            bond_dims,
            tf.convert_to_tensor(center_site, dtype=tf.int32),
        )
        return canonical_tensors, canonical_tensors[center_site]

    canonical_tensors = [None] * N

    # Left-orthogonalization sweep (sites 0 to center-1)
    for i in range(center_site):
        tensor = mps_tensors[i]
        bond_left, phys, bond_right = tensor.shape

        # Reshape to matrix: [bond_left * phys, bond_right]
        mat = tf.reshape(tensor, [bond_left * phys, bond_right])

        # QR decomposition
        q, r = tf.linalg.qr(mat)

        # Store left-orthogonal tensor
        canonical_tensors[i] = tf.reshape(q, [bond_left, phys, -1])

        # Absorb R into next tensor
        if i < N - 1:
            mps_tensors[i + 1] = tf.einsum("ij,jkl->ikl", r, mps_tensors[i + 1])

    # Right-orthogonalization sweep (sites N-1 to center+1)
    for i in range(N - 1, center_site, -1):
        tensor = mps_tensors[i]
        bond_left, phys, bond_right = tensor.shape

        # Reshape to matrix: [bond_left, phys * bond_right]
        mat = tf.reshape(tensor, [bond_left, phys * bond_right])
        mat_t = tf.transpose(mat)

        # QR on transposed matrix
        q, r = tf.linalg.qr(mat_t)

        # Store right-orthogonal tensor
        canonical_tensors[i] = tf.reshape(tf.transpose(q), [bond_left, phys, bond_right])

        # Absorb R into previous tensor
        if i > 0:
            mps_tensors[i - 1] = tf.einsum("ijk,kl->ijl", mps_tensors[i - 1], tf.transpose(r))

    # Center site
    canonical_tensors[center_site] = mps_tensors[center_site]

    return canonical_tensors, canonical_tensors[center_site]


def mps_expect_pauli(mps_tensors, pauli_indices, coefficients):
    """
    Compute expectation value of a sum of Pauli strings efficiently.

    Args:
        mps_tensors: List of MPS core tensors
        pauli_indices: Matrix of Pauli indices [num_strings, num_sites]
            0: I, 1: X, 2: Y (unsupported), 3: Z
        coefficients: Coefficients for each Pauli string [num_strings]

    Returns:
        expectation: Scalar float32 tensor
    """
    len(mps_tensors)
    return _mps_contract_module.mps_expect_pauli(mps_tensors, pauli_indices, coefficients)


def mps_feature_importance(entanglement_entropies):
    """
    Compute site-wise feature importance from entanglement entropies.

    Args:
        entanglement_entropies: Tensor of entropies at each bond [N-1]

    Returns:
        importance: Tensor of importance at each site [N]
    """
    return _mps_contract_module.mps_feature_importance(
        entanglement_entropies=entanglement_entropies
    )


def mps_trotter_step(
    mps_tensors, gate_sites, gates_real, gates_imag, max_bond_dim, truncation_threshold=1e-10
):
    """
    Apply a set of two-site gates to an MPS (Trotter step).

    Args:
        mps_tensors: List of MPS core tensors
        gate_sites: Indices of sites where gates are applied [M]
        gates_real: List of real parts of gate matrices [M]
        gates_imag: List of imaginary parts of gate matrices [M] (unused in real MPS)
        max_bond_dim: Maximum bond dimension for truncation
        truncation_threshold: Truncation error threshold

    Returns:
        out_mps_tensors: Updated list of MPS core tensors
    """
    len(mps_tensors)
    len(gates_real)
    return _mps_contract_module.mps_trotter_step(
        mps_tensors, gate_sites, gates_real, gates_imag, max_bond_dim, truncation_threshold
    )


# Export public API
__all__ = [
    "mps_contract",
    "mps_expect",
    "mps_expect_pauli",
    "mps_feature_importance",
    "mps_trotter_step",
    "canonical_mps",
]
