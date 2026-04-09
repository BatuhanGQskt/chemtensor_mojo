# =============================================================================
# Tensor Abstraction Layer
# =============================================================================
# Import order matters to avoid circular imports.

# 1. First import traits (no dependencies on other m_tensor modules)
from .tensor_traits import TensorOps, TensorBackend

# 2. Then import numeric utilities (no dependencies)
from .numeric import *

# 3. Import dense tensor (depends on tensor_traits)
from .dense_tensor import *

# 4. Import complex tensor (depends on dense_tensor)
from .complex_tensor import *

# 5. Import block sparse tensor (depends on tensor_traits, dense_tensor for conversion)
from .block_sparse_tensor import (
    BlockSparseTensor,
    QNumber,
    BlockIndex,
    Block,
    allocate_block_sparse_for_tensor_dot,
    block_sparse_to_dense,
    dense_to_block_sparse,
)

# 6. Import generic operations (function overloading handles dispatch)
from .tensor_ops import (
    # Compile-time backend selector
    TensorBackendType,
    Tensor,  # Type alias based on TensorBackendType
    
    # =========================================================================
    # Generic Operations (Overloaded for DenseTensor & BlockSparseTensor)
    # =========================================================================
    # The compiler automatically selects the correct implementation based on
    # argument types at compile time. Use the same function name for both backends.
    
    # Factory functions
    create_tensor,
    create_tensor_uninitialized,
    create_tensor_from_data,
    
    # Tensor contraction
    tensor_dot,
    
    # Decompositions
    tensor_qr,
    tensor_svd_trunc,
    
    # Shape manipulation
    tensor_transpose,
    tensor_reshape,
    tensor_flatten_dims,
    tensor_copy_to_contiguous,
    
    # In-place operations
    tensor_scale_in_place,
    tensor_axpy_in_place,
    
    # Numerical operations
    tensor_dot_product,
    tensor_norm,
    
    # =========================================================================
    # Runtime configuration
    # =========================================================================
    TensorConfig,
    get_tensor_config,
    set_tensor_config,
    
    # Backend query functions
    is_using_dense_backend,
    is_using_sparse_backend,
)