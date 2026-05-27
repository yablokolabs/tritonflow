"""TritonFlow: High-performance GPU kernels for modern AI workloads.

Built with OpenAI Triton for vector search, analytics, scientific computing,
and AI/ML acceleration.
"""

__version__ = "0.1.0"
__author__ = "Yabloko Labs Ltd"

from tritonflow.utils.gpu import get_device_info as get_device_info
from tritonflow.utils.gpu import is_gpu_available as is_gpu_available
