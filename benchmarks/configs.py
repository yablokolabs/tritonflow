"""Benchmark configuration constants."""

__all__ = [
    "TENSOR_SIZES",
    "MATRIX_SIZES",
    "BATCH_SIZES",
    "SEQUENCE_LENGTHS",
    "DEFAULT_DTYPES",
    "DEFAULT_WARMUP",
    "DEFAULT_ITERATIONS",
]

TENSOR_SIZES = {
    "small": (1024,),
    "medium": (65536,),
    "large": (1_048_576,),
    "xlarge": (16_777_216,),
}

MATRIX_SIZES = {
    "small": (256, 256),
    "medium": (1024, 1024),
    "large": (4096, 4096),
}

BATCH_SIZES = [1, 8, 32, 128]

SEQUENCE_LENGTHS = [128, 512, 1024, 2048]

DEFAULT_DTYPES = ["float32", "float16"]

DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100
