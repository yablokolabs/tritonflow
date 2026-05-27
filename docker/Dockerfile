FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# -------------------------------------------------------------------
FROM base AS dev

WORKDIR /workspace

COPY pyproject.toml .
COPY tritonflow/ tritonflow/
COPY benchmarks/ benchmarks/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[all]"

CMD ["bash"]

# -------------------------------------------------------------------
FROM base AS runtime

WORKDIR /app

COPY pyproject.toml .
COPY tritonflow/ tritonflow/

RUN pip install --no-cache-dir -e ".[bench]"

ENTRYPOINT ["tritonflow"]
CMD ["info"]
