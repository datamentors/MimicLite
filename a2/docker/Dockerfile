# MimicLite training image for AgiBot A2 (mjlab backend). Build context = a2-mimiclite/.
#   docker build -f docker/Dockerfile -t a2-mimiclite:latest .
FROM ubuntu:24.04

ARG AA_COMMIT=b6a6b3f101344dfe8d5bdd6a06e921f4333127eb
ARG ML_COMMIT=3253810ce21dbcc05d8f4942a017ea3eaffc6cf1

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/tmp/uv-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl unzip build-essential libgl1 libegl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (motion conversion streams the read-only S3 CSVs)
RUN curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscli.zip

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/

WORKDIR /opt/aa
RUN git clone https://github.com/Agent-3154/active-adaptation.git . \
    && git checkout ${AA_COMMIT} \
    && git clone https://github.com/EGalahad/mimic-lite projects/mimic-lite \
    && git -C projects/mimic-lite checkout ${ML_COMMIT}

COPY docker/pyproject.toml venv/a2/pyproject.toml
COPY docker/uv.lock* venv/a2/
RUN if [ -f venv/a2/uv.lock ]; then uv sync --project venv/a2 --frozen; \
    else uv sync --project venv/a2 --python 3.12; fi \
    && rm -rf /tmp/uv-cache

# A2 profile overlay + asset registration
COPY overlay/ projects/mimic-lite/
RUN sed -i 's/^from \. import a3, g1, h2, rp1$/from . import a2, a3, g1, h2, rp1/' \
        projects/mimic-lite/mimic_lite/assets/__init__.py \
    && grep -q "import a2, a3" projects/mimic-lite/mimic_lite/assets/__init__.py

COPY model/a2_serial /opt/a2/a2_serial
# self-collision variant (build_a2_serial.py --self-collision); used when A2_MODEL=a2_serial_selfcol
COPY model/a2_serial_selfcol /opt/a2/a2_serial_selfcol
COPY model/build_a2_serial.py model/stand_test.py /opt/a2/model/
COPY tools /opt/a2/tools
COPY scripts /opt/a2/scripts

# Training model baked in as A2_MJCF default:
#   docker build ... -t a2-mimiclite:latest                                         (a2_serial)
#   docker build ... --build-arg A2_MODEL=a2_serial_selfcol -t a2-mimiclite:selfcol (self-collision)
ARG A2_MODEL=a2_serial
ENV PATH=/opt/aa/venv/a2/.venv/bin:${PATH} \
    A2_MJCF=/opt/a2/${A2_MODEL}/a2_serial.xml \
    A2_MOTION_DATASET=/data/motions/bonesseed_a2 \
    HOME=/tmp/home \
    ANY4HDMI_QPOS_CACHE_ROOT=/cache/any4hdmi \
    WARP_CACHE_PATH=/cache/warp \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    MPLCONFIGDIR=/tmp/home/mpl

RUN aa-discover-projects && aa-project enable mimic_lite && aa-list-tasks \
    && chmod -R a+rX /opt/aa /opt/a2 /opt/uv-python \
    && mkdir -p /tmp/home && chmod 1777 /tmp/home

WORKDIR /opt/aa
