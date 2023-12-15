# Use the Ubuntu 22.04 base image
FROM --platform=linux/amd64 ubuntu:22.04

# Update package lists and install necessary packages
RUN apt-get update \
    && apt-get install -y \
        wget \
        vim \
        python3 \
        python3-pip \
        libjemalloc2 \
        libxi6 \
        libxxf86vm1 \
        libxrender1 \
        libfftw3-3 \
        libopenjp2-7 \
        libembree3-3 \
        libpotrace0 \
        libgl1 \
        libavcodec58 \
        libavdevice58 \
        libboost-all-dev \
        libpugixml1v5 \
        libhpdf-2.3.0 \
        libimath-3-1-29 \
        libpystring0 \
        libyaml-cpp0.7 \
        libwebpdemux2  \
        xvfb \
        # Add more packages as needed \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install python-dotenv

# Set the working directory
WORKDIR /tmp

# Copy your application files into the container
COPY . /tmp/geo2sigmap

WORKDIR /tmp/geo2sigmap
# RUN mv lib /opt

WORKDIR /tmp/geo2sigmap/build_linux_release/bin/3.3/python/bin
RUN wget "https://bootstrap.pypa.io/get-pip.py" \
    && ./python3.10 get-pip.py



RUN echo "/tmp/geo2sigmap/lib\n/opt/lib/alembic/lib\n/opt/lib/alembic/lib64\n/opt/lib/blosc/lib\n/opt/lib/blosc/lib64\n/opt/lib/ispc/lib\n/opt/lib/ispc/lib64\n/opt/lib/level-zero/lib\n/opt/lib/level-zero/lib64\n/opt/lib/ocio/lib\n/opt/lib/ocio/lib64\n/opt/lib/oidn/lib\n/opt/lib/oidn/lib64\n/opt/lib/oiio/lib\n/opt/lib/oiio/lib64\n/opt/lib/openexr/lib\n/opt/lib/openexr/lib64\n/opt/lib/openvdb/lib\n/opt/lib/openvdb/lib64\n/opt/lib/osd/lib\n/opt/lib/osd/lib64\n/opt/lib/osl/lib\n/opt/lib/osl/lib64\n/opt/lib/tbb/lib\n/opt/lib/tbb/lib64\n/opt/lib/usd/lib\n/opt/lib/usd/lib64\n/opt/lib/xr-openxr-sdk/lib\n/opt/lib/xr-openxr-sdk/lib64" >> /etc/ld.so.conf \
    && ldconfig


WORKDIR /tmp/geo2sigmap
# # Set any environment variables if needed
# ENV ENV_VARIABLE_NAME=value

# # Expose any necessary ports
# EXPOSE 8080

# # Define the command to run your application
# CMD ["executable_command", "arguments"]
