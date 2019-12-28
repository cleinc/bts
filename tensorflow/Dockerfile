FROM tensorflow/tensorflow:1.13.2-gpu-jupyter

# libcuda.so.1 is not available by default so we add what are probably stubs.
# See https://github.com/tensorflow/tensorflow/issues/25865
# If we leave the stubs linked later, then we get a weird error about CUDA
# versions not matching, so we have to remove it later.
ENV LD_LIBRARY_PATH_OLD="${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/compat"

# Load everything we need to build the custom layer and stuff required by opencv.
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  g++ \
  libsm6 \
  libxext6 \
  libxrender-dev \
  && rm -rf /var/lib/apt/lists/*

# Setup our build paths
RUN mkdir -p /build
COPY custom_layer /build/custom_layer 
RUN mkdir -p /build/custom_layer/build

# Compile the new layer
WORKDIR /build/custom_layer/build
RUN cmake -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
RUN make -j

# Install the python requirements.
COPY requirements.txt /
RUN  pip install -r /requirements.txt

# Copy in the full repo.
COPY . /bts
WORKDIR /bts

# Put the new layer we built into /bts/custom_layer
RUN cp -r /build/custom_layer/build custom_layer/.

# Download the model locally.
RUN mkdir -p models \ 
  && python utils/download_from_gdrive.py 1ipme-fkV4pIx87sOs31R9CD_Qg-85__h models/bts_nyu.zip \
  && cd models \
  && unzip bts_nyu.zip

# Set the path back to avoid error (see above).
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH_OLD}"

# Add relevant paths to the PYTHONPATH so they can be imported from anywhere.
ENV PYTHONPATH=/bts:/bts/models/bts_nyu
