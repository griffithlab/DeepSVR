FROM continuumio/miniconda3

RUN \
    # Create Conda virtual environment
    conda create -n env python=3.6 && \
    echo "source activate env" > ~/.bashrc && \
    \
    # Install BioConda Channels
    conda config --add channels defaults && \
    conda config --add channels conda-forge && \
    conda config --add channels bioconda && \
    \
    # Install DeepSVR
    conda install -y deepsvr

ENV PATH /opt/conda/envs/env/bin:$PATH

# create a persistant volume for output data
VOLUME ["/data"]

# create default ad-hoc command as entrypoint
ENTRYPOINT ["deepsvr"]
CMD ["--help"]
