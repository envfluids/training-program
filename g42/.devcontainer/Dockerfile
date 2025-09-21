FROM mambaorg/micromamba:1.5.8

SHELL ["/bin/bash", "-lc"]

# Put the environment spec in the image
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Create env from the spec
RUN micromamba create -y -n workshop -f /tmp/environment.yml && \
    micromamba clean -a -y

# Ensure the env auto-activates for interactive bash (as mambauser)
RUN echo 'eval "$(micromamba shell hook --shell=bash)"' >> /home/${MAMBA_USER}/.bashrc && \
    echo 'micromamba activate workshop' >> /home/${MAMBA_USER}/.bashrc

# Also make PATH available in non-login shells
ENV CONDA_DEFAULT_ENV=workshop \
    PATH=/opt/conda/envs/workshop/bin:$PATH

WORKDIR /workspace
CMD ["bash"]
