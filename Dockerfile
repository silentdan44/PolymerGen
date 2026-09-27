FROM mambaorg/micromamba:2.0.5

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-openff.yml /tmp/environment-openff.yml

RUN micromamba env create --yes --file /tmp/environment-openff.yml \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

RUN micromamba run -n polymer-generator-openff python -m pip install --no-deps --editable /app

ENV ENV_NAME=polymer-generator-openff

CMD ["python", "-m", "polymer_lib.runner"]
