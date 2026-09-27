FROM mambaorg/micromamba:2.0.5

ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba env create --yes --file /tmp/environment.yml \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

RUN micromamba run -n polymer-generator python -m pip install --no-deps --editable /app

ENV ENV_NAME=polymer-generator

CMD ["python", "-m", "polymer_lib.runner"]
