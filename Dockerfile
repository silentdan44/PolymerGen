FROM mambaorg/micromamba:2.0.5

WORKDIR /app

# The OpenFF environment file installs the project in editable mode, so the
# source tree must be present before creating the environment.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

RUN micromamba env create --yes --file environment-openff.yml \
    && micromamba clean --all --yes

ENTRYPOINT ["micromamba", "run", "--no-capture-output", "-n", "polymer-generator-openff"]
CMD ["python", "-m", "pytest", "-q"]
