# Computational environments

- `figure-environment.yml`: portable public environment for `make verify figures`.
- `production-environment.yml`: captured Linux production environment for upstream analysis.
- `production-session.txt`: operating-system, Slurm, Python, and package-version record from the
  production session.

Use the figure environment for the public build. The production environment documents the original
upstream run; AlphaGenome re-querying also depends on the hosted server, whose model version was not
persistently exposed by the API.
