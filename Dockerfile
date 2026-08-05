FROM python:3.10.19-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.2.3 \
    matplotlib==3.10.6 \
    pyarrow==12.0.1 \
    scikit-learn==1.7.2 \
    statsmodels==0.14.5 \
    seaborn==0.13.2 \
    pysam==0.23.3

WORKDIR /work
CMD ["make", "verify", "figures"]
