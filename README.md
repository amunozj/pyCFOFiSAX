# CFOF Calculation via iSAX

[![Documentation Status](https://readthedocs.org/projects/pycfofisax/badge/?version=main)](https://pycfofisax.readthedocs.io/fr/main/?badge=main)

[![PyPI](https://github.com/luk-f/pyCFOFiSAX/actions/workflows/python-publish.yml/badge.svg)](https://github.com/luk-f/pyCFOFiSAX/actions/workflows/python-publish.yml)

Project Thesis Lucas \
2017-2020

## To begin

### Installation via PyPI

Run `pip install pyCFOFiSAX` or `python3 -m pip install pyCFOFiSAX`.

### Manual installation

Run `pip install -r requirements.txt` or `python3 -m pip install -r requirements.txt`.

The package `tslearn` requires `Cython`, `numba` and `llvmlite` (`pip` will automatically download these packages).
However for `Cython`, it is [necessary to have a C compiler (as described on this site)](https://cython.readthedocs.io/en/latest/src/quickstart/install.html),
and the [package `llvmlite`](https://llvmlite.readthedocs.io/en/latest/) for this project, tested only with Python 3.7 and 3.8
that we recommend.

Otherwise, the use of `conda` simplifies the installation (as recommended on the site of[tslearn](https://tslearn.readthedocs.io/en/latest/index.html)), because does not require
the installation of compiler c and packages `Cython` and `llvmlite`.

### Use

The documentation is available here (in French) : [https://pycfofisax.readthedocs.io/fr/main/](https://pycfofisax.readthedocs.io/fr/main/)

## Thanks

Development as part of the work of :
 - [CFOF: A Concentration Free Measure for Anomaly Detection, par Fabrizio Angiulli](https://arxiv.org/abs/1901.04992),
 - [iSAX: Indexing and Mining Terabyte Sized Time Series, par Jin Shieh et Eamonn Keogh](http://www.cs.ucr.edu/~eamonn/iSAX/iSAX.html),
 - [iSAX 2.0: Indexing and Mining One Billion Time Series, par Alessandro Camerra, Themis Palpanas, Jin Shieh et Eamonn Keogh](https://www.cs.ucr.edu/~eamonn/iSAX_2.0.pdf).
 - [Scoring Message Stream Anomalies in Railway Communication Systems, par Lucas Foulon, Serge Fenet, Christophe Rigotti et Denis Jouvin](https://hal.archives-ouvertes.fr/hal-02357924/)

Using the code of:
 - [tslearn de Romain Tavenard et al. (2017)](https://tslearn.readthedocs.io/en/latest/index.html).

Use of the packages of:
 - [Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing, Available online 2 June 2017, ISSN 0925-2312](https://doi.org/10.1016/j.neucom.2017.04.070)
