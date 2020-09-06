###################################################################
dl - Deep Neural Networks in Python
###################################################################

Introduction
============
This library, dl, is an implementation of neural networks using Python.


Running and Development Environment
===================================
- Developed and (unit) tested locally on Mac OS 10.15.5. 
- GPU (CUDA) acceleration (unit) tested on Google Colaboratory.


Python Dependencies - Local (CPU)
=================================
- Python 3.6.10

- pandas 1.1.1
- pytest 6.0.1
- seaborn 0.10.1
- Sphinx 3.2.1 (installed using Anaconda: conda install -c anaconda sphinx)
- torch 1.6.0


Python Dependencies - Remote - Google Colaboratory (GPU)
========================================================
- Python 3.6.9

- pandas 1.0.5
- pytest 3.6.3
- seaborn 0.10.1
- torch 1.6.0+cu101


Download
========
To download dl from Github, do::

    $ git clone https://github.com/robotbrainyz/dl.git


Run Tests
=========
To run unit tests, do::

    $ pytest


View and Update Documentation
=============================
- dl's functions and API are available for browsing and searching in dl/doc/_build/html/index.html
- To update and auto-generate HTML documentation, do::

    $ cd doc
    $ . updateDocs.sh


Getting Started with dl on Google Colaboratory
==============================================
- Sign in to Google Drive.
- Create a folder to work in.

- Sign in to Google Colaboratory.
- Create a new notebook.

- Mount Google Drive in the notebook::

    from google.colab import drive
    drive.mount('/content/gdrive')
    
- This should bring up a URL. Click on the URL to open the link in a new browser tab or window. Select the account you used to create the Google Drive folder earlier. This allows Google Drive File Stream to access your Google Drive content.

- In the Google Colaboratory notebook, change directory to the folder::

    %cd /content/gdrive/My\ Drive/<my folder>

- Clone the repo::

    ! git clone https://www.github.com/robotbrainyz/dl.git

- Run pytest::

    ! python3 -m pytest
    
- The unit tests include basic training and evaluation of neural network models supported by dl. See files named dl_model*_test.py for examples on how to use dl.

- dl/dl_base.ipynb is a sample notebook that runs on Google Colaboratory.
