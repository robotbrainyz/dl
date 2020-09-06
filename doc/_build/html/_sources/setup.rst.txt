###################################################################
dl - Deep Neural Networks in Python
###################################################################

Python Dependencies
===================
- Python 3
- Numpy
- pytest
- Sphinx
- seaborn
- pandas
- torch

Download
=================================
To download dl from Github, do::

    $ git clone https://github.com/robotbrainyz/dl.git

Run Tests
=========
To run unit tests, do::

    $ pytest
    
Installation - Environment for auto-generating documentation
============================================================
Install Required Packages
Python 3
Pip install numpy
Pip install pytest
conda install -c anaconda sphinx


Documentation 
Install sphinx for documentation
mkdir doc
cd doc
sphinx-quickstart (all default options), version 1.0, release 1

In conf.py, ensure the following line points to the folder with python source files:
sys.path.insert(0, os.path.abspath('../src'))   

In conf.py, add sphinx.ext.autodoc to extensions:
extensions = ['sphinx.ext.autodoc']  

By default, the napoleon google docstring style is used:
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

In index.rst, add modules:
Contents:                                                                                                                                                                          
                                                                                                                                                                                   
.. toctree::                                                                                                                                                                       
   :maxdepth: 2                                                                                                                                                                    
                                                                                                                                                                                   
   modules  



Pull python docstring comments into restructuredtext files
cd doc
mkdir source
sphinx-apidoc -f -o . ../src
This should create a list of .rst files in the /doc folder

Generate html from restructuredtext (.rst) files
To generate html page: 
make html 

Google Colab
============
Sign in to Google Drive.
Create a folder where the files for dl can be cloned from github or copied to.

Sign in to Google Colaboratory.
Create a new notebook.

Mount Google Drive in the notebook:
from google.colab import drive
drive.mount('/content/gdrive')

This should bring up a URL. Click on the URL to open the link in a new browser tab or window. Select the account you used to create the Google Drive folder earlier. This allows Google Drive File Stream to access your Google Drive content.

In the Google Colaboratory notebook, change directory to the folder:
%cd /content/gdrive/My\ Drive/<my folder>
where <my folder> is the path to the folder.

To clone a private github repo, create an SSH key and link the SSH key to your github account:
https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh

Generate SSH key:
! ssh-keygen -t rsa -b 4096 -C "<your email>"

Print SSH key in console. Copy it to clipboard:
cat < <your .pub file path>

In a browser, sign in to github, add the public SSH key to your github account by pasting:
https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh

In Google Colab:
! chmod 700 /root/.ssh
Add to known hosts
! ssh-keyscan github.com >> /root/.ssh/known_hosts
! chmod 644 /root/.ssh/known_hosts
Configure git email and username
! git config --global user.email "email"
! git config --global user.name "username"
! ssh git@github.com

In Google Colab, clone the repo:
! git clone git@github.com:robotbrainyz/dl.git

Run pytest in Google Colab:
! python3 -m pytest

