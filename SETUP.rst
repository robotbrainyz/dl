###################################################################
dl - Neural Networks in Python
###################################################################


Requirements
============
**dl**, a neural network implementation in Python has the following dependencies:

- Python 3
- Numpy
- pytest
- Sphinx

Installation - Python environment
============
To download the Binary Trees library from Github, do::

    $ git clone https://github.com/shunsvineyard/python-sample-code.git


Installation - Environment for auto-generating documentation
============

    


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
sphinx-apidoc -f -o source/ ../src
This should create a list of .rst files in the /doc folder

Generate html from restructuredtext (.rst) files
To generate html page: 
make html 
