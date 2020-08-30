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
- seaborn
- pandas

-CUDA
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo apt update
sudo apt-get install build-essential gcc-multilib dkms
sudo chmod +x cuda_10.2.89_440.33.01_linux.run
sudo ./cuda_10.2.89_440.33.01_linux.run
(driver and toolkit mandatory, samples optional)
sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig

sudo nano /etc/environment
PATH="/usr/local/sbin:usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games:/usr/local/cuda/bin"

nvidia-smi
(should see GPU driver version and CUDA version)

sudo nano /etc/rc.local
#!/bin/bash
nvidia-smi -pm 1
nvidia-smi -e 0
exit 0

sudo chmod +x /etc/rc.local

sudo reboot -h now




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
sphinx-apidoc -f -o . ../src
This should create a list of .rst files in the /doc folder

Generate html from restructuredtext (.rst) files
To generate html page: 
make html 
