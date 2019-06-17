.. Installation

Installation
===============

Note: segregation supports python `3.5`_ and `3.6`_ only. Please make sure that you are
operating in a python 3 environment.

i) `pip` directly running in the prompt::

	pip install segregation

ii) Using the `conda-forge` channel as described in https://github.com/conda-forge/segregation-feedstock::

	conda config --add channels conda-forge
	conda install segregation

iii) Another recommended method for installing segregation is with [anaconda](https://www.anaconda.com/download/). Clone this repository or download it manually then `cd` into the directory and run the following commands (this will install the development version)::

	conda env create -f environment.yml
	source activate segregation
	python setup.py develop

iv) `pip` directly from this repository running in the prompt (if you experience an issue trying to install this way, take a look at this discussion: https://github.com/pysal/segregation/issues/15)::

	pip install git+https://github.com/pysal/segregation
