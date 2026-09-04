.. Installation

Installation
===============

``segregation`` requires Python >= 3.12.

i) ``pip`` directly running in the prompt::

	pip install segregation

ii) Using the ``conda-forge`` channel as described in https://github.com/conda-forge/segregation-feedstock::

	conda install -c conda-forge segregation

iii) Install the development version from a local clone of this repository (this is an editable install)::

	git clone https://github.com/pysal/segregation.git
	cd segregation
	pip install -e .

iv) To use the bundled conda environment for the development install, create it first and then install into it::

	conda env create -f environment.yml
	conda activate segregation
	pip install -e .

v) ``pip`` directly from this repository running in the prompt::

	pip install git+https://github.com/pysal/segregation

Building the documentation
==========================

The example notebooks are executed when the documentation is built, so
building the docs also runs every notebook::

	conda env create -f environment.yml
	conda activate segregation
	pip install -e .
	cd docs
	make html

To execute the notebooks on their own::

	jupyter nbconvert --execute --to notebook --inplace docs/notebooks/*.ipynb
