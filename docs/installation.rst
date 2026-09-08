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

Optional dependencies
=====================

The core installation covers every segregation index and the inference,
decomposition, batch, and multiscalar tools. A few features and the example
notebooks rely on extra packages that are **not** installed automatically:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Package
     - Needed for
     - Install
   * - ``pandarm``
     - the ``segregation.network`` module: network-based indices,
       ``SpatialMinMax``, and network multiscalar profiles
     - ``pip install pandarm`` or ``conda install -c conda-forge pandarm``
   * - ``quilt3``
     - downloading the prepackaged OpenStreetMap street networks used in
       the network examples and tests
     - ``pip install quilt3`` or ``conda install -c conda-forge quilt3``
   * - ``watermark``
     - the ``%load_ext watermark`` cell at the top of every example notebook
     - ``pip install watermark`` or ``conda install -c conda-forge watermark``
   * - ``ipywidgets``
     - progress bars (``tqdm``) rendering inside Jupyter
     - ``pip install ipywidgets`` or ``conda install -c conda-forge ipywidgets``

All four are bundled in the ``tests`` extra::

	pip install "segregation[tests]"

or, with ``conda``::

	conda install -c conda-forge pandarm quilt3 watermark ipywidgets

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
