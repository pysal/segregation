.. Installation

Installation
===============

segregation supports python `3.5`_ and `3.6`_ only. Please make sure that you are
operating in a python 3 environment.

Installing released version
---------------------------

segregation has not yet been released on the `Python Package Index`_. 


Installing development version
------------------------------
The recommended method for installing segregation is with `anaconda`_. To get started with the development version, clone this repository or download it manually then `cd` into the directory and run the following commands::

$ conda env create -f environment.yml
$ source activate segregation 
$ python setup.py develop

You can  also `fork`_ the `pysal/segregation`_ repo and create a local clone of
your fork. By making changes
to your local clone and submitting a pull request to `pysal/segregation`_, you can
contribute to the segregation development.

.. _3.5: https://docs.python.org/3.5/
.. _3.6: https://docs.python.org/3.6/
.. _Python Package Index: https://pypi.org/project/segregation/
.. _pysal/segregation: https://github.com/pysal/segregation
.. _fork: https://help.github.com/articles/fork-a-repo/
.. _anaconda: https://www.anaconda.com/download/ 
