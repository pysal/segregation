# coding: utf-8

from setuptools import setup, find_packages

from distutils.command.build_py import build_py

import os

with open('README.md', 'r', encoding='utf8') as file:
    long_description = file.read()

with open('segregation/__init__.py', 'r') as f:
    exec(f.readline())


# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def _get_requirements_from_files(groups_files):
    groups_reqlist = {}

    for k, v in groups_files.items():
        with open(v, 'r') as f:
            pkg_list = f.read().splitlines()
        groups_reqlist[k] = pkg_list

    return groups_reqlist


def setup_package():
    # get all file endings and copy whole file names without a file suffix
    # assumes nested directories are only down one level
    _groups_files = {
        'base': 'requirements.txt',
    }

    reqs = _get_requirements_from_files(_groups_files)
    install_reqs = reqs.pop('base')

    setup(
        name = 'segregation',
        version = __version__,
        description = "Methods for estimating and performing inference for spatial and non-spatial segregation in PySAL.",
        long_description = long_description,
		long_description_content_type = "text/markdown",
        maintainer = "Renan Xavier Cortes",
        maintainer_email = 'renanc@ucr.edu',
        url='https://segregation.readthedocs.io/en/latest/',
        download_url='https://pypi.org/project/segregation/',
        license = 'BSD',
        py_modules = ['segregation'],
        packages = find_packages(),
        setup_requires = ["pytest-runner"],
        tests_require = ["pytest"],
        keywords = ['spatial statistics', 'demography'],
        classifiers = [
            #'Development Status :: 1 - Alpha',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: GIS',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6'
        ],
        install_requires = install_reqs,
        cmdclass = {'build_py': build_py},
        python_requires = '>3.4')


if __name__ == '__main__':
    setup_package()
