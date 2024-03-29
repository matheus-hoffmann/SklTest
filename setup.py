import io
import os
import re

from setuptools import setup, find_packages


def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def version(path):
    """Obtain the package version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(
        r"""^__version__ = ['"]([^'"]*)['"]""", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


DESCRIPTION = "skl_test"
VERSION = version("skl_regressor_test/__init__.py")

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name="skl_test",
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matheus Hoffmann",
    author_email="mmatheushb@hotmail.com",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "xgboost",
        "scikit-learn",
        "openpyxl"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "coverage",
            "codecov",
            "sphinx",
            "nbsphinx",
            "numpydoc",
            "sphinxcontrib-bibtex",
            "black",
            "isort",
            "myst-nb",
            "linkify-it-py",
            "sphinx-book-theme",
            "sphinx-panels",
            "sphinx-copybutton",
        ],
    },
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
