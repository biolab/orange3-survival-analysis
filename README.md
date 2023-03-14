<h1 align="center">Orange3-Survival-Analysis</h1>


<p align="center">
<a href="https://github.com/biolab/orange3-survival-analysis/actions"><img alt="Actions Status" src="https://github.com/biolab/orange3-survival-analysis/actions/workflows/test.yml/badge.svg"></a>
<a href="https://orange3-survival-analysis.readthedocs.io/en/stable/?badge=stable"><img alt="Documentation Status" src="https://readthedocs.org/projects/orange3-survival-analysis/badge/?version=stable"></a>
<a href="https://codecov.io/gh/biolab/orange3-survival-analysis"><img alt="Coverage Status" src="https://codecov.io/gh/biolab/orange3-survival-analysis/branch/master/graph/badge.svg?token=H8PQO96TJJ"></a>
<a href="https://pypi.org/project/Orange3-Survival-Analysis/"><img alt="PyPI" src="https://img.shields.io/pypi/v/orange3-survival-analysis?color=blue"></a>
<a href="https://anaconda.org/conda-forge/orange3-survival-analysis/"><img alt="conda-forge" src="https://anaconda.org/conda-forge/orange3-survival-analysis/badges/version.svg"></a>
<a href="https://zenodo.org/badge/latestdoi/318492208"><img src="https://zenodo.org/badge/318492208.svg" alt="DOI"></a>
</p>

![Example Workflow](doc/readme-screenshot.png)

Survival analysis add-on for the [Orange](http://orange.biolab.si)
data mining suite. For more see the [widget documentation](https://orangedatamining.com/widget-catalog/survival-analysis/kaplan-meier-plot/)
and [example workflows](https://orangedatamining.com/workflows/Survival-Analysis/).


Blog posts:
- [An introduction to the Kaplan-Meier Estimator](https://orangedatamining.com/blog/2022/2022-05-25-KaplanMeier/)
- [Cox regression in Orange](https://orangedatamining.com/blog/2023/2023-01-27-cox-regression-in-orange/)


# Easy installation

First, [download](https://orange.biolab.si/download) the latest Orange release from
our website. Then, to install the survival analysis add-on, head to
`Options -> Add-ons...` in the menu bar.

# For developers


If you would like to install from cloned git repository, run

    pip install .

To register this add-on with Orange, but keep the code in the development directory
(do not copy it to Python's site-packages directory), run

    pip install -e .


###  Usage

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python -m Orange.canvas

The new widget appears in the toolbox bar under the section Survival Analysis.
Starting up for the first time may take a while.
