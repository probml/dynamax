# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'dynamax'
copyright = '2022, Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li, Scott Linderman, and Kevin Murphy'
author = 'Peter Chang, Giles Harper-Donnelly, Aleyna Kara, Xinglong Li, Scott Linderman, and Kevin Murphy'


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.myst': 'myst-nb',
    '.ipynb': 'myst-nb'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'notebooks/slds/rbpf_maneuver.ipynb']

nb_execution_allow_errors = False

# Myst-NB
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    # "html_admonition",
    # "html_image",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]
nb_execution_timeout = 600
nb_execution_mode = "cache"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = ""
html_logo = "../logo/logo.gif"
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/probml/dynamax',
    "use_repository_button": True,
    "use_download_button": False,
    'repository_branch': 'main',
    "path_to_docs": 'docs',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
        'binderhub_url': 'https://mybinder.org'
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


autosummary_generate = True
autodoc_typehints = "description"
add_module_names = False
autodoc_member_order = "bysource"