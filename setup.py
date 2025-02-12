# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 00:52:16 2019

@author: Jin Dou
"""

import setuptools


setuptools.setup(
  name="DynamicallyWarpedTRF",
  version="0.0.1",
  author="Powerfulbean",
  author_email="powerfulbean@gmail.com",
  long_description_content_type="text/markdown",
  url="",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  install_requires=[
    "matplotlib==3.6.1",
    "mne==0.19.1",
    "nntrf",
    "numpy==1.20.1",
    "scikit-learn==1.0.2",
    "scipy==1.9.3",
    "statsmodels==0.13.5",
    "StellarInfra",
    "StimRespFlow",
    "torch>=1.12.1,<2.0.0",
    "scikit-fda==0.7.1",
    "mtrf",
  ],
)