#!/bin/bash

# A script file for Google Colaboratory

# TeX
# apt-get install texlive-latex-extra  &> tex.log
# apt-get install ghostscript &>> tex.log
# apt-get install dvipng &>> tex.log


# Setting-up
rm -rf log
rm -rf config.py
mkdir logs


# Packages
pip install yellowbrick==1.3.post1 &> logs/yellow.log
pip install pymc3==3.11.2 &> logs/pymc3.log
pip install cloudpickle==1.6.0 &> logs/cloudpickle.log
pip install dask[complete]==2.30.0 &> logs/dask.log
pip install -U imbalanced-learn==0.8.0 &> logs/imblearn.log


# https://linux.die.net/man/1/wget
wget -q https://github.com/exhypotheses/beans/raw/develop/beans.zip
wget -q https://raw.githubusercontent.com/exhypotheses/beans/develop/config.py


# https://linux.die.net/man/1/unzip
rm -rf beans
unzip -u -q beans.zip
rm -rf beans.zip
