#!/usr/bin/env bash

source .bashrc_pypath

# disable QT session manager warnings
unset SESSION_MANAGER

exec ${BAG_PYTHON} -m IPython
