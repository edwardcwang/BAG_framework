#!/usr/bin/env bash

source .bashrc_pypath

# disable QT session manager warnings
unset SESSION_MANAGER

if [ -z ${BAG_PYTHON+x} ]
then
    echo "BAG_PYTHON is unset"
    exit 1
fi

${BAG_PYTHON} BAG_framework/run_scripts/netlist_config.py $@
