#!/usr/bin/env bash
# Generate configurations files needed to netlist from BAG.
# This script must be run from the working directory.

source .bashrc_pypath

# disable QT session manager warnings
unset SESSION_MANAGER

if [ -z ${BAG_PYTHON+x} ]
then
    echo "BAG_PYTHON is unset"
    exit 1
fi

export OUTDIR=${BAG_TECH_CONFIG_DIR##*/}/netlist_setup
export CONF=${OUTDIR}/gen_config.yaml

${BAG_PYTHON} BAG_framework/run_scripts/netlist_config.py ${CONF} ${OUTDIR}
