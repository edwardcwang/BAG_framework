#!/usr/bin/env bash

source .bashrc_pypath

if [ $# -lt 1 ]
then
    echo "Usage: ./pytest.sh <test directory> [options...]"
    exit 1
fi

exec ${BAG_PYTEST} $@
