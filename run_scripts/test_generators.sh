#!/usr/bin/env bash

source .bashrc_pypath

if [[ $# < 1 ]]
then
    echo "Usage: ./test_generators.sh <data dir> [--package <package name>] [options...]"
    exit 1
fi

exec ${BAG_PYTEST} BAG_framework/tests_gen --data_root $@
