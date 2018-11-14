#!/usr/bin/env bash

source .bashrc_pypath

if [[ $# < 1 ]]
then
    echo "Usage: ./test_generators.sh <repo name> [options...]"
    exit 1
fi

exec ${BAG_PYTEST} BAG_framework/tests_gen --package $@
