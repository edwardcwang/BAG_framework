#!/usr/bin/env bash

source .bashrc_pypath

if [[ $# < 1 ]]
then
    echo "Usage: ./pytest.sh <repo name> [options...]"
    exit 1
fi

exec ${BAG_PYTEST} BAG_framework/tests_gen --package $@
