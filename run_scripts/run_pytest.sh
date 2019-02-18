#!/usr/bin/env bash

source .bashrc_pypath

if [ -z ${BAG_PYTEST+x} ]
then
    echo "BAG_PYTEST is unset"
    exit 1
fi

exec ${BAG_PYTEST} $@
