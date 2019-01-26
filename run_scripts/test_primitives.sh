#!/usr/bin/env bash

source .bashrc_pypath

if [ -z ${BAG_PYTEST+x} ]
then
    echo "BAG_PYTEST is unset"
    exit 1
fi

echo ${BAG_PYTEST}
echo ${BAG_TECH_CONFIG_DIR}
exec ${BAG_PYTEST} ${BAG_TECH_CONFIG_DIR}/tests $@
