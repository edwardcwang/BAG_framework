#!/usr/bin/env bash

source .bashrc_pypath

exec ${BAG_PYTEST} $@
