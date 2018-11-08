#!/usr/bin/env bash

set -x

git submodule foreach --recursive git reset --hard
git submodule foreach --recursive git clean -fd
git submodule foreach --recursive git pull

git reset --hard
git clean -fd
git pull
