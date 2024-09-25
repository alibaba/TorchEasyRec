#!/usr/bin/env bash

release_type=$1

if [[ "${release_type}" == *"nightly"* ]]; then
  git_rev=$(git rev-parse --short HEAD)
  date=$(date +"%Y%m%d")
  export NIGHTLY_VERSION="${date}.${git_rev}"
fi

echo "building ${release_type} wheel... "

if [ -e build ]; then
  rm -rf build
fi
if [ -e dist ]; then
  rm -rf dist
fi
if [ -e rec_sln.egg-info ]; then
  rm -rf tzrec.egg-info
fi

bash scripts/gen_proto.sh
python setup.py sdist bdist_wheel
