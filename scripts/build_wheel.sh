#!/usr/bin/env bash

release_type=$1

if [[ "${release_type}" == *"nightly"* ]]; then
  git_rev=$(git rev-parse --short HEAD)
  date=$(date +"%Y%m%d")
  echo "git_rev: ${git_rev}"
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

if [[ "${release_type}" == *"nightly-oss"* ]]; then
  OSSUTIL="ossutil --mode EcsRamRole --ecs-role-name ${ALIBABA_CLOUD_ECS_METADATA} -e oss-cn-bejing.aliyuncs.com"
  whls=`${OSSUTIL} ls oss://tzrec/release/nightly/ -s | grep ".whl"`

  # check whl exist
  oss_exist=0
  for whl in ${whls[@]}; do
    if [[ $whl =~ $git_rev ]]
    then
        oss_exist=1
    fi
  done

  if [[ $exist == 0 ]]
  then
    # update wheel
    ${OSS_UTIL} cp -f ${wheel_path} oss://tzrec/release/nightly/
    ${OSS_UTIL} set-acl oss://tzrec/release/nightly/${wheel_file} public-read

    # update repo.html
    whls=`${OSS_UTIL} ls oss://tzrec/release/nightly/ -s| grep ".whl"`
    local_repo="repo.html"
    > ${local_repo}
    for whl in ${whls[@]}; do
        fname=`basename ${whl}`
        fname_encoded=`url_encode ${fname}`
        echo "<a href=\"${url_base}${fname_encoded}\">${fname}</a><br>" >> ${local_repo}
    done
    ${OSS_UTIL} cp -f ${local_repo} oss://tzrec/release/nightly/
    ${OSS_UTIL} set-acl oss://tzrec/release/nightly/${local_repo} public-read
  fi
fi
