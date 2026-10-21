#!/usr/bin/env bash

function url_encode() {
  local _length="${#1}"
  for (( _offset = 0 ; _offset < _length ; _offset++ )); do
    _print_offset="${1:_offset:1}"
    case "${_print_offset}" in
      [a-zA-Z0-9.~_-]) printf "${_print_offset}" ;;
      ' ') printf + ;;
      *) printf '%%%X' "'${_print_offset}" ;;
    esac
  done
}

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
wheel_file=`ls dist/ | grep .whl`
wheel_path="dist/$wheel_file"
echo "wheel path: $(realpath ${wheel_path})"

if [[ "${release_type}" == *"nightly-oss"* ]]; then
  OSS_UTIL="ossutil -e oss-accelerate.aliyuncs.com "
  if [[ -n "${ALIBABA_CLOUD_ECS_METADATA}" ]]; then
    OSS_UTIL="${OSS_UTIL} --mode EcsRamRole --ecs-role-name ${ALIBABA_CLOUD_ECS_METADATA} "
  fi
  url_prefix="https://tzrec.oss-accelerate.aliyuncs.com/release/nightly/"
  whls=`${OSS_UTIL} ls oss://tzrec/release/nightly/ -s | grep ".whl"`

  # check whl exist
  oss_exist=0
  for whl in ${whls[@]}; do
    if [[ $whl =~ $git_rev ]]
    then
        oss_exist=1
        echo "wheel already exists, skip upload. wheel url: ${url_prefix}$(url_encode `basename ${whl}`)"
    fi
  done

  if [[ $oss_exist == 0 ]]
  then
    # update wheel
    ${OSS_UTIL} cp -f ${wheel_path} oss://tzrec/release/nightly/
    ${OSS_UTIL} set-acl oss://tzrec/release/nightly/${wheel_file} public-read
    echo "wheel url: ${url_prefix}$(url_encode ${wheel_file})"

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
