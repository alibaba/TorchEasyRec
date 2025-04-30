# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Alibaba, Inc. and its affiliates.
import codecs
import os

from setuptools import find_packages, setup


def readme():
    """Parse readme content."""
    with codecs.open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


version_file = "tzrec/version.py"


def get_version():
    """Get TorchEasyRec version."""
    with codecs.open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    if "NIGHTLY_VERSION" in os.environ:
        return f"{locals()['__version__']}+{os.environ['NIGHTLY_VERSION']}"
    else:
        return locals()["__version__"]


def parse_requirements(fname="requirements.txt"):
    """Parse the package dependencies listed in a requirements file."""

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for line in parse_require_file(target):
                yield line
        else:
            yield line

    def parse_require_file(fpath):
        with codecs.open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for ll in parse_line(line):
                        yield ll

    packages = list(parse_require_file(fname))
    return packages


setup(
    name="tzrec",
    version=get_version(),
    description="An easy-to-use framework for Recommendation",
    long_description=readme(),
    author="EasyRec Team",
    author_email="easy_rec@alibaba-inc.com",
    url="http://gitlab.alibaba-inc.com/pai_biz_arch/TorchEasyRec",
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=["Programming Language :: Python :: 3"],
    tests_require=parse_requirements("requirements/test.txt"),
    install_requires=parse_requirements("requirements/runtime.txt"),
    extras_require={
        "all": parse_requirements("requirements.txt"),
        "tests": parse_requirements("requirements/test.txt"),
        "gpu": parse_requirements("requirements/gpu.txt"),
        "cpu": parse_requirements("requirements/cpu.txt"),
    },
)
