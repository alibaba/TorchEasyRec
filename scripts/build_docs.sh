# Make sure in python3 environment
DATE=`date +'%Y%m%d'`

# make proto
bash scripts/gen_proto.sh
sed -i 's#<p>#<pre>#g;s#</p>#</pre>#g' docs/source/proto.html

# install requirements
python setup.py install
pip install -r requirements/docs.txt

# copy intro
sed 's#(docs/source/#(#g;s#(docs/images/#(../images/#g' README.md > docs/source/intro.md

# replace wheel and docker version
LATEST_WHEEL_VERSION=$(pip index versions tzrec -f http://tzrec.oss-cn-beijing.aliyuncs.com/release/nightly/repo.html --trusted-host tzrec.oss-cn-beijing.aliyuncs.com | awk 'NR==1{print $2}' | sed 's/[()]//g')
LATEST_DOCKER_VERSION=$(grep DOCKER_TAG= scripts/build_docker.sh | awk -F= '{print $2}')
for f in docs/source/quick_start/*.md; do
    cp $f $f.bak
    sed -i 's/${TZREC_NIGHTLY_VERSION}/'"${LATEST_WHEEL_VERSION}"'/g;s/${TZREC_DOCKER_VERSION}/'"${LATEST_DOCKER_VERSION}"'/g' $f
done

# make sphinx
cd docs
rm -rf build
make html
rm -rf build/html/_modules
cd -

# revert wheel and docker version
for f in docs/source/quick_start/*.md; do
    mv $f.bak $f
done

# python post_fix.py build/html/search.html

echo "view docs: python -m http.server --directory=docs/build/html/ 8081"
