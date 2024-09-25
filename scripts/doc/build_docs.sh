# Make sure in python3 environment
DATE=`date +'%Y%m%d'`

# install requirements
python setup.py install
pip install -r requirements/docs.txt

# edit docs, e.g., replace version info
bash scripts/doc/build_doc_pre_work.sh

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
