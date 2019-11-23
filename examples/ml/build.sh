DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
rm -rf build
python3 setup.py build_ext -j8
python3 setup.py install
