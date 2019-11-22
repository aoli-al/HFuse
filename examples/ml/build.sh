DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
rm -rf build
/home/hao01/torch_extension/bin/python3 setup.py build_ext -j8
/home/hao01/torch_extension/bin/python3 setup.py install
