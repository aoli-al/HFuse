import argparse
import json
import subprocess
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("config_path")
args = parser.parse_args()


fuser_bin = "/tmp/tmp.Qz5yLOgO2H/cmake-build-debug-cvl/tools/llvm-smart-fuser/llvm-smart-fuser"
clang_base = "/home/hao01/scrach/llvm/build/lib/clang/10.0.0"
currentTime = 99999999999999999999999999999999999
currentImpl = ""
launchBound = False


def run(data, func, launch_bound=False):
    global currentTime
    global currentImpl
    global launchBound
    kernel_file = data["kernel_file"]
    copyfile(kernel_file, kernel_file + ".bak")
    f = open(kernel_file, "r")
    counter = 0
    source = ""
    for line in f:
        counter += 1
        source += line
        if counter == data['function_location']:
            source += func
        if counter == data['invocation_insert_location']:
            source += data['invocation_statement']
    f.close()
    f = open(kernel_file, "w")
    f.write(source)
    f.close()
    print("")
    env = {}
    if launch_bound:
        env["MAX_REG"] = str(data["launch_bound"])
    subprocess.check_output(data['build_script'].split(" "), env=env)
    subprocess.check_output(["/usr/local/cuda-10.1/bin/nvprof", "-f", "-o", "/tmp/out.nvprof",
                             *data['run_script'].split(" ")])

    subprocess.check_output(["/home/hao01/nvprof2json", "-f", "-o", "/tmp/out.json", "/tmp/out.nvprof"])
    copyfile(kernel_file + ".bak", kernel_file)

    with open('/tmp/out.json', "r") as jf:
        jd = json.load(jf)
        for event in jd['traceEvents']:
            name = event['name']
            if "FUNC" in name:
                if event['dur'] < currentTime:
                    currentTime = event['dur']
                    currentImpl = func
                    launchBound = launch_bound



with open(args.config_path) as f:
    data = json.load(f)
    kernel_file = data['kernel_file']
    copyfile(kernel_file, kernel_file+".bak")
    output = subprocess.check_output([fuser_bin, "--config="+data["kernel_config"],
                                      kernel_file, "-extra-arg=-resource-dir="+clang_base])
    copyfile(kernel_file+".bak", kernel_file)
    for func in json.loads(output):
        run(data, func, True)
        run(data, func, False)

print(currentImpl)
print(currentTime)
