import csv
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import re

order = [
    "sha256d_gpu_hash_shared(",
    "ethash_search(",
    "blake2b_gpu_hash(",
    "sia_blake2b_gpu_hash(",


    "ethash_search2_blake2b_gpu_hash_0",
    "sia_blake2b_gpu_hash_ethash_search4_0",
    "sha256d_gpu_hash_shared_ethash_search3_0",

    "blake2b_gpu_hash_sia_blake2b_gpu_hash_0",
    "blake2b_gpu_hash_sha256d_gpu_hash_shared_0",
    "sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_0",


    "ethash_search2_blake2b_gpu_hash_100",
    "sha256d_gpu_hash_shared_ethash_search3_100",
    "blake2b_gpu_hash_sia_blake2b_gpu_hash_100",
    "blake2b_gpu_hash_sha256d_gpu_hash_shared_100",
    "sia_blake2b_gpu_hash_ethash_search4_100",
    "sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_100",

    "kernelHistogram1D",
    "im2col_kernel",
    "MaxPoolForward",
    "batch_norm_collect_statistics_kernel",
    "upsample_bilinear2d_out_frame",
    #
    "im2col_kernel_MaxPoolForward_0",
    "im2col_kernel_batch_norm_collect_statistics_kernel_0",
    "im2col_kernel_upsample_bilinear2d_out_frame_0",
    "MaxPoolForward_batch_norm_collect_statistics_kernel_0",
    "max_pool_upsample_kernel",
    "kernelHistogram1D_MaxPoolForward_11",
    "kernelHistogram1D_batch_norm_collect_statistics_kernel_11",
    "im2col_kernel_kernelHistogram1D_0",
    "kernelHistogram1D_upsample_bilinear2d_out_frame_11",
    "upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_0",

    "im2col_kernel_MaxPoolForward_100",
    "im2col_kernel_batch_norm_collect_statistics_kernel_100",
    "im2col_kernel_upsample_bilinear2d_out_frame_100",
    "MaxPoolForward_batch_norm_collect_statistics_kernel_100",
    "MaxPoolForward_upsample_bilinear2d_out_frame_100",
    "kernelHistogram1D_MaxPoolForward_100",
    "kernelHistogram1D_batch_norm_collect_statistics_kernel_100",
    "im2col_kernel_kernelHistogram1D_100",
    "kernelHistogram1D_upsample_bilinear2d_out_frame_100",
    "upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_100",

    "kernelHistogram1D_MaxPoolForward_0",
    "kernelHistogram1D_batch_norm_collect_statistics_kernel_0",
    "im2col_kernel_kernelHistogram1D_1x",
    "kernelHistogram1D_upsample_bilinear2d_out_frame_0",

    "kernelHistogram1D_MaxPoolForward_bar_sync",
    "kernelHistogram1D_batch_norm_collect_statistics_kernel_bar_sync",
    "im2col_kernel_kernelHistogram1D_bar_sync",
    "kernelHistogram1D_upsample_bilinear2d_out_frame_bar_sync",
    "upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_bar_sync",
    "im2col_kernel_batch_norm_collect_statistics_kernel_bar_sync",
    "MaxPoolForward_batch_norm_collect_statistics_kernel_bar_sync",
]

kernels = {
    "sia_blake2b_gpu_hash": "Blake2B",
    "sha256d_gpu_hash_shared": "SHA256",
    "ethash_search": "Ethash",
    "blake2b_gpu_hash": "Blake256",
    "kernelHistogram1D": "Hist",
    "im2col_kernel": "Im2Col",
    "MaxPoolForward": "Maxpool",
    "batch_norm_collect_statistics_kernel": "Batchnorm",
    "upsample_bilinear2d_out_frame": "Upsample",
}

kernel_order = [
    "sia_blake2b_gpu_hash",
    "sha256d_gpu_hash_shared",
    "ethash_search",
    "blake2b_gpu_hash",
    "kernelHistogram1D",
    "im2col_kernel",
    "MaxPoolForward",
    "batch_norm_collect_statistics_kernel",
    "upsample_bilinear2d_out_frame",
]

tags = {
    "_lb": "LB",
    "_bar_sync": "BS",
    "_vfuse": "VF",
    "_hfuse": "HF",
    # "idx_1": "OP",
    "_imba": "PP",
}


event_order = [
    "elapsed_cycles_pm",
    "issue_slot_utilization",
    # "stall_inst_fetch",
    # "stall_exec_dependency",
    "stall_memory_dependency",
    "achieved_occupancy",
    # "eligible_warps_per_cycle",
]

events = {
    "elapsed_cycles_pm": "Cycles",
    "issue_slot_utilization": "Slot Utilization",
    "eligible_warps_per_cycle": "Eligible Warps/Cycle",
    "stall_inst_fetch": "Instruction Fetch",
    "stall_exec_dependency": "Execution Dependency",
    "stall_memory_dependency": "Data Request",
    "achieved_occupancy": "Occupancy",
}

TAG_ORDER = [
    "ST", "VF", "LB+VF", "HF", "LB+HF", "HF+OP", "LB+HF+OP", "BS+HF", "LB+BS+HF",
    # "HF+PP", "HF+OP+PP", "LB+HF+PP", 
    "LB+BS+HF+PP", "BS+HF+PP", 
    # "LB+HF+OP+PP"
]

###
# note maxpool issue slot util = 6.301776
#      upsample = 2072924527
#      mp clock = 1453350987
#      up sample clock = 2072924527

###
result = {}

LABEL = "Avg"
ITERS = 5

def build_name(names):
    return "+".join(sorted(names))

def build_tags(tag):
    t = None
    for _, v in tags.items():
        if v in tag:
            if not t:
                t = v
            else:
                t += "+" + v
    return t


def find_name(name):
    fused = "_fused_kernel" in name
    r = []
    t = ""
    for k in kernel_order:
        v = kernels[k]
        if k in name:
            name = name.replace(k, "")
            if not fused:
                return v, "ST", "0"
            else:
                r.append(v)
                # if not r:
                #     r = v
                # else:
                #     r += "+" + v
    if not len(r):
        return None, None, None
    for k, v in tags.items():
        if k in name:
            if not t:
                t = v
            else:
                t += "+" + v
    idx = re.search("_idx_(\d)", name).group(1)
    return build_name(r), t, idx

def analyze(file_name, result, ignore=[]):
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key, tag, idx = find_name(row['Kernel'])
            if not key:
                continue
            # key += app
            if key not in result:
                result[key] = {}
            if tag not in result[key]:
                result[key][tag] = {}
            # if key in split and idx != str(split[key]):
            #     continue
            if idx not in result[key][tag]:
                result[key][tag][idx] = {}
            if 'event' in file_name:
                result[key][tag][idx][row['Event Name']] = float(row[LABEL].strip("%"))
            else:
                if row['Metric Name'] == "achieved_occupancy":
                    result[key][tag][idx][row['Metric Name']] = float(row[LABEL].strip("%")) * 100
                else:
                    result[key][tag][idx][row['Metric Name']] = float(row[LABEL].strip("%"))
    for key in ignore:
        del result[key]

found_tags = set()

def analyze_execution_time(f, time_result, time_result_separate, ignore):
    visited_ts = set()
    # time_result = {}
    time_result_count = {}
    # time_result_separate = {}

    def update_time_result(key, tag, idx, time):
        if key not in time_result_count:
            time_result_count[key] = {}
        if tag not in time_result_count[key]:
            time_result_count[key][tag] = {}
        if idx not in time_result_count[key][tag]:
            time_result_count[key][tag][idx] = 1
        else:
            time_result_count[key][tag][idx] += 1
        if key not in time_result:
            time_result[key] = {}
            time_result_separate[key] = {}
        if tag not in time_result[key]:
            time_result[key][tag] = {}
            time_result_separate[key][tag] = {}
        if idx in time_result[key][tag]:
            time_result[key][tag][idx] += time
            time_result_separate[key][tag][idx].append(time)
        else:
            time_result[key][tag][idx] = time
            time_result_separate[key][tag][idx] = [time]

    with open(f) as json_file:
        data = json.load(json_file)
        prev = []
        prev_time = None
        last_time = None
        prev_dur = None
        dur_p = 0
        data['traceEvents'] = sorted(data['traceEvents'], key=lambda t: t['ts'])
        for event in data['traceEvents']:
            name = event['name']
            dur_p = event['ts']
            key, tag, idx = find_name(name)
            if not key:
                continue
            if 'ts' not in event or event['ts'] in visited_ts:
                continue
            found_tags.add(tag)
            visited_ts.add(event['ts'])
            update_time_result(key, tag, idx, event['dur'])
            if tag == "ST":
                prev.append(key)
                if not prev_time:
                    prev_time = event['ts']
                    prev_dur = event['dur']
                last_time = event['ts'] + event['dur']
                if len(prev) == 2:
                    name = build_name(prev)
                    update_time_result(name, "ST", idx, last_time - prev_time)
                    update_time_result(name, "RA", idx, prev_dur / event['dur'] if prev[0] < prev[1] else event['dur'] / prev_dur )
                    if prev[0] not in time_result_separate[name]:
                        time_result_separate[name][prev[0]] = (99999999999, -9999999)
                        time_result_separate[name][prev[1]] = (99999999999, -9999999)
                    
                    time_result_separate[name][prev[0]] = (min(time_result_separate[name][prev[0]][0], prev_dur), 
                                                           max(time_result_separate[name][prev[0]][1], prev_dur))
                    time_result_separate[name][prev[1]] = (min(time_result_separate[name][prev[1]][0], event['dur']), 
                                                           max(time_result_separate[name][prev[1]][1], event['dur']))
                    prev = []
                    prev_time = None
                    last_time = None

    for x, y in time_result_count.items():
        for k, v in y.items():
            for iii, count in v.items():
                time_result[x][k][iii] = time_result[x][k][iii] / 1000000 / count
        if "+" in x:
            prev = x.split("+")
            time_result[x]["RA"]["0"] = time_result[prev[0]]["ST"]["0"] / time_result[prev[1]]["ST"]["0"]
    for k in ignore:
        del time_result[k]
        del time_result_separate[k]
    return time_result, time_result_separate

r1 = {}
r1_s = {}
v1 = {}
v1_s = {}
pascal_selection = {
    "Batchnorm+Upsample": ["LB+HF", "LB+VF"],
    "Hist+Im2Col": ["LB+HF", "LB+VF"],
    "Hist+Maxpool": ["LB+HF", "LB+VF"],
    "Batchnorm+Hist": ["LB+HF", "LB+VF"],
    "Hist+Upsample": ["LB+HF", "VF"],
    "Batchnorm+Im2Col": ["LB+HF", "LB+VF"],
    "Im2Col+Maxpool": ["LB+HF", "LB+VF"],
    "Batchnorm+Maxpool": ["LB+VF", "LB+HF"],
    "Im2Col+Upsample": ["VF", "HF"],
    "Maxpool+Upsample": ["LB+HF", "LB+VF"],
    "Blake2B+Ethash": ["LB+HF", "LB+VF"],
    "Blake256+Ethash": ["LB+HF", "LB+VF"],
    "Ethash+SHA256": ["LB+HF", "LB+VF"],
    "Blake256+Blake2B": ["HF", "LB+VF"],
    "Blake256+SHA256": ["HF", "LB+VF"], 
    "Blake2B+SHA256": ["HF", "LB+VF"], 
}
volta_selection = {
    "Batchnorm+Upsample": ["LB+HF", "LB+VF"],
    "Hist+Im2Col": ["LB+VF", "LB+HF"],
    "Hist+Maxpool": ["HF", "LB+VF"],
    "Batchnorm+Hist": ["LB+HF", "VF"],
    "Hist+Upsample": ["LB+HF", "LB+VF"],
    "Batchnorm+Im2Col": ["LB+HF", "LB+VF"],
    "Im2Col+Maxpool": ["VF", "LB+HF"], 
    "Batchnorm+Maxpool": ["LB+HF", "VF"],
    "Im2Col+Upsample": ["HF", "VF"],
    "Maxpool+Upsample": ["LB+HF", "VF"],
    "Blake2B+Ethash": ["LB+HF", "LB+VF"],
    "Blake256+Ethash": ["LB+HF", "LB+VF"],
    "Ethash+SHA256": ["HF", "LB+VF"],
    "Blake256+Blake2B": ["HF", "LB+VF"],
    "Blake256+SHA256": ["HF", "LB+VF"], 
    "Blake2B+SHA256": ["HF", "LB+VF"], 
}
split = {
    "Batchnorm+Upsample": 3,
    "Hist+Im2Col": 3,
    "Hist+Maxpool": 3,
    "Batchnorm+Hist": 3,
    "Hist+Upsample": 3,
    "Batchnorm+Im2Col": 3,
    "Im2Col+Maxpool": 3, 
    "Batchnorm+Maxpool": 1,
    "Im2Col+Upsample": 3,
    "Maxpool+Upsample": 1,
}
order = [
    "Batchnorm+Upsample",
    "Batchnorm+Hist",
    "Batchnorm+Im2Col",
    "Batchnorm+Maxpool",
    "Hist+Im2Col",
    "Hist+Maxpool",
    "Hist+Upsample",
    "Im2Col+Maxpool",
    "Im2Col+Upsample",
    "Maxpool+Upsample",
    "Blake2B+Ethash",
    "Blake256+Ethash",
    "Ethash+SHA256",
    "Blake256+Blake2B",
    "Blake256+SHA256",
    "Blake2B+SHA256",
]

plot_shape = {
    "VFuse": {
        "1080Ti": "xb",
        "V100": "*g",
    },

    "HFuse": {
        "1080Ti": "+r",
        "V100": ".y",
    },
    "NTuse": {
        "1080Ti": "3c",
        "V100": "1m",
    },
}
r2 = {}
# analyze_execution_time("./data-new/ml-pascal-chart-1.json", r1, r1_s, ["Batchnorm+Im2Col"])
# analyze_execution_time("./data-new/ml-pascal-chart-2.json", r1, r1_s, [])
# r1_s = json.load(open("./data-new/ml-pascal-chart.json"))
# analyze_execution_time("./data-new/ml-volta-chart-1.json", v1, v1_s, ['Im2Col+Upsample'])
# analyze_execution_time("./data-new/ml-volta-chart-2.json", v1, v1_s, [])
# json.dump(v1_s, open("./data-new/ml-volta-chart.json", 'w'))
# exit(0)
# v1_s = json.load(open("./data-new/ml-volta-chart.json"))

r1 = {}


upsample = [
    "Batchnorm+Upsample",
    "Hist+Upsample",
    "Im2Col+Upsample",
    "Maxpool+Upsample",
    "Batchnorm",
    "Hist",
    "Im2Col",
    "Maxpool",
    "Upsample"
]

# analyze_execution_time("./data-new/ml_perf/p-1.json", r1,  r1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/v-1.json", v1, v1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/p-2.json", r1,  r1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/v-2.json", v1, v1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/p-3.json", r1,  r1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/v-3.json", v1, v1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/p-4.json", r1,  r1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/v-4.json", v1, v1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/p-5.json", r1,  r1_s, upsample)
# analyze_execution_time("./data-new/ml_perf/v-5.json", v1, v1_s, upsample)
# analyze_execution_time("./data-new/crypto-volta.json", {}, v1_s, [])
# analyze_execution_time("./data-new/crypto-pascal.json", {}, r1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-p-1.json", r1,  r1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-p-2.json", r1,  r1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-p-3.json", r1,  r1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-p-4.json", r1,  r1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-p-5.json", r1,  r1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-v-1.json", v1, v1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-v-2.json", v1, v1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-v-3.json", v1, v1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-v-4.json", v1, v1_s, [])
# analyze_execution_time("./data-new/ml_perf/u-v-5.json", v1, v1_s, [])



# r1 = {}
# v1 = {}
# dummy_r1 = {}
# dummy_v1 = {}
# analyze_execution_time("./data-new/crypto-single.json", r1, {}, [])
# analyze_execution_time("./data-new/crypto/crypto-single-volta.json", v1, {}, [])
# analyze_execution_time("./data-new/ml_perf/ml-single-pascal.json", r1, dummy_r1, upsample)
# analyze_execution_time("./data-new/ml_perf/ml-single-volta.json", v1, dummy_v1, upsample)
# analyze_execution_time("./data-new/ml_perf/upsample-single-pascal.json", r1, dummy_r1, [])
# analyze_execution_time("./data-new/ml_perf/upsample-single-volta.json", v1, dummy_v1, [])

# json.dump(r1_s, open("./data-new/ml_perf/graph-pascal.json", 'w'))
# json.dump(v1_s, open("./data-new/ml_perf/graph-volta.json", 'w'))
# json.dump(r1, open("./data-new/ml_perf/table-pascal.json", 'w'))
# json.dump(v1, open("./data-new/ml_perf/table-volta.json", 'w'))
# exit(0)

r1_s = json.load(open(sys.argv[1]))
v1_s = json.load(open(sys.argv[2]))
#  r1 = json.load(open("./data-new/ml_perf/table-pascal.json"))
#  v1 = json.load(open("./data-new/ml_perf/table-volta.json"))

# del r1['']
# del r1_s['']
#  r2 = analyze_execution_time("./data-new/ml-pascal.json")
print(found_tags)
# exit(0)
# analyze_execution_time("./data/regcap-barsync.json", r2)
# analyze_execution_time("./data/ml.json", r1)
# analyze_execution_time("./data/ml-regcap.json", r2)
# analyze_execution_time("./data/ml-regcap.json", r2)
# analyze_execution_time("./data/ml-regcap.json", r2)
#  analyze_execution_time("./data/miner.json", r1)
#  analyze_execution_time("./data/miner_regcap.json", r2)
# analyze_execution_time("./data/ml_regcap.json", r2)
# analyze_execution_time("./data/ml-naive.json", r2)

# fr = {}
# for x, y in r1.items():
#     if "_s" not in x:
#         fr[x] = min(y, r2[x])
#     else:
#         fr[x] = y


# print(fr)

def build_graph(result, selection, result_volta, selection_volta):
    # order = list(filter(lambda x: "+" in x, result_volta.keys())) 
    min_length = 99999999999999999999999
    for k in order:
        delta = 999999999999999999999
        delta_idx = 0
        for idx in range(int(len(result_volta[k]['RA']["0"])//ITERS)):
            if abs(result_volta[k]['RA']['0'][idx] - 1) < delta:
                delta = abs(result_volta[k]['RA']['0'][idx] - 1)
                delta_idx = idx
        print("delta: ", k, delta_idx)
        min_length = min(len(result[k]['RA']["0"]), min_length)
        min_length = min(len(result_volta[k]['RA']["0"]), min_length)
    min_length /= ITERS

    def get_average(data):
        data = np.split(np.array(data), ITERS)
        return np.asarray(np.matrix(data).mean(0))[0]
    def get_best(data, tag):
        candidates = []
        for k, v in data.items():
            if tag in k:
                candidates.extend(v.values())
        candidates = [get_average(x) for x in candidates]
        # return candidates[0]
        return np.amin(candidates, 0)
    def get_best_naive(data, tag, iii):
        candidates = []
        for k, v in data.items():
            if tag in k:
                candidates.append(v[iii])
        candidates = [get_average(x) for x in candidates]
        # return candidates[0]
        return np.amin(candidates, 0)
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 2, 1])
    fig, axs = plt.subplots(4, 4, figsize=(15,14))
    idx = 0

    for k in order:
        # if idx == 8:
        #     idx += 1
        v = result[k]
        print(k)
        if "+" not in k:
            continue
        vra = get_average(v['RA']["0"])
        p_max = max(vra)
        p_min = min(vra)
        volta_ra = get_average(result_volta[k]['RA']["0"])
        v_max = max(volta_ra)
        v_min = min(volta_ra)
        range_min = max(v_min, p_min)
        range_max = min(v_max, p_max)
        range_min = -9999
        range_max = 9999
        # plt.figure()

        vst = get_average(v['ST']["0"])
        volta_st = get_average(result_volta[k]['ST']["0"])
        ks = sorted(k.split('+'))
        if v[ks[0]][1] - v[ks[0]][0] > v[ks[1]][1] - v[ks[1]][0]:
            ks[0] = "*" + ks[0] + "*"
        else:
            ks[1] = "*" + ks[1] + "*"
        aaa = axs[idx // 4, idx % 4]
        # aaa = plt
        # plt.figure()
        # for tag in TAG_ORDER:
        def check(res, sel, st, ra, name, id="0"):
            # if sel and tag not in sel[k]:
            #     return
            # if tag not in res or tag == "ST":
            #     return
            a = np.array(st) / np.array(res)
            a -= 1
            lab = sel + "use"
            if id == "xx":
                fmt = plot_shape[lab][name]
            else:
                fmt = '.'
            if lab == "NTuse":
                lab = "Naive"
            lab += "(" + name + ")"
            arr1inds = ra.argsort()
            sorted_arr1 = ra[arr1inds[::-1]]
            a = a[arr1inds[::-1]]
            range_arr = (sorted_arr1 <= range_max) & (sorted_arr1 >= range_min)
            # lab = tag
            # sorted_arr2 = arr2[arr1inds[::-1]]
            if (len(a) > 40):
                sorted_arr1 = sorted_arr1[range_arr][::int(len(a)//min_length)]
                a = a[range_arr][::int(len(a)//min_length)]
            else:
                sorted_arr1 = sorted_arr1[range_arr]
                a = a[range_arr]
            if idx == 0:
                aaa.plot(sorted_arr1, a, fmt, label=lab, markersize=5, zorder=5)
            else:
                aaa.plot(sorted_arr1, a, fmt, markersize=5, zorder=5)
            # aaa.axvline(x=r1[k]["RA"]["0"], ymin=0, ymax=1, linewidth=1, color='r', linestyle='dashed')
            # aaa.axvline(x=v1[k]["RA"]["0"], ymin=0, ymax=1, linewidth=1, color='b', linestyle='dashed')
            print(k)

            yyy = np.average(a)
            offset = {
                "Batchnorm+Im2Col": {
                    "HFuse(1080Ti)": -0.01
                },
                "Hist+Upsample": {
                    "VFuse(1080Ti)": 0.01
                },
                "Blake256+Ethash": {
                    "VFuse(1080Ti)": 0.01
                },
                "Blake2B+SHA256": {
                    "HFuse(V100)": 0.012
                }
            }
            if k in offset and lab in offset[k]:
                yyy += offset[k][lab] / 10 * 2.6
            
            if "Naive" not in lab:
                aaa.axhline(y=(yyy), xmin=0, xmax=1, linewidth=1, color=fmt[-1], linestyle='dashed', zorder=10)
            # aaa.axhline(y=v1[k]["RA"]["0"], xmin=0, xmax=1, linewidth=1, color='b', linestyle='dashed')
            # print(k, lab, np.average(a))
        # for iii, bst in v["LB+HF"].items():
        #     check(get_average(bst), "HF", vst, vra, "Pascal", iii)
        check(get_best(v, "HF"), "HF", vst, vra, "1080Ti", "xx")
        check(get_best(v, "VF"), "VF", vst, vra, "1080Ti", "xx")
        check(get_best(result_volta[k], "HF"), "HF", volta_st, volta_ra, "V100", "xx")
        check(get_best(result_volta[k], "VF"), "VF", volta_st, volta_ra, "V100", "xx")
        if k in split:
            check(get_best_naive(v, "HF", str(split[k])), "NT", vst, vra, "1080Ti", "xx")
            check(get_best_naive(result_volta[k], "HF", str(split[k])), "NT", volta_st, volta_ra, "V100", "xx")

        # check(result_volta[k], selection_volta, volta_st, volta_ra, "Volta")
        # plt.legend()
        aaa.set_title("+".join(ks))
        # plt.title(k)
        # plt.show()
        # axs[idx // 4, idx % 4].legend()
        # plt.show()
        idx += 1
    # fig.
    # fig.show()
    # fig.legend(loc=8, ncol=4)
    fig.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.43, 0.025))
    # fig.legend()
    # fig.tight_layout()
    for i in range(16):
        # if i == 8:
        #     continue
        ax = axs[i // 4, i % 4]
        if i % 4 == 0:
            ax.set(ylabel='Speedup')
        if int(i // 4) == 3:
            print(i)
            ax.set(xlabel='Ratio')
        # if int(i // 4) == 2 and int (i % 4) == 1:
        #     ax.set(ylabel='Speedup')
        # if int(i // 4) == 1 and int (i % 4) == 0:
        #     ax.set(xlabel='Ratio')
        # if int(i // 4) == 1 and int (i % 4) == 3:
        #     ax.set(xlabel='Ratio')

    # axs[2, 3].set_visible(False)
    # axs[2, 0].set_visible(False)
    # set_visible
    # fig.align_ylabels(axs)
    plt.savefig('fused.png', quality=100, dpi=400, bbox_inches='tight', pad_inches=0)
    # plt.show()

def generate_table_1(result):
    s = " "
    st = 0

    def color_str(time):
        if time < st:
            return "\\textcolor{green}{%.2f}" % time
        elif time > st:
            return "\\textcolor{red}{%.2f}" % time
        else:
            return "\\textcolor{black}{%.2f}" % time
    for tag in TAG_ORDER:
        s += "& " + tag
    s += "\\\\ \n \\hline\n"
    for k, v in result.items():
        if "+" not in k:
            print(k)
            print(v)
            continue
        s += k + " &"
        st = v["ST"]
        for tag in TAG_ORDER:
            # if tag == "ST":
            #     st = v[tag]
            if tag not in v:
                s += " * &"
            else:
                s += color_str(v[tag]) + " &"
        s = s[:-1]
        s += "\\\\ \n \\hline\n"
    print(s)
    # print(su / suc)

# generate_table_1(r1)
# build_graph(r1_s, pascal_selection, v1_s, volta_selection)
build_graph(r1_s, None, v1_s, None)
#  generate_table_1(r2)


def generate_table_3(extime, metrics, e2, m2):
    def color_str(prec):
        prec *= 100
        if prec > 0:
            return "\\textcolor{green}{%.1f} &" % prec
        else:
            return "\\textcolor{red}{%.1f} &" % prec
    s = ""
    for k, v in metrics.items():
        v2 = m2[k]
        if "+" in k:
            continue
        s += "\\hline\n"
        tag = "ST"
        tag_order = ["ST"]
        s += k + " &"
        # s += "ST" + " &"
        # speed = extime[k]["ST"] / extime[k][tag] - 1
        s += "%.2f / %.2f &" % (extime[k]["ST"]["0"], e2[k]["ST"]["0"])
        s += "%.2f / %.2f &" % (
            v[tag]["0"]["issue_slot_utilization"],
            v2[tag]["0"]["issue_slot_utilization"],
        )
        # s += " &"
        s += "%.1f / %.1f & %.1f / %.1f \\\\\n" % (
            v[tag]["0"]["stall_memory_dependency"], 
            v2[tag]["0"]["stall_memory_dependency"], 
            v[tag]["0"]["achieved_occupancy"],
            v2[tag]["0"]["achieved_occupancy"]
            )
    print(s)

def generate_table_2(extime, metrics, e2, m2):
    def build_tag(tags, kernel):
        t = next(filter(lambda x: "HF" in x, tags))
        t = set(t.split("+"))
        lb = "LB" in t
        if "LB" in t:
            t.remove("LB")
        yield build_tags(t)
        t.add("LB")
        yield build_tags(t)
        

        # if "LB" in t:
        #     yield build_tags(t)
        #     t.remove("LB")
        #     yield build_tags(t)
        #     tag.add("LB")
        # else:
        #     t.add("LB")
        #     yield build_tags(t)
        #     t.remove("LB")
        #     yield build_tags(t)
        # if "Hist" in kernel:
        #     if "LB" in t:
        #         yield "LB+HF"
        #     else:
        #         yield "HF"
        
    def color_str(t1, t2):
        prec = (t1 / t2 - 1) * 100
        # prec *= 100
        if prec > 0:
            return "\\textcolor{green}{%.1f} " % prec
        else:
            return "\\textcolor{red}{%.1f} " % prec
    s = ""
    for k in order:
        v = metrics[k]
        v2 = m2[k]
        if "+" not in k:
            continue
        s += "\\hline\n"
        tag_order = list(build_tag(pascal_selection[k], k))
        s += "\\multirow{" + str(len(tag_order)) + "}{*}{" + k + "} &"
        native = False
        idx = 0
        for tag in build_tag(pascal_selection[k], k):
            if native:
                s += "\\cline{2-4} \\cline{6-7}\n"
                s += " &"
            if idx == 0:
                s += "N-RegCap" + " &"
            elif idx == 1:
                s += "RegCap" + " &"
            elif idx == 2:
                s += "Naive" + " &"
            # else:
            #     s += "BarSync" + " &"
            
            idx += 1
            # speed = extime[k]["ST"], extime[k][tag] - 1
            if k in split:
                iii = str(split[k])
                ii2 = ""
                def find_(extime):
                    ii = ""
                    ext = 99999999999999999999999999999999999
                    for ikey in extime[k][tag]:
                        if extime[k]["HF"][ikey] < ext:
                            ext = extime[k]["HF"][ikey]
                            ii = ikey
                        if extime[k]["LB+HF"][ikey] < ext:
                            ext = extime[k]["LB+HF"][ikey]
                            ii = ikey
                    return ii
                iii = find_(extime)
                ii2 = find_(e2)
                
            else:
                iii = "0"
                ii2 = "0"
            s += color_str(extime[k]["ST"]["0"], extime[k][tag][iii])
            s += "/ " + color_str(e2[k]["ST"]["0"], e2[k][tag][ii2]) + " & "
            s += "%.2f / " % v[tag][iii]["issue_slot_utilization"]
            s += "%.2f & " % v2[tag][ii2]["issue_slot_utilization"]
            if not native:
                kernels = k.split('+')
                def compute(me):
                    total_time = 0
                    total_util = 0
                    for kn in kernels:
                        ec = me[kn]['ST']["0"]['elapsed_cycles_pm']
                        isu = me[kn]['ST']["0"]['issue_slot_utilization']
                        total_util += ec * isu
                        total_time += ec
                    return total_util / total_time
                s += "\\multirow{" + str(len(tag_order)) + "}{*}{%.2f / %.2f} &" % (compute(metrics), compute(m2))
                native = True
            else:
                s += " &"
            s += "%.1f / %.1f & %.1f / %.1f \\\\\n" % (
                v[tag][iii]["stall_memory_dependency"], 
                v2[tag][ii2]["stall_memory_dependency"], 
                v[tag][iii]["achieved_occupancy"],
                v2[tag][ii2]["achieved_occupancy"])
    print(s)
        # for tag, value in v.items():
        #     pass
# analyze("./data/ml-event.csv")
# analyze("./data/ml-spill-event.csv")
# analyze("./data/ml-metrics.csv")
# analyze("./data/ml-spill-metrics.csv")
m1 = {}
analyze("./data-new/crypto-events.csv", m1)
analyze("./data-new/crypto-metrics.csv", m1)
analyze("./data-new/ml_perf/ml-events-pascal.csv", m1, upsample)
analyze("./data-new/ml_perf/ml-metrics-pascal.csv", m1, upsample)
analyze("./data-new/ml_perf/upsample-events-pascal.csv", m1)
analyze("./data-new/ml_perf/upsample-metrics-pascal.csv", m1)
m2 = {}
analyze("./data-new/crypto/crypto-events-volta.csv", m2)
analyze("./data-new/crypto/crypto-metrics-volta.csv", m2)
analyze("./data-new/ml_perf/ml-events-volta.csv", m2, upsample)
analyze("./data-new/ml_perf/ml-metrics-volta.csv", m2, upsample)
analyze("./data-new/ml_perf/upsample-events-volta.csv", m2)
analyze("./data-new/ml_perf/upsample-metrics-volta.csv", m2)

generate_table_2(r1, m1, v1, m2)
generate_table_3(r1, m1, v1, m2)

exit(0)
r2 = {}
# analyze("./data/barsync-spill-event.csv", r2)
# analyze("./data/barsync-spill-metrics.csv", r2)
# analyze("./data/ml-spill-metrics.csv")
#  analyze("./data/ethminer_event.csv")
#  analyze("./data/ethminer_metrics.csv")
#  analyze("./data/ethminer_spill_event.csv")
#  analyze("./data/ethminer_spill_metrics.csv")
# analyze("./data/ml-naive-event.csv")
# analyze("./data/ml-naive-metric.csv")

header_printed = False

t3 = ""

print(result)
for o in order:
    s1 = ""
    s2 = ""
    s3 = ""
    key = kernels[o]
    if "_" in key and "bar" not in key:
        continue
    if key not in result:
        continue
    m1 = result[key]
    m2 = result[key + "_p"]
    naive = "Hist" in key and "+" in key and "bar" not in key
    if naive:
        m3 = result[key + "_n"]
    #  for key, m in result.items():
    d = "Pairs & "
    multirow = "3" if naive else "2"
    if "+" in key:
        s1 += "\\multirow{" + multirow + "}{*}{{" + key.replace("_", "") + "}} & "
    else:
        s1 += "{" + key + "} & "
    s2 += " & "
    s3 += " & "

    if "+" in key:
        s1 += " N-RegCap & "
        s2 += " RegCap & "
        s3 += " Naive & "
    for k in event_order:
        if events[k] == "Cycles":
            d += "Time" + " & "
            def color_str(prec):
                prec *= 100
                if prec > 0:
                    return "\\textcolor{green}{%.2f} & " % prec
                else:
                    return "\\textcolor{red}{%.1f} & " % prec
            # s2 += ("%.1f" % r2[key]) + " & "
            if "+" in key:
                if "bar_sync" in key:
                    stream_key = key[:-9] + "_s"
                else:
                    stream_key = key + "_s"
                # s2 += color_str(r1[stream_key] / r2[key] - 1)
                # s1 += color_str(r1[stream_key] / r1[key] - 1)
                s2 += color_str(r2[key]/100)
                s1 += color_str(r1[key]/100)
                if naive:
                    perc = r1[key + "_s"] / r2[key + "_n"] - 1
                    s3 += color_str(perc)
                    # if r1[key + "_s"] > r2[key + "_n"]:
                    #     s3 += "\\textcolor{green}{%.1f} & " % r2[key+"_n"]
                    # else:
                    #     s3 += "\\textcolor{red}{%.1f} & " % r2[key+"_n"]
            else:
                s1 += ("%.1f" % r1[key]) + " & "
            continue
        v1 = m1[k]
        v2 = m2[k]
        d += events[k] + " & "
        s1 += ("%.2f" % v1) + " & "
        s2 += ("%.2f" % v2) + " & "
        if naive:
            s3 += ("%.2f" % m3[k]) + " & "
        if "+" in key and k == "issue_slot_utilization":
            [k1, k2] = key.split("+")
            if 'bar' in k2:
                k2 = k2[:-9]
            ek1 = result[k1]['elapsed_cycles_pm']
            ek2 = result[k2]['elapsed_cycles_pm']
            wk1 = result[k1]['issue_slot_utilization']
            wk2 = result[k2]['issue_slot_utilization']
            d += " Stream Slot Utilization & "

            # if "Maxpool+Upsample" in key:
            #     ## See comments above, maxpool uses different configurations because of memory limitation
            #     s1 += "\\multirow{" + multirow + "}{*}{%.2f} &" % (14.37)
            # else:
            s1 += "\\multirow{" + multirow + "}{*}{%.2f} &" % ((ek1 * wk1 + ek2 * wk2) / (ek1 + ek2))
            s2 += " & "
            s3 += " & "

    s1 = s1[:-2]
    s2 = s2[:-2]
     # d = d[:-2]
    s1 += " \\\\"
    s2 += " \\\\"
    if not header_printed:
        header_printed = True
        print(d + " \\\\ \n \\hline")
    if "+" in key:
        print(s1)
        print("\\cline{2-4} \\cline{6-7}")
        print(s2)
        if naive:
            print("\\cline{2-4} \\cline{6-7}")
            print(s3[:-2] + "\\\\")
        print("\\hline")
    else:
        t3 += s1 + "\n"

print(t3)
