import csv
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

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
    "sia_blake2b_gpu_hash": "SIA",
    "sha256d_gpu_hash_shared": "SHA256",
    "ethash_search": "Ethash",
    "blake2b_gpu_hash": "Blake",
    "kernelHistogram1D": "Hist",
    "im2col_kernel": "Im2Col",
    "MaxPoolForward": "Maxpool",
    "batch_norm_collect_statistics_kernel": "Batchnorm",
    "upsample_bilinear2d_out_frame": "Upsample",
}

tags = {
    "_lb": "LB",
    "_bar_sync": "BS",
    "_vfuse": "VF",
    "_hfuse": "HF",
    "idx_1": "OP",
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
    # "HF+PP", "HF+OP+PP", "LB+HF+PP", "LB+BS+HF+PP", "BS+HF+PP", "LB+HF+OP+PP"
]

###
# note maxpool issue slot util = 6.301776
#      upsample = 2072924527
#      mp clock = 1453350987
#      up sample clock = 2072924527

###
result = {}

LABEL = "Avg"

def build_name(names):
    return "+".join(sorted(names))

def find_name(name):
    fused = "_fused_kernel" in name
    r = []
    t = ""
    for k, v in kernels.items():
        if k in name:
            if not fused:
                return v, "ST"
            else:
                r.append(v)
                # if not r:
                #     r = v
                # else:
                #     r += "+" + v
    for k, v in tags.items():
        if k in name:
            if not t:
                t = v
            else:
                t += "+" + v
    return build_name(r), t

def analyze(file_name, result):
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key, tag = find_name(row['Kernel'])
            if not key:
                continue
            # key += app
            if key not in result:
                result[key] = {}
            if tag not in result[key]:
                result[key][tag] = {}
            if 'event' in file_name:
                result[key][tag][row['Event Name']] = float(row[LABEL].strip("%"))
            else:
                if row['Metric Name'] == "achieved_occupancy":
                    result[key][tag][row['Metric Name']] = float(row[LABEL].strip("%")) * 100
                else:
                    result[key][tag][row['Metric Name']] = float(row[LABEL].strip("%"))

found_tags = set()

def analyze_execution_time(f):
    visited_ts = set()
    time_result = {}
    time_result_count = {}
    time_result_separate = {}

    def update_time_result(key, tag, time):
        if key not in time_result:
            time_result[key] = {}
            time_result_count[key] = {}
            time_result_separate[key] = {}
        if tag in time_result[key]:
            time_result[key][tag] += time
            time_result_count[key][tag] += 1
            time_result_separate[key][tag].append(time)
        else:
            time_result[key][tag] = time
            time_result_count[key][tag] = 1
            time_result_separate[key][tag] = [time]

    with open(f) as json_file:
        data = json.load(json_file)
        prev = []
        prev_time = None
        last_time = None
        prev_dur = None
        for event in data['traceEvents']:
            name = event['name']
            key, tag = find_name(name)
            if not key:
                continue
            if 'ts' not in event or event['ts'] in visited_ts:
                continue
            found_tags.add(tag)
            visited_ts.add(event['ts'])
            update_time_result(key, tag, event['dur'])
            if tag == "ST":
                prev.append(key)
                if not prev_time:
                    prev_time = event['ts']
                    prev_dur = event['dur']
                last_time = event['ts'] + event['dur']
                if len(prev) == 2:
                    name = build_name(prev)
                    update_time_result(name, "ST", last_time - prev_time)
                    update_time_result(name, "RA", prev_dur / event['dur'] if prev[0] < prev[1] else event['dur'] / prev_dur )
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

    for x, y in time_result.items():
        for k, v in y.items():

            y[k] = v / 1000000 / time_result_count[x][k]
    return time_result, time_result_separate

# r1 = {}
# r2 = {}
r1, r1_s = analyze_execution_time("./data-new/ml-volta-chart.json")
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

def build_graph(result):
    for k, v in result.items():
        if "+" not in k:
            continue
        plt.figure()
        ks = sorted(k.split('+'))
        if v[ks[0]][1] - v[ks[0]][0] > v[ks[1]][1] - v[ks[1]][0]:
            ks[0] = "*" + ks[0] + "*"
        else:
            ks[1] = "*" + ks[1] + "*"
        for tag in TAG_ORDER:
            if tag not in v or tag == "ST":
                continue
            a = np.array(v["ST"]) / np.array(v[tag]) 
            a -= 1
            plt.plot(v['RA'], a, '.', label=tag)
        plt.legend()
        plt.xlabel("ratio: " + ":".join(ks))
        plt.ylabel("speed up")
        plt.title(k)
        plt.savefig(k+'.png', quality=100, figsize=(1280, 960), dpi=400)
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

generate_table_1(r1)
build_graph(r1_s)
#  generate_table_1(r2)


def generate_table_3(extime, metrics):
    def color_str(prec):
        prec *= 100
        if prec > 0:
            return "\\textcolor{green}{%.1f} &" % prec
        else:
            return "\\textcolor{red}{%.1f} &" % prec
    s = ""
    for k, v in metrics.items():
        if "+" in k:
            continue
        s += "\\hline\n"
        tag = "ST"
        tag_order = ["ST"]
        s += k + " &"
        s += "ST" + " &"
        # speed = extime[k]["ST"] / extime[k][tag] - 1
        s += "%.2f &" % extime[k]["ST"]
        s += "%.2f &" % v[tag]["issue_slot_utilization"]
        s += " &"
        s += "%.1f & %.1f \\\\\n" % (v[tag]["stall_memory_dependency"], v[tag]["achieved_occupancy"])
    print(s)

def generate_table_2(extime, metrics):
    def color_str(t1, t2):
        # prec *= 100
        if t2 < t1:
            return "\\textcolor{green}{%.2f} &" % t2
        else:
            return "\\textcolor{red}{%.2f} &" % t2
    s = ""
    for k, v in metrics.items():
        if "+" not in k:
            continue
        s += "\\hline\n"
        tag_order = list(filter(lambda x: x in v.keys() and x != "ST" and "VF" not in x, TAG_ORDER))
        s += "\\multirow{" + str(len(tag_order)) + "}{*}{" + k + "} &"
        native = False
        for tag in tag_order:
            if native:
                s += "\\cline{2-4} \\cline{6-7}\n"
                s += " &"
            s += tag + " &"
            # speed = extime[k]["ST"], extime[k][tag] - 1
            s += color_str(extime[k]["ST"], extime[k][tag])
            s += "%.2f &" % v[tag]["issue_slot_utilization"]
            if not native:
                kernels = k.split('+')
                total_time = 0
                total_util = 0
                for kn in kernels:
                    ec = metrics[kn]['ST']['elapsed_cycles_pm']
                    isu = metrics[kn]['ST']['issue_slot_utilization']
                    total_util += ec * isu
                    total_time += ec
                s += "\\multirow{" + str(len(tag_order)) + "}{*}{%.2f} &" % (total_util / total_time)
                native = True
            else:
                s += " &"
            s += "%.1f & %.1f \\\\\n" % (v[tag]["stall_memory_dependency"], v[tag]["achieved_occupancy"])
    print(s)
        # for tag, value in v.items():
        #     pass
# analyze("./data/ml-event.csv")
# analyze("./data/ml-spill-event.csv")
# analyze("./data/ml-metrics.csv")
# analyze("./data/ml-spill-metrics.csv")
m1 = {}
analyze("./data-new/ml-events.csv", m1)
analyze("./data-new/ml-metrics.csv", m1)

generate_table_2(r1, m1)
generate_table_3(r1, m1)

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
