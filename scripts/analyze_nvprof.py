import csv
import sys
import json

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
]

kernels = {
    "sia_blake2b_gpu_hash(": "SIA",
    "sha256d_gpu_hash_shared(": "SHA256",
    "ethash_search(": "Ethash",
    "blake2b_gpu_hash(": "Blake",

    "ethash_search2_blake2b_gpu_hash_0": "Ethash+Blake",
    "sia_blake2b_gpu_hash_ethash_search4_0": "Ethash+SIA",
    "sha256d_gpu_hash_shared_ethash_search3_0": "Ethash+SHA256",

    "blake2b_gpu_hash_sia_blake2b_gpu_hash_0": "Blake+SIA",
    "blake2b_gpu_hash_sha256d_gpu_hash_shared_0": "Blake+SHA256",
    "sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_0":"SIA+SHA256",

    "sia_blake2b_gpu_hash_ethash_search4_100": "Ethash+SIA_",
    "ethash_search2_blake2b_gpu_hash_100": "Ethash+Blake_",
    "sha256d_gpu_hash_shared_ethash_search3_100": "Ethash+SHA256_",

    "blake2b_gpu_hash_sia_blake2b_gpu_hash_100": "Blake+SIA_",
    "blake2b_gpu_hash_sha256d_gpu_hash_shared_100": "Blake+SHA256_",
    "sha256d_gpu_hash_shared_sia_blake2b_gpu_hash_100": "SIA+SHA256_",

    "kernelHistogram1D" : "Hist",
    "im2col_kernel" : "Im2Col",
    "MaxPoolForward": "Maxpool",
    "batch_norm_collect_statistics_kernel": "Batchnorm",
    "upsample_bilinear2d_out_frame": "Upsample",

    "im2col_kernel_MaxPoolForward_0": "Maxpool+Im2Col",
    "im2col_kernel_batch_norm_collect_statistics_kernel_0": "Batchnorm+Im2Col",
    "im2col_kernel_upsample_bilinear2d_out_frame_0": "Upsample+Im2Col",
    "MaxPoolForward_batch_norm_collect_statistics_kernel_0": "Maxpool+Batchnorm",
    "max_pool_upsample_kernel": "Maxpool+Upsample",
    "kernelHistogram1D_MaxPoolForward_11": "Maxpool+Hist",
    "kernelHistogram1D_batch_norm_collect_statistics_kernel_11": "Batchnorm+Hist",
    "im2col_kernel_kernelHistogram1D_0": "Im2Col+Hist",
    "kernelHistogram1D_upsample_bilinear2d_out_frame_11": "Upsample+Hist",
    "upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_0": "Upsample+Batchnorm",

    "im2col_kernel_MaxPoolForward_100": "Maxpool+Im2Col_",
    "im2col_kernel_batch_norm_collect_statistics_kernel_100": "Batchnorm+Im2Col_",
    "im2col_kernel_upsample_bilinear2d_out_frame_100": "Upsample+Im2Col_",
    "MaxPoolForward_batch_norm_collect_statistics_kernel_100": "Maxpool+Batchnorm_",
    "MaxPoolForward_upsample_bilinear2d_out_frame_100": "Maxpool+Upsample_",
    "kernelHistogram1D_MaxPoolForward_100": "Maxpool+Hist_",
    "kernelHistogram1D_batch_norm_collect_statistics_kernel_100": "Batchnorm+Hist_",
    "im2col_kernel_kernelHistogram1D_100": "Im2Col+Hist_",
    "kernelHistogram1D_upsample_bilinear2d_out_frame_100": "Upsample+Hist_",
    "upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_100": "Upsample+Batchnorm_",
}

event_order = [
    "elapsed_cycles_pm",
    "issue_slot_utilization",
    # "stall_inst_fetch",
    # "stall_exec_dependency",
    "stall_memory_dependency",
    "achieved_occupancy",
    "eligible_warps_per_cycle",
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

result =  {}

def find_name(kernel):
    candidate = False
    for name in order:
        if name in kernel:
            candidate = kernels[name]
    # if not candidate:
    #     print(kernel)
    return candidate

def analyze(file_name):
    app = ""
    if "spill" in file_name:
        app = "_p"
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = find_name(row['Kernel'])
            if not key:
                continue
            key += app
            if key not in result:
                result[key] = {}
            if 'event' in file_name:
                result[key][row['Event Name']] = float(row['Avg'].strip("%"))
            else:
                if row['Metric Name'] == "achieved_occupancy":
                    result[key][row['Metric Name']] = float(row['Avg'].strip("%")) * 100
                else:
                    result[key][row['Metric Name']] = float(row['Avg'].strip("%"))



def analyze_execution_time(f, res):
    visited_ts = set()
    time_result = {}
    time_result_count = {}
    with open(f) as json_file:
        data = json.load(json_file)
        prev = None
        prev_time = None
        for event in data['traceEvents']:
            name = event['name']
            key = find_name(name)
            if not key:
                continue
            if 'ts' not in event or event['ts'] in visited_ts:
                continue
            visited_ts.add(event['ts'])
            if key in time_result:
                time_result[key] += event['dur']
                time_result_count[key] += 1
            else:
                time_result[key] = event['dur']
                time_result_count[key] = 1
            if "+" not in key:
                if not prev:
                    prev = key
                    prev_time = event['ts']
                else:
                    if prev + "+" + key in kernels.values():
                        new_name = prev + "+" + key
                    else:
                        new_name = key + "+" + prev
                    new_name += "_s"
                    if new_name in time_result:
                        time_result[new_name] += event['ts'] + event['dur'] - prev_time
                        time_result_count[new_name] += 1
                    else:
                        time_result[new_name] = event['ts'] + event['dur'] - prev_time
                        time_result_count[new_name] = 1
                    prev = None
                    prev_time = None

    for x, y in time_result.items():
        res[x] = y / 1000000 / time_result_count[x]

r1 = {}
r2 = {}
analyze_execution_time("./data/ml.json", r1)
analyze_execution_time("./data/miner.json", r1)
analyze_execution_time("./data/miner_regcap.json", r2)
analyze_execution_time("./data/ml_regcap.json", r2)

fr = {}
for x, y in r1.items():
    if "_s" not in x:
        fr[x] = min(y, r2[x])
    else:
        fr[x] = y


def generate_table_1(result):
    s = ""

    for kernel in kernels.values():
        if "+" not in kernel:
            continue
        if "_" in kernel:
            continue
        if kernel not in result:
            continue
        prt = lambda p: "%.2f" % result[p]
        s += "\\textbf{" + kernel + "} & " + prt(kernel + "_s") \
             + " & " + prt(kernel + "_") \
             + " & " + prt(kernel)
        s += " & " + "%.2f" % (((result[kernel + "_s"] / result[kernel]) - 1) * 100)
        s += "\\\\ \n \\hline\n"
    print(s)

generate_table_1(fr)

analyze("./data/ml_event.csv")
analyze("./data/ml_metrics.csv")
analyze("./data/ml_spill_event.csv")
analyze("./data/ml_spill_metrics.csv")
analyze("./data/ethminer_event.csv")
analyze("./data/ethminer_metrics.csv")
analyze("./data/ethminer_spill_event.csv")
analyze("./data/ethminer_spill_metrics.csv")

header_printed = False

t3 = ""

print(result)
for o in order:
    s1 = ""
    s2 = ""
    key = kernels[o]
    if "_" in key:
        continue
    m1 = result[key]
    m2 = result[key + "_p"]
    #  for key, m in result.items():
    d = "Pairs & "
    if "+" in key:
        s1 += "\\multirow{2}{*}{\\textbf{" + key + "}} & "
    else:
        s1 += "\\textbf{" + key + "} & "
    s2 += " & "
    for k in event_order:
        if events[k] == "Cycles":
            d += "Time" + " & "
            def color_str(dic, k):
                if r1[k + "_s"] > dic[k]:
                    return "\\textcolor{green}{%.1f} & " % dic[k]
                else:
                    return "\\textcolor{red}{%.1f} & " % dic[k]
            # s2 += ("%.1f" % r2[key]) + " & "
            if "+" in key:
                s2 += color_str(r2, key)
                s1 += color_str(r1, key)
                # s1 += ("%.1f" % r1[key]) + " & "
            continue
        v1 = m1[k]
        v2 = m2[k]
        d += events[k] + " & "
        s1 += ("%.2f" % v1) + " & "
        s2 += ("%.2f" % v2) + " & "
        if "+" in key and k == "issue_slot_utilization":
            [k1, k2] = key.split("+")
            ek1 = result[k1]['elapsed_cycles_pm']
            ek2 = result[k2]['elapsed_cycles_pm']
            wk1 = result[k1]['issue_slot_utilization']
            wk2 = result[k2]['issue_slot_utilization']
            d += " Stream Slot Utilization & "
            s1 += "\\multirow{2}{*}{%.2f} &" % ((ek1 * wk1 + ek2 * wk2) / (ek1 + ek2))
            s2 += " & "

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
        print("\\cline{2-3} \\cline{5-7}")
        print(s2)
        print("\\hline")
    else:
        t3 += s1 + "\n"

print(t3)
