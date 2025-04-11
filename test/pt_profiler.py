import json
import argparse
import sys


def analyze_high_level(log_json):
    # Check if the JSON data is a dictionary or a list
    if isinstance(log_json, dict):
        print("Keys of a dictionary:", list(log_json.keys()))
        for key in log_json.keys():
            print(key)
            analyze_high_level(log_json[key])
    elif isinstance(log_json, list):
        print("Length of a list:", len(log_json))
    else:
        print("JSON contains data of type:", type(log_json))


def analyze_trace_events(trace_events):
    # for event in trace_events[:2]:
    #     analyze_high_level(event)
    # analyze_high_level(trace_events[0])

    for event in trace_events:
        # if "YJH_PROFILE_Transformer_Layer_0" in event.get("name"):
        if "YJH_PROFILE" in event.get("name"):
            print("Found YJH_PROFILE in event name:", event["name"])
            for k, v in event.items():
                print(f"Key: {k}, Value: {v}")


def find_record_functions(trace_events):
    # Initialize a dictionary to hold the wrappers
    wrappers = {}

    # Iterate through the trace events
    for event in trace_events:
        # Check if the event has the 'args' key and if it contains 'wrapper'
        if "args" in event and "wrapper" in event["args"]:
            wrapper_name = event["args"]["wrapper"]
            # Add the wrapper to the dictionary
            wrappers[wrapper_name] = event

    # Print the found wrappers
    for wrapper_name, wrapper_event in wrappers.items():
        print(f"Wrapper: {wrapper_name}, Event: {wrapper_event}")


def seperate_events(log):
    cpu_ops = []
    kernel_events = []
    flow_events = []

    # Check if the JSON data is a list
    if not isinstance(log, list):
        raise ValueError("log should be a list")

    for event in log:
        if "cat" in event:
            # Check if the event is a CPU operation
            if event["ph"] == "X":
                if event["cat"] == "cpu_op":
                    cpu_ops.append(event)
                elif event["cat"] == "kernel":
                    kernel_events.append(event)
            elif event["ph"] == "f" or event["ph"] == "s" or event["ph"] == "t":
                flow_events.append(event)
    return cpu_ops, kernel_events, flow_events


def collect_record_functions(log):
    record_functions = []
    fwd_results = []
    # Check if the JSON data is a list
    if not isinstance(log, list):
        raise ValueError("log should be a list")

    for event in log:
        if "cat" in event:
            if event["cat"] == "user_annotation" and "YJH_PROFILE" in event["name"]:
                record_functions.append(event)
            elif (
                event["cat"] == "gpu_user_annotation" and "YJH_PROFILE" in event["name"]
            ):
                fwd_results.append(event)
    return record_functions, fwd_results


def collect_forward_cpu_ops(cpu_ops, record_function):
    forward_cpu_ops = []
    # Check if the JSON data is a list
    if not isinstance(record_function, dict):
        raise ValueError("record_function should be a dict")
    start = record_function["ts"]
    end = start + record_function["dur"]
    seq_num = []

    for event in cpu_ops:
        if (
            event["ts"] >= start
            and event["ts"] <= end
            and "args" in event
            and "Sequence number" in event["args"]
        ):
            if event["args"]["Sequence number"] in seq_num:
                # Remove the event already in the list
                for i in range(len(forward_cpu_ops)):
                    if (
                        forward_cpu_ops[i]["args"]["Sequence number"]
                        == event["args"]["Sequence number"]
                    ):
                        del forward_cpu_ops[i]
                        break
            else:
                seq_num.append(event["args"]["Sequence number"])
            forward_cpu_ops.append(event)
    return forward_cpu_ops


def collect_backward_cpu_ops(cpu_ops, forward_cpu_ops):
    backward_cpu_ops = []
    times = {"start": sys.float_info.max, "end": 0}
    for forward_cpu_op in forward_cpu_ops:
        if "args" in forward_cpu_op and "Sequence number" in forward_cpu_op["args"]:
            seq_num = forward_cpu_op["args"]["Sequence number"]
            candidates = []
            for event in cpu_ops:
                if (
                    "args" in event
                    and "Sequence number" in event["args"]
                    and event["args"]["Sequence number"] == seq_num
                    # and "evaluate_function" not in event["name"]
                    and "evaluate_function" in event["name"]
                    and event["name"] != forward_cpu_op["name"]
                    and "Backward" in event["name"]
                ):
                    candidates.append(event)
                    # backward_cpu_ops.append(event)
                    # break
            # assert len(candidates) == 1 or len(candidates) == 2
            # if len(candidates) == 2:
            #     candidates.sort(key=lambda x: x["ts"])
            #     backward_cpu_ops.append(candidates[0])
            # else:
            #     backward_cpu_ops.append(candidates[0])
            assert len(candidates) == 1 
            backward_cpu_ops = backward_cpu_ops + candidates
            times["start"] = min(times["start"], candidates[0]["ts"])
            times["end"] = max(times["end"], candidates[0]["ts"] + candidates[0]["dur"])

    return times, backward_cpu_ops


def collect_flow_events(times, flow_events):
    collected_flow_events = []
    start = times["start"]
    end = times["end"]
    for event in flow_events:
        if (
            event["ph"] == "s"
            and event["cat"] == "ac2g"
            and event["ts"] >= start
            and event["ts"] <= end
        ):
            collected_flow_events.append(event)
    return collected_flow_events


def collect_kernel_events(collected_flow_events, kernel_events):
    kernel_times = {"start": sys.float_info.max, "end": 0}
    kernels_record_function = []
    flow_event_ids = [event["id"] for event in collected_flow_events]

    for event in kernel_events:
        if "args" in event and event["args"]["correlation"] in flow_event_ids:
            kernels_record_function.append(event)
            kernel_times["start"] = min(kernel_times["start"], event["ts"])
            kernel_times["end"] = max(kernel_times["end"], event["ts"] + event["dur"])

    return kernel_times, kernels_record_function


def test_main():
    # Load JSON file
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    with open(args.logfile, "r", encoding="utf-8") as file:
        # Load the JSON data
        log_json = json.load(file)

    # [x] Seperate trace events into cpu_op, kernel, and flow events
    cpu_ops, kernel_events, flow_events = seperate_events(log_json["traceEvents"])

    # [x] Find the YJH_PROFILE events(record functions)
    record_functions, fwd_results = collect_record_functions(log_json["traceEvents"])

    # Test for all layers considering only forward and backward
    for idx, record_function in enumerate(record_functions):
        # [x] Colloct all the forward cpu_op events in a YJH_PROFILE event
        forward_cpu_ops = collect_forward_cpu_ops(cpu_ops, record_function)

        # [x] Find the corresponding backward cpu_op events using fwdbwd flow events
        times, backward_cpu_ops = collect_backward_cpu_ops(
            cpu_ops, forward_cpu_ops
        )

        # [x] Collect all the flow events with "ac2g"+"s"
        collected_flow_events = collect_flow_events(times, flow_events)

        # [x] "ac2g" + "s" -> "correlation" -> "kernel" events
        kernel_times, kernels_record_function = collect_kernel_events(
            collected_flow_events, kernel_events
        )
        if "YJH_PROFILE_Transformer_Layer_0" in  record_function["name"]:
            print(record_function["name"])
            for fwd_result in fwd_results:
                if (
                    fwd_result["name"] == record_function["name"]
                    and fwd_result["args"]["External id"]
                    == record_function["args"]["External id"]
                ):
                    print(f"Forward duration: {fwd_result["dur"] / 1000:.3f} ms", )
                    break
            back_dur = kernel_times["end"] - kernel_times["start"]
            print(f"Backward duration: {back_dur / 1000:.3f} ms")
            # sort kernels_record_function by "ts" in "args"
            # kernels_record_function.sort(key=lambda x: x["ts"])
            # for event in kernels_record_function:
            #     print(event)
            #     print("")
            print("=======================================")


def find_top_api(subject, cpu_ops):
    found = False
    # forward=False
    found_start_ts = subject["ts"]
    found_end_ts = found_start_ts + subject["dur"]
    first_top = subject
    second_top = subject
    for co in cpu_ops:
        if co["ts"] <= found_start_ts and co["ts"] + co["dur"] >= found_end_ts:
            second_top = first_top
            first_top = co
            found = True
            found_start_ts = co["ts"]
            found_end_ts = co["ts"] + co["dur"]
            # in case of forward kernels
            if "YJH_PROFILE" in co["name"]:
                break
    if found == False:
        print("ERROR3")
    return first_top, second_top


def main():
    test_main()
    return
    # Load JSON file
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    file = open(args.logfile)
    jsonString = json.load(file)["traceEvents"]
    cpu_ops = []
    for idx, l in enumerate(jsonString):
        if "cat" in l and l["cat"] == "cpu_op" and "ts" in l and "dur" in l:
            cpu_ops.append(l)

    gpu_kernels = []
    # corresponding_bw_top=list()
    corresponding_bw_top = dict()
    for idx, l in enumerate(jsonString):
        if "cat" in l and l["cat"] == "fwdbwd" and "ph" in l and l["ph"] == "s":
            if "ph" not in jsonString[idx + 1] or jsonString[idx + 1]["ph"] != "f":
                print("ERROR4")
            if jsonString[idx - 1]["ts"] != jsonString[idx + 1]["ts"]:
                print("ERROR6")

            for idx2, l2 in enumerate(jsonString):
                if "cat" in l2 and l2["cat"] == "cpu_op" and l2["ts"] == l["ts"]:
                    forward_top, _ = find_top_api(l2, cpu_ops)
                    name = (
                        forward_top["name"]
                        if "YJH_PROFILE" in forward_top["name"]
                        else "YJH_MISC"
                    )
                    backward_top, _ = find_top_api(jsonString[idx - 1], cpu_ops)
                    # corresponding_bw_top.append([name,backward_top])   # forward top name, backward top
                    if name not in corresponding_bw_top:
                        corresponding_bw_top[name] = list()
                    corresponding_bw_top[name].append(backward_top)
                    break
        if "cat" in l and l["cat"] == "Kernel":
            if jsonString[idx + 1]["ph"] != "f":
                print("ERROR")
                exit(0)
            found = False
            for ii in range(idx + 2, len(jsonString)):
                if (
                    jsonString[ii]["name"] == "cudaLaunchKernel"
                    and "dur" in jsonString[ii]
                ):
                    gpu_kernels.append([jsonString[ii], l])  # From(cpu) -to(gpu kernel)
                    found = True
                    break
            if found == False:
                print("ERROR2")

    print("# of gpu kernels: ", len(gpu_kernels))
    print("# of cpu kernels: ", len(cpu_ops))

    forward_result_count = 0
    backward_result_count = 0
    profile_results = dict()
    backward_stacks = list()
    for gk in gpu_kernels:
        first_top, second_top = find_top_api(gk[0], cpu_ops)
        # in case of forward kernels
        if "YJH_PROFILE" in first_top["name"]:
            if "[FORWARD]" + first_top["name"] not in profile_results:
                profile_results["[FORWARD]" + first_top["name"]] = list()
            profile_results["[FORWARD]" + first_top["name"]].append(gk)
            forward_result_count = forward_result_count + 1
        # in case of backward kernels
        else:
            found = False
            for k, v in corresponding_bw_top.items():
                for vv in v:
                    if first_top == vv:
                        found = True
                        key = "[BACKWARD]" + k if k != "YJH_MISC" else "YJH_MISC"
                        if key not in profile_results:
                            profile_results[key] = list()
                        profile_results[key].append(gk)
                        backward_result_count = backward_result_count + 1
                        break
                if found == True:
                    break
            if found == False:
                # print("1 ", first_top)
                # print("2 ", second_top)
                # print("3 ", gk[1])
                if "YJH_MISC" not in profile_results:
                    profile_results["YJH_MISC"] = list()
                profile_results["YJH_MISC"].append(gk)
    print("forward_result_count: ", forward_result_count)
    print("backward_result_count: ", backward_result_count)
    final = dict()
    total_time = 0
    for key, val in profile_results.items():
        profile_time = 0
        for v in val:
            profile_time = profile_time + v[1]["dur"]
        final[key] = profile_time
        total_time = total_time + profile_time
    for k, v in final.items():
        print(k, " ", v / 1000, "ms", v / total_time * 100, "%")
    print("Total Time", total_time / 1000, "ms")
    # for k,v in corresponding_bw_top.items():
    #    print(k)
    #    for vv in v:
    #        print(vv["name"])


if __name__ == "__main__":
    main()
