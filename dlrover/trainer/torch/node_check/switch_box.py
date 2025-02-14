import subprocess
import time
import argparse
import re
import os

cpu_nodes = 1

sys_layout_dir = "/sys/class/mxcd/mxcd/layout/"


def parse_int_pair(pair_str):
    try:
        a, b = pair_str.split(":")
        return (int(a), int(b))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{pair_str} is not a valid pair of integers separated by ':'"
        )

def get_num_cpu_nodes():
    try:
        with open(f"{sys_layout_dir}/properties", "r") as file:
            for line in file:
                if "num_cpu_nodes" in line:
                    parts = line.split()
                    if len(parts) > 1:
                        return int(parts[1])  # 返回num_cpu_nodes的值
    except FileNotFoundError:
        print(f"The file {sys_layout_dir}/properties does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_gpu_id_map():
    gpu_ip_map = {}
    for dirs in os.scandir(f"{sys_layout_dir}/nodes"):
        node_properties_path = f"{sys_layout_dir}/nodes/{dirs.name}/properties"
        try:
            with open(node_properties_path, "r") as file:
                for line in file:
                    if "mgpu_id" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            if int(parts[1]) not in gpu_ip_map:
                                gpu_ip_map[int(parts[1])] = int(dirs.name) - cpu_nodes
                            else:
                                print(
                                    f"GPUID {parts[1]} dupplicated in gpu_id_map "
                                    f"{gpu_ip_map} in {sys_layout_dir}/nodes"
                                )
                                exit(1)
        except FileNotFoundError:
            print(f"The file {node_properties_path} does not exist.")
            continue
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
    return gpu_ip_map


def is_slow(output, bandwidth: int) -> bool:
    bandwidth_pattern = r"bandwidth:(\d+\.\d+)MB/s"

    match = re.search(bandwidth_pattern, output)
    if match:
        if float(match.group(1)) > bandwidth:
            return False
    return True

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process pairs of integers.")

    # 添加 --pairs 参数来接收数组对
    parser.add_argument(
        "--pairs",
        metavar="PAIR",
        type=parse_int_pair,
        nargs="+",
        help="an array of pairs of integers separated by colons",
    )

    parser.add_argument(
        "--min-bandwidth",
        type=int,
        default=10000,
        help="The minimal bandwith (MB/s) of switchbox to GPU.",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=2,
        help="The minimal pass channels of switchbox to GPU.",
    )

    # 解析命令行参数
    args = parser.parse_args()

    exec_path = os.getenv("SWITCHBOX_CHECK_TOOL_PATH", "/opt/mxmap/bin/mxom-client")

    # 输入box
    global cpu_nodes
    cpu_nodes = get_num_cpu_nodes()
    gpu_ip_map = get_gpu_id_map()
    processes = []
    commands = []
    start = time.time()
    index = 1
    err_count: int = 0
    for pair in args.pairs:
        (socketA, socketB) = pair
        print(f"socket Pair for Box{index}: {socketA}, {socketB}")

        gpuAID = gpu_ip_map[socketA]
        gpuBID = gpu_ip_map[socketB]
        print(f"gpu pair is: {gpuAID}: {gpuBID}")
        if gpuAID is None or gpuBID is None:
            err_count += 1
            continue

        # 执行测试
        command = [
            exec_path,
            "-l",
            f"{gpuAID}",
            "-p",
            f"{gpuBID}",
            "-t",
            "1",
            "-c",
            "-w",
            "1",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        processes.append(process)
        commands.append(command)
        index += 1

    for process, command in zip(processes, commands):
        try:
            # 等待命令完成或超时，超时时间为10秒
            remain_time = 10 + start - time.time()
            if remain_time > 0:
                output, errors = process.communicate(timeout=remain_time)
            else:
                output, errors = process.communicate(timeout=1)
            if process.returncode:
                print(" ".join(command), f" failed. Output  {output} \nError {errors}")
                err_count += 1
            else:
                if is_slow(output, args.min_bandwidth):
                    print(" ".join(command), f" too slow. Output {output}")
                    err_count += 1
                else:
                    print(" ".join(command), f" succeeded. Output {output}")
        except subprocess.TimeoutExpired:
            process.kill()
            print(" ".join(command), " execute timout")
            err_count += 1
    if err_count > len(args.pairs) - args.min_channels:
        print(f"Error count {err_count} exceeded!")
        exit(1)


if __name__ == "__main__":
    main()
