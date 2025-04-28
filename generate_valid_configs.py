#!/usr/bin/env python3
import argparse
import random
import re
import ast
import sys
import time

def parse_configs(file_path):
    start_re = re.compile(r'^Arm\s+\d+\s+Start Config\s+(\d+):\s*(\[.*\])')
    end_re   = re.compile(r'^Arm\s+\d+\s+End\s+Config\s+(\d+):\s*(\[.*\])')
    starts, ends = {}, {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = start_re.match(line)
            if m:
                starts[int(m.group(1))] = ast.literal_eval(m.group(2))
            m = end_re.match(line)
            if m:
                ends[int(m.group(1))] = ast.literal_eval(m.group(2))
    idxs = sorted(set(starts) & set(ends))
    if not idxs:
        print("No paired configs found.", file=sys.stderr)
        sys.exit(1)
    return [(starts[i], ends[i]) for i in idxs]

def main():
    parser = argparse.ArgumentParser(
        description="Randomly select paired arm configurations and print them."
    )
    parser.add_argument("-f", "--file",   required=True,
                        help="Input text file with Arm Start/End Configs")
    parser.add_argument("-a", "--arms",   type=int, required=True,
                        help="Number of arms")
    parser.add_argument("-c", "--configs",type=int, required=True,
                        help="Configs per arm")
    parser.add_argument("-s", "--seed",   type=int, default=None,
                        help="(optional) integer seed for reproducibility")
    args = parser.parse_args()

    # seed RNG: if no seed given, use nanosecond time to get a fresh random seed each run
    if args.seed is None:
        random.seed(time.time_ns() & ((1<<64)-1))
    else:
        random.seed(args.seed)

    pool = parse_configs(args.file)
    if args.configs > len(pool):
        print(f"Error: only {len(pool)} pairs available, asked for {args.configs}.",
              file=sys.stderr)
        sys.exit(1)

    for arm in range(args.arms):
        chosen = random.sample(pool, args.configs)
        for idx, (start, end) in enumerate(chosen, start=1):
            print(f"Arm {arm} Start Config {idx}: {start}")
            print(f"Arm {arm} End   Config {idx}: {end}")
        if arm != args.arms - 1:
            print()

if __name__ == "__main__":
    main()
