from bench import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pinned", action="store_true")
parser.add_argument("--streamed", action="store_true")
args = parser.parse_args()

benchmark_with_profiler(args.pinned, args.streamed)