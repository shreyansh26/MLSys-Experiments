import triton.profiler.viewer as proton_viewer

def show_profile(precision, profile_name):
    metric_names = ["time/ms"] # Can also be - `gbyte/s` or `byte/s` - Supports at most two metrics while showing the tree
    if precision == 'fp8':
        metric_names = ["tflop8/s"] + metric_names
    elif precision == 'fp16':
        metric_names = ["tflop16/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)

if __name__ == "__main__":
    show_profile("fp16", "proton_results/matmul_fp16")