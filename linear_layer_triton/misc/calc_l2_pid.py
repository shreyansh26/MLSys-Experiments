pid = 5
M = 16
N = 16
BLOCK_SIZE_M = 4
BLOCK_SIZE_N = 4
GROUP_SIZE_M = 2
print(((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N))
for pid in range(((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)):
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    # Number of programs ids along the N axis
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group
    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + (pid % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m

    print(f"{pid} -> {pid_m}, {pid_n}")