# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=bytecode python graph_break.py
import torch

@torch.compile
def fn(a):
    b = a + 2
    print("Hi")
    return b + a

fn(torch.randn(4))


'''
We can see that the modified bytecode is split into two functions, fn, the original function, and a function called resume_in_fn. 
This second function is a function created by Dynamo to implement the execution of the program starting at the graph break. This is often called a **continuation function**. 
This continuation function simply calls the second compiled function with the right arguments. The code for the initial function is rewritten implementing the strategy that we described before

L0-4. Call the compiled function (a + 2).
L6. Store its result in a local variable called graph_out_0. graph_out_0 is a tuple
L8-22. Leave the stack as it would be at the point of the graph break
L26. Execute the code that caused the graph break
L28-38. Call the compiled continuation function (a + b)


V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode] ORIGINAL BYTECODE fn /mnt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo/graph_break.py line 5 
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]   7           0 LOAD_FAST                0 (a)
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]               2 LOAD_CONST               1 (2)
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]               4 BINARY_ADD
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]               6 STORE_FAST               1 (b)
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode] 
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]   8           8 LOAD_GLOBAL              0 (print)
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]              10 LOAD_CONST               2 ('Hi')
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]              12 CALL_FUNCTION            1
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]              14 POP_TOP
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode] 
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]   9          16 LOAD_FAST                1 (b)
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]              18 LOAD_FAST                0 (a)
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]              20 BINARY_ADD
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode]              22 RETURN_VALUE
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode] 
V0123 18:55:58.460000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0] [__bytecode] 
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode] MODIFIED BYTECODE fn /mnt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo/graph_break.py line 5 
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]   5           0 LOAD_GLOBAL              2 (__compiled_fn_2)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]               2 LOAD_FAST                0 (a)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]               4 CALL_FUNCTION            1
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]               6 STORE_FAST               2 (graph_out_0)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]               8 LOAD_GLOBAL              1 (__builtins_dict___1)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              10 LOAD_CONST               3 ('print')
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              12 BINARY_SUBSCR
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              14 LOAD_CONST               2 ('Hi')
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              16 LOAD_FAST                2 (graph_out_0)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              18 LOAD_CONST               4 (0)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              20 BINARY_SUBSCR
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              22 STORE_FAST               1 (b)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              24 DELETE_FAST              2 (graph_out_0)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode] 
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]   8          26 CALL_FUNCTION            1
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              28 LOAD_GLOBAL              3 (__resume_at_14_3)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              30 ROT_TWO
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              32 LOAD_FAST                0 (a)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              34 LOAD_FAST                1 (b)
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              36 CALL_FUNCTION            3
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode]              38 RETURN_VALUE
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode] 
V0123 18:56:04.966000 140355673236608 torch/_dynamo/convert_frame.py:621] [0/0_1] [__bytecode] 
Hi
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode] ORIGINAL BYTECODE torch_dynamo_resume_in_fn_at_8 /mnt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo/graph_break.py line 8 
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]   8           0 LOAD_FAST                0 (___stack0)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               2 JUMP_ABSOLUTE            9 (to 18)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               4 LOAD_FAST                1 (a)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               6 LOAD_CONST               1 (2)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               8 BINARY_ADD
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              10 STORE_FAST               2 (b)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              12 LOAD_GLOBAL              0 (print)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              14 LOAD_CONST               2 ('Hi')
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              16 CALL_FUNCTION            1
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]         >>   18 POP_TOP
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode] 
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]   9          20 LOAD_FAST                2 (b)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              22 LOAD_FAST                1 (a)
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              24 BINARY_ADD
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              26 RETURN_VALUE
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode] 
V0123 18:56:04.969000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode] 
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode] MODIFIED BYTECODE torch_dynamo_resume_in_fn_at_8 /mnt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo/graph_break.py line 8 
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]   8           0 LOAD_GLOBAL              1 (__compiled_fn_5)
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               2 LOAD_FAST                2 (b)
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               4 LOAD_FAST                1 (a)
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               6 CALL_FUNCTION            2
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]               8 UNPACK_SEQUENCE          1
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]              10 RETURN_VALUE
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode] 
V0123 18:56:06.060000 140355673236608 torch/_dynamo/convert_frame.py:621] [1/0] [__bytecode]
'''