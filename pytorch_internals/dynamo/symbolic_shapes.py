# https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html
# TORCH_LOGS=graph_code,graph_sizes python symbolic_shapes.py
import torch

@torch.compile
def fn(a, b):
    return a.shape[0] * a * b

fn(torch.randn(4, 3), torch.randn(4, 3))
fn(torch.randn(8, 3), torch.randn(8, 3))

'''
In the first graph the shape is traced as a constant, but once it changes, it traces it symbolically using a SymInts

V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code] TRACED GRAPH
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]  ===== __compiled_fn_1 =====
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]  /home/shreyansh/miniconda3/envs/shreyansh-env-py10-torch24/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]     def forward(self, L_a_: "f32[4, 3][3, 1]cpu", L_b_: "f32[4, 3][3, 1]cpu"):
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         l_a_ = L_a_
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         l_b_ = L_b_
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         # File: /mnt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo/symbolic_shapes.py:7 in fn, code: return a.shape[0] * a * b
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         mul: "f32[4, 3][3, 1]cpu" = 4 * l_a_;  l_a_ = None
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         mul_1: "f32[4, 3][3, 1]cpu" = mul * l_b_;  mul = l_b_ = None
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         return (mul_1,)
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code]         
V0123 10:01:52.787000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/0] [__graph_code] 
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code] TRACED GRAPH
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]  ===== __compiled_fn_3 =====
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]  /home/shreyansh/miniconda3/envs/shreyansh-env-py10-torch24/lib/python3.10/site-packages/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]     def forward(self, s0: "Sym(s0)", L_a_: "f32[s0, 3][3, 1]cpu", L_b_: "f32[s0, 3][3, 1]cpu"):
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         l_a_ = L_a_
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         l_b_ = L_b_
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         # File: /mnt/ssd1/shreyansh/home_dir/misc_experiments/pytorch_internals/dynamo/symbolic_shapes.py:7 in fn, code: return a.shape[0] * a * b
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         size = l_a_.size()
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         getitem: "Sym(s0)" = size[0];  size = None
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         mul: "f32[s0, 3][3, 1]cpu" = getitem * l_a_;  getitem = l_a_ = None
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         mul_1: "f32[s0, 3][3, 1]cpu" = mul * l_b_;  mul = l_b_ = None
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         return (mul_1,)
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code]         
V0123 10:01:56.913000 140455623955584 torch/_dynamo/output_graph.py:1291] [0/1] [__graph_code] 
'''