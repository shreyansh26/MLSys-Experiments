from mlp_models import MLP1, MLP2
from torch.fx import symbolic_trace, subgraph_rewriter

gm = symbolic_trace(MLP1())
graph = gm.graph
print(gm.graph)
print("******")
print(graph.owning_module)
print("******")

for node in graph.nodes:
    print(node)
    print(node.op)
    print(node.target)
    print(node.args)
    print("******")

print("**************************************")

gm = symbolic_trace(MLP2())
graph = gm.graph
print(gm.graph)
print("******")
print(graph.owning_module)
print("******")

for node in graph.nodes:
    print(node)
    print(node.op)
    print(node.target)
    print(node.args)
    print(dict(node.graph.owning_module.named_modules()))
    print("******")

mlp = MLP2()
mod = mlp.get_submodule("ll")
print(mod)
attr = getattr(mod, "ll", None)
print(attr)
# mod = mlp.get_submodule("ll2") # AttributeError
# print(mod)
mod = getattr(mlp, "ll")
print(mod)