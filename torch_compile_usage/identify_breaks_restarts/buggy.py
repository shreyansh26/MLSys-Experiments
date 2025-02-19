import torch

# Bug 1: Using print() in a compiled function causes a graph break.
def f1(x):
    y = x + 1
    # print("f1 logging: x shape:", x.shape)  # <-- Graph break due to printing.
    return torch.sin(y)

# Bug 2: Data-dependent control flow with a condition based on a tensor and .item().
def f2(x):
    s = x.sum()
    # The comparison "s > 0" is data-dependent and will cause a graph break.
    if s > 0:
        # .item() forces extraction of a Python scalar, another source of graph break.
        return x * s.item()
    else:
        return x - s.item()

# Bug 3: Changing a constant integer argument forces recompilation.
def f3(x, k: int):
    # The branch taken depends on the value of k.
    # Each different integer value is treated as a constant and changes the graph.
    if k % 2 == 0:
        out = x * k
    else:
        out = x + k
    return out

def main():
    x = torch.randn(3, 3, requires_grad=True)

    # Run f1 to trigger a print-induced graph break.
    f1_compiled = torch.compile(f1)
    out1 = f1_compiled(x)
    print("Output from f1:", out1)

    # Run f2 to trigger data-dependent control flow and .item() graph breaks.
    f2_compiled = torch.compile(f2)
    out2 = f2_compiled(x)
    print("Output from f2:", out2)

    # Call f3 several times with a changing constant argument to force recompiles.
    f3_compiled = torch.compile(f3)
    for k in range(1, 4):
        out3 = f3_compiled(x, k)
        print(f"Output from f3 with constant {k}: ", out3)

    f3_compiled_dynamic = torch.compile(f3, dynamic=True)
    for k in range(1, 4):
        out3 = f3_compiled_dynamic(x, k)
        print(f"Output from f3 with constant {k}: ", out3)

if __name__ == "__main__":
    main()