#!/usr/bin/env python3
"""
Simulate CUDA-style shared-memory bank mapping for a 32 × 32 tile
(4-byte elements, 32 banks) and illustrate how XOR-swizzling removes
column-access conflicts.

Author:  you :-)
"""

TILE = 32                 # one warp in each dimension
BANKS = 32                # NV shared mem: 32 banks, 4 B wide


def bank(word_addr: int) -> int:
    """Map a *word* address to a bank ID (32-bit ops)."""
    return word_addr % BANKS


def naive_phys_x(x: int, y: int) -> int:
    """Row-major physical column index (no swizzle)."""
    return x


def xor_swizzled_phys_x(x: int, y: int) -> int:
    """Physical column index after XOR swizzle x' = x ^ y."""
    return x ^ y


def analyse(mapping):
    """
    Print the worst-case bank conflict degree for
    - writing a row  (x varies, y fixed)
    - reading a col  (y varies, x fixed)
    """
    # --- row write ---
    conflicts_row = 0
    for y in range(TILE):
        banks = [bank(y * TILE + mapping(x, y)) for x in range(TILE)]
        conflicts_row = max(conflicts_row,
                            max(banks.count(b) for b in set(banks)))
    # --- column read ---
    conflicts_col = 0
    for x in range(TILE):
        banks = [bank(y * TILE + mapping(x, y)) for y in range(TILE)]
        conflicts_col = max(conflicts_col,
                            max(banks.count(b) for b in set(banks)))
    return conflicts_row, conflicts_col


if __name__ == "__main__":
    naive_row, naive_col = analyse(naive_phys_x)
    xor_row,   xor_col   = analyse(xor_swizzled_phys_x)

    print("=== 32 × 32 tile, one 32-thread warp ===")
    print("               write-row   read-column")
    print(f"naïve layout :   {naive_row:2d}-way      {naive_col:2d}-way")
    print(f"XOR swizzle  :   {xor_row:2d}-way      {xor_col:2d}-way")

    # Optional: show the bank table for one tile
    SHOW_TABLE = True
    if SHOW_TABLE:
        def dump(title, mapping):
            print(f"\n{title}")
            for y in range(TILE):
                row = [f"{bank(y*TILE + mapping(x,y)):2d}" for x in range(TILE)]
                print(" ".join(row))
        dump("Naïve bank IDs", naive_phys_x)
        dump("Swizzled bank IDs (x^y)", xor_swizzled_phys_x)