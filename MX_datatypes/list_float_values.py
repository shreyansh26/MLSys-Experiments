#!/usr/bin/env python3
import argparse
import math

def generate_values(E: int, M: int):
    """
    Yield tuples (sign, exp_raw, mant_raw, value) for every bit‐pattern
    in an E‐bit exponent, M‐bit mantissa float format.
    """
    bias = (1 << (E - 1)) - 1
    max_exp = (1 << E) - 1
    for sign in (0, 1):
        for exp_raw in range(0, max_exp + 1):
            for mant_raw in range(0, 1 << M):
                # decode
                if exp_raw == 0:
                    if mant_raw == 0:
                        v = 0.0
                    else:
                        # subnormal
                        v = (mant_raw / (1 << M)) * 2.0 ** (1 - bias)
                elif exp_raw == max_exp:
                    if mant_raw == 0:
                        v = math.inf
                    else:
                        v = math.nan
                else:
                    # normal
                    v = (1.0 + mant_raw / (1 << M)) * 2.0 ** (exp_raw - bias)
                if sign == 1:
                    v = -v
                yield sign, exp_raw, mant_raw, v

def main():
    p = argparse.ArgumentParser(
        description="List all representable values in an ExMy floating format."
    )
    p.add_argument("E", type=int, help="number of exponent bits")
    p.add_argument("M", type=int, help="number of mantissa bits")
    p.add_argument(
        "--csv", "-c", action="store_true",
        help="output as CSV with columns sign,exp_raw,mant_raw,value"
    )
    args = p.parse_args()

    total = 2 * (1 << args.E) * (1 << args.M)
    print(f"Enumerating {total} values for E={args.E}, M={args.M}...\n")

    if args.csv:
        print("sign,exp_raw,mant_raw,value")
        for s, e, m, v in generate_values(args.E, args.M):
            # NaNs print oddly; normalize
            if math.isnan(v):
                v_str = "NaN"
            elif math.isinf(v):
                v_str = "Infinity" if v > 0 else "-Infinity"
            else:
                v_str = repr(v)
            print(f"{s},{e},{m},{v_str}")
    else:
        for s, e, m, v in generate_values(args.E, args.M):
            flag = ("±∞","NaN","subnormal","normal","zero")
            # simply print in a human‐readable form
            print(f"s={s} e={e:0{args.E}b} m={m:0{args.M}b} → {v!r}")

if __name__ == "__main__":
    main()
