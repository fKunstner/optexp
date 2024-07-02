import math

import numpy as np
import torch

from optexp.utils import nice_logspace

if __name__ == "__main__":

    if False:
        possible_values = [
            [
                0.1,
                1e-1,
                10**-1,
                1.0e-1,
                0.1e0,
                np.logspace(-1, -1, 1)[0],
                10 ** np.linspace(-1, -1, 1)[0],
                10 ** np.linspace(-7, 0, 15)[-3],
                10 ** np.linspace(-3, 0, 13)[-5],
                10 ** np.linspace(-1.0, -1.0, 1)[0],
                10 ** np.linspace(-7.0, 0.0, 15)[-3],
                10 ** np.linspace(-3.0, 0.0, 13)[-5],
                # torch.logspace(-1, -1, 1)[0].item(),
            ],
            [
                10.0,
                1e1,
                10**1,
                10**1.0,
                10.0**1,
                10.0**1.0,
                10.0e0,
                0.1e2,
                np.logspace(1, 1, 1)[0],
                10 ** np.linspace(-1, -1, 1)[0],
                10 ** np.linspace(-7, 0, 15)[-3],
                10 ** np.linspace(-3, 0, 13)[-5],
                10 ** np.linspace(-1.0, -1.0, 1)[0],
                10 ** np.linspace(-7.0, 0.0, 15)[-3],
                10 ** np.linspace(-3.0, 0.0, 13)[-5],
                # torch.logspace(-1, -1, 1)[0].item(),
            ],
            [
                10**-3.5,
                np.logspace(-4, -3, 3)[-2],
                10 ** np.linspace(-4, -3, 3)[1],
                10 ** np.linspace(-7, 0, 15)[7],
                10 ** np.linspace(-4, -1, 13)[2],
                # torch.logspace(-4, -3, 3)[1].item(),
                # np.logspace(-4, -3, 3)[-2],
                10 ** np.linspace(-4.0, -3.0, 3)[1],
                10 ** np.linspace(-7.0, 0.0, 15)[7],
                10 ** np.linspace(-4.0, -1.0, 13)[2],
            ],
        ]

        def npval(val):
            return np.float32(val).view(np.int32)

        for list in possible_values:
            for value in list:
                print(
                    value,
                    str(value),
                    repr(value),
                    npval(value),
                    f"{value:.60f}",
                    type(value),
                )
            print(
                "All the same?",
                all([value == other_value for value in list for other_value in list]),
            )

    def continued_fraction_complexity(value, MAX=10):
        """
        https://en.wikipedia.org/wiki/Continued_fraction#Calculating_continued_fraction_representations
        """
        i = math.floor(value)
        f = value - i

        for step in range(0, 10):
            if np.allclose(f, 0):
                return step
            i = math.floor(1 / f)
            f = 1 / f - i
        return MAX

    def power_repr(value):
        MAX = 6
        log10_complexity = continued_fraction_complexity(math.log10(value), MAX)
        log2_complexity = continued_fraction_complexity(math.log2(value), MAX)
        if log10_complexity == MAX and log2_complexity == MAX:
            return str(value)
        if log10_complexity < log2_complexity:
            return f"10**{math.log10(value):g}"
        else:
            return f"2**{math.log2(value):g}"

    if False:
        values_to_test = [
            *np.logspace(-7, 3, 11),
            *np.logspace(-7, 3, 11, base=2),
            *nice_logspace(-7, 7, base=10, density=3),
            *nice_logspace(-7, 7, base=10, density=3),
            *nice_logspace(-7, 7, base=2, density=3),
            *nice_logspace(-7, 7, base=2, density=3),
            *torch.logspace(-7, 3, 11).numpy(),
            *torch.logspace(-7, 3, 11, base=2).numpy(),
            0.9,
            0.99,
            0.999,
            0.9999,
            0.99999,
            0.999999,
            3e-4,
            *np.logspace(-7, -6, 4),
            *np.logspace(-7, -6, 4, base=2),
        ]
        print("======================")
        for value in values_to_test:
            print(
                f"{value:4.5f}",
                f"{value:.5e}",
                power_repr(value),
                continued_fraction_complexity(math.log10(value), 10),
                continued_fraction_complexity(math.log2(value), 10),
            )

        for value in values_to_test:
            other = eval(f"{value:.5e}")
            if not np.allclose(other, value):
                print(f"Failed for {value}, {other}")

    def test_frac_comp(val):
        print(val)
        print(f"{val:.10g}")
        a = continued_fraction_complexity(math.log10(val), 10)
        b = continued_fraction_complexity(math.log2(val), 10)
        print(a, b)

    val = 10**-4
    test_frac_comp(val)

    val = torch.logspace(-7, 3, 11).numpy()[3]
    test_frac_comp(val)
