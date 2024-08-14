from typing import List, Tuple


def rgb_to_unit(xs):
    """Convert a list of RGB numbers [1, 255] to a list of unit [0, 1]"""
    if len(xs) == 3:
        return [x / 255.0 for x in xs]
    return rgb_to_unit(xs[:3]) + xs[3:]


class BaseColorScheme:
    as_list: List[Sequence] = []

    @classmethod
    def get(cls, i: int):
        return cls.as_list[i % len(cls.as_list)]


class Colors:

    class HighContrast(BaseColorScheme):
        """
        High contrast color scheme from Paul Tol
        https://personal.sron.nl/~pault/
        """

        blue = rgb_to_unit([0, 68, 136])
        yellow = rgb_to_unit([221, 170, 51])
        red = rgb_to_unit([187, 85, 102])
        as_list = [blue, yellow, red]

    class Vibrant(BaseColorScheme):
        """
        Vibrant color scheme from Paul Tol
        https://personal.sron.nl/~pault/
        """

        orange = rgb_to_unit([238, 119, 51])
        blue = rgb_to_unit([0, 119, 187])
        cyan = rgb_to_unit([51, 187, 238])
        magenta = rgb_to_unit([238, 51, 119])
        red = rgb_to_unit([204, 51, 17])
        teal = rgb_to_unit([0, 153, 136])
        grey = rgb_to_unit([187, 187, 187])
        as_list = [orange, blue, cyan, magenta, red, teal, grey]
