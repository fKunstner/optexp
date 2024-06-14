from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Hyperparameter:
    """Representation of a hyperparmeter in scientific notation, base^exponent"""

    exponent: Fraction
    base: int = 10

    def as_float(self) -> float:
        """The value of the learning rate to be used within the optimizer

        Returns:
            float: the float value of the learning rate
        """
        return float(self.base**self.exponent)

    def __str__(self) -> str:
        return f"{self.base}^{self.exponent}"

    def as_latex_str(self) -> str:
        return f"${self.base}^{{{self.exponent}}}$"

    def __lt__(self, other):
        return self.as_float() < other.as_float()

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return self.as_float() > other.as_float()

    def __ge__(self, other):
        return self == other or self > other


@dataclass(frozen=True)
class LearningRate(Hyperparameter):
    pass
