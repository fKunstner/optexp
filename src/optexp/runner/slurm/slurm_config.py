from dataclasses import dataclass
from typing import Literal, Optional

GPU = Literal["p100", "v100l", "v100", "h100", "a100", "a100l"]


@dataclass
class SlurmConfig:
    hours: int
    gb_ram: int
    n_cpus: int = 1
    gpu: Optional[bool | GPU] = False
    n_gpus: Optional[int] = 1

    @property
    def time_str(self) -> str:
        return f"0-{self.hours:02d}:00"

    @property
    def mem_str(self) -> str:
        return f"{self.gb_ram}G"

    @property
    def gpu_str(self) -> str:
        if self.gpu is None:
            return ""
        else:
            match self.gpu:
                case None:
                    return ""
                case False:
                    return ""
                case True:
                    return f"#SBATCH --gpus-per-node={self.n_gpus}"
                case str():
                    return f"#SBATCH --gpus-per-node={self.gpu}:{self.n_gpus}"
