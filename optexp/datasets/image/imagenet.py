from attrs import frozen

from optexp.datasets.dataset import Dataset, HasClassCounts


@frozen
class ImageNet(Dataset, HasClassCounts):
    raise NotImplementedError()
