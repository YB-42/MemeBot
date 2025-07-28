from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    def forward(self, x):
        return self.backbone(x)
