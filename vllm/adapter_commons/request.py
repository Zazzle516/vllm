# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

# 强制子类实现某些接口，防止不完整的继承
class AdapterRequest(ABC):
    """
    Base class for adapter requests.
    """

    @property
    @abstractmethod
    def adapter_id(self) -> int:
        raise NotImplementedError

    def __post_init__(self) -> None:
        if self.adapter_id < 1:
            raise ValueError(f"id must be > 0, got {self.adapter_id}")

    def __eq__(self, value: object) -> bool:
        return isinstance(
            value, self.__class__) and self.adapter_id == value.adapter_id

    def __hash__(self) -> int:
        return hash(self.adapter_id)
