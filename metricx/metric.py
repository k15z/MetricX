class Metric:
    """This class represents an objective to be optimized.

    Attributes:
        name: The name of the metric.
        description: A human-readible description of the metric.
        is_higher_better: Whether larger values are better.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        is_higher_better: bool = True,
    ):
        self.name = name
        self.description = description
        self.is_higher_better = is_higher_better

    def __hash__(self):
        return hash(self.name) + hash(self.description) + hash(self.is_higher_better)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.name == other.name
            and self.description == other.description
            and self.is_higher_better == other.is_higher_better
        )

    def __repr__(self):
        return f"Metric({self.name!r}, {self.description!r}, {self.is_higher_better!r})"
