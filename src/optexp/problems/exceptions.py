class DivergingException(Exception):
    """Called when loss is NAN or INF."""

    def __init__(self, message="Live training loss is NAN or INF") -> None:
        self.message = message
        super().__init__(self.message)
