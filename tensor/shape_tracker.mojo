alias dims_size = 12


class ShapeTracker:
    var dims: StaticIntTuple[dims_size]
    var strides: StaticIntTuple[dims_size]
    var is_contiguous: Bool

    def __init__(self):
        pass

    def __eq__(self, other):
        pass

    def matmul_eq(self, other):
        pass

    def __is_contiguous(self):
        pass

    def contiguous(self):
        pass

    def reshape(self):
        pass

    def permute(self):
        pass

    def view(self):
        pass

    def transpose(self):
        pass
