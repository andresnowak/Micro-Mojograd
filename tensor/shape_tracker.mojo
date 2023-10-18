alias dims_size = 12


struct ShapeTracker:
    var dims: StaticIntTuple[dims_size]
    var strides: StaticIntTuple[dims_size]
    var is_contiguous: Bool

    fn __init__(inout self):
        pass

    fn __eq__(self, other: Self):
        pass

    fn matmul_eq(self, other: Self):
        pass

    fn __is_contiguous(self):
        pass

    fn contiguous(self):
        pass

    fn reshape(self):
        pass

    fn permute(self):
        pass

    fn view(self):
        pass

    fn transpose(self):
        pass
