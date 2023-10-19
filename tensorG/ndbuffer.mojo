from memory import memset_zero
from math import min
from sys.info import simdwidthof

from .shape_tracker import ShapeTracker


struct NDBuffer[type: DType]:
    alias simd_width = simdwidthof[type]()
    var shape: ShapeTracker
    var data: DTypePointer[type]

    fn __init__(inout self, *dims: Int):
        self.__init__(dims)

    fn __init__(inout self, dims: VariadicList[Int]):
        self.shape = ShapeTracker[](dims)
        self.data = DTypePointer[type].alloc(self.shape.num_elements())
        self.zero()

    fn zero(self):
        """Set all the elements of the tensor to zero."""
        memset_zero(self.data, self.num_elements())

    fn print_all(self):
        """Print The dimension of the tensor."""

        let size = self.num_elements()
        let dims = self.shape.dims()

        var count = 0
        for i in range(size + 1):
            var suffix_product = 1
            count = 0

            @unroll
            for j in range(self.shape._dims_size - 1, -1, -1):
                if j < self.shape.rank():
                    suffix_product *= dims[j]
                    if i % suffix_product == 0 and i != 0:
                        print_no_newline("]")
                        count += 1

            if i > 0 and i < size:
                print_no_newline(",")

            # print the new lines between dimensions
            if i < size:
                for i in range(min(count, 3)):
                    print()

            suffix_product = 1

            @unroll
            for j in range(self.shape.rank() - 1, -1, -1):
                if j < self.shape.rank():
                    suffix_product *= dims[j]
                    if i % suffix_product == 0 and i != size:
                        print_no_newline("[")

            if i < size:
                print_no_newline(self.simd_load[1](i))

        print()

    fn num_elements(self) -> Int:
        """Return the number of elements in the tensor."""
        return self.shape.num_elements()

    fn simd_load[width: Int](self, index: Int) -> SIMD[type, width]:
        """
        Loads a value from the buffer at the specified index.

        **Constraints**:
        - The buffer must be contiguous or width must be 1.
        """
        return self.data.simd_load[width](index)
    