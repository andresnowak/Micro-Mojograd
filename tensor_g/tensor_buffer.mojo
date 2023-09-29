from utils.vector import InlinedFixedVector
from utils.index import StaticIntTuple
from random import rand
from memory import memset_zero

from .helpers import __check_bounds, __negative_pos_to_positive

alias dims_average_size = 5
alias DIMS_SIZE = 12


struct TensorBuffer[type: DType]:
    var data: DTypePointer[type]
    var _dims: StaticIntTuple[DIMS_SIZE]
    var _strides: StaticIntTuple[DIMS_SIZE]
    var _rank: Int
    var is_contiguous: Bool

    fn __init__(inout self, random: Bool, *dims: Int):
        self.data = DTypePointer[type].alloc(0)
        self._dims = StaticIntTuple[DIMS_SIZE](dims)
        self._strides = StaticIntTuple[DIMS_SIZE](0)
        self._rank = 1
        self.is_contiguous = True

        if random:
            rand(self.data, self.num_elements())
        else:
            self.zero()
        self._rank = len(VariadicList(dims))
        self.__suffix_product()
        self.is_contiguous = self.__check_is_contiguous()

    fn __init__(inout self, dims: VariadicList[Int], random: Bool = False):
        self.data = DTypePointer[type].alloc(0)
        self._dims = StaticIntTuple[DIMS_SIZE](dims)
        self._strides = StaticIntTuple[DIMS_SIZE](0)
        self._rank = 1
        self.is_contiguous = True

        if random:
            rand(self.data, self.num_elements())
        else:
            self.zero()
        self._rank = len(dims)
        self.__suffix_product()
        self.is_contiguous = self.__check_is_contiguous()

    fn __init__(inout self, data: DTypePointer[type], dims: VariadicList[Int]):
        self.data = DTypePointer[type](data.address)
        self._dims = StaticIntTuple[DIMS_SIZE](dims)
        self._strides = StaticIntTuple[DIMS_SIZE](0)
        self._rank = 1
        self.is_contiguous = True

        self._rank = len(dims)
        self.__suffix_product()
        self.is_contiguous = self.__check_is_contiguous()

    fn __copyinit__(inout self, existing: TensorBuffer[type]):
        self._dims = existing._dims
        self._strides = existing._strides
        self._rank = existing._rank
        self.is_contiguous = existing.is_contiguous
        self.data = existing.data

    fn __moveinit__(inout self, owned existing: TensorBuffer[type]):
        self._dims = existing._dims
        self._strides = existing._strides
        self._rank = existing._rank
        self.is_contiguous = existing.is_contiguous
        self.data = existing.data

    fn zero(self):
        """Set all the elements of the tensor to zero."""
        memset_zero(self.data, self.num_elements())

    fn num_elements(self) -> Int:
        """Get the number of elements in the tensor."""
        var size = 1

        for i in range(self._rank):
            size *= self._dims[i]
        return size

    fn __suffix_product(inout self):
        var size = 1
        for i in range(self._rank):
            self._strides[i] = size
            size *= self._dims[i]

    # fn print_all(self):
    #     """Print The dimension of the tensor."""
    #     print("[")
    #     for i in range(self._rank):
    #         print_no_newline(self[i], ",")
    #     print("]")

    fn __eq__(self, other: Self) -> Bool:
        """Check that both tensors have the same dimension."""
        return self._dims == other._dims

    fn matmul_eq(self, other: Self) -> Bool:
        """Check if the tensor is compatible with the other tensor for a matrix multiplication.
        """
        if self._rank != other._rank:
            return False

        # Check that all dimension are equal except the last two
        for i in range(self._rank - 2):
            if self._dims[i] != other._dims[i]:
                return False

        # check that the last dimension of the first tensor is equal to the second last dimension of the second tensor (ex. 2x3 * 3x4, 3 == 3)
        return self._dims[self._rank - 1] != other._dims[self._rank - 2]

    # fn __getitem__(self, *index: Int) -> SIMD[type, 1]:
    #     """Gets an element from the buffer from the specified index."""
    #     pass

    # fn __getitem__(self, index: VariadicList[Int]) -> SIMD[type, 1]:
    #     """Gets an element from the buffer from the specified index."""
    #     pass

    # fn __getitem__[len: Int](self, index: StaticIntTuple[len]) -> SIMD[type, 1]:
    #     """Gets an element from the buffer from the specified index."""
    #     pass

    # fn __getitem__(
    #     self, index: InlinedFixedVector[dims_average_size, Int]
    # ) -> SIMD[type, 1]:
    #     """Gets an element from the buffer from the specified index."""
    #     pass

    # fn __setitem__(self, value: SIMD[type, 1], *index: Int):
    #     """Sets an element in the buffer at the specified index."""
    #     pass

    # fn simd_load[width: Int](self, index: Int) -> SIMD[type, width]:
    #     """
    #     Loads a value from the buffer at the specified index.

    #     **Constraints**:
    #     - The buffer must be contiguous or width must be 1.
    #     """
    #     pass

    # fn simd_store[width: Int](self, index: Int, value: SIMD[type, width]):
    #     """
    #     Stores a value in the buffer at the specified index.

    #     **Constraints**:
    #     - The buffer must be contiguous or width must be 1.
    #     """
    #     pass

    fn rank(self) -> Int:
        """Get the rank of the tensor."""
        return self._rank

    fn __len__(self) -> Int:
        return self._rank

    fn strides(self) -> StaticIntTuple[DIMS_SIZE]:
        """Get the strides of the tensor."""
        return self._strides

    fn dims(self) -> StaticIntTuple[DIMS_SIZE]:
        """Get the dimensions of the tensor."""
        return self._dims

    fn dim(self, index: Int) -> Int:
        """Get the dimension at the given index."""
        return self._dims[index]

    fn stride(self, index: Int) -> Int:
        """Get the stride at the given index."""
        return self._strides[index]

    fn __check_is_contiguous(self) -> Bool:
        var c_contiguous = 1  # we save values in C row major order
        for i in range(self._rank):
            if self._strides[i] != c_contiguous:
                return False
            c_contiguous *= self._dims[i]
        return True
