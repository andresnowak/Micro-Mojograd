from utils.vector import InlinedFixedVector
from utils.index import StaticIntTuple

from .helpers import __check_bounds, __negative_pos_to_positive

alias dims_average_size = 5


struct TensorView:
    var _dims: Pointer[Int]
    var _strides: Pointer[Int]
    var _rank: Int
    var _size: Int
    var is_contiguous: Bool

    fn __init(inout self, rank: Int, get_value_from_index: fn (Int) capturing -> Int):
        # This is a hack to get around the fact that we don't have traits for now in mojo
        self._dims.free()
        self._strides.free()

        self._dims = Pointer[Int].alloc(rank)
        self._strides = Pointer[Int].alloc(rank)

        # get the dimensions of the tensor
        for i in range(rank):
            self._dims.store(i, get_value_from_index(i))

        # get the total amount of values in tensor
        for i in range(rank):
            self._size *= self._dims[i]

        self._rank = rank
        # get the strides of the tensor
        self.__suffix_product()

        # By default is contiguous
        self.is_contiguous = True

    fn __init__(inout self, *dims: Int):
        self._strides = Pointer[Int].alloc(0)
        self._dims = Pointer[Int].alloc(0)
        self._rank = 1
        self._size = 1
        self.is_contiguous = True

        fn get_value_from_index(index: Int) -> Int:
            return dims[index]

        self.__init(len(VariadicList(dims)), get_value_from_index)

    fn __init__(inout self, dims: VariadicList[Int]):
        self._strides = Pointer[Int].alloc(0)
        self._dims = Pointer[Int].alloc(0)
        self._rank = 1
        self._size = 1
        self.is_contiguous = True

        fn get_value_from_index(index: Int) -> Int:
            return dims[index]

        self.__init(len(dims), get_value_from_index)

    fn __init__(inout self, dims: InlinedFixedVector[dims_average_size, Int]):
        self._strides = Pointer[Int].alloc(0)
        self._dims = Pointer[Int].alloc(0)
        self._rank = 1
        self._size = 1
        self.is_contiguous = True

        fn get_value_from_index(index: Int) -> Int:
            return dims[index]

        self.__init(len(dims), get_value_from_index)

    fn __suffix_product(inout self):
        var size = 1
        for i in range(self._rank):
            self._strides.store(i, size)
            size *= self._dims[i]

    fn __copyinit__(inout self, existing: TensorView):
        self._dims = existing._dims
        self._strides = existing._strides
        self._rank = existing._rank
        self._size = existing._size
        self.is_contiguous = existing.is_contiguous

    fn __moveinit__(inout self, owned existing: TensorView):
        self._dims = existing._dims
        self._strides = existing._strides
        self._rank = existing._rank
        self._size = existing._size
        self.is_contiguous = existing.is_contiguous

    fn __getitem__(self, index: Int) -> Int:
        """Get the size of the dimension at the given index."""
        return self._dims[index]

    fn print_all(self):
        """Print The dimension of the tensor."""
        print("[")
        for i in range(self._rank):
            print_no_newline(self[i], ",")
        print("]")

    fn __eq__(self, other: TensorView) -> Bool:
        """Check that both tensors have the same dimension."""
        return self._dims == other._dims

    fn matmul_eq(self, other: TensorView) -> Bool:
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

    fn rank(self) -> Int:
        """Get the rank of the tensor."""
        return self._rank

    fn __len__(self) -> Int:
        return self._rank

    fn stride(self) -> InlinedFixedVector[dims_average_size, Int]:
        """Get the strides of the tensor."""
        var strides = InlinedFixedVector[dims_average_size, Int](self._rank)
        for i in range(self._rank):
            strides[i] = self._strides[i]

        return strides

    fn check_is_contiguous(self) -> Bool:
        var c_contiguous = 1
        var f_contiguous = self._size
        for i in range(self._rank):
            if (
                self._strides[i] != c_contiguous
                or self._strides[self._rank - i - 1] != f_contiguous
            ):
                return False
            c_contiguous *= self._dims[i]
            f_contiguous /= self._dims[i]
        return True

    fn __check_reshape_possible(
        self, new_rank: Int, get_value_from_index: fn (Int) capturing -> Int
    ) -> Bool:
        var new_size = 1

        for i in range(new_rank):
            new_size *= get_value_from_index(i)

        return new_size == self._size

    fn reshape(self, *dims: Int) -> TensorView:
        fn get_value_from_index(index: Int) -> Int:
            return dims[index]

        debug_assert(
            self.__check_reshape_possible(
                len(VariadicList(dims)), get_value_from_index
            ),
            "The new shape must have the same number of elements as the old shape",
        )

        return TensorView(dims)

    fn reshape(self, dims: VariadicList[Int]) -> TensorView:
        fn get_value_from_index(index: Int) -> Int:
            return dims[index]

        debug_assert(
            self.__check_reshape_possible(
                len(VariadicList(dims)), get_value_from_index
            ),
            "The new shape must have the same number of elements as the old shape",
        )

        return TensorView(dims)

    fn reshape(self, dims: InlinedFixedVector[dims_average_size, Int]) -> TensorView:
        fn get_value_from_index(index: Int) -> Int:
            return dims[index]

        debug_assert(
            self.__check_reshape_possible(
                len(VariadicList(dims)), get_value_from_index
            ),
            "The new shape must have the same number of elements as the old shape",
        )

        return TensorView(dims)

    fn permute(self, *dims: Int) -> TensorView:
        pass

    fn permute(self, dims: VariadicList[Int]) -> TensorView:
        pass

    fn transpose(self, *dims: Int):
        pass

    fn transpose(self, dims: VariadicList[Int]):
        pass
