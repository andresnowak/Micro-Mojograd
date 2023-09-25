from utils.vector import InlinedFixedVector
from utils.index import StaticIntTuple

from .helpers import __check_bounds, __negative_pos_to_positive

alias dims_average_size = 5


struct TensorView[rank_size: Int]:
    var tensor_shape: StaticIntTuple[rank_size]
    var size: Int  # amount of values in tensor
    var len: Int  # rank of tensor
    var strides: StaticIntTuple[rank_size]

    fn __init__(inout self, *dims: Int):
        let temp = VariadicList(
            dims
        )  # because i dont know how to get the size with the variadic MLIR
        self.len = len(temp)
        self.tensor_shape = StaticIntTuple[rank_size](dims)
        self.size = 1
        self.strides = StaticIntTuple[rank_size]()

        self.size = self.product_dimensions()
        self.__suffix_product()

    fn __init__(inout self, dims: VariadicList[Int]):
        self.tensor_shape = StaticIntTuple[rank_size](dims)
        self.len = len(dims)
        self.size = 1
        self.strides = StaticIntTuple[rank_size]()

        self.size = self.product_dimensions()
        self.__suffix_product()

    fn __init__[size: Int](inout self, dims: InlinedFixedVector[size, Int]):
        self.tensor_shape = StaticIntTuple[rank_size]()
        self.len = len(dims)
        self.size = 1
        self.strides = StaticIntTuple[rank_size]()

        @parameter
        fn get_index_value(i: Int) -> Int:
            return dims[i]

        self.__fill_tensor_shape(self.len, get_index_value)
        self.size = self.product_dimensions()
        self.__suffix_product()

    fn __copyinit__(inout self: Self, existing: Self):
        """Creates a deep copy of an existing shape."""
        self.tensor_shape = existing.tensor_shape
        self.size = existing.size
        self.len = existing.len
        self.strides = existing.strides

    fn __moveinit__(inout self: Self, owned existing: Self):
        """Moves exsiting shape into new shape."""
        self.tensor_shape = existing.tensor_shape ^
        self.size = existing.size
        self.len = existing.len
        self.strides = existing.strides ^

    # fn __del__(owned self):
    #     self.tensor_shape.free()

    fn __fill_tensor_shape(
        inout self, size: Int, get_index_value: fn (Int) capturing -> Int
    ):
        for i in range(size):
            self.tensor_shape[i] = get_index_value(i)

    fn product_dimensions(self) -> Int:
        var size = 1
        for i in range(self.rank()):
            size *= self.tensor_shape[i]
        return size

    fn __suffix_product(inout self) -> None:
        """Strides has to be already initialized with the correct size."""
        self.strides[self.rank() - 1] = 1  # the first value has to be 1

        for index in range(self.rank() - 1):
            self.strides[self.rank() - 2 - index] = (
                self.strides[index] * self.tensor_shape[self.rank() - 1 - index]
            )

    fn stride(self) -> StaticIntTuple[rank_size]:
        return self.strides

    fn __get_position(self, get_index_value: fn (Int) capturing -> Int) -> Int:
        """Convert position from tuple of index dimensions to 1D position."""
        var pos = 0
        var dims_product_acum = 1
        for i in range(self.rank() - 1, 0, -1):
            dims_product_acum *= self.tensor_shape[i]
            pos += (
                __negative_pos_to_positive(
                    get_index_value(i - 1), self.tensor_shape[i - 1]
                )
                * dims_product_acum
            )

        pos += __negative_pos_to_positive(
            get_index_value(self.rank() - 1), self.tensor_shape[self.rank() - 1]
        )
        return pos

    @always_inline
    fn get_position[len: Int](self, index: StaticIntTuple[len]) -> Int:
        """Convert position from tuple of index dimensions to 1D position."""

        @parameter
        fn get_index_value(i: Int) -> Int:
            return index[i]

        return self.__get_position(get_index_value)

    @always_inline
    fn get_position[len: Int](self, index: InlinedFixedVector[len, Int]) -> Int:
        """Convert position from tuple of index dimensions to 1D position."""

        @parameter
        fn get_index_value(i: Int) -> Int:
            return index[i]

        return self.__get_position(get_index_value)

    @always_inline
    fn get_dimension_position(self, index: Int, dim_index: Int) -> Int:
        """Convert position of a dimension to it's 1D position."""

        debug_assert(
            index >= self.tensor_shape[dim_index],
            "Index out of bounds of it's dimension",
        )
        debug_assert(dim_index >= self.rank(), "Dimension index out of bounds")

        return (
            __negative_pos_to_positive(index, self.tensor_shape[dim_index])
            * self.strides[dim_index]
        )

    fn __getitem__(self, index: Int) -> Int:
        let pos = __negative_pos_to_positive(index, self.len)
        __check_bounds(pos, self.len)
        return self.tensor_shape[pos]

    fn __len__(self: Self) -> Int:
        """Get rank of tensor view."""
        return self.len

    @always_inline
    fn __eq__(self, other: TensorView[rank_size]) -> Bool:
        if self.rank() != other.rank():
            return False

        for i in range(self.rank()):
            if self[i] != other[i]:
                return False

        return True

    @always_inline
    fn eq_matmul(self, other: TensorView[rank_size]) -> Bool:
        if self.rank() != other.rank():
            return False

        # if rank is 1 we only check the dimension for both tensors are the same (to do a dot product)
        if self.rank() == 1:
            return self == other

        for i in range(self.rank() - 2):
            if self[i] != other[i]:
                return False

        if self[self.rank() - 2] != other[self.rank() - 1]:
            return False

        return True

    @always_inline
    fn rank(self) -> Int:
        """Get rank of tensor view."""
        return self.len

    fn num_elements(self) -> Int:
        """Get number of elements in tensor view."""
        return self.size

    fn shape(self) -> StaticIntTuple[rank_size]:
        """Get shape of tensor view."""
        return self.tensor_shape

    fn print_all(self):
        print("[")
        for i in range(self.rank()):
            print_no_newline(self[i], ",")
        print("]")

    fn permute(self):
        pass

    fn reshape[new_rank_size: Int](inout self, *dims: Int) -> TensorView[new_rank_size]:
        return self.reshape[new_rank_size](dims)

    fn reshape[
        new_rank_size: Int
    ](inout self, dims: VariadicList[Int]) -> TensorView[new_rank_size]:
        var size = 1

        for i in range(len(dims)):
            size *= dims[i]

        debug_assert(size == self.size, "New shape must have same number of elements")

        return TensorView[new_rank_size](dims)
