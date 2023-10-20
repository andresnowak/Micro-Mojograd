struct ShapeTracker[dims_size: Int = 12]:
    alias _dims_size = dims_size
    var _dims: StaticIntTuple[dims_size]
    var _strides: StaticIntTuple[dims_size]
    var _rank: Int
    var is_contiguous: Bool

    fn __init__(inout self, *dims: Int):
        self.__init__(dims)

    fn __init__(inout self, dims: VariadicList[Int]):
        self._rank = 0
        self.is_contiguous = True
        self._dims = StaticIntTuple[self._dims_size](0)
        self._strides = StaticIntTuple[self._dims_size](0)

        self._rank = len(dims)
        for i in range(self._rank):
            self._dims[i] = dims[i]
        self.__suffix_product()

    fn __init__(inout self, rank: Int, dims: StaticIntTuple[dims_size]):
        self._rank = 0
        self.is_contiguous = True
        self._dims = StaticIntTuple[self._dims_size](0)
        self._strides = StaticIntTuple[self._dims_size](0)

        self._rank = rank
        for i in range(self._rank):
            self._dims[i] = dims[i]
        self.__suffix_product()

    fn __init__(
        inout self,
        rank: Int,
        dims: StaticIntTuple[dims_size],
        strides: StaticIntTuple[dims_size],
    ):
        self._rank = 0
        self.is_contiguous = True
        self._dims = StaticIntTuple[self._dims_size](0)
        self._strides = StaticIntTuple[self._dims_size](0)

        self._rank = rank
        for i in range(self._rank):
            self._dims[i] = dims[i]
            self._strides[i] = strides[i]
        self.__is_contiguous()

    fn __copyinit__(inout self, existing: Self):
        self._dims = existing._dims
        self._strides = existing._strides
        self._rank = existing._rank
        self.is_contiguous = existing.is_contiguous

    fn __suffix_product(inout self):
        var product = 1
        for i in range(self._rank - 1, -1, -1):
            self._strides[i] = product
            product *= self._dims[i]

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        if self._rank != other._rank:
            return False
        for i in range(self._rank):
            if self._dims[i] != other._dims[i]:
                return False
        return True

    @always_inline
    fn matmul_eq(self, other: Self):
        pass

    fn __len__(self) -> Int:
        return self._rank

    fn rank(self) -> Int:
        return self._rank

    fn shape(self) -> StaticIntTuple[dims_size]:
        return self._dims

    fn strides(self) -> StaticIntTuple[dims_size]:
        return self._strides

    fn dim(self, index: Int) -> Int:
        """Get the dimension at the given index."""
        return self._dims[index]

    fn stride(self, index: Int) -> Int:
        """Get the stride at the given index."""
        return self._strides[index]

    fn __is_contiguous(inout self):
        var product = 1
        for i in range(self._rank - 1, -1, -1):
            if self._strides[i] != product:
                self.is_contiguous = False
            product *= self._dims[i]

        self.is_contiguous = True

    fn contiguous(self) -> Self:
        if self.is_contiguous:
            return self
        else:
            var new_shape = ShapeTracker(self._rank, self._dims)
            new_shape.is_contiguous = True
            return new_shape

    fn reshape(self):
        pass

    fn permute(self):
        pass

    fn view(self):
        pass

    fn transpose(self) -> Self:
        var new_dims = StaticIntTuple[self._dims_size](0)
        var new_strides = StaticIntTuple[self._dims_size](0)
        for i in range(self._rank):
            new_dims[i] = self._dims[self._rank - i - 1]
            new_strides[i] = self._strides[self._rank - i - 1]

        let new_shape = ShapeTracker(self._rank, new_dims, new_strides)

        return new_shape

    fn num_elements(self) -> Int:
        var product = 1
        for i in range(self._rank):
            product *= self._dims[i]

        return product
