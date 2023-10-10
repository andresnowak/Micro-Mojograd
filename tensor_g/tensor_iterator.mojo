# The problem with the tensor iterator is when we have parallelization, we have to create a tensor iterator for each thread

# When we get to the end of a dimension, we restart at 0 and we substract the total value in index variable with the value of the dimension by its strides value


struct TensorIterator:
    var _rank: Int
    var _dims: StaticIntTuple[12]
    var _strides: StaticIntTuple[12]
    var _indexes: StaticIntTuple[12]
    var _index: Int
    var _size: Int
    var _is_contiguous: Bool
    var _broadcasted: Bool

    fn __init__(
        inout self,
        rank: Int,
        size: Int,
        dims: StaticIntTuple[12],
        strides: StaticIntTuple[12],
        contiguous: Bool,
        broadcasted: Bool,
    ):
        self._rank = rank
        self._dims = dims
        self._strides = strides
        self._indexes = StaticIntTuple[12](0)
        self._index = 0
        self._size = size
        self._is_contiguous = contiguous
        self._broadcasted = broadcasted

        # for i in range(rank):
        #     self._indexes[i] = 0
        #     self._size *= dims[i]

    @always_inline
    fn next(inout self):
        if self._index == self._size - 1:
            return

        if self._is_contiguous and not self._broadcasted:
            self._index += 1
            return

        for i in range(self._rank - 1, -1, -1):
            self.next_index(i)
            if self._indexes[i] != 0:
                break

    @always_inline
    fn next_index(inout self, dim_index: Int):
        self._indexes[dim_index] += 1
        if self._indexes[dim_index] >= self._dims[dim_index]:
            self._indexes[dim_index] = 0
            self._index -= self._strides[dim_index] * (self._dims[dim_index] - 1)
        else:
            self._index += self._indexes[dim_index] * self._strides[dim_index]

    @always_inline
    fn get_index(self) -> Int:
        return self._index


fn get_real_index(
    index: Int,
    dims: StaticIntTuple[12],
    strides: StaticIntTuple[12],
    rank: Int,
    contiguous: Bool,
    broadcasted: Bool,
) -> Int:
    if contiguous and not broadcasted:
        return index

    var pos = 0
    var suffix_product = 1

    @unroll
    for i in range(12 - 1, -1, -1):
        if i < rank:
            pos += ((index // suffix_product) % dims[i]) * strides[i]
            suffix_product *= dims[i]

    return pos
