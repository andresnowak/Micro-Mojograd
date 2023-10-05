from algorithm import vectorize

from .tensor_buffer import TensorBuffer

alias dims_average_size = 5
alias DIMS_SIZE = 12


struct TensorG[type: DType]:
    var data: TensorBuffer[type]
    alias simd_width = simdwidthof[type]()

    fn __init__(inout self, random: Bool, *dims: Int):
        self.data = TensorBuffer[type](dims, random)

    fn __init__(inout self, dims: VariadicList[Int], random: Bool = False):
        self.data = TensorBuffer[type](dims, random)

    fn __init__(
        inout self, dims: StaticIntTuple[DIMS_SIZE], rank: Int, random: Bool = False
    ):
        self.data = TensorBuffer[type](dims, rank, random)

    fn __copyinit__(inout self, existing: Self):
        self.data = existing.data

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data ^

    fn print_all(self):
        self.data.print_all()

    fn __getitem__(self, index: Int) -> SIMD[type, 1]:
        return self.data.get_1d_item(index)

    fn simd_load[simd_width: Int](self, index: Int) -> SIMD[type, simd_width]:
        return self.data.simd_load[simd_width](index)

    fn simd_strided_load[
        simd_width: Int
    ](self, index: Int, stride: Int) -> SIMD[type, simd_width]:
        return self.data.simd_strided_load[simd_width](index, stride)

    fn __setitem__(self, index: Int, value: SIMD[type, 1]):
        self.data.set_1d_item(index, value)

    fn simd_store[simd_width: Int](self, index: Int, value: SIMD[type, simd_width]):
        self.data.simd_store[simd_width](index, value)

    fn simd_strided_store[
        simd_width: Int
    ](self, index: Int, stride: Int, value: SIMD[type, simd_width]):
        self.data.simd_strided_store[simd_width](index, stride, value)

    fn __add__(self, other: Self) -> Self:
        let size = self.data.num_elements()

        let result = Self(self.shape(), self.data.rank(), False)  # is contiguous

        @parameter
        fn v_add[nelts: Int](i: Int):
            if self.data.is_contiguous and other.data.is_contiguous:
                result.simd_store[nelts](
                    i,
                    self.simd_load[nelts](i) + other.simd_load[nelts](i),
                )
            else:
                # change the strided load to not use vectorize function, because we can have the last position in the tensor at the middle of the for loop and not at the end, depending on the order of strides
                result.simd_store[nelts](
                    i,
                    self.simd_strided_load[nelts](
                        i, self.data.stride(self.data.rank() - 1)
                    )
                    + other.simd_strided_load[nelts](
                        i, other.data.stride(other.data.rank() - 1)
                    ),
                )

        vectorize[TensorG[type].simd_width, v_add](size)

        return result ^

    fn shape(self) -> StaticIntTuple[DIMS_SIZE]:
        return self.data.shape()

    fn strides(self) -> StaticIntTuple[DIMS_SIZE]:
        return self.data.strides()
