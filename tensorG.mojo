from random import rand
from memory import memset_zero
from memory.buffer import Buffer
from utils.index import StaticIntTuple, Index
from utils.list import Dim, DimList
from tensor import TensorShape
from utils.vector import DynamicVector, InlinedFixedVector
from runtime.llcl import num_cores, Runtime
from algorithm import vectorize, parallelize
import math

alias dims_average_size = 5


struct TensorView:
    var tensor_shape: Pointer[Int]
    var size: Int
    var len: Int

    fn __init__(inout self, *dims: Int):
        let temp = VariadicList(dims) # because i dont know how to get the size with the variadic MLIR
        self.tensor_shape = Pointer[Int].alloc(len(temp))
        for i in range(len(temp)):
            self.tensor_shape.store(i, temp[i])
        self.len = len(temp)
        self.size = 1
        self.size = self.product_dimensions()

    fn __init__(inout self, dims: VariadicList[Int]):
        self.tensor_shape = Pointer[Int].alloc(len(dims))
        for i in range(len(dims)):
            self.tensor_shape.store(i, dims[i])
        self.len = len(dims)
        self.size = 1
        self.size = self.product_dimensions()

    fn __init__[size: Int](inout self, dims: InlinedFixedVector[size, Int]):
        self.tensor_shape = Pointer[Int].alloc(len(dims))
        for i in range(len(dims)):
            self.tensor_shape.store(i, dims[i])
        self.len = len(dims)
        self.size = 1
        self.size = self.product_dimensions()

    fn __copyinit__(inout self: Self, existing: Self):
        """Creates a deep copy of an existing shape."""
        self.tensor_shape = existing.tensor_shape
        self.size = existing.size
        self.len = existing.len

    fn __moveinit__(inout self: Self, owned existing: Self):
        """Moves exsiting shape into new shape."""
        self.tensor_shape = existing.tensor_shape ^
        self.size = existing.size
        self.len = existing.len

    fn product_dimensions(self) -> Int:
        var size = 1
        for i in range(self.rank()):
            size *= self.tensor_shape[i]
        return size

    @always_inline
    fn get_position[len: Int](self, index: StaticIntTuple[len]) -> Int:
        var pos = 0
        var dims_product_acum = 1
        for i in range(self.rank() - 1, 0, -1):
            dims_product_acum *= self.tensor_shape[i]
            pos += index[i - 1] * dims_product_acum

        pos += index[self.rank() - 1]
        return pos

    @always_inline
    fn get_position[len: Int](self, index: InlinedFixedVector[len, Int]) -> Int:
        var pos = 0
        var dims_product_acum = 1
        for i in range(self.rank() - 1, 0, -1):
            dims_product_acum *= self.tensor_shape[i]
            pos += index[i - 1] * dims_product_acum

        pos += index[self.rank() - 1]
        return pos

    fn __getitem__(self, index: Int) -> Int:
        return self.tensor_shape[index]

    fn __len__(self: Self) -> Int:
        """Get rank of tensor view."""
        return self.len

    @always_inline
    fn __eq__(self, other: TensorView) -> Bool:
        if self.rank() != other.rank():
            return False

        for i in range(self.rank()):
            if self[i] != other[i]:
                return False

        return True

    @always_inline
    fn eq_matmul(self, other: TensorView) -> Bool:
        if self.rank() != other.rank():
            return False

        if self.rank() == 1:
            return True

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

    fn shape(self) -> Pointer[Int]:
        """Get shape of tensor view."""
        return self.tensor_shape

    fn print_all(self):
        print("[")
        for i in range(self.rank()):
            print_no_newline(self[i], ",")
        print("]")


struct TensorG[Type: DType]:
    var data: DTypePointer[Type]
    var dims: TensorView

    fn __init__(inout self, random: Bool, *dims: Int):
        self.dims = TensorView(dims)
        let size = self.dims.num_elements()

        self.data = DTypePointer[Type].alloc(size)
        if random:
            rand(self.data, size)
        else:
            self.zero()

    fn __init__(inout self, random: Bool, dims: TensorView):
        self.dims = dims
        let size = self.dims.num_elements()

        self.data = DTypePointer[Type].alloc(size)
        if random:
            rand(self.data, size)
        else:
            self.zero()

    fn __init__(
        inout self,
        data: VariadicList[FloatLiteral],
        dims: TensorView,
    ):
        self.dims = dims
        let size = self.dims.num_elements()

        let dims_area_correct = size == len(data)
        debug_assert(
            dims_area_correct,
            "Error, the size of the data doesn't match the size of the tensor.",
        )

        self.data = DTypePointer[Type].alloc(size)
        for i in range(size):
            self.data.simd_store[1](i, data[i])

    fn __init__(
        inout self,
        data: DynamicVector[FloatLiteral],
        dims: TensorView,
    ):
        self.dims = dims
        let size = self.dims.num_elements()

        let dims_area_correct = size == len(data)
        debug_assert(
            dims_area_correct,
            "Error, the size of the data doesn't match the size of the tensor.",
        )

        self.data = DTypePointer[Type].alloc(size)
        for i in range(size):
            self.data.simd_store[1](i, data[i])

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.dims.num_elements())

    fn __copyinit__(inout self, existing: Self):
        self.dims = existing.dims
        let size = self.dims.num_elements()
        self.data = DTypePointer[Type].alloc(size)

        for i in range(size):
            self.data.simd_store[1](i, existing.data.simd_load[1](i))

    fn __moveinit__(inout self, owned existing: Self):
        self.dims = existing.dims
        self.data = existing.data

    fn byte_count(self) -> Int:
        return sizeof[Type]() * self.dims.num_elements()

    @always_inline
    fn __getitem__[len: Int](self, index: StaticIntTuple[len]) -> SIMD[Type, 1]:
        return self.load[1](index)

    @always_inline
    fn __getitem__[
        len: Int
    ](self, index: InlinedFixedVector[len, Int]) -> SIMD[Type, 1]:
        return self.load[1](index)

    @always_inline
    fn __getitem__(self, index: Int) -> SIMD[Type, 1]:
        """Access the data as a 1D array."""
        return self.load[1](index)

    @always_inline
    fn load[
        nelts: Int, len: Int
    ](self, index: StaticIntTuple[len]) -> SIMD[Type, nelts]:
        return self.data.simd_load[nelts](
            self.dims.get_position(index)
        )  # cols = 5. so if y is 0 position and x is 1 position we acces 1 position, if position y is 1 and x is 0 we acces 5 position

    @always_inline
    fn load[
        nelts: Int, len: Int
    ](self, index: InlinedFixedVector[len, Int]) -> SIMD[Type, nelts]:
        return self.data.simd_load[nelts](self.dims.get_position(index))

    @always_inline
    fn load[nelts: Int](self, index: Int) -> SIMD[Type, nelts]:
        """Access the data as a 1D array."""
        return self.data.simd_load[nelts](index)

    @always_inline
    fn __setitem__[len: Int](self, index: StaticIntTuple[len], val: SIMD[Type, 1]):
        return self.store[1](index, val)

    @always_inline
    fn __setitem__[
        len: Int
    ](self, index: InlinedFixedVector[len, Int], val: SIMD[Type, 1]):
        return self.store[1](index, val)

    @always_inline
    fn __setitem__(self, index: Int, val: SIMD[Type, 1]):
        return self.store[1](index, val)

    @always_inline
    fn store[
        nelts: Int, len: Int
    ](self, index: StaticIntTuple[len], val: SIMD[Type, nelts]):
        self.data.simd_store[nelts](self.dims.get_position(index), val)

    @always_inline
    fn store[
        nelts: Int, len: Int
    ](self, index: InlinedFixedVector[len, Int], val: SIMD[Type, nelts]):
        self.data.simd_store[nelts](self.dims.get_position(index), val)

    @always_inline
    fn store[nelts: Int](self, index: Int, val: SIMD[Type, nelts]):
        """Access and store the data as a 1D array."""
        self.data.simd_store[nelts](index, val)

    fn __dim_suffix_product[len: Int](self) -> InlinedFixedVector[len, Int]:
        var suffix_product = InlinedFixedVector[len, Int](self.dims.rank())
        suffix_product.append(1)  # the first value has to be 1

        for index in range(self.dims.rank() - 1):
            suffix_product.append(
                suffix_product[index] * self.dims[self.dims.rank() - 1 - index]
            )

        return suffix_product

    fn __dim_matmul_suffix_product[
        len: Int
    ](self, other: Self) -> InlinedFixedVector[len, Int]:
        var suffix_product = InlinedFixedVector[len, Int](self.dims.rank() + 1)
        suffix_product.append(1)  # the first value has to be 1

        suffix_product.append(suffix_product[0] * other.dims[other.dims.rank() - 1])
        suffix_product.append(suffix_product[1] * self.dims[self.dims.rank() - 1])

        for index in range(self.dims.rank() - 2):
            suffix_product.append(
                suffix_product[index + 2] * self.dims[self.dims.rank() - 2 - index]
            )

        return suffix_product

    fn __matmul_num_elements(self, other: Self) -> Int:
        var size = 1
        for i in range(self.dims.rank() - 2):
            size *= self.dims[i]

        size *= self.dims[self.dims.rank() - 2] # The different dimension of first tensor
        size *= other.dims[other.dims.rank() - 1] # the different dimension of second tensor
        size *= self.dims[self.dims.rank() - 1] # The dimension that is the same for both tensors

        return size

    fn print_all(self):
        let suffix_product = self.__dim_suffix_product[dims_average_size]()

        # the first dimension uses the last value of the suffix_product,and the last dimension uses the first value of the suffix_product (so 1)

        print("[")

        var index_values_old = InlinedFixedVector[dims_average_size, Int](
            self.dims.rank()
        )
        let size = self.dims.num_elements()

        for i in range(size):
            var index_values = InlinedFixedVector[dims_average_size, Int](
                self.dims.rank()
            )
            for j in range(self.dims.rank()):
                let index = i // suffix_product[self.dims.rank() - 1 - j] % self.dims[j]

                index_values.append(index)

            for i in range(self.dims.rank() - 1):
                if index_values[i] > index_values_old[i]:
                    print("],\n[")
                    break

            print_no_newline(self[index_values], ",")

            index_values_old = index_values

        print("]")

    fn rank(self) -> Int:
        return self.dims.rank()

    @always_inline
    fn __iterate_tensor[
        nelts: Int
    ](self, other: Self, res: Self, inout flag: Bool, func: StringLiteral):
        let size = self.dims.num_elements()

        @parameter
        fn iterate_vectorize[nelts: Int](i: Int):
            if func == "add":
                res.store[nelts](
                    i,
                    self.load[nelts](i) + other.load[nelts](i),
                )
            elif func == "mul":
                res.store[nelts](
                    i,
                    self.load[nelts](i) * other.load[nelts](i),
                )
            elif func == "eq":
                if self.load[nelts](i) != other.load[nelts](i):
                    flag = False
            else:
                debug_assert(False, "Error, function not implemented.")

        vectorize[nelts, iterate_vectorize](size)

    @always_inline
    fn __iterate_tensor[
        nelts: Int
    ](
        self,
        other: Self,
        res: Self,
        inout flag: Bool,
        func: StringLiteral,
        rt: Runtime,
        n_cores: Int,
    ):

        var last_dim = 1
        if self.dims.rank() > 1:
            last_dim = self.dims[self.dims.rank() - 1]
        let size = self.dims.num_elements()

        @parameter
        fn iterate_parallel(n: Int):
            @parameter
            fn iterate_vectorize[nelts: Int](k: Int):
                let i = n * last_dim + k

                if func == "add":
                    res.store[nelts](
                        i,
                        self.load[nelts](i)
                        + other.load[nelts](i),
                    )
                elif func == "mul":
                    res.store[nelts](
                        i,
                        self.load[nelts](i)
                        * other.load[nelts](i),
                    )
                elif func == "eq":
                    if self.load[nelts](i) != other.load[nelts](i):
                        flag = False
                else:
                    debug_assert(False, "Error, function not implemented.")

            vectorize[nelts, iterate_vectorize](last_dim)

        parallelize[iterate_parallel](rt, size // last_dim, n_cores)

    @always_inline
    fn __add__(self, other: Self) -> Self:
        return self.add[1](other)

    @always_inline
    fn add[nelts: Int](self, other: Self) -> Self:
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, self.dims)

        @parameter
        fn add_v(index_values: InlinedFixedVector[dims_average_size, Int]):
            res.store[nelts](
                index_values,
                self.load[nelts](index_values) + other.load[nelts](index_values),
            )

        var flag = True
        self.__iterate_tensor[nelts](other, res, flag, "add")

        return res ^

    @always_inline
    fn add[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Self:
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, self.dims)

        var flag = True
        self.__iterate_tensor[nelts](other, res, flag, "add", rt, n_cores)

        return res ^

    fn __mul__(self, other: Self) -> Self:
        return self.mul[1](other)

    @always_inline
    fn mul[nelts: Int](self, other: Self) -> Self:
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, self.dims)

        var flag = True
        self.__iterate_tensor[nelts](other, res, flag, "mul")

        return res ^

    @always_inline
    fn mul[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Self:
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, self.dims)

        var flag = True
        self.__iterate_tensor[nelts](other, res, flag, "mul", rt, n_cores)

        return res ^

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.eq[1](other)

    @always_inline
    fn eq[nelts: Int](self, other: Self) -> Bool:
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, 0)
        var flag = True

        self.__iterate_tensor[nelts](other, res, flag, "eq")

        return flag

    @always_inline
    fn eq[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Bool:
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, 0)
        var flag = True

        self.__iterate_tensor[nelts](other, res, flag, "eq", rt, n_cores)

        return flag

    @always_inline
    fn dot[nelts: Int](self, other: Self) -> Self:
        let dims_1d = self.dims.rank() == 1 and other.dims.rank() == 1
        debug_assert(dims_1d, "Error dimensions aren't 1D can't dot tensors.")
        let dims_eq = self.dims == other.dims
        debug_assert(dims_eq, "Error dimension aren't equal can't sum tensors.")

        let res = Self(False, 1)

        let size = self.dims.num_elements()

        @parameter
        fn dot_v[nelts: Int](index: Int):
            res[0] = res[0] + (self.data.simd_load[nelts](index) * other.data.simd_load[nelts](index)).reduce_add()

        vectorize[nelts, dot_v](size)

        return res ^

    @always_inline
    fn __matmul__(self, other: Self) -> Self:
        return self.matmul[1](other)

    @always_inline
    fn matmul[nelts: Int](self, other: Self) -> Self:
        if self.dims.rank() == 1 and other.dims.rank() == 1:
            return self.dot[nelts](other)

        let dims_eq = self.dims.eq_matmul(other.dims)
        debug_assert(dims_eq, "Error dimensions don't conform for a matmul.")

        var res_dims = InlinedFixedVector[dims_average_size, Int](self.dims.rank())
        for i in range(self.dims.rank() - 2):
            res_dims.append(self.dims[i])
        res_dims.append(self.dims[self.dims.rank() - 2])
        res_dims.append(other.dims[other.dims.rank() - 1])

        let res = Self(False, TensorView(res_dims))
        var flag = True

        let matmul_suffix_product = self.__dim_matmul_suffix_product[dims_average_size](
            other
        )

        let size = self.__matmul_num_elements(other)

        # we use a for, so only vectorize works in the las dimension (the dimension were the values are stored), because if not vectorized can grab position that area outside of the memory of our data (because size is bigger than the size of the tensor)
        let res_last_dim = res.dims[res.dims.rank() - 1]
        for i in range(0, size // res_last_dim):    
            @parameter
            fn matmul_v[nelts: Int](j: Int):
                let index = i * res_last_dim + j

                var index_values_self = InlinedFixedVector[dims_average_size, Int](
                    self.dims.rank()
                )
                var index_values_other = InlinedFixedVector[dims_average_size, Int](
                    self.dims.rank()
                )
                var index_values_res = InlinedFixedVector[dims_average_size, Int](
                    self.dims.rank()
                )

                for i in range(self.dims.rank() - 2):
                    let index_pos = index
                        // matmul_suffix_product[self.dims.rank() - i]
                        % self.dims[i]
                    index_values_self.append(
                        index_pos
                    )
                    index_values_other.append(
                        index_pos
                    )
                    index_values_res.append(
                        index_pos
                    )
                
                index_values_self.append(
                    index // matmul_suffix_product[2] % self.dims[self.dims.rank() - 2]
                )
                index_values_self.append(
                    index // matmul_suffix_product[1] % self.dims[self.dims.rank() - 1]
                )

                index_values_other.append(
                    index // matmul_suffix_product[1] % other.dims[other.dims.rank() - 2]
                )
                index_values_other.append(
                    index // matmul_suffix_product[0] % other.dims[other.dims.rank() - 1]
                )

                index_values_res.append(
                    (index // matmul_suffix_product[2]) % res.dims[res.dims.rank() - 2]
                )
                index_values_res.append(
                    (index // matmul_suffix_product[0]) % res.dims[res.dims.rank() - 1]
                )

                res.store[nelts](
                    index_values_res,
                    res.load[nelts](index_values_res)
                    + self.load[1](index_values_self)
                    * other.load[nelts](index_values_other),
                )

            vectorize[nelts, matmul_v](res_last_dim)

        return res ^

    @always_inline
    fn matmul[nelts: Int](self, other: Self, rt: Runtime, n_cores: Int) -> Self:
        if self.dims.rank() == 1 and other.dims.rank() == 1:
            return self.dot[nelts](other)

        let dims_eq = self.dims.eq_matmul(other.dims)
        debug_assert(dims_eq, "Error dimensions don't conform for a matmul.")

        var res_dims = InlinedFixedVector[dims_average_size, Int](self.dims.rank())
        for i in range(self.dims.rank() - 2):
            res_dims.append(self.dims[i])
        res_dims.append(self.dims[self.dims.rank() - 2])
        res_dims.append(other.dims[other.dims.rank() - 1])

        let res = Self(False, TensorView(res_dims))
        var flag = True

        let matmul_suffix_product = self.__dim_matmul_suffix_product[dims_average_size](
            other
        )

        let size = self.__matmul_num_elements(other)

        let res_last_dim = res.dims[res.dims.rank() - 1] # The dimension that is different for self and other
        let self_last_dim = self.dims[self.dims.rank() - 1] # the dimension that is the same for self and other

        # We use the for inside the parallel function to remove data races, so the vectorize function works in the last dimension of the res tensor and the for makes it so the for and vectorize function work on the penultimate dimension of the res tensor (so parallel works on the penultimate dimension of the res tensor)
        @parameter
        fn matmul_p(i: Int):
            for j in range(0, self_last_dim):
                @parameter
                fn matmul_v[nelts: Int](k: Int):
                    let index = i * (res_last_dim * self_last_dim) + j * res_last_dim + k # remove data races of parallel

                    var index_values_self = InlinedFixedVector[dims_average_size, Int](
                    self.dims.rank()
                    )
                    var index_values_other = InlinedFixedVector[dims_average_size, Int](
                        self.dims.rank()
                    )
                    var index_values_res = InlinedFixedVector[dims_average_size, Int](
                        self.dims.rank()
                    )

                    for i in range(self.dims.rank() - 2):
                        let index_pos = index
                            // matmul_suffix_product[self.dims.rank() - i]
                            % self.dims[i]
                        index_values_self.append(
                            index_pos
                        )
                        index_values_other.append(
                            index_pos
                        )
                        index_values_res.append(
                            index_pos
                        )
                    
                    index_values_self.append(
                        index // matmul_suffix_product[2] % self.dims[self.dims.rank() - 2]
                    )
                    index_values_self.append(
                        index // matmul_suffix_product[1] % self.dims[self.dims.rank() - 1]
                    )

                    index_values_other.append(
                        index // matmul_suffix_product[1] % other.dims[other.dims.rank() - 2]
                    )
                    index_values_other.append(
                        index // matmul_suffix_product[0] % other.dims[other.dims.rank() - 1]
                    )

                    index_values_res.append(
                        (index // matmul_suffix_product[2]) % res.dims[res.dims.rank() - 2]
                    )
                    index_values_res.append(
                        (index // matmul_suffix_product[0]) % res.dims[res.dims.rank() - 1]
                    )

                    res.store[nelts](
                        index_values_res,
                        res.load[nelts](index_values_res)
                        + self.load[1](index_values_self)
                        * other.load[nelts](index_values_other),
                    )

                vectorize[nelts, matmul_v](res_last_dim)

        parallelize[matmul_p](rt, size // (res_last_dim * self_last_dim), n_cores)

        return res ^
