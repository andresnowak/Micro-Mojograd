from tensor_g import TensorG, TensorView
from python import Python, PythonObject
from runtime.llcl import num_cores, Runtime
from time import time

alias nelts = simdwidthof[DType.float64]()
alias type_f = DType.float64
alias mul_pool = 4


# Todo: Fix eq simd implementation, it seems to be wrong


fn test_matmul[
    rank_size: Int, Type: DType, nelts: Int
](
    A: TensorG[rank_size, Type],
    B: TensorG[rank_size, Type],
    C: TensorG[rank_size, Type],
) -> Bool:
    var flag = True

    var res = A @ B
    print("Matmul normal:", res == C)
    flag = flag and (res == C)
    print("Matmul normal, eq simd: ", res.eq[nelts](C))
    flag = flag and (res.eq[nelts](C))

    res = A.matmul[nelts](B)
    print("Matmul simd:", res == C)
    flag = flag and (res == C)
    print("Matmul simd, eq simd:", res.eq[nelts](C))
    flag = flag and (res.eq[nelts](C))

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            let res = A.matmul[nelts](B, rt, mul_pool * rt.parallelism_level())

            print("Matmul simd parallel:", res == C)
            flag = flag and (res == C)
            print("Matmul parallel, eq simd:", res.eq[nelts](C))
            flag = flag and (res.eq[nelts](C))
            print(
                "Matmul parallel, eq parallel:",
                res.eq[nelts](C, rt, mul_pool * rt.parallelism_level()),
            )
            flag = flag and (res.eq[nelts](C, rt, mul_pool * rt.parallelism_level()))

        bench_parallel()
        _ = (A, B, C, flag)

    return flag


fn test_add[
    rank_size: Int, Type: DType, nelts: Int
](
    A: TensorG[rank_size, Type],
    B: TensorG[rank_size, Type],
    C: TensorG[rank_size, Type],
) -> Bool:
    var flag = True

    var res = A + B
    print("Add normal:", res == C)
    flag = flag and (res == C)
    print("Add normal, eq simd: ", res.eq[nelts](C))
    flag = flag and (res.eq[nelts](C))

    res = A.add[nelts](B)
    print("Add simd:", res == C)
    flag = flag and (res == C)
    print("Add simd, eq simd:", res.eq[nelts](C))
    flag = flag and (res.eq[nelts](C))

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            let res = A.add[nelts](B, rt, mul_pool * rt.parallelism_level())

            print("Add simd parallel:", res == C)
            flag = flag and (res == C)
            print("Add parallel, eq simd:", res.eq[nelts](C))
            flag = flag and (res.eq[nelts](C))
            print(
                "Add parallel, eq parallel:",
                res.eq[nelts](C, rt, mul_pool * rt.parallelism_level()),
            )
            flag = flag and (res.eq[nelts](C, rt, mul_pool * rt.parallelism_level()))

        bench_parallel()
        _ = (A, B, C)

    return flag


fn test_mul[
    rank_size: Int, Type: DType, nelts: Int
](
    A: TensorG[rank_size, Type],
    B: TensorG[rank_size, Type],
    C: TensorG[rank_size, Type],
) -> Bool:
    var flag = True

    var res = A * B
    print("Mul normal:", res == C)
    flag = flag and (res == C)
    print("Mul normal, eq simd: ", res.eq[nelts](C))
    flag = flag and (res.eq[nelts](C))

    res = A.mul[nelts](B)
    print("Mul simd:", res == C)
    flag = flag and (res == C)
    print("Mul simd, eq simd:", res.eq[nelts](C))
    flag = flag and (res.eq[nelts](C))

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            let res = A.mul[nelts](B, rt, mul_pool * rt.parallelism_level())

            print("Mul simd parallel:", res == C)
            flag = flag and (res == C)
            print("Mul parallel, eq simd:", res.eq[nelts](C))
            flag = flag and (res.eq[nelts](C))
            print(
                "Mul parallel, eq parallel:",
                res.eq[nelts](C, rt, mul_pool * rt.parallelism_level()),
            )
            flag = flag and (res.eq[nelts](C, rt, mul_pool * rt.parallelism_level()))

        bench_parallel()
        _ = (A, B, C)

    return flag


fn create_tensor_from_numpy[
    rank_size: Int, Type: DType
](np: PythonObject, shape: TensorView[rank_size]) -> TensorG[rank_size, Type]:
    let size = shape.num_elements()

    var res = DynamicVector[FloatLiteral](size)
    res.reserve(
        size
    )  # the init function should do this, but it seems it doesn't (error)

    try:
        for i in range(size):
            if size == 1:
                res.push_back(np.__index__())
            else:
                res.push_back(np[i].__index__())
    except:
        print("Error converting numpy array to tensor")
        return TensorG[rank_size, Type](False, shape)

    return TensorG[rank_size, Type](res, shape)


fn test_same_dim[
    rank_size: Int
](I: Int, np_shape: PythonObject, tensor_shape: TensorView[rank_size]) -> Bool:
    from python import Python

    var flag = True
    try:
        let np = Python.import_module("numpy")

        var test_1 = np.random.randint(0, 20, tensor_shape.num_elements())
        var test_2 = np.random.randint(0, 20, tensor_shape.num_elements())

        let A = create_tensor_from_numpy[rank_size, type_f](test_1, tensor_shape)
        let B = create_tensor_from_numpy[rank_size, type_f](test_2, tensor_shape)

        # check shape size
        var shape_correct = A.dims.num_elements() == test_1.shape.__len__().__index__()
        debug_assert(shape_correct, "Shape size is not correct")
        shape_correct = B.dims.num_elements() == test_2.shape.__len__().__index__()
        debug_assert(shape_correct, "Shape size is not correct")

        test_1 = test_1.reshape(np_shape)
        test_2 = test_2.reshape(np_shape)

        # check rank
        var rank_correct = A.rank() == test_1.ndim.__index__()
        debug_assert(rank_correct, "Rank is not correct")
        rank_correct = B.rank() == test_2.ndim.__index__()
        debug_assert(rank_correct, "Rank is not correct")

        # Add test
        var res = test_1 + test_2

        res = res.reshape(tensor_shape.num_elements())
        var C = create_tensor_from_numpy[rank_size, type_f](res, tensor_shape)

        var res_correct = test_add[rank_size, type_f, nelts](A, B, C)
        flag = res_correct and flag
        print("Add test ", I, ":", res_correct)
        print()

        # Mul Test
        res = test_1 * test_2

        res = res.reshape(tensor_shape.num_elements())
        C = create_tensor_from_numpy[rank_size, type_f](res, tensor_shape)

        res_correct = test_mul[rank_size, type_f, nelts](A, B, C)
        flag = res_correct and flag
        print("Mul test ", I, ":", res_correct)
        print()

        # Matmul test
        res = np.matmul(test_1, test_2)

        if tensor_shape.rank() > 1:
            res = res.reshape(tensor_shape.num_elements())
            C = create_tensor_from_numpy[rank_size, type_f](res, tensor_shape)
        else:
            C = create_tensor_from_numpy[rank_size, type_f](
                res, TensorView[rank_size](1)
            )

        res_correct = test_matmul[rank_size, type_f, nelts](A, B, C)
        flag = res_correct and flag
        print("Matmul test ", I, ":", res_correct)
        print()
    except:
        print("Error importing numpy ", I)
        return False

    return flag


fn test_different_dim[
    rank_size: Int
](
    I: Int,
    np_shape_1: PythonObject,
    np_shape_2: PythonObject,
    tensor_shape_1: TensorView[rank_size],
    tensor_shape_2: TensorView[rank_size],
    tensor_shape_3: TensorView[rank_size],
) -> Bool:
    from python import Python

    var flag = True
    try:
        let np = Python.import_module("numpy")

        # Matmul test 2
        var test_1 = np.random.randint(0, 20, tensor_shape_1.num_elements())
        var test_2 = np.random.randint(0, 20, tensor_shape_2.num_elements())

        let A = create_tensor_from_numpy[rank_size, type_f](test_1, tensor_shape_1)
        let B = create_tensor_from_numpy[rank_size, type_f](test_2, tensor_shape_2)

        test_1 = test_1.reshape(np_shape_1)
        test_2 = test_2.reshape(np_shape_2)

        var res = np.matmul(test_1, test_2)

        res = res.reshape((tensor_shape_3.num_elements()))
        let C = create_tensor_from_numpy[rank_size, type_f](res, tensor_shape_3)

        let res_correct = test_matmul[rank_size, type_f, nelts](A, B, C)
        flag = res_correct and flag
        print("Matmul test ", I, ":", res_correct)
        print()
    except:
        print("Error importing numpy", I)
        return False

    return flag


fn main():
    print("nelts size:", nelts)

    var flag = True
    let start = time.now()

    flag = (
        test_same_dim(1, (5, 5, 5, 5, 5, 5), TensorView[6](5, 5, 5, 5, 5, 5)) and flag
    )
    flag = test_same_dim(2, (25, 25, 25), TensorView[3](25, 25, 25)) and flag
    flag = test_same_dim(3, (125, 125), TensorView[2](125, 125)) and flag
    flag = test_same_dim(4, (15_625), TensorView[1](15_625)) and flag

    flag = (
        test_different_dim(
            5,
            (2, 2, 3),
            (2, 3, 2),
            TensorView[3](2, 2, 3),
            TensorView[3](2, 3, 2),
            TensorView[3](2, 2, 2),
        )
        and flag
    )
    flag = (
        test_different_dim(
            6,
            (2, 4, 3),
            (2, 3, 5),
            TensorView[3](2, 4, 3),
            TensorView[3](2, 3, 5),
            TensorView[3](2, 4, 5),
        )
        and flag
    )
    flag = (
        test_different_dim(
            7,
            (5, 2, 4, 3),
            (5, 2, 3, 5),
            TensorView[4](5, 2, 4, 3),
            TensorView[4](5, 2, 3, 5),
            TensorView[4](5, 2, 4, 5),
        )
        and flag
    )

    let end = time.now()

    print("All tests passed:", flag)
    print("Time elapsed: ", (end - start) // 1000000, "ms")
