from time import now
from tensorG import TensorG, TensorView
from benchmark import Benchmark
from runtime.llcl import num_cores, Runtime

alias simd_size_float = simdwidthof[DType.float32]()
alias mul_pool = 4
alias type_test = DType.float32


fn bench_sum(matrix_1: TensorG[type_test], matrix_2: TensorG[type_test]):
    _ = matrix_1 + matrix_1


fn bench_sum_vectorized(matrix_1: TensorG[type_test], matrix_2: TensorG[type_test]):
    _ = matrix_1.add[simd_size_float](matrix_2)


fn bench_sum_parallelized(
    matrix_1: TensorG[type_test],
    matrix_2: TensorG[type_test],
    rt: Runtime,
    num_workers: Int,
):
    _ = matrix_1.add[simd_size_float](matrix_2, rt, num_workers)


fn benchmark_sum(dims: TensorView):
    let matrix_1 = TensorG[type_test](True, dims)
    let matrix_2 = TensorG[type_test](True, dims)

    @parameter
    fn bench():
        _ = matrix_1 + matrix_2

    let normal = Benchmark().run[bench]() / 1e6
    print("normal:", normal, "ms")

    @parameter
    fn bench_vectorized():
        _ = matrix_1.add[simd_size_float](matrix_2)

    let vectorized = Benchmark().run[bench_vectorized]() / 1e6
    print("vectorized:", vectorized, "ms")

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            _ = matrix_1.add[simd_size_float](
                matrix_2, rt, mul_pool * rt.parallelism_level()
            )

        let parallelized = Benchmark().run[bench_parallel]() / 1e6
        # Prevent the tensors from being freed before the benchmark run
        _ = (matrix_1, matrix_2)
        print("Parallelized:", parallelized, "ms")


fn benchmark_mul(dims: TensorView):
    let matrix_1 = TensorG[type_test](True, dims)
    let matrix_2 = TensorG[type_test](True, dims)

    @parameter
    fn bench():
        _ = matrix_1 * matrix_2

    let normal = Benchmark().run[bench]() / 1e6
    print("normal:", normal, "ms")

    @parameter
    fn bench_vectorized():
        _ = matrix_1.mul[simd_size_float](matrix_2)

    let vectorized = Benchmark().run[bench_vectorized]() / 1e6
    print("vectorized:", vectorized, "ms")

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            _ = matrix_1.mul[simd_size_float](
                matrix_2, rt, mul_pool * rt.parallelism_level()
            )

        let parallelized = Benchmark().run[bench_parallel]() / 1e6
        # Prevent the tensors from being freed before the benchmark run
        _ = (matrix_1, matrix_2)
        print("Parallelized:", parallelized, "ms")


fn benchmark_matmul(dims: TensorView):
    let matrix_1 = TensorG[type_test](True, dims)
    let matrix_2 = TensorG[type_test](True, dims)

    @parameter
    fn bench():
        _ = matrix_1 @ matrix_2

    let normal = Benchmark().run[bench]() / 1e6
    print("normal:", normal, "ms")

    @parameter
    fn bench_vectorized():
        _ = matrix_1.matmul[simd_size_float](matrix_2)

    let vectorized = Benchmark().run[bench_vectorized]() / 1e6
    print("vectorized:", vectorized, "ms")

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            _ = matrix_1.matmul[simd_size_float](
                matrix_2, rt, mul_pool * rt.parallelism_level()
            )

        let parallelized = Benchmark().run[bench_parallel]() / 1e6
        # Prevent the tensors from being freed before the benchmark run
        _ = (matrix_1, matrix_2)
        print("Parallelized:", parallelized, "ms")


fn main():
    print("Benchmarking sum")
    benchmark_sum(TensorView(512, 512, 512))
    print("Benchmarking mul")
    benchmark_mul(TensorView(512, 512, 512))
    print("Benchmarking matmul")
    benchmark_matmul(TensorView(512, 512))
