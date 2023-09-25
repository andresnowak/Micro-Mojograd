from time import now
from tensor_g import TensorG, TensorView
from benchmark import Benchmark
from runtime.llcl import num_cores, Runtime
from time import time

alias simd_size_float = simdwidthof[DType.float32]()
alias mul_pool = 4
alias type_test = DType.float32


fn benchmark_sum[rank_size: Int](dims: TensorView[rank_size]):
    let matrix_1 = TensorG[rank_size, type_test](True, dims)
    let matrix_2 = TensorG[rank_size, type_test](True, dims)

    @parameter
    fn bench():
        _ = matrix_1 + matrix_2

    let normal = Benchmark().run[bench]()
    print("normal:", normal / 1e6, "ms")
    print("normal:", normal, "ns")

    @parameter
    fn bench_vectorized():
        _ = matrix_1.add[simd_size_float](matrix_2)

    let vectorized = Benchmark().run[bench_vectorized]()
    print("vectorized:", vectorized / 1e6, "ms")
    print("vectorized:", vectorized, "ns")

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            _ = matrix_1.add[simd_size_float](
                matrix_2, rt, mul_pool * rt.parallelism_level()
            )

        let parallelized = Benchmark().run[bench_parallel]()
        # Prevent the tensors from being freed before the benchmark run
        _ = (matrix_1, matrix_2)
        print("Parallelized:", parallelized / 1e6, "ms")
        print("Parallelized:", parallelized, "ns")


fn benchmark_mul[rank_size: Int](dims: TensorView[rank_size]):
    let matrix_1 = TensorG[rank_size, type_test](True, dims)
    let matrix_2 = TensorG[rank_size, type_test](True, dims)

    @parameter
    fn bench():
        _ = matrix_1 * matrix_2

    let normal = Benchmark().run[bench]()
    print("normal:", normal / 1e6, "ms")
    print("normal:", normal, "ns")

    @parameter
    fn bench_vectorized():
        _ = matrix_1.mul[simd_size_float](matrix_2)

    let vectorized = Benchmark().run[bench_vectorized]()
    print("vectorized:", vectorized / 1e6, "ms")
    print("vectorized:", vectorized, "ns")

    with Runtime(num_cores()) as rt:

        @always_inline
        @parameter
        fn bench_parallel():
            _ = matrix_1.mul[simd_size_float](
                matrix_2, rt, mul_pool * rt.parallelism_level()
            )

        let parallelized = Benchmark().run[bench_parallel]()
        # Prevent the tensors from being freed before the benchmark run
        _ = (matrix_1, matrix_2)
        print("Parallelized:", parallelized / 1e6, "ms")
        print("Parallelized:", parallelized, "ns")


fn benchmark_matmul[rank_size: Int](dims: TensorView[rank_size]):
    let matrix_1 = TensorG[rank_size, type_test](True, dims)
    let matrix_2 = TensorG[rank_size, type_test](True, dims)

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
    let start = time.now()
    print("Benchmarking sum")
    benchmark_sum(TensorView[3](512, 512, 512))
    print("Benchmarking mul")
    benchmark_mul(TensorView[3](512, 512, 512))
    print("Benchmarking matmul")
    benchmark_matmul(TensorView[3](512, 512))
    let end = time.now()

    print("Elapsed time:", (end - start) // 1_000_000, "ms")
