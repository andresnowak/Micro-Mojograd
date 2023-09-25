# Micro-Mojograd

For now this project is just about trying to recreate [Micrograd from andrej karpathy](https://github.com/karpathy/micrograd) but in mojo and with tensors.

The Tensors for now need to know the rank size at compile time, because when we do a permute we can change the strides of the tensor, and we need to know the position of each of the dimension and instead of doing a for for each of the dimensions, we can do an unroll instead, but we need to know the rank at compile time for this to work (for now this is the only way I know how to do it for it to be efficient).

## To-do

-   [x] Tensor: using 1d memory and not using MLIR
-   [x] Tensor Matmul
-   [x] Tensor sum elementwise
-   [x] Tensor Mul elementwise
-   [x] Tensor equal operator
-   [ ] Tensor broadcasting
-   [ ] Tensor reshape
-   [ ] Tensor transpose
-   [ ] Fix problem where parallel version of element-wise operations (add, mul, eq, etc.) are slower than the sequential version.
-   [ ] Autograd: _For now doing the autogradient part is difficult, because mojo still doesn't have complete ownership and lifetime support._
