# Micro-Mojograd

This is for now going to be a **Toy project** to learn about how do tensors work, how to make them efficient, how to do the computational graph efficiently and the Autograd part.

When the language is a bit more mature and has complete lifetime and ownership support and Traits support, I will try to make a more complete version of this library, or maybe help another mojo project like [numojo](https://github.com/MadAlex1997/Mojo-Arrays)

And it is going to be inspired by:

-   [Micrograd](https://github.com/karpathy/micrograd)
-   [Pytorch](https://github.com/pytorch/pytorch)
-   [Tinygrad](https://github.com/tinygrad/tinygrad)

## To-do

-   [x] Tensor: using 1d memory and not using MLIR
-   [x] Tensor Matmul
-   [x] Tensor sum elementwise
-   [x] Tensor Mul elementwise
-   [x] Tensor equal operator
-   [ ] Tensor broadcasting
-   [ ] Fix problem where parallel version of element-wise operations (add, mul, eq, etc.) are slower than the sequential version.
-   [ ] Autograd: _For now doing the autogradient part is difficult, because mojo still doesn't have complete ownership and lifetime support._
