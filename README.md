# Micro-Mojograd

For now this project is just about trying to recreate [Micrograd from andrej karpathy](https://github.com/karpathy/micrograd) but in mojo and with tensors.

## To-do
- [x] Tensor: using 1d memory and not using MLIR
- [x] Tensor Matmul
- [x] Tensor sum elementwise
- [x] Tensor Mul elementwise
- [x] Tensor equal operator
- [ ] Tensor broadcasting
- [ ] Autograd: *For now doing the autogradient part is difficult, because mojo still doesn't have complete ownership and lifetime support.*
