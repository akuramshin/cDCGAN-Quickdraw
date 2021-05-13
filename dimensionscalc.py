stride = 2
padding = 1
kernel_size = 4
dilation = 1
output_padding = 0
H_in = 7



H_out = (H_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
print(H_out)