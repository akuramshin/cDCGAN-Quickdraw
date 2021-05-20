stride = 1
padding = 0
kernel_size = 4
dilation = 1
output_padding = 0
H_in =4


# ConvTranspose
#H_out = (H_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1

#Conv2D
H_out = ((H_in + 2*padding - dilation*(kernel_size-1)-1)/stride)+1

print(H_out)