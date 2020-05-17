def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


if __name__ == "__main__":
    H, W = 4, 4
    KH, KW = 3, 3
    SH, SW = 1, 1
    PH, PW = 1, 1

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KH, SW, PW)

    print(OH, OW)
