def MPAF(gradient):
    for k in gradient:
        gradient[k] = -gradient[k] * 10
    return gradient