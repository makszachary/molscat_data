def A_plus_squared(mf, i = 3/2, B = 2.97, B_hfs = 2441):
    return 0.5*(1+mf/(i+0.5))*(1+B/B_hfs*(1-mf/(i+0.5)))

def A_minus_squared(mf, i = 3/2, B = 2.97, B_hfs = 2441):
    return 0.5*(1-mf/(i+0.5))*(1-B/B_hfs*(1+mf/(i+0.5)))

def probabilities_from_matrix_elements_hot(f, mf, ms, i = 3/2, B = 2.97, B_hfs = 2441):
    if f != i+1/2:
        return 0
    probability = 1/4*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    if ms == 1/2:
        probability += 1/4*A_minus_squared(mf = mf + 1, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf,  i = i, B = B, B_hfs = B_hfs)
    elif ms == -1/2:
        probability += 1/4*A_plus_squared(mf = mf - 1, i = i, B = B, B_hfs = B_hfs)*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    return probability

def probabilities_from_matrix_elements_cold(f, mf, ms, i = 3/2, B = 2.97, B_hfs = 2441):
    if f == i + 1/2:
        if ms == 1/2:
            return 1/4*A_plus_squared(mf = mf + 1, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
        elif ms == -1/2:
            return 1/4*A_minus_squared(mf = mf - 1, i = i, B = B, B_hfs = B_hfs)*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    elif f == i - 1/2:
        if ms == 1/2:
            return 1/4*A_minus_squared(mf = mf + 1, i = i, B = B, B_hfs = B_hfs)*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
        elif ms == -1/2:
            return 1/4*A_plus_squared(mf = mf - 1, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    return None
