def get_alpha_value(alpha, base):
    '''
    Nhận alpha_value từ alpha_value và Rope_freq_base
    '''
    if base > 0:
        return (base / 10000.) ** (63 / 64.)
    else:
        return alpha


def get_rope_freq_base(alpha, base):
    '''
    Nhận Rope_freq_base từ alpha_value và Rope_freq_base
    '''
    if base > 0:
        return base
    else:
        return 10000 * alpha ** (64 / 63.)
