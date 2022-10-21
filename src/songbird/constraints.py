
def none_is_zero(params):

    LA_presence = params["lewis_acid"]
    LA_amount = float(params["lewis_acid_eq"])
    NU_amount = float(params["nucleophile_eq"])
    
    if LA_presence == 'none' and LA_amount > 0.05:
        return False
    elif LA_presence != 'none' and LA_amount < 0.05:
        return False
    elif LA_amount % 0.1 > 0.3:
        return False
    elif NU_amount % 0.5 > 0.3:
        return False
    else:
        return True

