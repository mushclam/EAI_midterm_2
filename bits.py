def fourbit(string): 
    if len(string) != 100:
        raise Exception("Error: The length of every chromosome must be 100")
    f = 0
    for i in range(0,99,4):
        if sum(string[i:i+4]) == 4:
            f += 4
        else:
            f += 3 - sum(string[i:i+4])
    return f