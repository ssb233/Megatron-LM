from calculate_division import *


if __name__ == "__main__":
    # 默认参数
    PP = 8
    DP = 64
    M = 48
    CM = 6 / DP
    K = 16
    Delay = 4
    Ft = 1
    Bt = 2
    Memory_limit = [16] * PP
    
    # 通过模型获取参数
    
    
    x_vals = get_best_devision(PP, M, DP, CM, K, Delay, Memory_limit, Ft, Bt)
    print(x_vals)