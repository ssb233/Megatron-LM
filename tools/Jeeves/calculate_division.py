import gurobipy as gp
from gurobipy import GRB
from itertools import product
import concurrent.futures
from tqdm import tqdm  # 可选：用于显示进度条
print(gp.gurobi.version())
print(gp.__file__)


def get_division_result(PP, M, DP, CM, K, Delay, Memory_limit, comm_aware=True, memory_aware=True, Ft=1, Bt=2):
    # 根据comm_aware和memory-aware的参数，选择对应的求解方式
    
    x_val = get_best_devision(PP, M, DP, CM, K, Delay, Memory_limit, comm_aware, memory_aware, Ft, Bt)
    
    divide_result = [['decoder'] * i for i in x_val] # 确保x_val返回的是整数列表
    divide_result[0].insert(0, 'embedding')
    divide_result[-1].append('loss')
    return divide_result
def get_optimal_num(PP,  DP, CM, stage, delay, ):
    forward_time = 2
    backward_time = 2 * forward_time
    bound = (PP - stage) * (forward_time + backward_time) + (2) * CM + 2 * delay
    if CM * DP > forward_time + backward_time:
        # 如果通信时间大于前向和后向计算时间之和，则内存限制为通信时间
        for i in range(100):
            if CM * DP * i  >= bound:
                return (i)
    else:
        # 如果通信时间小于等于前向和后向计算时间之和，则内存限制为前向和后向计算时间之和
        for i in range(100):
            if (forward_time + backward_time) * i >= bound:
                return (i)

def get_num_delta(As, M):
    for i in range(100):
        if (As + 1) * i >= M:
            return i
        
def check_Ks_list(x_vals, Ks_list, As_list, PP,  DP, CM,  delay):
    if CM * DP > max(x_vals) * 3:
        vital = CM * DP
    else:
        vital = max(x_vals) * 3
    Real_Ks_list = []
    Real = True
    for i in range(PP // 2):
        bound = 0
        for j in range(i, PP):
            bound += 3 * x_vals[j]
        bound += 2 * CM + 2 * delay
        for k in range(100):
            if vital * k >= bound:
                break
        Real_Ks_list.append(k)
        if k != Ks_list[i] and k >= As_list[i]:
            print(f"Stage {i} has a mismatch in Ks_list: expected {k}, got {Ks_list[i]}")
            Real = False
    print("Real Ks_list:", Real_Ks_list)

    return Real

def get_heuristic_division_total_concurrent(PP, M, DP, CM, K, Delay, Memory_limit):
    candidate_pairs = get_Ks_As_list(PP, M, DP, CM, K, Delay, Memory_limit)  # 获取所有候选 (K_list, A_list) 对
    consist_pair_list = []
    
    # 并行处理每个候选对
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                get_heuristic_division_single, 
                PP, M, DP, CM, K, Delay, Memory_limit, 
                pair[0], pair[1]
            ): pair for pair in candidate_pairs
        }
        
        # 使用 tqdm 显示进度（可选）
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(candidate_pairs),
            desc="Processing pairs"
        ):
            pair = futures[future]
            try:
                x_vals, t_val = future.result()
                if x_vals:
                    consist_pair_list.append({
                        "pair": pair,
                        "x_vals": x_vals,
                        "t_val": t_val
                    })
            except Exception as e:
                print(f"Error processing pair {pair}: {e}")

    if not consist_pair_list:
        print("No consistent pairs found.")
        return [], None
    
    # 找到最小 t_val 的结果
    for entry in consist_pair_list:
        print(f"Pair: {entry['pair']}, x_vals: {entry['x_vals']}, t_val: {entry['t_val']}")
    min_result = min(consist_pair_list, key=lambda x: x["t_val"])
    # print(f"Minimum t_val found: {min_result['t_val']} for pair {min_result['pair']}")
    return min_result["x_vals"], min_result["t_val"]
            
def get_heuristic_division_total(PP, M, DP, CM, K, Delay, Memory_limit):
    candidata_pair = get_Ks_As_list(PP, M, DP, CM, K, Delay, Memory_limit)
    consist_pair_list = []
    consist_pair_count = 0
    for pair in candidata_pair:
        print("Testing pair:", pair)
        x_vals, t_val = get_heuristic_division_single(PP, M, DP, CM, K, Delay, Memory_limit, pair[0], pair[1])
        if x_vals != []:
            consist_pair_list.append({})
            consist_pair_list[consist_pair_count]["pair"] = pair
            consist_pair_list[consist_pair_count]["x_vals"] = x_vals
            consist_pair_list[consist_pair_count]["t_val"] = t_val
            consist_pair_count += 1
    if consist_pair_count == 0:
        print("No consistent pairs found.")
        return []
    else:
        min_t_val = float('inf')
        min_x = None
        for i in range(consist_pair_count):
            if consist_pair_list[i]['t_val'] <=  min_t_val:
                min_t_val = consist_pair_list[i]['t_val']
                min_x = consist_pair_list[i]['x_vals']
        return min_x, min_t_val
        
    

def get_heuristic_division_single(PP, M, DP, CM, K, Delay, Memory_limit, Ks_list, As_list):
    ####
    ### Ks_list: 每个stage的期待的warm_up中 micro batch的数量,
    ### As_list: 每个stage最多放的micro batch的数量
    ####
    
    
    model = gp.Model()
    x = model.addVars(PP, lb=0, name="x")
    # for i in range(PP):
    #     model.addConstr(x[i] == 2, name=f"x_eq_2_{i}")
    num_delta = [get_num_delta(As_list[i], M) for i in range(PP)]
    # print("delay:", Delay)
    delta_time = model.addVar(lb=0, name="delta_time")
    t = model.addVar(lb=0, name="t")
    x_max = model.addVar(lb=0, name="x_max")
    z = model.addVars(PP, vtype=GRB.BINARY, name="z_max")
    BIG_M = 1e6

    for i in range(PP):
        model.addConstr(x[i] <= x_max, name=f"x_leq_xmax_{i}")
        model.addConstr(x_max <= x[i] + (1 - z[i]) * BIG_M, name=f"xmax_leq_xi_if_z_{i}")

    model.addConstr(gp.quicksum(z[i] for i in range(PP)) == 1, name="only_one_z")
    # Inter = max(x_max, CM * DP)
    z_inter = model.addVar(vtype=GRB.BINARY, name="z_inter")
    Inter = model.addVar(lb=0, name="Inter")
    CM_DP = CM * DP
    # print("CM_DP:", CM_DP)
    model.addConstr(Inter <= x_max + (1 - z_inter) * 1e6, name="inter_leq_xmax_if_z1")
    model.addConstr(Inter <= CM_DP + z_inter * 1e6, name="inter_leq_cmdp_if_z0")
    model.addConstr(Inter >= x_max, name="Inter_ge_x_max")
    model.addConstr(Inter >= CM_DP, name="Inter_ge_CM_DP")
    
    #####内存限制
    for s in range(int(PP / 2 -2)):
        model.addConstr(As_list[s] * x[s] <= Memory_limit, name=f"mem_limit_s{s}")
        f1 = gp.quicksum(3 * x[i] for i in range(s, PP)) + 2 * CM + 2 * Delay
        f2 = 3 * Inter * Ks_list[s]
        # f2 = 6 * Ks_list[s]
        model.addConstr(f1 <= f2, name=f"opt_bound_s{s}")
        
        
    ######Extra Time by bubble
    for s in range(int(PP / 2 - 2)):
        if As_list[s] < Ks_list[s]:
            expr_part_1 = model.addVar(lb=0, name=f"expr_part_{s}")
            model.addConstr(expr_part_1 == num_delta[s] * (
                gp.quicksum(3 * x[i] for i in range(s, PP))
                + (2) * CM
                + 2 * Delay
                - As_list[s] * Inter * 3
            ), name=f"expr_part_def_{s}")
            model.addConstr(expr_part_1  <= delta_time, name=f"delta_leq_dt_{s}")
            expr_par_2 = model.addVar(lb=0, name=f"expr_part_2_{s}")
            model.addConstr(expr_par_2 == num_delta[s] * (
                (Ks_list[s] - As_list[s] - 1) * 3 * Inter + 2 * x[s]
            ), name=f"expr_part_2_def_{s}")
            model.addConstr(expr_par_2 <= delta_time, name=f"delta_leq_dt_2_{s}")
    
    ######objective function########
    for stage in range(PP):
        # obj_1
        obj_1 = gp.quicksum(3 * x[i] for i in range(stage + 1))
        obj_1 += (M - 1) * 3 * x[stage]
        if stage >= PP / 2:
            obj_1 += CM * (DP + 1) + 2 * Delay
        model.addConstr(t >= obj_1, name=f"MaxConstraint_{stage}_1")
        obj_2 = gp.quicksum(3 * x[i] for i in range(PP))
        obj_2 += (DP + 1) * CM + 2 * Delay
        obj_2 += 3 * x[stage] * (M - 1)
        obj_2 += delta_time
        model.addConstr(t >= obj_2, name=f"MaxConstraint_{stage}_2")
        
    # 通信上限
    obj_cm = gp.quicksum(3 * x[stage] for stage in range(PP))
    obj_cm += (DP + 1) * CM + (M - 1) * DP * CM + 2 * Delay
    obj_cm += delta_time
    model.addConstr(t >= obj_cm, name="MaxConstraint_Communication")
    # 总数量限制
    model.addConstr(gp.quicksum(x[i] for i in range(PP)) == K, name="SumConstraint")

    # 目标函数
    model.setObjective(t, GRB.MINIMIZE)
    # gurobi不显示计算过程
    model.Params.OutputFlag = 0  # 关闭所有输出
    
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        return [], 0
    x_vals = [x[i].X for i in range(PP)]
    # print("Optimal values of x:", x_vals)
    t_val = t.X
    # print("Optimal value of t:", t_val)
    delta_time_val = delta_time.X
    # print("delta_time:", delta_time_val)
    # print("Inter:", Inter.X)
    # print("x_max:", x_max.X)
    # print("z_inter:", z_inter.X)
    
    # #####check Ks_list
    # if not check_Ks_list(x_vals, Ks_list, As_list, PP, DP, CM, Delay):
    #     print("Ks_list is not consistent with x_vals.")
    #     return [], 0
    # else:
    #     print("Ks_list is consistent with x_vals.")
    return x_vals, t_val
        
    
    

def get_best_devision(PP, M, DP, CM, K, Delay, Memory_limit, Ft, Bt):
    """
    该函数使用Gurobi优化器建立并求解一个混合整数规划模型，用于寻找最优的划分方案。
    
    参数说明：
    - PP: 总阶段数（pipeline stages）
    - M: 总的microbatch的数量
    - DP: 数据并行度（data parallelism degree）
    - CM: 通信开销（communication cost）
    - K: 所有的模型层数总和（sum of x values）
    - Delay: 延迟常量（delay constant）
    - Memory_limit: 内存上限（memory constraint）, list[]，各个stage的limit不一定相同，单位为总剩余内存 / 单层激活占用
    - Ft: 前向时间（forward time）单层
    - Bt: 后向时间（backward time）

    返回值：
    - x_vals: 每个阶段分配的资源数量列表（最优解）
    
    前后向时间应该当作1和2了
    """
    # 每个stage划分到的模型的层数量，这里可能需要修改为整数
    model = gp.Model()
    x = model.addVars(PP, lb=0, name="x") 
    
    # optimal_num: 每个stage的，无内存限制下的不引入额外的延迟的最优的激活的microbatch的数量（这个一开始应该也是不确定的）
    optimal_num = model.addVars(PP, lb=0, vtype=GRB.INTEGER, name="optimal_num")
    
    # delta_time: 由于内存限制导致的microbatch的延后，带来的额外延迟时间
    delta_time = model.addVar(lb=0, name="delta_time")
    
    # actual_num: 每个stage的，实际使用的激活的microbatch的数量
    actual_num = model.addVars(PP, lb=0, vtype=GRB.INTEGER, name="actual_num")
    
    # num_delta: extra_time的周期
    num_delta = model.addVars(PP, lb=0, vtype=GRB.INTEGER, name="num_delta")
    
    
    # t: 最终目标函数变量，表示最大执行时间
    t = model.addVar(lb=0, name="t")

    BIG_M = 1e6  # 大M法中的大常数，用于线性化逻辑约束
    
    # z: 二进制变量，用于判断optimal_num与actual_num是否不同
    z = model.addVars(PP, vtype=GRB.BINARY, name="z")

    # ========== 添加内存相关约束 ==========
    for s in range(int(PP / 2)):
        # 实际的是要考虑内存的，最优的是不考虑内存的，因此要小于等于
        model.addConstr(actual_num[s] <= optimal_num[s], name=f"actual_num_s{s}_leq_optimal")
        
        # 实际的激活值的数量*模型层数不能超过内存限制，要统一单位
        model.addConstr(actual_num[s] * x[s] <= Memory_limit[s], name=f"mem_limit_s{s}")
        
        # 上界约束：确保资源分配不会导致过高的计算负载
        f1 = gp.quicksum((Ft + Bt) * x[i] for i in range(s, PP)) + 2 * CM + 2 * Delay
        # f2 = 3 * 2 * optimal_num[s] # 这里应该是把模型层数当作2了？？？，这是stage0上第一个microbatch的反向的最晚结束时间???
        f2 = (Ft + Bt) * (K // PP) * optimal_num[s]
        model.addConstr(f1 <= f2, name=f"opt_bound_s{s}")

    # ========== 添加时间相关约束 ==========
    for stage in range(PP):
        # 第一类时间约束：计算bound公式
        obj_1 = gp.quicksum((Ft + Bt) * x[i] for i in range(stage + 1)) # 到当前stage的前后向
        obj_1 += (M - 1) * (Ft + Bt) * x[stage] # 中间必算的M-1个microbatch的前后向
        if stage >= PP / 2:
            obj_1 += CM * (DP + 1) + 2 * Delay
        model.addConstr(t >= obj_1, name=f"MaxConstraint_{stage}_1")

        # 第二类时间约束：通讯bound公式
        obj_2 = gp.quicksum((Ft + Bt) * x[i] for i in range(PP))
        obj_2 += (DP + 1) * CM + 2 * Delay
        obj_2 += (Ft + Bt) * x[stage] * (M - 1)

        if stage < PP / 2:
            # 当optimal_num != actual_num时，引入额外的时间开销
            
            # expr_part: 计算划分差异带来的额外时间项
            expr_part = model.addVar(lb=-gp.GRB.INFINITY, name=f"expr_part_{stage}")
            model.addConstr(expr_part == (
                (optimal_num[stage] - actual_num[stage] - 1) * (Ft + Bt) * x[stage]
                + gp.quicksum((Ft + Bt) * x[i] for i in range(stage, PP))
                + (2) * CM
                + 2 * Delay
                - actual_num[stage] * x[stage] * (Ft + Bt)
            ), name=f"expr_part_def_{stage}")
            
            # product: 周期 * 单周期extraTime
            product = model.addVar(lb=-gp.GRB.INFINITY, name=f"product_{stage}")
            model.addConstr(product == num_delta[stage] * expr_part, name=f"product_def_{stage}")
            
            # num_delta 约束：num_delta就是重复周期数
            model.addConstr((num_delta[stage] + 1) * actual_num[stage] >= M, 
                       name=f"num_delta_constraint_{stage}")
                
            # 使用z[stage]控制是否激活delta_time项
            diff = model.addVar(lb=-BIG_M, ub=BIG_M, name=f"diff_{stage}")
            model.addConstr(diff == optimal_num[stage] - actual_num[stage], name=f"diff_def_{stage}")
            
            # 如果optimal_num != actual_num，则z=1
            model.addGenConstrIndicator(z[stage], True, diff >= 1e-5, name=f"z1_if_diff_{stage}")
            model.addGenConstrIndicator(z[stage], True, diff <= -1e-5, name=f"z1_if_diff_neg_{stage}")
            
            # 如果optimal_num == actual_num，则z=0
            model.addGenConstrIndicator(z[stage], False, diff == 0, name=f"z0_if_equal_{stage}")
            
            # activated_product: 只在z=1时才将product加入目标函数
            activated_product = model.addVar(lb=0, name=f"activated_product_{stage}")
            model.addConstr(activated_product == z[stage] * product)

            # 将激活后的delta_time加入obj_2
            model.addConstr(activated_product <= delta_time, name=f"delta_leq_dt_{stage}")
            obj_2 += activated_product

        model.addConstr(t >= obj_2, name=f"MaxConstraint_{stage}_2")

    # ========== 通信时间约束，通用的通信总时间计算 ==========
    obj_cm = gp.quicksum((Ft + Bt) * x[stage] for stage in range(PP))
    obj_cm += (DP + 1) * CM + (M - 1) * DP * CM + 2 * Delay
    obj_cm += delta_time
    model.addConstr(t >= obj_cm, name="MaxConstraint_Communication")

    # ========== 总模型层数约束 ==========
    model.addConstr(gp.quicksum(x[i] for i in range(PP)) == K, name="SumConstraint")

    # ========== 设置目标函数，最小化总时间==========
    model.setObjective(t, GRB.MINIMIZE)

    # ========== 求解设置 ==========
    model.Params.DualReductions = 0
    model.Params.NonConvex = 2
    model.optimize()

    # ========== 输出求解结果 ==========
    print("Solver status code:", model.Status)
    if model.status == GRB.INFEASIBLE:
        # 若模型不可行，输出冲突约束
        model.computeIIS()
        model.write("model.ilp")
        print("不可行模型已保存为 model.ilp")
        
        print("\n冲突约束列表:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"- {c.ConstrName}: {model.getRow(c)} {c.Sense} {c.RHS}")

    x_vals = [x[i].X for i in range(PP)]
    print("Optimal values of x:", x_vals)
    t_val = t.X
    print("Optimal value of t:", t_val)
    num_vals = [actual_num[i].X for i in range(PP)]
    print("actual values of num:", num_vals)
    optimal_num_vals = [optimal_num[i].X for i in range(PP)]
    print("optimal values of num:", optimal_num_vals)
    print("delta_time:", delta_time.X)
    num_delta_vals = [num_delta[i].X for i in range(PP)]
    print("num_delta values:", num_delta_vals)
    
    return x_vals

def generate_variants(K_list, A_list, limit):
    PP = len(K_list)
    assert len(A_list) == PP, "K_list and A_list must have the same length"
    
    # Step 1: Generate all possible K_variants
    K_options = []
    for K, A in zip(K_list, A_list):
        if K >= limit:
            K_options.append([K - 1, K, K + 1])
        else:
            K_options.append([K])
    
    # Generate all K_variants using Cartesian product
    K_variants = product(*K_options)
    
    variants = []
    for K_variant in K_variants:
        # Step 2: For each K_variant, generate all possible A_variants
        A_options = []
        for A, K in zip(A_list, K_variant):
            min_val = min(A, K)
            max_val = K

            A_options.append(range(min_val, max_val + 1))
        
        # Generate all A_variants for this K_variant
        A_variants = product(*A_options)
        
        # Step 3: Pair K_variant with each A_variant
        for A_variant in A_variants:
            variants.append((list(K_variant), list(A_variant)))
    
    return variants

def get_Ks_As_list(PP, M, DP, CM, K, Delay, Memory_limit):
    Ks_list = []
    for i in range(PP):
        if i < PP // 2:
            Ks_list.append(get_optimal_num(PP, DP, CM, i, Delay))
        else:
            Ks_list.append(0)
    print("Ks_list:", Ks_list)
    As_limit = Memory_limit // (K // PP)
    As_list = []
    for i in range(PP):
        if Ks_list[i] > As_limit:
            As_list.append(As_limit)
        else:
            As_list.append(Ks_list[i])
    candidata_pair = generate_variants(Ks_list, As_list, As_limit)
    return candidata_pair

# if __name__ == "__main__":
    # DP = 64
    # PP = 8
    # M = 48 ##micro batch数量
    # # CM = 1.5 ###通信的时间
    # CM = 6 / DP
    # K = 16##总的layer的数量
    # Delay = 4
    # Memory_limit = 16  ####内存的限制
    # x_vals = get_best_devision(PP, M, DP, CM, K, Delay, Memory_limit)
    
    # x_vals, t_val = get_heuristic_division_total_concurrent(PP, M, DP, CM, K, Delay, Memory_limit)
    # print(x_vals, t_val)
    
    # Ks_list = []
    # for i in range(PP):
    #     if i < PP // 2:
    #         Ks_list.append(get_optimal_num(PP, DP, CM, i, Delay))
    #     else:
    #         Ks_list.append(0)
    # print("Ks_list:", Ks_list)
    # As_limit = Memory_limit // (K // PP)
    # As_list = []
    # for i in range(PP):
    #     if Ks_list[i] > As_limit:
    #         As_list.append(As_limit)
    #     else:
    #         As_list.append(Ks_list[i])
            
    # candidata_pair = generate_variants(Ks_list, As_list, As_limit)
    # print("Generated candidate pairs:", len(candidata_pair))
    # consist_pair_count = 0
    # consist_pair_list = []
    # for pair in candidata_pair:
    #     print("Testing pair:", pair)
    #     x_vals, t_val = get_heuristic_division_single(PP, M, DP, CM, K, Delay, Memory_limit, pair[0], pair[1])
    #     if x_vals != []:
    #         consist_pair_list.append({})
    #         consist_pair_list[consist_pair_count]["pair"] = pair
    #         consist_pair_list[consist_pair_count]["x_vals"] = x_vals
    #         consist_pair_list[consist_pair_count]["t_val"] = t_val
    #         consist_pair_count += 1
            
    # print(consist_pair_count, "pairs are consistent with the model.")
    # for i in range(consist_pair_count):
    #     print(f"Pair {i + 1}: {consist_pair_list[i]['pair']}, x_vals: {consist_pair_list[i]['x_vals']}, t_val: {consist_pair_list[i]['t_val']}")
