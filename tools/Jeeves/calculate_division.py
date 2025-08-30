import gurobipy as gp
from gurobipy import GRB
from itertools import product
import concurrent.futures
from tqdm import tqdm  # 可选：用于显示进度条
print(gp.gurobi.version())
print(gp.__file__)

# 定义print_rank_0函数，如果不在megatron环境中则使用普通print
def print_rank_0(message):
    try:
        from megatron.training import print_rank_0 as megatron_print
        megatron_print(message)
    except ImportError:
        print(message)


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
        
    
    

def get_best_devision(PP, M, DP, CM, K, Delay, Memory_limit, comm_aware=True, memory_aware=True, Ft=1, Bt=2):
    """
    该函数使用Gurobi优化器建立并求解一个混合整数规划模型，用于寻找最优的划分方案。
    
    参数说明：
    - PP: 总阶段数（pipeline stages）
    - M: 总的microbatch的数量
    - DP: 数据并行度（data parallelism degree）
    - CM: 通信开销（communication cost）ms
    - K: 所有的模型层数总和（sum of x values）
    - Delay: 延迟常量（delay constant）ms
    - Memory_limit: 内存上限（memory constraint）, list[]，各个stage的limit不一定相同，单位为总剩余内存 / 单层激活占用
    - comm_aware: 是否考虑通信感知优化
    - memory_aware: 是否考虑内存感知优化
    - Ft: 前向时间（forward time）单层 ms
    - Bt: 后向时间（backward time）单层 ms

    返回值：
    - x_vals: 每个阶段分配的资源数量列表（最优解）
    """
    try:
        # 创建Gurobi模型
        model = gp.Model("CrossDC_OptimalPartition")
        model.Params.OutputFlag = 0  # 关闭输出
        model.Params.TimeLimit = 300  # 设置5分钟时间限制
        model.Params.MIPGap = 0.01   # 设置1%的优化间隙
        model.Params.NumericFocus = 3  # 提高数值稳定性
        
        # 决策变量：每个stage划分到的模型层数量
        x = model.addVars(PP, lb=1, vtype=GRB.INTEGER, name="x") 
        
        # 目标函数变量：最大执行时间
        t = model.addVar(lb=0, name="t")

        
        # ========== 添加基本约束 ==========
        # 1. 总层数约束
        model.addConstr(gp.quicksum(x[i] for i in range(PP)) == K, name="TotalLayers")
        
        # 2. 内存约束 - 只对pipeline前半部分的stage应用（因为后半部分主要是backward）
        if memory_aware:
            for s in range(PP):
                # 确保Memory_limit是列表，如果是单个值则复制
                if isinstance(Memory_limit, (int, float)):
                    mem_limit = Memory_limit
                else:
                    mem_limit = Memory_limit[s] if s < len(Memory_limit) else Memory_limit[0]
                
                # 内存约束：层数 * 平均激活microbatch数 <= 内存限制
                # 这里使用简化的激活microbatch估算：M // PP
                avg_active_microbatches = max(1, M // PP)
                model.addConstr(x[s] * avg_active_microbatches <= mem_limit, 
                               name=f"MemoryConstraint_{s}")
        
        # ========== 时间约束 - 管道并行执行时间模型 ==========
        # 3. 对每个stage，计算其执行时间上界
        for stage in range(PP):
            # 计算该stage的总计算时间（前向+后向）
            stage_compute_time = (Ft + Bt) * x[stage] * M
            
            # 添加通信时间（如果是跨DC的stage）
            comm_time = 0
            if comm_aware:
                # 假设stage >= PP//2的为跨DC通信
                if stage >= PP // 2:
                    comm_time = CM * DP + Delay  # 通信延迟
            
            # stage总时间约束
            total_stage_time = stage_compute_time + comm_time
            model.addConstr(t >= total_stage_time, name=f"StageTime_{stage}")
        
        # 4. 全局通信时间约束（管道并行的关键路径）
        if comm_aware:
            # 管道并行的临界路径时间
            critical_path_time = gp.quicksum((Ft + Bt) * x[s] for s in range(PP)) \
                               + CM * (DP + M - 1) + Delay * 2  # 双向延迟
            model.addConstr(t >= critical_path_time, name="CriticalPath")
        
        # ========== 设置目标函数 ==========
        model.setObjective(t, GRB.MINIMIZE)
        
        # ========== 求解 ==========
        model.optimize()
        
        # ========== 处理求解结果 ==========
        if model.status == GRB.OPTIMAL:
            x_vals = [int(x[i].X) for i in range(PP)]
            t_val = t.X
            print_rank_0(f"最优解找到: x={x_vals}, 总时间={t_val:.2f}ms")
            return x_vals
            
        elif model.status == GRB.INFEASIBLE:
            print_rank_0("模型不可行，尝试分析冲突约束...")
            model.computeIIS()
            print_rank_0("不可行约束:")
            for c in model.getConstrs():
                if c.IISConstr:
                    print_rank_0(f"- {c.ConstrName}")
            # 返回均匀分割作为fallback
            return [K // PP + (1 if i < K % PP else 0) for i in range(PP)]
            
        elif model.status == GRB.TIME_LIMIT:
            print_rank_0("求解超时，返回当前最优解")
            try:
                x_vals = [int(x[i].X) for i in range(PP)]
                return x_vals
            except:
                # 返回均匀分割作为fallback
                return [K // PP + (1 if i < K % PP else 0) for i in range(PP)]
                
        else:
            print_rank_0(f"求解失败，状态码: {model.status}")
            # 返回均匀分割作为fallback
            return [K // PP + (1 if i < K % PP else 0) for i in range(PP)]
            
    except Exception as e:
        print_rank_0(f"求解器异常: {e}")
        # 返回均匀分割作为fallback
        return [K // PP + (1 if i < K % PP else 0) for i in range(PP)]

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
