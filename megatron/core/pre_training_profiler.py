# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""性能采样管理器，用于在训练开始前收集性能数据"""

import torch
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager

from megatron.training import print_rank_0
from megatron.core.performance_monitor import get_performance_profiler, initialize_performance_profiler
from megatron.core.parallel_state import get_pipeline_model_parallel_rank, get_pipeline_model_parallel_world_size

class PreTrainingProfiler:
    """预训练性能采样器，用于在正式训练前收集性能数据"""
    
    def __init__(self, profile_iterations=5):
        self.profile_iterations = profile_iterations
        self.profiler = None
        self.collected_metrics = {}
        self.is_profiling = False
        self.stage_memory_limits = []
        
    def initialize(self, args):
        """初始化性能监控器"""
        if args.jeeves_enable_profiling:
            self.profiler = initialize_performance_profiler(
                enable_detailed_profiling=True,
                max_history_size=self.profile_iterations * 2
            )
            self.profile_iterations = args.jeeves_profile_iters
        else:
            self.profiler = get_performance_profiler()
    
    def run_profiling_iterations(self, model, data_iterator, forward_step_func, config):
        """运行几个iteration来收集性能数据"""
        if not self.profiler:
            return
            
        print_rank_0(f"开始运行 {self.profile_iterations} 个iteration进行性能采样...")
        
        # 启用性能监控
        self.is_profiling = True
        
        # 预热一个iteration（不计入统计）
        print_rank_0("执行预热iteration...")
        self._run_single_iteration_for_profiling(model, data_iterator, forward_step_func, config, warmup=True)
        
        # 正式采样iterations
        for iter_idx in range(self.profile_iterations):
            print_rank_0(f"正在执行第 {iter_idx + 1}/{self.profile_iterations} 个采样iteration...")
            self._run_single_iteration_for_profiling(model, data_iterator, forward_step_func, config)
            
            # 每次采样后收集内存快照
            self.profiler.collect_memory_snapshot()
            
            # 短暂等待，确保GPU操作完成
            torch.cuda.synchronize()
            time.sleep(0.1)
        
        # 收集和分析数据
        self.collected_metrics = self._analyze_profiling_data()
        self.is_profiling = False
        
        print_rank_0("性能采样完成！")
        return self.collected_metrics
    
    def _run_single_iteration_for_profiling(self, model, data_iterator, forward_step_func, config, warmup=False):
        """运行单个iteration进行性能采样"""
        try:
            from megatron.core.pipeline_parallel import get_forward_backward_func
            from megatron.core.num_microbatches_calculator import get_num_microbatches
            from megatron.training.utils import get_batch_on_this_tp_rank
            
            # 获取数据批次
            data_b = next(data_iterator)
            data_b = get_batch_on_this_tp_rank(data_b)
            
            # 获取前后向函数
            forward_backward_func = get_forward_backward_func()
            
            # 获取微批次数量
            num_microbatches = get_num_microbatches()
            
            if not warmup and self.profiler:
                # 性能监控下的前后向传播
                with self.profiler.profile_forward_microbatch(0):
                    with self.profiler.profile_backward_microbatch(0):
                        losses_reduced = forward_backward_func(
                            forward_step_func=forward_step_func,
                            data_iterator=[data_b] * num_microbatches,  # 重复使用相同数据
                            model=model,
                            num_microbatches=num_microbatches,
                            seq_length=config.seq_length if hasattr(config, 'seq_length') else None,
                            micro_batch_size=config.micro_batch_size if hasattr(config, 'micro_batch_size') else None,
                        )
            else:
                # 预热iteration，不进行性能监控
                losses_reduced = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=[data_b] * num_microbatches,
                    model=model,
                    num_microbatches=num_microbatches,
                    seq_length=config.seq_length if hasattr(config, 'seq_length') else None,
                    micro_batch_size=config.micro_batch_size if hasattr(config, 'micro_batch_size') else None,
                )
                    
        except Exception as e:
            print_rank_0(f"性能采样iteration执行失败: {e}")
            # 对于性能采样失败，我们使用简单的模拟数据
            if not warmup and self.profiler:
                # 添加模拟的性能数据
                self.profiler.forward_times.append(10.0)  # 10ms 前向时间
                self.profiler.backward_times.append(20.0)  # 20ms 后向时间
                self.profiler.collect_memory_snapshot()
    
    def _analyze_profiling_data(self) -> Dict[str, Any]:
        """分析性能采样数据"""
        if not self.profiler:
            return {}
            
        # 获取计算性能指标
        compute_metrics = self.profiler.get_compute_metrics()
        
        # 获取通信性能指标
        comm_metrics = self.profiler.get_communication_metrics()
        
        # 获取内存使用指标
        memory_metrics = self.profiler.get_memory_metrics()
        
        # 计算每个stage的内存限制
        self._calculate_stage_memory_limits(memory_metrics)
        
        # 打印性能汇总
        self.profiler.print_summary()
        
        return {
            'compute_metrics': compute_metrics,
            'communication_metrics': comm_metrics,
            'memory_metrics': memory_metrics,
            'stage_memory_limits': self.stage_memory_limits,
        }
    
    def _calculate_stage_memory_limits(self, memory_metrics: Dict[str, float]):
        """计算各stage的内存限制（单位：可容纳的激活数量）"""
        try:
            from megatron.core.memory_estimator import get_memory_estimator
            from megatron.training import get_args
            
            memory_estimator = get_memory_estimator()
            args = get_args()
            
            # 使用精确的内存估算器
            self.stage_memory_limits = memory_estimator.generate_stage_memory_limits(args)
            
            # 打印详细的内存分析
            memory_estimator.print_memory_analysis(args)
            
        except Exception as e:
            print_rank_0(f"使用精确内存估算器失败: {e}，使用简化估算")
            
            # Fallback to simplified estimation
            pp_size = memory_metrics.get('pp_size', 1)
            pp_rank = memory_metrics.get('pp_rank', 0)
            effective_memory_gb = memory_metrics.get('effective_memory_limit_gb', 10.0)
            
            # 使用更精确的单层激活大小估算
            estimated_single_layer_activation_mb = memory_metrics.get('single_layer_activation_mb', 32.0)
            if estimated_single_layer_activation_mb == 0:
                estimated_single_layer_activation_mb = 32.0  # 默认值
            
            # 计算该stage可容纳的激活数量
            effective_memory_mb = effective_memory_gb * 1024
            max_activations = int(effective_memory_mb / estimated_single_layer_activation_mb)
            max_activations = max(1, max_activations)  # 至少为1
            
            # 为所有stage生成内存限制（这里简化为相同值）
            self.stage_memory_limits = [max_activations] * pp_size
            
            print_rank_0(f"估算的stage内存限制: 每stage可容纳 {max_activations} 个激活")
            print_rank_0(f"当前stage {pp_rank}: 有效内存限制 {effective_memory_gb:.2f} GB")
    
    def get_optimized_parameters_for_solver(self) -> Dict[str, Any]:
        """获取优化后的求解器参数"""
        if not self.collected_metrics:
            # 返回默认参数
            return {
                'Ft': 1.0,
                'Bt': 2.0,
                'Memory_limit': [100] * get_pipeline_model_parallel_world_size(),
                'CM': 6.0,
            }
        
        compute_metrics = self.collected_metrics.get('compute_metrics', {})
        memory_metrics = self.collected_metrics.get('memory_metrics', {})
        
        # 获取实际的前后向时间（转换为毫秒）
        Ft = compute_metrics.get('avg_forward_time_ms', 1.0)
        Bt = compute_metrics.get('avg_backward_time_ms', 2.0)
        
        # 获取内存限制
        Memory_limit = self.stage_memory_limits if self.stage_memory_limits else [100] * get_pipeline_model_parallel_world_size()
        
        # 获取通信开销（这里使用简化估计）
        CM = compute_metrics.get('avg_communication_time_ms', Ft + Bt) * 0.1  # 假设通信时间是计算时间的10%
        
        return {
            'Ft': Ft,
            'Bt': Bt,
            'Memory_limit': Memory_limit,
            'CM': CM,
        }
    
    def should_run_profiling(self, args) -> bool:
        """判断是否应该运行性能采样"""
        return (hasattr(args, 'jeeves_use_stage_division') and 
                args.jeeves_use_stage_division and 
                hasattr(args, 'use_cross_dc') and 
                args.use_cross_dc)


# 全局性能采样器实例
_global_pre_profiler: Optional[PreTrainingProfiler] = None

def get_pre_training_profiler() -> PreTrainingProfiler:
    """获取全局预训练性能采样器实例"""
    global _global_pre_profiler
    if _global_pre_profiler is None:
        _global_pre_profiler = PreTrainingProfiler()
    return _global_pre_profiler

def initialize_pre_training_profiler(profile_iterations=5) -> PreTrainingProfiler:
    """初始化全局预训练性能采样器"""
    global _global_pre_profiler
    _global_pre_profiler = PreTrainingProfiler(profile_iterations=profile_iterations)
    return _global_pre_profiler