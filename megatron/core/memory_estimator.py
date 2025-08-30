# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""更精确的内存使用估算工具"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)

class AccurateMemoryEstimator:
    """精确的内存使用估算器，考虑PyTorch动态分配和各种开销"""
    
    def __init__(self):
        self.base_memory_overhead_gb = 2.0  # PyTorch基础开销
        self.fragmentation_factor = 1.15    # 内存碎片化因子
        self.safety_margin = 0.10           # 安全边际（10%）
        
        # 不同stage的额外内存因子
        self.embedding_stage_factor = 1.25  # embedding层额外25%
        self.loss_stage_factor = 1.20       # loss层额外20%
        self.middle_stage_factor = 1.10     # 中间层额外10%
        
    def estimate_model_memory_usage(self, model, args) -> Dict[str, float]:
        """估算模型的内存使用情况（参数、梯度、优化器状态）"""
        if not model:
            return {'parameters_gb': 0, 'gradients_gb': 0, 'optimizer_states_gb': 0}
        
        # 计算参数内存
        total_params = 0
        if isinstance(model, list):
            for model_chunk in model:
                total_params += sum(p.numel() for p in model_chunk.parameters())
        else:
            total_params = sum(p.numel() for p in model.parameters())
        
        # 根据参数精度计算内存
        if args.bf16:
            bytes_per_param = 2  # BF16
        elif args.fp16:
            bytes_per_param = 2  # FP16
        else:
            bytes_per_param = 4  # FP32
        
        parameters_gb = total_params * bytes_per_param / (1024**3)
        # 梯度内存（通常与参数相同精度）
        gradients_gb = parameters_gb
        # 优化器状态内存（如Adam需要额外的momentum和variance）
        if getattr(args, 'optimizer', 'adam').lower() == 'adam':
            optimizer_states_gb = parameters_gb * 2  # momentum + variance
        else:
            optimizer_states_gb = parameters_gb  # 其他优化器通常需要1x参数内存
        
        return {
            'parameters_gb': parameters_gb,
            'gradients_gb': gradients_gb,
            'optimizer_states_gb': optimizer_states_gb,
        }
    
    def estimate_activation_memory_per_layer(self, args) -> float:
        """估算单层的激活内存大小（MB）"""
        seq_len = getattr(args, 'seq_length', 2048)
        hidden_size = getattr(args, 'hidden_size', 4096)
        micro_batch_size = getattr(args, 'micro_batch_size', 1)
        
        # 根据数据类型确定字节数
        if args.bf16 or args.fp16:
            bytes_per_element = 2
        else:
            bytes_per_element = 4
        
        # 单层激活内存估算（经验公式）
        # 包括：输入激活 + 注意力权重 + MLP中间激活
        attention_activation = seq_len * hidden_size * micro_batch_size * bytes_per_element
        attention_weights = seq_len * seq_len * micro_batch_size * getattr(args, 'num_attention_heads', 32) * bytes_per_element / getattr(args, 'tensor_model_parallel_size', 1)
        mlp_activation = seq_len * hidden_size * 4 * micro_batch_size * bytes_per_element  # 4x for MLP expansion
        
        total_activation_bytes = attention_activation + attention_weights + mlp_activation
        return total_activation_bytes / (1024 * 1024)  # 转换为MB
    
    def get_stage_memory_factor(self, stage_id: int, total_stages: int) -> float:
        """获取特定stage的内存因子"""
        if stage_id == 0:  # embedding层
            return self.embedding_stage_factor
        elif stage_id == total_stages - 1:  # loss层
            return self.loss_stage_factor
        else:  # 中间层
            return self.middle_stage_factor
    
    def calculate_effective_memory_limit(self, args, model=None) -> Dict[str, Any]:
        """计算每个stage的有效内存限制"""
        # 获取GPU内存信息
        try:
            if torch.cuda.is_available():
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_allocated = torch.cuda.memory_allocated()
                gpu_memory_reserved = torch.cuda.memory_reserved()
            else:
                # 默认值（用于测试）
                gpu_memory_total = 24 * 1024**3  # 24GB
                gpu_memory_allocated = 0
                gpu_memory_reserved = 0
        except:
            # 如果没有torch，使用默认值
            gpu_memory_total = 24 * 1024**3  # 24GB
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0
        
        gpu_memory_total_gb = gpu_memory_total / (1024**3)
        
        # 计算模型静态内存使用
        model_memory = self.estimate_model_memory_usage(model, args) if model else {
            'parameters_gb': 0, 'gradients_gb': 0, 'optimizer_states_gb': 0
        }
        
        model_static_memory_gb = (
            model_memory['parameters_gb'] + 
            model_memory['gradients_gb'] + 
            model_memory['optimizer_states_gb']
        )
        
        # 获取当前stage信息
        try:
            pp_rank = get_pipeline_model_parallel_rank()
            pp_size = get_pipeline_model_parallel_world_size()
        except:
            pp_rank = 0
            pp_size = 1
        
        # 计算stage特定的内存因子
        stage_factor = self.get_stage_memory_factor(pp_rank, pp_size)
        
        # 计算可用内存
        base_overhead = self.base_memory_overhead_gb * stage_factor
        safety_reserved = gpu_memory_total_gb * self.safety_margin
        
        available_memory_gb = (
            gpu_memory_total_gb - 
            model_static_memory_gb - 
            base_overhead - 
            safety_reserved
        ) / self.fragmentation_factor
        
        # 确保可用内存为正数
        available_memory_gb = max(0.1, available_memory_gb)  # 至少100MB
        
        # 估算单层激活内存
        single_layer_activation_mb = self.estimate_activation_memory_per_layer(args)
        
        # 计算可容纳的microbatch数量
        available_memory_mb = available_memory_gb * 1024
        max_microbatches = int(available_memory_mb / single_layer_activation_mb)
        max_microbatches = max(1, max_microbatches)  # 至少为1
        
        return {
            'gpu_memory_total_gb': gpu_memory_total_gb,
            'model_static_memory_gb': model_static_memory_gb,
            'available_memory_gb': available_memory_gb,
            'single_layer_activation_mb': single_layer_activation_mb,
            'max_microbatches_per_stage': max_microbatches,
            'stage_memory_factor': stage_factor,
            'pp_rank': pp_rank,
            'pp_size': pp_size,
            'model_memory_breakdown': model_memory,
        }
    
    def generate_stage_memory_limits(self, args, model=None) -> List[int]:
        """为所有stage生成内存限制列表"""
        try:
            pp_size = get_pipeline_model_parallel_world_size()
        except:
            pp_size = 1
            
        current_stage_limits = self.calculate_effective_memory_limit(args, model)
        
        # 简化假设：所有stage使用相似的内存限制
        # 实际使用中，可以为每个stage单独计算
        base_limit = current_stage_limits['max_microbatches_per_stage']
        
        stage_limits = []
        for stage_id in range(pp_size):
            stage_factor = self.get_stage_memory_factor(stage_id, pp_size)
            
            # 根据stage因子调整限制
            adjusted_limit = int(base_limit / stage_factor)
            adjusted_limit = max(1, adjusted_limit)  # 至少为1
            
            stage_limits.append(adjusted_limit)
        
        return stage_limits
    
    def print_memory_analysis(self, args, model=None):
        """打印详细的内存分析"""
        try:
            from megatron.training import print_rank_0
        except ImportError:
            def print_rank_0(msg):
                print(msg)
        
        analysis = self.calculate_effective_memory_limit(args, model)
        
        print_rank_0("=" * 80)
        print_rank_0("详细内存分析")
        print_rank_0("=" * 80)
        
        print_rank_0(f"GPU总内存: {analysis['gpu_memory_total_gb']:.2f} GB")
        print_rank_0(f"模型静态内存: {analysis['model_static_memory_gb']:.2f} GB")
        print_rank_0(f"  - 参数: {analysis['model_memory_breakdown']['parameters_gb']:.2f} GB")
        print_rank_0(f"  - 梯度: {analysis['model_memory_breakdown']['gradients_gb']:.2f} GB")
        print_rank_0(f"  - 优化器状态: {analysis['model_memory_breakdown']['optimizer_states_gb']:.2f} GB")
        
        print_rank_0(f"可用激活内存: {analysis['available_memory_gb']:.2f} GB")
        print_rank_0(f"单层激活大小: {analysis['single_layer_activation_mb']:.2f} MB")
        print_rank_0(f"当前stage最大microbatch数: {analysis['max_microbatches_per_stage']}")
        print_rank_0(f"Stage内存因子: {analysis['stage_memory_factor']:.2f}")
        print_rank_0(f"Pipeline Rank: {analysis['pp_rank']}/{analysis['pp_size']}")
        
        # 显示所有stage的内存限制
        stage_limits = self.generate_stage_memory_limits(args, model)
        print_rank_0(f"所有stage内存限制: {stage_limits}")
        
        print_rank_0("=" * 80)


# 全局内存估算器实例
_global_memory_estimator: Optional[AccurateMemoryEstimator] = None

def get_memory_estimator() -> AccurateMemoryEstimator:
    """获取全局内存估算器实例"""
    global _global_memory_estimator
    if _global_memory_estimator is None:
        _global_memory_estimator = AccurateMemoryEstimator()
    return _global_memory_estimator