# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""精确性能监控系统，用于跨数据中心训练优化"""

import time
import torch
import torch.cuda.nvtx as nvtx
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict, deque
from contextlib import contextmanager
import threading

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_data_parallel_rank,
)

class PrecisePerformanceProfiler:
    """精确的性能分析器，用于监控前后向计算时间、通信时间和内存使用"""
    
    def __init__(self, enable_detailed_profiling=True, max_history_size=100):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.max_history_size = max_history_size
        
        # 计算时间记录 (毫秒)
        self.forward_times = deque(maxlen=max_history_size)
        self.backward_times = deque(maxlen=max_history_size)
        self.layer_forward_times = defaultdict(lambda: deque(maxlen=max_history_size))
        self.layer_backward_times = defaultdict(lambda: deque(maxlen=max_history_size))
        
        # 通信时间记录 (毫秒)
        self.communication_times = deque(maxlen=max_history_size)
        self.send_times = deque(maxlen=max_history_size)
        self.recv_times = deque(maxlen=max_history_size)
        
        # 内存使用记录
        self.memory_snapshots = deque(maxlen=max_history_size)
        
        # microbatch 级别的时间记录
        self.microbatch_forward_times = deque(maxlen=max_history_size)
        self.microbatch_backward_times = deque(maxlen=max_history_size)
        
        # 锁，用于多线程安全
        self._lock = threading.Lock()
        
        # 当前正在进行的计时
        self._active_timers = {}
    
    @contextmanager
    def profile_forward_microbatch(self, microbatch_id: int):
        """监控单个microbatch的前向传播时间"""
        if not self.enable_detailed_profiling:
            yield
            return
            
        timer_key = f"forward_mb_{microbatch_id}"
        start_time = self._start_cuda_timer(timer_key)
        
        try:
            yield
        finally:
            duration = self._end_cuda_timer(timer_key, start_time)
            with self._lock:
                self.microbatch_forward_times.append(duration)
                self.forward_times.append(duration)
    
    @contextmanager 
    def profile_backward_microbatch(self, microbatch_id: int):
        """监控单个microbatch的后向传播时间"""
        if not self.enable_detailed_profiling:
            yield
            return
            
        timer_key = f"backward_mb_{microbatch_id}"
        start_time = self._start_cuda_timer(timer_key)
        
        try:
            yield
        finally:
            duration = self._end_cuda_timer(timer_key, start_time)
            with self._lock:
                self.microbatch_backward_times.append(duration)
                self.backward_times.append(duration)
    
    @contextmanager
    def profile_layer_forward(self, layer_id: int):
        """监控单层前向传播时间"""
        if not self.enable_detailed_profiling:
            yield
            return
            
        timer_key = f"layer_forward_{layer_id}"
        start_time = self._start_cuda_timer(timer_key)
        
        try:
            yield
        finally:
            duration = self._end_cuda_timer(timer_key, start_time)
            with self._lock:
                self.layer_forward_times[layer_id].append(duration)
    
    @contextmanager
    def profile_layer_backward(self, layer_id: int):
        """监控单层后向传播时间"""
        if not self.enable_detailed_profiling:
            yield
            return
            
        timer_key = f"layer_backward_{layer_id}"
        start_time = self._start_cuda_timer(timer_key)
        
        try:
            yield
        finally:
            duration = self._end_cuda_timer(timer_key, start_time)
            with self._lock:
                self.layer_backward_times[layer_id].append(duration)
    
    @contextmanager
    def profile_communication(self, comm_type: str, tensor_size_mb: float = 0.0):
        """监控通信时间"""
        if not self.enable_detailed_profiling:
            yield
            return
            
        timer_key = f"comm_{comm_type}_{time.time()}"
        start_time = self._start_cuda_timer(timer_key)
        
        try:
            yield
        finally:
            duration = self._end_cuda_timer(timer_key, start_time)
            with self._lock:
                self.communication_times.append(duration)
                if 'send' in comm_type.lower():
                    self.send_times.append(duration)
                elif 'recv' in comm_type.lower():
                    self.recv_times.append(duration)
    
    def _start_cuda_timer(self, timer_key: str) -> float:
        """开始CUDA计时器，确保精确性"""
        torch.cuda.synchronize()  # 确保之前的操作完成
        start_time = time.perf_counter()
        
        # 使用NVTX标记
        nvtx.range_push(timer_key)
        
        self._active_timers[timer_key] = {
            'start_time': start_time,
            'start_event': torch.cuda.Event(enable_timing=True)
        }
        self._active_timers[timer_key]['start_event'].record()
        
        return start_time
    
    def _end_cuda_timer(self, timer_key: str, start_time: float) -> float:
        """结束CUDA计时器并返回持续时间（毫秒）"""
        # 记录结束事件
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()  # 确保当前操作完成
        
        # 计算持续时间
        if timer_key in self._active_timers:
            start_event = self._active_timers[timer_key]['start_event']
            cuda_duration = start_event.elapsed_time(end_event)  # CUDA事件时间（毫秒）
            del self._active_timers[timer_key]
        else:
            # Fallback to CPU timing
            cpu_duration = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
            cuda_duration = cpu_duration
        
        # 结束NVTX标记
        nvtx.range_pop()
        
        return cuda_duration
    
    def collect_memory_snapshot(self):
        """收集详细的内存使用快照"""
        if not torch.cuda.is_available():
            return {}
        
        # 获取GPU内存信息
        gpu_memory_allocated = torch.cuda.memory_allocated()
        gpu_memory_reserved = torch.cuda.memory_reserved()
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
        
        # 获取详细的内存统计
        memory_stats = torch.cuda.memory_stats()
        
        # 获取并行信息
        pp_rank = get_pipeline_model_parallel_rank()
        pp_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = get_data_parallel_rank()
        
        snapshot = {
            'timestamp': time.time(),
            'gpu_memory_allocated': gpu_memory_allocated,
            'gpu_memory_reserved': gpu_memory_reserved,
            'gpu_memory_total': gpu_memory_total,
            'memory_utilization': gpu_memory_allocated / gpu_memory_total,
            'reserved_utilization': gpu_memory_reserved / gpu_memory_total,
            
            # 详细统计
            'active_bytes_all_time_peak': memory_stats.get('active_bytes.all.peak', 0),
            'allocated_bytes_all_time_peak': memory_stats.get('allocated_bytes.all.peak', 0),
            'reserved_bytes_all_time_peak': memory_stats.get('reserved_bytes.all.peak', 0),
            
            # 并行信息
            'pp_rank': pp_rank,
            'pp_size': pp_size,
            'tp_rank': tp_rank,
            'dp_rank': dp_rank,
        }
        
        with self._lock:
            self.memory_snapshots.append(snapshot)
        
        return snapshot
    
    def get_compute_metrics(self) -> Dict[str, float]:
        """获取计算性能指标"""
        with self._lock:
            if not self.forward_times or not self.backward_times:
                return {
                    'avg_forward_time_ms': 0.0,
                    'avg_backward_time_ms': 0.0,
                    'forward_backward_ratio': 2.0,  # 默认比例
                    'compute_variance': 0.0,
                }
            
            forward_times = list(self.forward_times)
            backward_times = list(self.backward_times)
            
            avg_forward = np.mean(forward_times)
            avg_backward = np.mean(backward_times)
            
            # 计算方差来衡量时间稳定性
            forward_var = np.var(forward_times) if len(forward_times) > 1 else 0.0
            backward_var = np.var(backward_times) if len(backward_times) > 1 else 0.0
            
            return {
                'avg_forward_time_ms': avg_forward,
                'avg_backward_time_ms': avg_backward,
                'forward_backward_ratio': avg_backward / avg_forward if avg_forward > 0 else 2.0,
                'forward_variance_ms2': forward_var,
                'backward_variance_ms2': backward_var,
                'compute_variance': (forward_var + backward_var) / 2,
                'samples_count': min(len(forward_times), len(backward_times)),
            }
    
    def get_communication_metrics(self) -> Dict[str, float]:
        """获取通信性能指标"""
        with self._lock:
            if not self.communication_times:
                return {
                    'avg_communication_time_ms': 0.0,
                    'avg_send_time_ms': 0.0,
                    'avg_recv_time_ms': 0.0,
                }
            
            comm_times = list(self.communication_times)
            send_times = list(self.send_times)
            recv_times = list(self.recv_times)
            
            return {
                'avg_communication_time_ms': np.mean(comm_times),
                'avg_send_time_ms': np.mean(send_times) if send_times else 0.0,
                'avg_recv_time_ms': np.mean(recv_times) if recv_times else 0.0,
                'communication_variance_ms2': np.var(comm_times) if len(comm_times) > 1 else 0.0,
                'comm_samples_count': len(comm_times),
            }
    
    def get_memory_metrics(self) -> Dict[str, float]:
        """获取内存使用指标"""
        latest_snapshot = None
        with self._lock:
            if self.memory_snapshots:
                latest_snapshot = self.memory_snapshots[-1]
        
        if not latest_snapshot:
            self.collect_memory_snapshot()
            with self._lock:
                latest_snapshot = self.memory_snapshots[-1] if self.memory_snapshots else {}
        
        if not latest_snapshot:
            return {}
        
        # 使用更精确的内存估算器
        try:
            from megatron.core.memory_estimator import get_memory_estimator
            from megatron.training import get_args
            
            memory_estimator = get_memory_estimator()
            args = get_args()
            
            # 计算精确的内存使用情况
            memory_analysis = memory_estimator.calculate_effective_memory_limit(args)
            
            return {
                'gpu_memory_total_gb': memory_analysis['gpu_memory_total_gb'],
                'gpu_memory_allocated_gb': latest_snapshot['gpu_memory_allocated'] / (1024**3),
                'gpu_memory_reserved_gb': latest_snapshot['gpu_memory_reserved'] / (1024**3),
                'effective_memory_limit_gb': memory_analysis['available_memory_gb'],
                'model_static_memory_gb': memory_analysis['model_static_memory_gb'],
                'memory_utilization': latest_snapshot['memory_utilization'],
                'reserved_utilization': latest_snapshot['reserved_utilization'],
                'pp_rank': memory_analysis['pp_rank'],
                'pp_size': memory_analysis['pp_size'],
                'extra_memory_factor': memory_analysis['stage_memory_factor'],
                'safety_factor': 0.10,  # 10% safety margin
                'max_microbatches_per_stage': memory_analysis['max_microbatches_per_stage'],
                'single_layer_activation_mb': memory_analysis['single_layer_activation_mb'],
                'model_memory_breakdown': memory_analysis['model_memory_breakdown'],
            }
            
        except Exception as e:
            # Fallback to original logic if new estimator fails
            pp_rank = latest_snapshot.get('pp_rank', 0)
            pp_size = latest_snapshot.get('pp_size', 1)
            
            # 内存安全系数
            safety_factor = 0.15  # 预留15%作为安全边界
            
            # 针对第一个和最后一个stage的额外内存需求
            extra_memory_factor = 1.0
            if pp_rank == 0:  # embedding层
                extra_memory_factor = 1.25  # 额外25%用于embedding层
            elif pp_rank == pp_size - 1:  # loss层
                extra_memory_factor = 1.20  # 额外20%用于loss计算
            
            gpu_memory_total = latest_snapshot['gpu_memory_total']
            gpu_memory_reserved = latest_snapshot['gpu_memory_reserved']
            
            # 可用内存计算
            available_memory = gpu_memory_total * (1 - safety_factor) / extra_memory_factor
            effective_memory_limit = available_memory - gpu_memory_reserved
            
            return {
                'gpu_memory_total_gb': gpu_memory_total / (1024**3),
                'gpu_memory_allocated_gb': latest_snapshot['gpu_memory_allocated'] / (1024**3),
                'gpu_memory_reserved_gb': gpu_memory_reserved / (1024**3),
                'effective_memory_limit_gb': effective_memory_limit / (1024**3),
                'memory_utilization': latest_snapshot['memory_utilization'],
                'reserved_utilization': latest_snapshot['reserved_utilization'],
                'pp_rank': pp_rank,
                'pp_size': pp_size,
                'extra_memory_factor': extra_memory_factor,
                'safety_factor': safety_factor,
            }
    
    def estimate_activation_memory_capacity(self, single_layer_activation_size_mb: float) -> int:
        """估算当前stage可容纳的最大microbatch数量"""
        memory_metrics = self.get_memory_metrics()
        
        if not memory_metrics or single_layer_activation_size_mb <= 0:
            return 1  # 默认最小值
        
        effective_limit_gb = memory_metrics['effective_memory_limit_gb']
        effective_limit_mb = effective_limit_gb * 1024
        
        # 计算可容纳的microbatch数量
        max_microbatches = int(effective_limit_mb / single_layer_activation_size_mb)
        
        return max(1, max_microbatches)  # 至少为1
    
    def get_layer_performance_breakdown(self) -> Dict[str, Dict[str, float]]:
        """获取各层性能详细分解"""
        breakdown = {}
        
        with self._lock:
            # 前向时间分解
            for layer_id, times in self.layer_forward_times.items():
                if times:
                    if layer_id not in breakdown:
                        breakdown[layer_id] = {}
                    breakdown[layer_id]['avg_forward_ms'] = np.mean(times)
                    breakdown[layer_id]['forward_samples'] = len(times)
            
            # 后向时间分解
            for layer_id, times in self.layer_backward_times.items():
                if times:
                    if layer_id not in breakdown:
                        breakdown[layer_id] = {}
                    breakdown[layer_id]['avg_backward_ms'] = np.mean(times)
                    breakdown[layer_id]['backward_samples'] = len(times)
        
        return breakdown
    
    def reset_statistics(self):
        """重置所有统计数据"""
        with self._lock:
            self.forward_times.clear()
            self.backward_times.clear()
            self.layer_forward_times.clear()
            self.layer_backward_times.clear()
            self.communication_times.clear()
            self.send_times.clear()
            self.recv_times.clear()
            self.memory_snapshots.clear()
            self.microbatch_forward_times.clear()
            self.microbatch_backward_times.clear()
    
    def print_summary(self):
        """打印性能汇总"""
        from megatron.training import print_rank_0
        
        compute_metrics = self.get_compute_metrics()
        comm_metrics = self.get_communication_metrics()
        memory_metrics = self.get_memory_metrics()
        
        print_rank_0("=" * 80)
        print_rank_0("性能监控汇总")
        print_rank_0("=" * 80)
        
        print_rank_0("计算性能:")
        print_rank_0(f"  平均前向时间: {compute_metrics['avg_forward_time_ms']:.2f} ms")
        print_rank_0(f"  平均后向时间: {compute_metrics['avg_backward_time_ms']:.2f} ms")
        print_rank_0(f"  前后向时间比例: {compute_metrics['forward_backward_ratio']:.2f}")
        print_rank_0(f"  采样数量: {compute_metrics['samples_count']}")
        
        print_rank_0("\n通信性能:")
        print_rank_0(f"  平均通信时间: {comm_metrics['avg_communication_time_ms']:.2f} ms")
        print_rank_0(f"  平均发送时间: {comm_metrics['avg_send_time_ms']:.2f} ms")
        print_rank_0(f"  平均接收时间: {comm_metrics['avg_recv_time_ms']:.2f} ms")
        
        print_rank_0("\n内存使用:")
        print_rank_0(f"  GPU总内存: {memory_metrics.get('gpu_memory_total_gb', 0):.2f} GB")
        print_rank_0(f"  已分配内存: {memory_metrics.get('gpu_memory_allocated_gb', 0):.2f} GB")
        print_rank_0(f"  有效内存限制: {memory_metrics.get('effective_memory_limit_gb', 0):.2f} GB")
        print_rank_0(f"  内存利用率: {memory_metrics.get('memory_utilization', 0)*100:.1f}%")
        print_rank_0(f"  Pipeline Rank: {memory_metrics.get('pp_rank', 0)}/{memory_metrics.get('pp_size', 1)}")
        
        print_rank_0("=" * 80)


# 全局性能监控器实例
_global_profiler: Optional[PrecisePerformanceProfiler] = None

def get_performance_profiler() -> PrecisePerformanceProfiler:
    """获取全局性能监控器实例"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PrecisePerformanceProfiler()
    return _global_profiler

def initialize_performance_profiler(enable_detailed_profiling=True, max_history_size=100):
    """初始化全局性能监控器"""
    global _global_profiler
    _global_profiler = PrecisePerformanceProfiler(
        enable_detailed_profiling=enable_detailed_profiling,
        max_history_size=max_history_size
    )
    return _global_profiler

def destroy_performance_profiler():
    """销毁全局性能监控器"""
    global _global_profiler
    _global_profiler = None