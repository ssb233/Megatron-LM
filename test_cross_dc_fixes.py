#!/usr/bin/env python3
"""
测试修复后的跨数据中心训练优化模块功能
"""

import sys
import os
import argparse
from types import SimpleNamespace

# 添加Megatron路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_memory_estimator():
    """测试内存估算器功能"""
    print("=" * 60)
    print("测试内存估算器...")
    
    try:
        from megatron.core.memory_estimator import AccurateMemoryEstimator, get_memory_estimator
        
        # 创建实例
        estimator = AccurateMemoryEstimator()
        print("✓ AccurateMemoryEstimator 创建成功")
        
        # 测试全局实例
        global_estimator = get_memory_estimator()
        print("✓ get_memory_estimator 成功")
        
        # 测试内存估算方法
        # 创建模拟的args对象
        mock_args = SimpleNamespace(
            bf16=True,
            fp16=False,
            optimizer='adam',
            seq_length=2048,
            hidden_size=4096,
            micro_batch_size=1,
            num_attention_heads=32,
            tensor_model_parallel_size=1
        )
        
        # 测试模型内存估算
        memory_usage = estimator.estimate_model_memory_usage(None, mock_args)
        print(f"✓ estimate_model_memory_usage 成功: {memory_usage}")
        
        # 测试激活内存估算
        activation_mem = estimator.estimate_activation_memory_per_layer(mock_args)
        print(f"✓ estimate_activation_memory_per_layer 成功: {activation_mem:.2f} MB")
        
        # 测试stage内存因子
        stage_factor = estimator.get_stage_memory_factor(0, 4)
        print(f"✓ get_stage_memory_factor 成功: {stage_factor}")
        
        print("✓ 内存估算器所有功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 内存估算器测试失败: {e}")
        return False

def test_performance_monitor():
    """测试性能监控器功能"""
    print("=" * 60)
    print("测试性能监控器...")
    
    try:
        from megatron.core.performance_monitor import PrecisePerformanceProfiler, get_performance_profiler
        
        # 创建实例
        profiler = PrecisePerformanceProfiler()
        print("✓ PrecisePerformanceProfiler 创建成功")
        
        # 测试全局实例
        global_profiler = get_performance_profiler()
        print("✓ get_performance_profiler 成功")
        
        # 测试性能监控方法
        profiler.start_timing('test')
        profiler.end_timing('test')
        print("✓ start_timing/end_timing 成功")
        
        profiler.record_forward_time(0, 10.5)
        profiler.record_backward_time(0, 15.2)
        print("✓ record_forward_time/record_backward_time 成功")
        
        # 测试内存快照
        profiler.collect_memory_snapshot()
        print("✓ collect_memory_snapshot 成功")
        
        # 测试获取指标
        compute_metrics = profiler.get_compute_metrics()
        print(f"✓ get_compute_metrics 成功: samples={compute_metrics.get('samples_count', 0)}")
        
        print("✓ 性能监控器所有功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 性能监控器测试失败: {e}")
        return False

def test_pre_training_profiler():
    """测试预训练性能采样器功能"""
    print("=" * 60)
    print("测试预训练性能采样器...")
    
    try:
        from megatron.core.pre_training_profiler import PreTrainingProfiler, get_pre_training_profiler
        
        # 创建实例
        profiler = PreTrainingProfiler()
        print("✓ PreTrainingProfiler 创建成功")
        
        # 测试全局实例
        global_profiler = get_pre_training_profiler()
        print("✓ get_pre_training_profiler 成功")
        
        # 测试参数获取（模拟）
        mock_metrics = {
            'avg_forward_time_ms': 100.0,
            'avg_backward_time_ms': 150.0,
            'avg_communication_time_ms': 20.0,
        }
        
        params = profiler.get_optimized_parameters_for_solver()
        print(f"✓ get_optimized_parameters_for_solver 成功: {type(params)}")
        
        print("✓ 预训练性能采样器所有功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 预训练性能采样器测试失败: {e}")
        return False

def test_arguments_parsing():
    """测试参数解析功能"""
    print("=" * 60)
    print("测试参数解析...")
    
    try:
        from megatron.training.arguments import _add_profiler_args, _add_Jeeves_args, _add_cross_datacenter_args
        
        # 创建解析器
        parser = argparse.ArgumentParser()
        
        # 测试添加参数组
        parser = _add_profiler_args(parser)
        print("✓ _add_profiler_args 成功")
        
        parser = _add_Jeeves_args(parser)
        print("✓ _add_Jeeves_args 成功")
        
        parser = _add_cross_datacenter_args(parser)
        print("✓ _add_cross_datacenter_args 成功")
        
        # 测试解析
        test_args = [
            '--jeeves-enable-profiling',
            '--jeeves-use-stage-division',
            '--jeeves-comm-aware',
            '--use-cross-dc',
            '--cross-dc-propagation-delay', '50.0'
        ]
        
        args = parser.parse_args(test_args)
        print(f"✓ 参数解析成功: jeeves_enable_profiling={args.jeeves_enable_profiling}")
        print(f"✓ 参数解析成功: use_cross_dc={args.use_cross_dc}")
        
        print("✓ 参数解析所有功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 参数解析测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始跨数据中心训练优化模块Bug修复验证...")
    print("=" * 80)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("内存估算器", test_memory_estimator()))
    test_results.append(("性能监控器", test_performance_monitor()))
    test_results.append(("预训练性能采样器", test_pre_training_profiler()))
    test_results.append(("参数解析", test_arguments_parsing()))
    
    # 汇总结果
    print("=" * 80)
    print("测试结果汇总:")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！跨数据中心训练优化模块Bug修复成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关模块")
        return 1

if __name__ == "__main__":
    sys.exit(main())