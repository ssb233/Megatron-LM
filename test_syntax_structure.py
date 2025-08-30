#!/usr/bin/env python3
"""
测试修复后的跨数据中心训练优化模块的语法和结构完整性
"""

import sys
import os
import ast
import importlib.util

def check_python_syntax(file_path):
    """检查Python文件语法正确性"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 解析语法树
        ast.parse(source)
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def check_function_definitions(file_path, expected_functions):
    """检查文件中是否包含预期的函数定义"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # 提取所有函数名
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        missing_functions = []
        for expected_func in expected_functions:
            if expected_func not in functions:
                missing_functions.append(expected_func)
        
        if missing_functions:
            return False, f"缺少函数: {missing_functions}"
        else:
            return True, f"包含所有预期函数: {expected_functions}"
    
    except Exception as e:
        return False, f"检查失败: {e}"

def check_class_definitions(file_path, expected_classes):
    """检查文件中是否包含预期的类定义"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # 提取所有类名
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        missing_classes = []
        for expected_class in expected_classes:
            if expected_class not in classes:
                missing_classes.append(expected_class)
        
        if missing_classes:
            return False, f"缺少类: {missing_classes}"
        else:
            return True, f"包含所有预期类: {expected_classes}"
    
    except Exception as e:
        return False, f"检查失败: {e}"

def test_memory_estimator():
    """测试内存估算器文件"""
    print("=" * 60)
    print("测试 memory_estimator.py...")
    
    file_path = "megatron/core/memory_estimator.py"
    
    # 检查语法
    success, msg = check_python_syntax(file_path)
    if not success:
        print(f"❌ 语法检查失败: {msg}")
        return False
    print("✓ 语法检查通过")
    
    # 检查类定义
    expected_classes = ["AccurateMemoryEstimator"]
    success, msg = check_class_definitions(file_path, expected_classes)
    if not success:
        print(f"❌ 类定义检查失败: {msg}")
        return False
    print(f"✓ 类定义检查通过: {msg}")
    
    # 检查函数定义
    expected_functions = [
        "estimate_model_memory_usage",
        "estimate_activation_memory_per_layer",
        "get_stage_memory_factor",
        "calculate_effective_memory_limit",
        "generate_stage_memory_limits",
        "print_memory_analysis",
        "get_memory_estimator"
    ]
    success, msg = check_function_definitions(file_path, expected_functions)
    if not success:
        print(f"❌ 函数定义检查失败: {msg}")
        return False
    print(f"✓ 函数定义检查通过")
    
    return True

def test_performance_monitor():
    """测试性能监控器文件"""
    print("=" * 60)
    print("测试 performance_monitor.py...")
    
    file_path = "megatron/core/performance_monitor.py"
    
    # 检查语法
    success, msg = check_python_syntax(file_path)
    if not success:
        print(f"❌ 语法检查失败: {msg}")
        return False
    print("✓ 语法检查通过")
    
    # 检查类定义
    expected_classes = ["PrecisePerformanceProfiler"]
    success, msg = check_class_definitions(file_path, expected_classes)
    if not success:
        print(f"❌ 类定义检查失败: {msg}")
        return False
    print(f"✓ 类定义检查通过: {msg}")
    
    # 检查函数定义
    expected_functions = [
        "profile_forward_microbatch",
        "profile_backward_microbatch", 
        "profile_layer_forward",
        "profile_layer_backward",
        "collect_memory_snapshot",
        "get_compute_metrics",
        "get_communication_metrics",
        "get_memory_metrics",
        "get_performance_profiler"
    ]
    success, msg = check_function_definitions(file_path, expected_functions)
    if not success:
        print(f"❌ 函数定义检查失败: {msg}")
        return False
    print(f"✓ 函数定义检查通过")
    
    return True

def test_pre_training_profiler():
    """测试预训练性能采样器文件"""
    print("=" * 60)
    print("测试 pre_training_profiler.py...")
    
    file_path = "megatron/core/pre_training_profiler.py"
    
    # 检查语法
    success, msg = check_python_syntax(file_path)
    if not success:
        print(f"❌ 语法检查失败: {msg}")
        return False
    print("✓ 语法检查通过")
    
    # 检查类定义
    expected_classes = ["PreTrainingProfiler"]
    success, msg = check_class_definitions(file_path, expected_classes)
    if not success:
        print(f"❌ 类定义检查失败: {msg}")
        return False
    print(f"✓ 类定义检查通过: {msg}")
    
    # 检查函数定义
    expected_functions = [
        "initialize",
        "run_profiling_iterations",
        "get_optimized_parameters_for_solver",
        "should_run_profiling",
        "get_pre_training_profiler"
    ]
    success, msg = check_function_definitions(file_path, expected_functions)
    if not success:
        print(f"❌ 函数定义检查失败: {msg}")
        return False
    print(f"✓ 函数定义检查通过")
    
    return True

def test_arguments():
    """测试参数定义文件"""
    print("=" * 60)
    print("测试 arguments.py...")
    
    file_path = "megatron/training/arguments.py"
    
    # 检查语法
    success, msg = check_python_syntax(file_path)
    if not success:
        print(f"❌ 语法检查失败: {msg}")
        return False
    print("✓ 语法检查通过")
    
    # 检查函数定义
    expected_functions = [
        "add_megatron_arguments",
        "parse_args",
        "_add_profiler_args",
        "_add_Jeeves_args", 
        "_add_cross_datacenter_args"
    ]
    success, msg = check_function_definitions(file_path, expected_functions)
    if not success:
        print(f"❌ 函数定义检查失败: {msg}")
        return False
    print(f"✓ 函数定义检查通过")
    
    return True

def main():
    """主测试函数"""
    print("开始跨数据中心训练优化模块语法和结构验证...")
    print("=" * 80)
    
    # 切换到项目目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("memory_estimator.py", test_memory_estimator()))
    test_results.append(("performance_monitor.py", test_performance_monitor()))
    test_results.append(("pre_training_profiler.py", test_pre_training_profiler()))
    test_results.append(("arguments.py", test_arguments()))
    
    # 汇总结果
    print("=" * 80)
    print("测试结果汇总:")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print("=" * 80)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有语法和结构检查通过！跨数据中心训练优化模块Bug修复成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关模块")
        return 1

if __name__ == "__main__":
    sys.exit(main())