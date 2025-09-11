#!/usr/bin/env python3
"""
共享工具定义模块
包含文件操作、计算器等工具函数和LangChain工具定义
"""

import os
import math
import random
from pathlib import Path
from typing import List, Any
from langchain.tools import StructuredTool

# --- 配置 ---
# 基础目录，所有文件操作都限制在此目录内
BASE_DIR = Path("./").resolve()

def set_base_directory(base_dir: str) -> None:
    """设置基础目录"""
    global BASE_DIR
    BASE_DIR = Path(base_dir).resolve()

def get_base_directory() -> Path:
    """获取当前基础目录"""
    return BASE_DIR

# --- 工具函数定义 ---

def read_file(name: str) -> str:
    """
    Read the contents of a file within the base directory.
    
    Args:
        name (str): The name of the file to read (relative to the base directory).
        
    Returns:
        str: The content of the file if successful, None if failed.
    """
    print(f"(read_file {name})")
    try:
        file_path = BASE_DIR / name
        # 安全检查：确保文件在 BASE_DIR 内
        if not str(file_path.resolve()).startswith(str(BASE_DIR)):
            print(f"错误：访问被拒绝，文件 '{name}' 在允许的目录之外")
            return None
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
        return f"成功读取文件 '{name}'，内容如下：\n{content}"
    except FileNotFoundError:
        print(f"错误：文件 '{name}' 未找到")
        return None
    except Exception as e:
        print(f"读取文件 '{name}' 时发生错误：{e}")
        return None

def list_files(directory: str = ".") -> str:
    """
    List all files in the specified directory and its subdirectories.
    
    Args:
        directory (str): The directory to list files from (relative to the base directory).
                        Default is "." (current directory).
    
    Returns:
        str: A formatted list of files if successful, None if failed.
    """
    print(f"(list_files {directory})")
    file_list = []
    try:
        # 构建目标目录路径
        target_dir = BASE_DIR / directory
        
        # 安全检查：确保目标目录在 BASE_DIR 内
        if not str(target_dir.resolve()).startswith(str(BASE_DIR)):
            print(f"错误：访问被拒绝，目录 '{directory}' 在允许的目录之外")
            return None
        
        # 检查目录是否存在
        if not target_dir.exists():
            print(f"错误：目录 '{directory}' 不存在")
            return None
        
        if not target_dir.is_dir():
            print(f"错误：'{directory}' 不是一个目录")
            return None
        
        # 递归查找所有文件
        for item in target_dir.rglob("*"):
            if item.is_file():
                # 计算相对于 BASE_DIR 的路径
                relative_path = item.relative_to(BASE_DIR)
                file_list.append(str(relative_path))
        
        if file_list:
            file_list_str = "\n".join([f"  - {file}" for file in file_list])
            return f"成功列出目录 '{directory}' 中的文件，共找到 {len(file_list)} 个文件：\n{file_list_str}"
        else:
            return f"目录 '{directory}' 中没有找到文件"
                
    except Exception as e:
        print(f"列出目录 '{directory}' 中的文件时发生错误：{e}")
        return None

def rename_file(name: str, new_name: str) -> str:
    """
    Rename a file within the base directory.
    
    Args:
        name (str): The current name of the file (relative to the base directory).
        new_name (str): The new name for the file (relative to the base directory).
        
    Returns:
        str: A success message if successful, None if failed.
    """
    print(f"(rename_file {name} -> {new_name})")
    try:
        old_path = BASE_DIR / name
        new_path = BASE_DIR / new_name
        
        # 安全检查：确保旧文件和新文件都在 BASE_DIR 内
        if not str(old_path.resolve()).startswith(str(BASE_DIR)):
            print(f"错误：访问被拒绝，原文件 '{name}' 在允许的目录之外")
            return None
        if not str(new_path.resolve()).startswith(str(BASE_DIR)):
            print(f"错误：访问被拒绝，新文件路径 '{new_name}' 在允许的目录之外")
            return None
        if new_path.resolve() == BASE_DIR:
            print(f"错误：不能重命名为基础目录本身")
            return None

        # 创建新文件路径的父目录（如果不存在）
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行重命名
        old_path.rename(new_path)
        return f"成功将文件 '{name}' 重命名为 '{new_name}'"
    except FileNotFoundError:
        print(f"错误：文件 '{name}' 未找到")
        return None
    except FileExistsError:
        print(f"错误：名为 '{new_name}' 的文件或目录已存在")
        return None
    except Exception as e:
        print(f"重命名文件 '{name}' 时发生错误：{e}")
        return None


# --- 导航和办公设备控制工具 ---

def navigate_to_office(office_number: str) -> str:
    """
    Navigate to a specific office room by room number. Use this when user asks to go to a specific office number like "521办公室" or "301办公室".
    
    Args:
        office_number (str): The office room number to navigate to (e.g., "521", "301").
        
    Returns:
        str: Navigation status and instructions if successful, None if failed.
    """
    print(f"(navigate_to_office {office_number})")
    
    try:
        # 模拟导航逻辑
        if office_number == "521":
            return f"正在前往521办公室...\n  位置：5楼东侧\n  路线：从当前位置直走50米，左转进入走廊，第二个门\n  预计时间：2分钟\n  已到达521办公室"
        elif office_number == "301":
            return f"正在前往301办公室...\n  位置：3楼西侧\n  路线：从当前位置上楼梯到3楼，右转直走30米\n  预计时间：3分钟\n  已到达301办公室"
        else:
            return f"正在前往{office_number}办公室...\n  位置：{office_number}办公室\n  路线：正在规划最优路径\n  预计时间：计算中...\n  已到达{office_number}办公室"
    except Exception as e:
        print(f"导航到办公室 {office_number} 时发生错误：{e}")
        return None

def navigate_to_teacher_office(teacher_name: str) -> str:
    """
    Navigate to a specific teacher's office by teacher name. Use this when user asks to go to a teacher's office like "康老师办公室" or "李老师办公室".
    
    Args:
        teacher_name (str): The name of the teacher (e.g., "康老师", "李老师").
        
    Returns:
        str: Navigation status and office information if successful, None if failed.
    """
    print(f"(navigate_to_teacher_office {teacher_name})")
    
    try:
        # 模拟教师办公室导航
        teacher_offices = {
            "康老师": "康老师办公室位于4楼南侧，房间号408",
            "李老师": "李老师办公室位于2楼北侧，房间号205", 
            "王老师": "王老师办公室位于3楼东侧，房间号312",
            "张老师": "张老师办公室位于5楼西侧，房间号502"
        }
        
        if teacher_name in teacher_offices:
            office_info = teacher_offices[teacher_name]
            return f"正在前往{teacher_name}办公室...\n  {office_info}\n  路线：正在规划路径\n  预计时间：2-4分钟\n  已到达{teacher_name}办公室"
        else:
            return f"正在查找{teacher_name}的办公室位置...\n  正在查询教师信息\n  位置：查询中...\n  预计时间：1-2分钟\n  已找到{teacher_name}办公室位置"
    except Exception as e:
        print(f"导航到{teacher_name}办公室时发生错误：{e}")
        return None

def open_display_board(board_type: str = "展示白板") -> str:
    """
    Open and activate a display board or whiteboard. Use this when user asks to open a whiteboard, display board, or projection screen.
    
    Args:
        board_type (str): The type of display board to open (e.g., "展示白板", "电子白板", "投影屏幕").
        
    Returns:
        str: Display board activation status if successful, None if failed.
    """
    print(f"(open_display_board {board_type})")
    
    try:
        if "白板" in board_type or "展示" in board_type:
            return f"正在打开{board_type}...\n  设备：智能展示白板\n  电源：已接通\n  显示：正在启动\n  功能：触控、书写、投影\n  {board_type}已成功打开并准备就绪"
        elif "投影" in board_type:
            return f"正在打开{board_type}...\n  设备：投影仪系统\n  电源：已接通\n  显示：正在启动\n  功能：投影、缩放、调节\n  {board_type}已成功打开并准备就绪"
        else:
            return f"正在打开{board_type}...\n  设备：{board_type}\n  电源：已接通\n  显示：正在启动\n  功能：显示、控制、交互\n  {board_type}已成功打开并准备就绪"
    except Exception as e:
        print(f"打开{board_type}时发生错误：{e}")
        return None

def control_air_conditioner(action: str = "打开", temperature: str = "26") -> str:
    """
    Control the air conditioning system. Use this when user asks to turn on, turn off, or adjust the air conditioner.
    
    Args:
        action (str): The action to perform ("打开", "关闭", "调节").
        temperature (str): The target temperature in Celsius.
        
    Returns:
        str: Air conditioning control status if successful, None if failed.
    """
    print(f"(control_air_conditioner {action} {temperature})")
    
    try:
        if action == "打开":
            return f"正在打开空调...\n  设备：中央空调系统\n  电源：已接通\n  目标温度：{temperature}°C\n  模式：自动调节\n  风速：中等\n  空调已成功打开，正在调节到{temperature}°C"
        elif action == "关闭":
            return f"正在关闭空调...\n  设备：中央空调系统\n  电源：正在关闭\n  当前温度：{temperature}°C\n  模式：关闭\n  风速：停止\n  空调已成功关闭"
        elif action == "调节":
            return f"正在调节空调温度...\n  设备：中央空调系统\n  电源：已接通\n  目标温度：{temperature}°C\n  模式：温度调节\n  风速：自动\n  空调温度已调节到{temperature}°C"
        else:
            return f"正在执行空调操作：{action}...\n  设备：中央空调系统\n  电源：已接通\n  温度设置：{temperature}°C\n  模式：{action}\n  风速：自动\n  空调操作完成"
    except Exception as e:
        print(f"控制空调时发生错误：{e}")
        return None

# --- 使用 StructuredTool.from_function 创建工具 ---
# 函数的 docstring 会被自动用作工具的 description
ReadFileTool = StructuredTool.from_function(read_file)
ListFilesTool = StructuredTool.from_function(list_files)
RenameFileTool = StructuredTool.from_function(rename_file)

# 导航和办公设备控制工具
NavigateToOfficeTool = StructuredTool.from_function(navigate_to_office)
NavigateToTeacherOfficeTool = StructuredTool.from_function(navigate_to_teacher_office)
OpenDisplayBoardTool = StructuredTool.from_function(open_display_board)
ControlAirConditionerTool = StructuredTool.from_function(control_air_conditioner)

# --- 工具列表 ---
ALL_TOOLS = [
    ReadFileTool, ListFilesTool, RenameFileTool,
    NavigateToOfficeTool, NavigateToTeacherOfficeTool, 
    OpenDisplayBoardTool, ControlAirConditionerTool
]

def get_all_tools():
    """获取所有工具列表"""
    return ALL_TOOLS

def get_tool_names():
    """获取所有工具名称列表"""
    return [tool.name for tool in ALL_TOOLS]

def get_tool_by_name(name: str):
    """根据名称获取工具"""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None

# --- 工具信息 ---
def get_tools_info():
    """获取工具信息字典"""
    return [
        {
            "name": tool.name,
            "description": tool.description
        }
        for tool in ALL_TOOLS
    ]

if __name__ == "__main__":

    # 显示工具信息
    print("\n  可用工具:")
    for tool_info in get_tools_info():
        print(f"  - {tool_info['name']}: {tool_info['description'][:50]}...")
    
    print("\n  工具模块测试完成！")
