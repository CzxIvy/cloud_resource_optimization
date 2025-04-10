# service.py - 资源优化和任务调度服务
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import time
import logging
from collections import defaultdict, deque

# 假设这是您已经开发好的深度强化学习模型导入
# from drl_model import ResourceOptimizationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceOptimizer:
    def __init__(self):
        self.resources = []  # 资源列表
        self.tasks = []  # 任务列表
        self.pending_tasks = deque()  # 待处理任务队列
        self.running_tasks = []  # 正在运行的任务
        self.completed_tasks = []  # 已完成的任务
        self.failed_tasks = []  # 失败的任务
        
        self.resource_history = defaultdict(list)  # 资源利用率历史
        self.task_history = []  # 任务执行历史
        
        # 初始化深度强化学习模型
        # self.drl_model = ResourceOptimizationModel()
        
        # 启动调度线程
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

    def reset_resources(self, resources):
        self.resources = resources

    def add_resource(self, resource):
        """添加新资源"""
        self.resources.append(resource)
        logger.info(f"Added new resource: {resource.name} (ID: {resource.id})")

    def remove_resource(self, resource):
        """从系统中移除资源"""
        # 检查资源是否存在于资源列表中
        if resource in self.resources:
            # 从资源列表中移除
            self.resources.remove(resource)
            # 清除该资源的利用率历史记录
            if resource.id in self.resource_history:
                del self.resource_history[resource.id]

            logger.info(f"Removed resource: {resource.name} (ID: {resource.id})")
        else:
            logger.warning(f"Attempted to remove non-existing resource: {resource.name} (ID: {resource.id})")

    def reset_tasks(self, tasks):
        self.tasks = tasks
        for task in self.tasks:
            self.pending_tasks.append(task)

    def add_task(self, task):
        """添加新任务"""
        self.tasks.append(task)
        self.pending_tasks.append(task)
        logger.info(f"Added new task: {task.name} (ID: {task.id})")
    
    def optimize(self):
        """执行资源优化算法"""
        # 这里应调用您的深度强化学习模型
        # result = self.drl_model.optimize(self.resources, self.pending_tasks)
        
        # 模拟优化结果
        result = {
            "optimized_assignments": [],
            "expected_performance_gain": 0.15,  # 15% 性能提升
            "resource_utilization_improvement": 0.2  # 20% 资源利用率提升
        }
        
        return result
    
    def _assign_task(self, task, resource):
        """分配任务到资源"""
        task.resource_id = resource.id
        task.resource = resource
        task.status = "running"
        task.started_at = datetime.utcnow()
        
        # 更新资源使用率
        requirements = task.resource_requirements
        if resource.type == "cpu" and "cpu" in requirements:
            resource.used += requirements["cpu"]
        elif resource.type == "memory" and "memory" in requirements:
            resource.used += requirements["memory"]
        elif resource.type == "gpu" and "gpu" in requirements:
            resource.used += requirements["gpu"]
        
        # 从待处理队列移到运行队列
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
        self.running_tasks.append(task)
        
        logger.info(f"Assigned task {task.id} to resource {resource.id}")
    
    def _complete_task(self, task):
        """完成任务"""
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        
        # 释放资源
        if task.resource:
            resource = task.resource
            requirements = task.resource_requirements
            if resource.type == "cpu" and "cpu" in requirements:
                resource.used -= requirements["cpu"]
            elif resource.type == "memory" and "memory" in requirements:
                resource.used -= requirements["memory"]
            elif resource.type == "gpu" and "gpu" in requirements:
                resource.used -= requirements["gpu"]
        
        # 从运行队列移到完成队列
        if task in self.running_tasks:
            self.running_tasks.remove(task)
        self.completed_tasks.append(task)
        
        # 记录任务执行历史
        self.task_history.append({
            "task_id": task.id,
            "name": task.name,
            "type": task.type,
            "start_time": task.started_at,
            "end_time": task.completed_at,
            "duration": (task.completed_at - task.started_at).total_seconds()
        })
        
        logger.info(f"Completed task {task.id}")
    
    def _scheduler_loop(self):
        """调度器循环，负责任务分配和执行"""
        while True:
            # 优先级排序
            sorted_pending = sorted(self.pending_tasks, key=lambda t: t.priority, reverse=True)
            
            # 尝试为待处理任务分配资源
            for task in sorted_pending:
                # 找到合适的资源
                suitable_resource = self._find_suitable_resource(task)
                if suitable_resource:
                    self._assign_task(task, suitable_resource)
            
            # 检查运行中的任务状态
            for task in list(self.running_tasks):
                # 模拟任务执行完成
                # 实际应用中，这里应该检查真实的任务执行状态
                # 这里只是简单模拟，随机完成一些任务
                if np.random.random() < 0.2:  # 20% 概率完成
                    self._complete_task(task)
            
            # 记录资源利用率历史
            current_time = datetime.utcnow()
            for resource in self.resources:
                utilization = resource.get_utilization()
                self.resource_history[resource.id].append({
                    "timestamp": current_time,
                    "utilization": utilization
                })
                
                # 只保留最近30分钟的历史
                cutoff_time = current_time - timedelta(minutes=30)
                self.resource_history[resource.id] = [
                    h for h in self.resource_history[resource.id] 
                    if h["timestamp"] > cutoff_time
                ]
            
            # 休眠1秒
            time.sleep(1)
    
    def _find_suitable_resource(self, task):
        """找到适合任务的资源"""
        requirements = task.resource_requirements
        suitable_resources = []
        
        for resource in self.resources:
            if resource.status != "available":
                continue
                
            # 检查资源类型和容量
            if resource.type == "cpu" and "cpu" in requirements:
                if resource.capacity - resource.used >= requirements["cpu"]:
                    suitable_resources.append(resource)
            elif resource.type == "memory" and "memory" in requirements:
                if resource.capacity - resource.used >= requirements["memory"]:
                    suitable_resources.append(resource)
            elif resource.type == "gpu" and "gpu" in requirements:
                if resource.capacity - resource.used >= requirements["gpu"]:
                    suitable_resources.append(resource)
        
        # 如果找到多个合适的资源，选择利用率最低的
        if suitable_resources:
            return min(suitable_resources, key=lambda r: r.get_utilization())
        
        return None
    
    def get_resource_status(self):
        """获取当前资源状态"""
        return [
            {
                "id": r.id,
                "name": r.name,
                "type": r.type.value if hasattr(r.type, 'value') else r.type,
                "capacity": r.capacity,
                "used": r.used,
                "utilization": r.get_utilization(),
                "status": r.status
            }
            for r in self.resources
        ]
    
    def get_task_status(self):
        """获取当前任务状态"""
        all_tasks = (
            [("pending", t) for t in self.pending_tasks] +
            [("running", t) for t in self.running_tasks] +
            [("completed", t) for t in self.completed_tasks] +
            [("failed", t) for t in self.failed_tasks]
        )
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "type": t.type,
                "status": status,
                "priority": t.priority,
                "resource_id": t.resource_id,
                "waiting_time": (datetime.utcnow() - t.created_at).total_seconds() if status == "pending" else None,
                "execution_time": t.get_execution_time() if status == "completed" else None
            }
            for status, t in all_tasks
        ]
    
    def get_resource_utilization(self):
        """获取资源利用率统计"""
        result = []
        for resource in self.resources:
            task_count = len([t for t in self.running_tasks if t.resource_id == resource.id])
            result.append({
                "id": resource.id,
                "name": resource.name,
                "type": resource.type.value if hasattr(resource.type, 'value') else resource.type,
                "utilization": resource.get_utilization(),
                "tasks_count": task_count,
                "history": self.resource_history.get(resource.id, [])
            })
        return result
    
    def get_task_queue_status(self):
        """获取任务队列状态统计"""
        # 计算平均等待时间
        if self.pending_tasks:
            avg_wait = sum(
                (datetime.utcnow() - t.created_at).total_seconds()
                for t in self.pending_tasks
            ) / len(self.pending_tasks)
        else:
            avg_wait = 0
        
        # 计算平均执行时间
        execution_times = [t.get_execution_time() for t in self.completed_tasks if t.get_execution_time()]
        avg_exec = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "pending": len(self.pending_tasks),
            "running": len(self.running_tasks),
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "average_wait_time": avg_wait,
            "average_execution_time": avg_exec,
            "task_history": self.task_history[-100:]  # 最近100条任务历史
        }
