# schemas.py - Pydantic 模型，用于数据验证和API响应
from pydantic import BaseModel, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResourceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class TaskBase(BaseModel):
    name: str
    type: str
    priority: int = 1
    resource_requirements: Dict[str, Any]


class TaskCreate(TaskBase):
    pass


class TaskResponse(TaskBase):
    id: int
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    resource_id: Optional[int] = None
    result: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ResourceBase(BaseModel):
    name: str
    type: ResourceType
    capacity: float
    meta_data: Optional[Dict[str, Any]] = None


class ResourceCreate(ResourceBase):
    name: str
    type: str
    capacity: float
    meta_data: dict


class ResourceResponse(ResourceBase):
    id: int
    used: float
    status: str

    class Config:
        from_attributes = True


class ResourceUtilization(BaseModel):
    id: int
    name: str
    type: ResourceType
    utilization: float
    tasks_count: int


class TaskQueueStatus(BaseModel):
    pending: int
    running: int
    completed: int
    failed: int
    average_wait_time: float
    average_execution_time: float
