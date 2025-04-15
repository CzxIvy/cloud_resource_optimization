# models.py - SQLAlchemy 数据模型
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base
import enum
from datetime import datetime

class TaskStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ResourceType(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

class TaskResourceAssociation(Base):
    __tablename__ = "task_resource_association"

    task_id = Column(Integer, ForeignKey("tasks.id"), primary_key=True)
    resource_id = Column(Integer, ForeignKey("resources.id"), primary_key=True)

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    type = Column(String(255), index=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    resource_requirements = Column(JSON)  # 例如: {"cpu": 2, "memory": 4, "len": 10}
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)
    
    # 关联到分配的资源
    resources = relationship("Resource", secondary="task_resource_association", back_populates="tasks")
    
    def get_execution_time(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

class Resource(Base):
    __tablename__ = "resources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    type = Column(Enum(ResourceType))
    capacity = Column(Float)
    used = Column(Float, default=0.0)
    status = Column(String(255), default="available")
    meta_data = Column(JSON, nullable=True)
    
    # 关联到分配的任务
    tasks = relationship("Task", secondary="task_resource_association", back_populates="resources")
    
    def get_utilization(self):
        if self.capacity > 0:
            return (self.used / self.capacity) * 100
        return 0.0

