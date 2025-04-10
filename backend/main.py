# main.py - FastAPI 主应用
from datetime import datetime

from fastapi import FastAPI, WebSocket, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import uvicorn
import json
from typing import List, Dict, Any
import asyncio

# 导入自定义模块
from models import Task, Resource, TaskStatus
from database import get_db, engine, Base
from schemas import TaskCreate, TaskResponse, ResourceCreate, ResourceResponse
from service import ResourceOptimizer

# 创建数据库表
Base.metadata.create_all(bind=engine)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时同步数据库和optimizer
    with next(get_db()) as db:
        resources = db.query(Resource).all()
        optimizer.reset_resources(resources)
        db.query(Task).update({Task.created_at: datetime.utcnow()})
        tasks = db.query(Task).all()
        optimizer.reset_tasks(tasks)

    # 启动时执行的代码
    task = asyncio.create_task(periodic_updates())
    yield
    # 关闭时执行的代码
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="云资源优化供应系统 API", lifespan=lifespan)

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket 客户端列表
active_connections: List[WebSocket] = []

# 资源优化器实例
optimizer = ResourceOptimizer()

def sync_optimizer_with_db(db: Session):
    """从数据库同步所有资源到optimizer"""
    resources = db.query(Resource).all()
    optimizer.reset_resources(resources)  # 完全重置optimizer中的资源

    tasks = db.query(Task).all()
    optimizer.reset_tasks(tasks)  # 完全重置optimizer中的任务

    return {"status": "success", "resource_count": len(resources), "task_count": len(tasks)}

# WebSocket 连接管理
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # 保持连接活跃
            await websocket.receive_text()
    except Exception:
        active_connections.remove(websocket)

# 广播资源和任务更新
async def broadcast_updates(data: Dict[str, Any]):
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception:
            active_connections.remove(connection)

async def periodic_updates():
    while True:
        # 获取最新的资源和任务状态
        resource_status = optimizer.get_resource_status()
        task_status = optimizer.get_task_status()
        
        # 广播更新
        await broadcast_updates({
            "resource_status": resource_status,
            "task_status": task_status
        })
        
        # 每秒更新一次
        await asyncio.sleep(1)

# API 路由 - 任务管理
@app.post("/tasks/", response_model=TaskResponse)
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    db_task = Task(**task.dict())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    # 通知优化器有新任务
    optimizer.add_task(db_task)
    
    return db_task

@app.get("/tasks/", response_model=List[TaskResponse])
def get_tasks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tasks = db.query(Task).offset(skip).limit(limit).all()
    return tasks

@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="任务未找到")
    return task

# API 路由 - 资源管理
@app.post("/resources/", response_model=ResourceResponse)
def create_resource(resource: ResourceCreate, db: Session = Depends(get_db)):
    print(resource)
    db_resource = Resource(**resource.dict())
    db.add(db_resource)
    db.commit()
    db.refresh(db_resource)
    
    # 通知优化器有新资源
    optimizer.add_resource(db_resource)
    
    return db_resource

# 添加删除资源的API端点
@app.delete("/resources/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resource(resource_id: int, db: Session = Depends(get_db)):
    # 查找要删除的资源
    db_resource = db.query(Resource).filter(Resource.id == resource_id).first()
    if db_resource is None:
        raise HTTPException(status_code=404, detail="资源未找到")
    
    # 检查资源是否有关联的任务
    if db_resource.assigned_tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无法删除此资源，因为它有正在运行的关联任务"
        )
    
    try:
        # 从优化器中移除资源
        optimizer.remove_resource(db_resource)
        
        # 从数据库中删除资源
        db.delete(db_resource)
        db.commit()

        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除资源时发生错误: {str(e)}"
        )

@app.get("/resources/", response_model=List[ResourceResponse])
def get_resources(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """获取所有资源并添加计算的utilization字段"""
    resources = db.query(Resource).offset(skip).limit(limit).all()

    # 创建包含utilization字段的响应
    response_resources = []
    for resource in resources:
        resource_dict = {
            "id": resource.id,
            "name": resource.name,
            "type": resource.type,
            "capacity": resource.capacity,
            "used": resource.used,
            "status": resource.status,
            # 添加计算的utilization字段
            "utilization": resource.get_utilization(),
            "meta_data": resource.meta_data
        }
        response_resources.append(resource_dict)

    return response_resources

# API 路由 - 资源优化和任务调度
@app.post("/optimize")
def optimize_resources(db: Session = Depends(get_db)):
    result = optimizer.optimize()
    return {"status": "success", "details": result}

# 获取当前资源利用率
@app.get("/metrics/resource-utilization")
def get_resource_utilization():
    return optimizer.get_resource_utilization()

# 获取任务队列状态
@app.get("/metrics/task-queue")
def get_task_queue_status():
    return optimizer.get_task_queue_status()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
