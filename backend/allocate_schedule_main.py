from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any
import asyncio
from contextlib import asynccontextmanager
import logging

from Optimizer import Optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

optimizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global optimizer
    if optimizer is None:
        optimizer = Optimizer()
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
        except Exception as e:
            logger.info(e)
            active_connections.remove(connection)


async def periodic_updates():
    while True:
        # 获取最新的资源和任务状态
        resource_status = optimizer.env.get_resource_status()
        resource_costs = optimizer.env.get_cost()
        task_status = optimizer.env.get_job_status()
        task_queue = optimizer.env.get_task_queue_status()

        # 广播更新
        await broadcast_updates({
            "resource_status": resource_status,
            "resource_costs": resource_costs,
            "task_status": task_status,
            "task_queue": task_queue
        })

        # 每两秒更新一次
        await asyncio.sleep(2)

@app.get("/tasks")
async def get_tasks():
    tasks = optimizer.env.get_job_status()
    if tasks is None:
        raise HTTPException(status_code=404, detail="任务未找到")
    return tasks

@app.get("/resources")
async def get_resources():
    resources = optimizer.env.get_resource_status()
    if resources is None:
        raise HTTPException(status_code=404, detail="资源未找到")
    return resources

@app.get("/canvas")
async def get_canvas():
    canvas = optimizer.env.get_canvas()
    return canvas

@app.get("/metrics/resource-utilization")
async def get_resource_utilization():
    return optimizer.env.get_resource_utilization()

@app.get("/metrics/task-queue")
def get_task_queue_status():
    return optimizer.env.get_task_queue_status()

@app.get("/metrics/resource-costs")
async def get_resource_costs():
    return optimizer.env.get_cost()

if __name__ == "__main__":
    uvicorn.run("allocate_schedule_main:app", host="0.0.0.0", port=8000)








