import React, { useState, useEffect, useRef } from "react";
import { useCallback } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
// import ResourceTaskForms from './ResourceTaskForms';
import ResourceList from "./ResourceList";

const Dashboard = () => {
  // 数据状态
  const [resources, setResources] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [resourceUtilization, setResourceUtilization] = useState([]);
  const [resourceCosts, setResourceCosts] = useState([]);
  const [taskQueueStatus, setTaskQueueStatus] = useState({
    pending: 0,
    running: 0,
    completed: 0,
    failed: 0,
    average_wait_time: 0,
    average_execution_time: 0,
  });

  // WebSocket连接状态
  const [connected, setConnected] = useState(false);
  const socketRef = useRef(null);

  // 处理从WebSocket接收的数据
  const handleWebSocketData = useCallback((data) => {
    // 更新资源状态
    if (data.resource_status) {
      setResources(data.resource_status);

      // 更新资源利用率历史
      const utilizationHistory = data.resource_status.map((resource) => {
        return {
          ...resource,
          history:
            resource.history ||
            Array.from({ length: 30 }, (_, i) => ({
              time: new Date(
                Date.now() - (29 - i) * 60000
              ).toLocaleTimeString(),
              value: resource.utilization,
            })),
        };
      });

      setResourceUtilization(utilizationHistory);

      // 更新资源租赁开销历史
      if (data.resource_costs) {
        setResourceCosts(data.resource_costs);
      } else {
        // 如果没有成本数据，基于资源利用率生成模拟成本数据
        const costHistory = data.resource_status.map((resource) => {
          // 假设成本与利用率相关，但有不同的基准值和波动
          const hourlyRate = getHourlyRateByType(resource.type);
          return {
            ...resource,
            hourlyRate: hourlyRate,
            history:
              resource.history ||
              Array.from({ length: 30 }, (_, i) => ({
                time: new Date(
                  Date.now() - (29 - i) * 60000
                ).toLocaleTimeString(),
                value:
                  (resource.utilization / 100) *
                  hourlyRate *
                  (0.8 + Math.random() * 0.4),
              })),
          };
        });

        setResourceCosts(costHistory);
      }
    }

    // 更新任务状态
    if (data.task_status) {
      setTasks(data.task_status);

      // 更新任务队列状态
      const queueStatus = {
        pending: data.task_status.filter((t) => t.status === "pending").length,
        running: data.task_status.filter((t) => t.status === "running").length,
        completed: data.task_status.filter((t) => t.status === "completed")
          .length,
        failed: data.task_status.filter((t) => t.status === "failed").length,
        average_wait_time: 0,
        average_execution_time: 0,
      };

      // 计算平均等待时间
      const pendingTasks = data.task_status.filter(
        (t) => t.status === "pending" && t.waiting_time
      );
      if (pendingTasks.length > 0) {
        queueStatus.average_wait_time =
          pendingTasks.reduce((sum, task) => sum + task.waiting_time, 0) /
          pendingTasks.length;
      }

      // 计算平均执行时间
      const completedTasks = data.task_status.filter(
        (t) => t.status === "completed" && t.execution_time
      );
      if (completedTasks.length > 0) {
        queueStatus.average_execution_time =
          completedTasks.reduce((sum, task) => sum + task.execution_time, 0) /
          completedTasks.length;
      }

      setTaskQueueStatus(queueStatus);
    }
  }, []);

  // 获取数据的函数
  const fetchData = useCallback(async () => {
    try {
      // 获取资源列表
      const resourcesResponse = await fetch(
        "http://localhost:8000/resources/",
        {
          method: "GET",
          headers: {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            Pragma: "no-cache",
            Expires: "0",
          },
        }
      );
      if (resourcesResponse.ok) {
        const resourcesData = await resourcesResponse.json();
        console.log("获取资源数据:", resourcesData);
        setResources(resourcesData);
      }

      // 获取任务列表
      const tasksResponse = await fetch("http://localhost:8000/tasks/");
      if (tasksResponse.ok) {
        const tasksData = await tasksResponse.json();
        console.log("获取资源数据:", tasksData);
        setTasks(tasksData);
      }

      // 获取资源利用率（默认方式）
      const utilizationResponse = await fetch(
        "http://localhost:8000/metrics/resource-utilization"
      );
      if (utilizationResponse.ok) {
        const utilizationData = await utilizationResponse.json();
        setResourceUtilization(utilizationData);
      }

      // const utilizationResponse = await fetch('http://api.example.com/metrics/resource-utilization', {
      //   method: 'GET',
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      // });

      // 获取资源租赁开销数据
      try {
        const costsResponse = await fetch(
          "http://localhost:8000/metrics/resource-costs"
        );
        if (costsResponse.ok) {
          const costsData = await costsResponse.json();
          setResourceCosts(costsData);
        }
      } catch (error) {
        console.log("获取资源租赁开销数据失败，将使用模拟数据:", error);
      }

      // 获取任务队列状态
      const queueResponse = await fetch(
        "http://localhost:8000/metrics/task-queue"
      );
      if (queueResponse.ok) {
        const queueData = await queueResponse.json();
        setTaskQueueStatus(queueData);
      }
    } catch (error) {
      console.error("获取初始数据失败:", error);
    }
  }, []);

  // 初始化WebSocket连接
  useEffect(() => {
    // 创建WebSocket连接
    const connectWebSocket = () => {
      // 获取当前主机和端口，假设WebSocket服务与React应用在同一域
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.hostname;
      const port = "8000"; // 使用你的FastAPI后端端口
      const wsUrl = `${protocol}//${host}:${port}/ws`;

      console.log(`Connecting to WebSocket at ${wsUrl}`);

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      // WebSocket事件处理
      socket.onopen = () => {
        console.log("WebSocket连接已建立");
        setConnected(true);

        // 连接后立即获取初始数据
        fetchData();
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketData(data);
        } catch (error) {
          console.error("处理WebSocket消息时出错:", error);
        }
      };

      socket.onclose = () => {
        console.log("WebSocket连接已关闭");
        setConnected(false);

        // 尝试重新连接
        setTimeout(connectWebSocket, 3000);
      };

      socket.onerror = (error) => {
        console.error("WebSocket错误:", error);
        setConnected(false);
      };
    };

    // 尝试连接
    connectWebSocket();

    // 组件卸载时关闭WebSocket连接
    return () => {
      if (
        socketRef.current &&
        socketRef.current.readyState === WebSocket.OPEN
      ) {
        socketRef.current.close();
      }
    };
  }, [handleWebSocketData, fetchData]);

  // 根据资源类型获取每小时成本
  const getHourlyRateByType = (type) => {
    const rates = {
      cpu: 0.5, // CPU 每小时0.5元
      gpu: 4.0, // GPU 每小时4元
      memory: 0.2, // 内存 每小时0.2元
      storage: 0.1, // 存储 每小时0.1元
      network: 0.3, // 网络 每小时0.3元
    };
    return rates[type] || 0.5; // 默认0.5元/小时
  };

  // 保持WebSocket连接活跃
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (
        socketRef.current &&
        socketRef.current.readyState === WebSocket.OPEN
      ) {
        // 发送ping消息以保持连接
        socketRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000); // 每30秒ping一次

    return () => clearInterval(pingInterval);
  }, []);

  // 模拟数据 - 当WebSocket未连接时使用
  useEffect(() => {
    // 仅在WebSocket未连接时使用模拟数据
    if (connected) return;

    // 原有的模拟数据逻辑
    const mockResources = [
      {
        id: 1,
        name: "CPU资源池",
        type: "cpu",
        capacity: 64, // 总CPU核心数
        used: 28, // 已使用的CPU核心数
        utilization: 43.75, // 使用率
        status: "available",
      },
      {
        id: 2,
        name: "内存资源池",
        type: "memory",
        capacity: 256, // 总内存容量(GB)
        used: 128, // 已使用内存(GB)
        utilization: 50, // 使用率
        status: "available",
      },
    ];

    const mockTasks = [
      {
        id: 1,
        name: "ML训练任务-1",
        type: "training",
        status: "running",
        priority: 3,
        resource_requirements: { cpu: 4, memory: 16 }, // 新增：任务所需资源
        resource_id: null, // 不再指向具体服务器，改为null或资源池ID
        waiting_time: null,
        execution_time: 120,
      },
      {
        id: 2,
        name: "数据处理-1",
        type: "processing",
        status: "pending",
        priority: 2,
        resource_requirements: { cpu: 2, memory: 6 },
        resource_id: null,
        waiting_time: null,
        execution_time: null,
      },
      {
        id: 3,
        name: "日志分析-1",
        type: "analysis",
        status: "completed",
        priority: 1,
        resource_requirements: { cpu: 4, memory: 4 },
        resource_id: 1,
        waiting_time: null,
        execution_time: 78,
      },
      {
        id: 4,
        name: "API服务-1",
        type: "service",
        status: "running",
        priority: 3,
        resource_requirements: { cpu: 16, memory: 4 },
        resource_id: 2,
        waiting_time: null,
        execution_time: 360,
      },
      {
        id: 5,
        name: "数据库备份",
        type: "backup",
        status: "pending",
        priority: 1,
        resource_requirements: { cpu: 6, memory: 6 },
        resource_id: null,
        waiting_time: 12,
        execution_time: null,
      },
    ];

    const mockUtilizationHistory = mockResources.map((resource) => {
      const history = Array.from({ length: 30 }, (_, i) => ({
        time: new Date(Date.now() - (29 - i) * 60000).toLocaleTimeString(),
        value: Math.floor(Math.random() * 20) + resource.utilization - 10,
      }));

      return {
        ...resource,
        history,
      };
    });

    // 新增：模拟资源租赁开销历史数据
    const mockCostsHistory = mockResources.map((resource) => {
      const hourlyRate = getHourlyRateByType(resource.type);
      const history = Array.from({ length: 30 }, (_, i) => {
        // 假设成本与利用率和资源类型相关
        const baseUtilization =
          mockUtilizationHistory.find((r) => r.id === resource.id)?.history[i]
            ?.value || resource.utilization;
        return {
          time: new Date(Date.now() - (29 - i) * 60000).toLocaleTimeString(),
          value:
            (baseUtilization / 100) * hourlyRate * (0.8 + Math.random() * 0.4),
        };
      });

      return {
        ...resource,
        hourlyRate,
        history,
      };
    });

    const mockQueueStatus = {
      pending: 2,
      running: 2,
      completed: 1,
      failed: 0,
      average_wait_time: 28.5,
      average_execution_time: 99,
    };

    setResources(mockResources);
    setTasks(mockTasks);
    setResourceUtilization(mockUtilizationHistory);
    setResourceCosts(mockCostsHistory);
    setTaskQueueStatus(mockQueueStatus);

    // 模拟定期更新
    const timer = setInterval(() => {
      // 随机更新资源使用率
      setResources((prev) =>
        prev.map((resource) => ({
          ...resource,
          used: Math.min(
            resource.capacity,
            resource.used + (Math.random() > 0.5 ? 1 : -1)
          ),
          utilization: Math.min(
            100,
            Math.max(0, resource.utilization + (Math.random() > 0.5 ? 5 : -5))
          ),
        }))
      );

      // 随机更新任务状态
      setTasks((prev) => {
        const updated = [...prev];
        // 随机选择一个任务更新状态
        if (updated.length > 0) {
          const index = Math.floor(Math.random() * updated.length);
          if (updated[index].status === "pending") {
            updated[index].status = "running";
            updated[index].resource_id = Math.floor(Math.random() * 4) + 1;
            updated[index].waiting_time = null;
          } else if (updated[index].status === "running") {
            updated[index].status = "completed";
            updated[index].execution_time =
              Math.floor(Math.random() * 100) + 50;
          }
        }
        return updated;
      });

      // 任务状态计数 - 在setState回调中访问当前状态
      setTasks((prevTasks) => {
        // 首先更新任务
        const updatedTasks = [...prevTasks];
        // ... (原有任务状态更新逻辑)

        // 然后根据更新后的任务计算状态计数
        setTaskQueueStatus((prev) => ({
          ...prev,
          pending: updatedTasks.filter((t) => t.status === "pending").length,
          running: updatedTasks.filter((t) => t.status === "running").length,
          completed: updatedTasks.filter((t) => t.status === "completed")
            .length,
        }));

        return updatedTasks;
      });

      // 更新利用率历史
      setResourceUtilization((prev) =>
        prev.map((resource) => {
          const newHistory = [...resource.history];
          newHistory.shift();
          newHistory.push({
            time: new Date().toLocaleTimeString(),
            value: Math.min(100, Math.max(0, resource.utilization)),
          });

          return {
            ...resource,
            history: newHistory,
          };
        })
      );

      // 更新资源租赁开销历史
      setResourceCosts((prev) =>
        prev.map((resource) => {
          const newHistory = [...resource.history];
          newHistory.shift();

          // 获取最新的资源利用率
          const currentUtilization =
            resources.find((r) => r.id === resource.id)?.utilization || 50;

          // 计算新的成本值（基于利用率和随机波动）
          const newCost =
            (currentUtilization / 100) *
            resource.hourlyRate *
            (0.8 + Math.random() * 0.4);

          newHistory.push({
            time: new Date().toLocaleTimeString(),
            value: newCost,
          });

          return {
            ...resource,
            history: newHistory,
          };
        })
      );
    }, 2000);

    return () => clearInterval(timer);
  }, [connected]); // 移除tasks依赖

  // 计算任务类型分布
  // const taskTypes = tasks.reduce((acc, task) => {
  //   acc[task.type] = (acc[task.type] || 0) + 1;
  //   return acc;
  // }, {});

  // const taskTypeData = Object.entries(taskTypes).map(([name, value]) => ({ name, value }));

  // 任务状态分布数据
  const taskStatusData = [
    { name: "待处理", value: taskQueueStatus.pending, color: "#FFBB28" },
    { name: "运行中", value: taskQueueStatus.running, color: "#0088FE" },
    { name: "已完成", value: taskQueueStatus.completed, color: "#00C49F" },
    { name: "失败", value: taskQueueStatus.failed, color: "#FF8042" },
  ];

  // 状态颜色映射
  const statusColors = {
    pending: "#FFBB28",
    running: "#0088FE",
    completed: "#00C49F",
    failed: "#FF8042",
  };

  // 资源类型图标映射
  const typeIcons = {
    cpu: "💻",
    gpu: "🖥️",
    memory: "🧠",
    storage: "💾",
    network: "🌐",
  };

  // 处理新资源和任务添加
  // const handleResourceAdded = (newResource) => {
  //   // 更新资源列表
  //   // setResources(prev => [...prev, newResource]);
  //   // 也可以直接重新获取数据
  //   fetchData();
  // };

  // const handleTaskAdded = (newTask) => {
  //   // 更新任务列表
  //   // setTasks(prev => [...prev, newTask]);
  //   // 也可以直接重新获取数据
  //   fetchData();
  // };

  // 处理资源和任务删除
  const handleDeleteResource = async (resourceId) => {
    if (!window.confirm("确定要减少此资源池容量吗？")) {
      return;
    }

    try {
      const response = await fetch(
        `http://localhost:8000/resources/${resourceId}`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        // 资源删除成功，从状态中移除
        setResources((prev) =>
          prev.filter((resource) => resource.id !== resourceId)
        );
        setResourceUtilization((prev) =>
          prev.filter((resource) => resource.id !== resourceId)
        );
        alert("资源删除成功");
      } else {
        // 尝试解析错误信息
        let errorMessage = "删除失败";
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          // 如果响应不是JSON格式，使用默认错误信息
        }
        alert(errorMessage);
      }
    } catch (error) {
      console.error("删除资源时出错:", error);
      alert("删除资源时出错，请检查网络连接");
    }
  };

  // 在渲染前计算总开销数据
  const calculateTotalCostData = () => {
    // 确保有数据可用
    if (!resourceCosts.length) return [];

    // 假设所有资源的历史时间点相同
    const timePoints = resourceCosts[0].history.map((point) => point.time);

    // 创建总成本数据数组
    return timePoints.map((time, index) => {
      // 对每个时间点，计算所有资源的成本总和
      const totalValue = resourceCosts.reduce((sum, resource) => {
        return sum + (resource.history[index]?.value || 0);
      }, 0);

      return {
        time: time,
        value: totalValue,
      };
    });
  };

  // 计算总成本数据
  const totalCostData = calculateTotalCostData();

  return (
    <div className="p-4 bg-gray-100 min-h-screen">
      {/* 标题栏 */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">面向SaaS的云资源优化供应系统</h1>
        <div className="flex items-center">
          <span
            className={`w-3 h-3 rounded-full mr-2 ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          ></span>
          <span>{connected ? "已连接" : "未连接"}</span>
        </div>
      </div>

      {/* 添加新资源和任务的表单 */}
      {/* <ResourceTaskForms
        onResourceAdded={handleResourceAdded}
        onTaskAdded={handleTaskAdded}
      /> */}

      {/* 状态卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-4 rounded shadow">
          <div className="text-gray-500 mb-2">资源类型</div>
          <div className="text-3xl font-bold">{resources.length}</div>
        </div>
        <div className="bg-white p-4 rounded shadow">
          <div className="text-gray-500 mb-2">任务总数</div>
          <div className="text-3xl font-bold">{tasks.length}</div>
        </div>
        <div className="bg-white p-4 rounded shadow">
          <div className="text-gray-500 mb-2">平均等待时间</div>
          <div className="text-3xl font-bold">
            {taskQueueStatus.average_wait_time.toFixed(1)}秒
          </div>
        </div>
        <div className="bg-white p-4 rounded shadow">
          <div className="text-gray-500 mb-2">平均执行时间</div>
          <div className="text-3xl font-bold">
            {taskQueueStatus.average_execution_time.toFixed(1)}秒
          </div>
        </div>
      </div>

      {/* 资源利用率图表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-4">资源利用率</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={resources}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis unit="%" />
              <Tooltip formatter={(value) => [`${value}%`, "利用率"]} />
              <Legend />
              <Bar dataKey="utilization" name="利用率" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 任务状态分布 */}
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-4">任务状态分布</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={taskStatusData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                {taskStatusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [value, "任务数量"]} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 资源利用率详情 */}
      <div className="bg-white p-4 rounded shadow mb-6">
        <h2 className="text-lg font-semibold mb-4">资源利用率历史</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {resourceUtilization.map((resource) => (
            <div key={resource.id} className="border p-3 rounded">
              <div className="flex justify-between items-center mb-2">
                <h3 className="font-medium">
                  {typeIcons[resource.type]} {resource.name}
                </h3>
                <div className="flex items-center">
                  <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm mr-2">
                    {resource.utilization.toFixed(1)}%
                  </span>
                  {/* <button
                    className="text-red-500 hover:text-red-700"
                    onClick={() => handleDeleteResource(resource.id)}
                    title="删除资源"
                  >
                    🗑️
                  </button> */}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={resource.history}>
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#8884d8"
                    dot={false}
                  />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip
                    formatter={(value) => [`${value.toFixed(1)}%`, "利用率"]}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </div>

      {/* 资源租赁开销动态变化曲线 */}
      <div className="bg-white p-4 rounded shadow mb-6">
        <h2 className="text-lg font-semibold mb-4">资源租赁总开销动态变化</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={totalCostData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis name="总成本 (元/小时)" />
            <Tooltip
              formatter={(value) => [
                `${value.toFixed(2)}元/小时`,
                "总租赁成本",
              ]}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="value"
              name="资源总开销"
              stroke="#8884d8"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 8 }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* 可选：显示当前总开销 */}
        <div className="mt-4 text-center">
          <span className="text-xl font-semibold">
            当前总开销:{" "}
            {totalCostData[totalCostData.length - 1]?.value.toFixed(2) || 0}{" "}
            元/小时
          </span>
        </div>
      </div>

      {/* 资源列表 */}
      <ResourceList
        resources={resources}
        onDeleteResource={handleDeleteResource}
      />

      {/* 任务列表 */}
      <div className="bg-white p-4 rounded shadow">
        <h2 className="text-lg font-semibold mb-4">任务列表</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-2 px-4 border-b text-left">ID</th>
                <th className="py-2 px-4 border-b text-left">名称</th>
                <th className="py-2 px-4 border-b text-left">类型</th>
                <th className="py-2 px-4 border-b text-left">状态</th>
                <th className="py-2 px-4 border-b text-left">优先级</th>
                <th className="py-2 px-4 border-b text-left">资源</th>
                <th className="py-2 px-4 border-b text-left">等待/执行时间</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((task) => (
                <tr key={task.id}>
                  <td className="py-2 px-4 border-b">{task.id}</td>
                  <td className="py-2 px-4 border-b">{task.name}</td>
                  <td className="py-2 px-4 border-b">{task.type}</td>
                  <td className="py-2 px-4 border-b">
                    <span
                      className="px-2 py-1 rounded text-sm"
                      style={{
                        backgroundColor: `${statusColors[task.status]}20`,
                        color: statusColors[task.status],
                      }}
                    >
                      {task.status === "pending"
                        ? "待处理"
                        : task.status === "running"
                        ? "运行中"
                        : task.status === "completed"
                        ? "已完成"
                        : "失败"}
                    </span>
                  </td>
                  <td className="py-2 px-4 border-b">{task.priority}</td>
                  <td className="py-2 px-4 border-b">
                    {task.resource_requirements
                      ? `CPU: ${task.resource_requirements.cpu}核, 内存: ${task.resource_requirements.memory}GB`
                      : "-"}
                  </td>
                  <td className="py-2 px-4 border-b">
                    {task.waiting_time
                      ? `等待: ${task.waiting_time}秒`
                      : task.execution_time
                      ? `执行: ${task.execution_time}秒`
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
