import React, { useState } from 'react';

// 添加资源和任务的表单组件
const ResourceTaskForms = ({ onResourceAdded, onTaskAdded }) => {
  // 资源表单状态
  const [resourceForm, setResourceForm] = useState({
    name: '',
    type: 'cpu', // 默认值
    capacity: 0,
    meta_data: {}
  });

  // 任务表单状态
  const [taskForm, setTaskForm] = useState({
    name: '',
    type: '',
    priority: 1,
    resource_requirements: { cpu: 0, memory: 0, gpu: 0 }
  });

  // 显示/隐藏表单的状态
  const [showResourceForm, setShowResourceForm] = useState(false);
  const [showTaskForm, setShowTaskForm] = useState(false);

  // 处理资源表单变化
  const handleResourceChange = (e) => {
    const { name, value } = e.target;
    setResourceForm({
      ...resourceForm,
      [name]: name === 'capacity' ? parseFloat(value) : value
    });
  };

  // 处理任务表单变化
  const handleTaskChange = (e) => {
    const { name, value } = e.target;
    if (name.startsWith('req_')) {
      const reqType = name.split('_')[1]; // 例如，req_cpu => cpu
      setTaskForm({
        ...taskForm,
        resource_requirements: {
          ...taskForm.resource_requirements,
          [reqType]: parseFloat(value) || 0
        }
      });
    } else {
      setTaskForm({
        ...taskForm,
        [name]: name === 'priority' ? parseInt(value, 10) : value
      });
    }
  };

  // 提交新资源
  const handleResourceSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/resources/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(resourceForm)
      });

      if (response.ok) {
        const newResource = await response.json();
        setResourceForm({
          name: '',
          type: 'cpu',
          capacity: 0,
          meta_data: {}
        });
        setShowResourceForm(false);
        if (onResourceAdded) onResourceAdded(newResource);
        alert('资源添加成功');
      } else {
        const error = await response.json();
        alert(`添加失败: ${error.detail || '未知错误'}`);
      }
    } catch (error) {
      console.error('添加资源出错:', error);
      alert('添加资源时出错，请检查网络连接');
    }
  };

  // 提交新任务
  const handleTaskSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/tasks/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(taskForm)
      });

      if (response.ok) {
        const newTask = await response.json();
        setTaskForm({
          name: '',
          type: '',
          priority: 1,
          resource_requirements: { cpu: 0, memory: 0, gpu: 0 }
        });
        setShowTaskForm(false);
        if (onTaskAdded) onTaskAdded(newTask);
        alert('任务添加成功');
      } else {
        const error = await response.json();
        alert(`添加失败: ${error.detail || '未知错误'}`);
      }
    } catch (error) {
      console.error('添加任务出错:', error);
      alert('添加任务时出错，请检查网络连接');
    }
  };

  return (
    <div className="mb-6">
      {/* 按钮区域 */}
      <div className="flex space-x-4 mb-4">
        <button 
          onClick={() => setShowResourceForm(!showResourceForm)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          {showResourceForm ? '取消添加资源' : '添加新资源'}
        </button>
        <button 
          onClick={() => setShowTaskForm(!showTaskForm)}
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
        >
          {showTaskForm ? '取消添加任务' : '添加新任务'}
        </button>
      </div>

      {/* 添加资源表单 */}
      {showResourceForm && (
        <div className="bg-white p-4 rounded shadow mb-4">
          <h2 className="text-lg font-semibold mb-4">添加新资源</h2>
          <form onSubmit={handleResourceSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-700 mb-2">资源名称</label>
                <input
                  type="text"
                  name="name"
                  value={resourceForm.name}
                  onChange={handleResourceChange}
                  className="w-full px-3 py-2 border rounded"
                  required
                />
              </div>
              
              <div>
                <label className="block text-gray-700 mb-2">资源类型</label>
                <select
                  name="type"
                  value={resourceForm.type}
                  onChange={handleResourceChange}
                  className="w-full px-3 py-2 border rounded"
                  required
                >
                  <option value="cpu">CPU</option>
                  <option value="gpu">GPU</option>
                  <option value="memory">内存</option>
                  <option value="storage">存储</option>
                  <option value="network">网络</option>
                </select>
              </div>
              
              <div>
                <label className="block text-gray-700 mb-2">容量</label>
                <input
                  type="number"
                  name="capacity"
                  value={resourceForm.capacity}
                  onChange={handleResourceChange}
                  className="w-full px-3 py-2 border rounded"
                  min="0"
                  step="0.1"
                  required
                />
              </div>
            </div>
            
            <div className="mt-4">
              <button
                type="submit"
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                提交
              </button>
            </div>
          </form>
        </div>
      )}

      {/* 添加任务表单 */}
      {showTaskForm && (
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-4">添加新任务</h2>
          <form onSubmit={handleTaskSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-700 mb-2">任务名称</label>
                <input
                  type="text"
                  name="name"
                  value={taskForm.name}
                  onChange={handleTaskChange}
                  className="w-full px-3 py-2 border rounded"
                  required
                />
              </div>
              
              <div>
                <label className="block text-gray-700 mb-2">任务类型</label>
                <input
                  type="text"
                  name="type"
                  value={taskForm.type}
                  onChange={handleTaskChange}
                  className="w-full px-3 py-2 border rounded"
                  placeholder="如：training, processing, analysis"
                  required
                />
              </div>
              
              <div>
                <label className="block text-gray-700 mb-2">优先级（1-5）</label>
                <input
                  type="number"
                  name="priority"
                  value={taskForm.priority}
                  onChange={handleTaskChange}
                  className="w-full px-3 py-2 border rounded"
                  min="1"
                  max="5"
                  required
                />
              </div>
            </div>
            
            <div className="mt-4">
              <h3 className="font-medium mb-2">资源需求</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-gray-700 mb-2">CPU需求</label>
                  <input
                    type="number"
                    name="req_cpu"
                    value={taskForm.resource_requirements.cpu}
                    onChange={handleTaskChange}
                    className="w-full px-3 py-2 border rounded"
                    min="0"
                    step="0.1"
                  />
                </div>
                
                <div>
                  <label className="block text-gray-700 mb-2">内存需求 (GB)</label>
                  <input
                    type="number"
                    name="req_memory"
                    value={taskForm.resource_requirements.memory}
                    onChange={handleTaskChange}
                    className="w-full px-3 py-2 border rounded"
                    min="0"
                    step="0.1"
                  />
                </div>
                
                <div>
                  <label className="block text-gray-700 mb-2">GPU需求</label>
                  <input
                    type="number"
                    name="req_gpu"
                    value={taskForm.resource_requirements.gpu}
                    onChange={handleTaskChange}
                    className="w-full px-3 py-2 border rounded"
                    min="0"
                    step="0.1"
                  />
                </div>
              </div>
            </div>
            
            <div className="mt-4">
              <button
                type="submit"
                className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
              >
                提交
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};

export default ResourceTaskForms;