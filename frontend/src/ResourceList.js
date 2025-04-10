import React, { useState } from 'react';

// 资源列表组件
const ResourceList = ({ resources, onDeleteResource }) => {
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [resourceToDelete, setResourceToDelete] = useState(null);
  const [deleteError, setDeleteError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // 资源类型图标映射
  const typeIcons = {
    cpu: '💻',
    gpu: '🖥️',
    memory: '🧠',
    storage: '💾',
    network: '🌐'
  };

  // 打开删除确认模态框
  const openDeleteModal = (resource) => {
    setResourceToDelete(resource);
    setDeleteError(null);
    setIsDeleteModalOpen(true);
  };

  // 关闭删除确认模态框
  const closeDeleteModal = () => {
    setIsDeleteModalOpen(false);
    setResourceToDelete(null);
    setDeleteError(null);
  };

  // 确认删除资源
  const confirmDelete = async () => {
    if (!resourceToDelete) return;
    
    setIsLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/resources/${resourceToDelete.id}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // 资源删除成功
        onDeleteResource(resourceToDelete.id);
        closeDeleteModal();
      } else {
        // 尝试解析错误信息
        let errorMessage = '删除失败';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          // 如果响应不是JSON格式，使用默认错误信息
        }
        setDeleteError(errorMessage);
      }
    } catch (error) {
      console.error('删除资源时出错:', error);
      setDeleteError('删除资源时出错，请检查网络连接');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded shadow mb-6">
      <h2 className="text-lg font-semibold mb-4">资源列表</h2>
      
      {resources.length === 0 ? (
        <div className="text-center py-4 text-gray-500">暂无资源</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead>
              <tr className="bg-gray-100">
                <th className="py-2 px-4 border-b text-left">ID</th>
                <th className="py-2 px-4 border-b text-left">名称</th>
                <th className="py-2 px-4 border-b text-left">类型</th>
                <th className="py-2 px-4 border-b text-left">容量</th>
                <th className="py-2 px-4 border-b text-left">已使用</th>
                <th className="py-2 px-4 border-b text-left">利用率</th>
                <th className="py-2 px-4 border-b text-left">状态</th>
                <th className="py-2 px-4 border-b text-left">操作</th>
              </tr>
            </thead>
            <tbody>
              {resources.map(resource => {
                // 确保 utilization 有值，如果没有则计算或设为默认值
                const utilization = resource.utilization !== undefined ? 
                  resource.utilization : 
                  (resource.capacity > 0 ? (resource.used / resource.capacity) * 100 : 0);
                
                return (
                  <tr key={resource.id}>
                    <td className="py-2 px-4 border-b">{resource.id}</td>
                    <td className="py-2 px-4 border-b">{resource.name}</td>
                    <td className="py-2 px-4 border-b">
                      {typeIcons[resource.type]} {resource.type === 'cpu' ? 'CPU' :
                        resource.type === 'gpu' ? 'GPU' :
                        resource.type === 'memory' ? '内存' :
                        resource.type === 'storage' ? '存储' : '网络'}
                    </td>
                    <td className="py-2 px-4 border-b">{resource.capacity}</td>
                    <td className="py-2 px-4 border-b">{resource.used?.toFixed(1) || '0.0'}</td>
                    <td className="py-2 px-4 border-b">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${Math.min(utilization, 100)}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-600">{utilization.toFixed(1)}%</span>
                    </td>
                    <td className="py-2 px-4 border-b">
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        resource.status === 'available' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {resource.status === 'available' ? '可用' : '不可用'}
                      </span>
                    </td>
                    <td className="py-2 px-4 border-b">
                      <button
                        className="text-red-500 hover:text-red-700"
                        onClick={() => openDeleteModal(resource)}
                        title="删除资源"
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* 删除确认模态框 */}
      {isDeleteModalOpen && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full flex items-center justify-center z-50">
          <div className="bg-white p-5 rounded-md shadow-xl max-w-md mx-auto">
            <h3 className="text-lg font-medium mb-4">确认删除</h3>
            
            <p className="mb-4">
              确定要删除资源 "{resourceToDelete?.name}" 吗？此操作无法撤销。
            </p>
            
            {deleteError && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                {deleteError}
              </div>
            )}
            
            <div className="flex justify-end space-x-3">
              <button
                className="px-4 py-2 bg-gray-300 text-gray-800 rounded hover:bg-gray-400"
                onClick={closeDeleteModal}
                disabled={isLoading}
              >
                取消
              </button>
              <button
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
                onClick={confirmDelete}
                disabled={isLoading}
              >
                {isLoading ? '删除中...' : '确认删除'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResourceList;