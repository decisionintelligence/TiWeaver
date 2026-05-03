import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
# from dtaidistance import dtw
import warnings
warnings.filterwarnings('ignore')

class DWTClustering:
    """
    基于动态时间规整(DTW)距离的聚类算法
    适用于时间序列、序列数据或具有时序结构的patch数据
    """
    
    def __init__(self, n_clusters=3, method='hierarchical', 
                 dtw_window=None, dtw_use_c=False):
        """
        参数:
        - n_clusters: 聚类数量
        - method: 聚类方法 'hierarchical'或'kmedoids'
        - dtw_window: DTW窗口大小，None表示无限制
        - dtw_use_c: 是否使用C加速的DTW计算
        """
        self.n_clusters = n_clusters
        self.method = method
        self.dtw_window = dtw_window
        self.dtw_use_c = dtw_use_c
        self.labels_ = None
        self.cluster_centers_ = None
        self.distance_matrix_ = None
        
    def _compute_dtw_distance_matrix(self, X):
        """
        计算DTW距离矩阵
        X: [n_samples, seq_len, n_features] 或 [n_samples, seq_len]
        """
        n_samples = X.shape[0]
        
        # 如果输入是3D，转换为2D用于DTW计算（展平特征维度）
        if X.ndim == 3:
            seq_len, n_features = X.shape[1], X.shape[2]
            X_flat = X.reshape(n_samples, seq_len * n_features)
        else:
            X_flat = X
        
        # 计算DTW距离矩阵
        # if self.dtw_use_c and dtw is not None:
        #     # 使用C加速的DTW
        #     self.distance_matrix_ = dtw.distance_matrix_fast(
        #         X_flat, window=self.dtw_window
        #     )
        # else:
        # 使用Python实现的DTW
        self.distance_matrix_ = self._dtw_distance_matrix_python(X_flat)
        
        return self.distance_matrix_
    
    def _dtw_distance_matrix_python(self, X):
        """Python实现的DTW距离矩阵计算"""
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self._dtw_distance(X[i], X[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _dtw_distance(self, s1, s2):
        """计算两个序列之间的DTW距离"""
        n, m = len(s1), len(s2)
        dtw_matrix = np.zeros((n + 1, m + 1))
        
        # 初始化
        dtw_matrix[0, 0] = 0
        for i in range(1, n + 1):
            dtw_matrix[i, 0] = np.inf
        for j in range(1, m + 1):
            dtw_matrix[0, j] = np.inf
        
        # 填充DTW矩阵
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                # 寻找最小路径
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],     # 插入
                    dtw_matrix[i, j - 1],     # 删除  
                    dtw_matrix[i - 1, j - 1]  # 匹配
                )
        
        return dtw_matrix[n, m]
    
    def _kmedoids_clustering(self, distance_matrix):
        """K-Medoids聚类算法"""
        n_samples = distance_matrix.shape[0]
        
        # 随机初始化medoids
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for iteration in range(100):  # 最大迭代次数
            # 分配样本到最近的medoid
            labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
            
            new_medoid_indices = np.copy(medoid_indices)
            
            # 更新每个簇的medoid
            for cluster_idx in range(self.n_clusters):
                cluster_mask = (labels == cluster_idx)
                if np.sum(cluster_mask) > 0:
                    # 选择距离和最小的点作为medoid
                    cluster_distances = distance_matrix[cluster_mask][:, cluster_mask]
                    total_distances = np.sum(cluster_distances, axis=1)
                    new_medoid_idx = np.argmin(total_distances)
                    new_medoid_indices[cluster_idx] = np.where(cluster_mask)[0][new_medoid_idx]
            
            # 检查收敛
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
                
            medoid_indices = new_medoid_indices
        
        return labels, medoid_indices
    
    def fit(self, X):
        """
        对数据进行DTW聚类
        
        参数:
        - X: 输入数据 [n_samples, seq_len, n_features] 或 [n_samples, seq_len]
        """
        # 计算DTW距离矩阵
        distance_matrix = self._compute_dtw_distance_matrix(X)
        
        # 根据选择的方法进行聚类
        if self.method == 'hierarchical':
            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='precomputed',
                linkage='average'
            )
            self.labels_ = clustering.fit_predict(distance_matrix)
            
        elif self.method == 'kmedoids':
            # K-Medoids聚类
            self.labels_, medoid_indices = self._kmedoids_clustering(distance_matrix)
            # 存储medoids作为簇中心
            if X.ndim == 3:
                self.cluster_centers_ = X[medoid_indices]
            else:
                self.cluster_centers_ = X[medoid_indices]
        
        return self
    
    def fit_predict(self, X):
        """拟合数据并返回聚类标签"""
        return self.fit(X).labels_
    
    def get_cluster_centers(self):
        """获取簇中心（对于K-Medoids返回medoids，对于层次聚类返回平均序列）"""
        return self.cluster_centers_