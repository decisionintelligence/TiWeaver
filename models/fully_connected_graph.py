import torch

class FullyConnectedGraph:
    def __init__(self, num_nodes, device=None):
        """
        初始化全连接图（有自环），并添加一个全局节点，与所有节点相连，所有其他节点指向全局节点。
        全局节点自身不自连接，且全局节点只接受边，不指出边。
        Args:
            num_nodes (int): 原始节点数（不包括全局节点）
            device: torch 设备，可选
        """
        self.num_nodes = num_nodes  # 原始节点数
        self.device = device
        self.total_nodes = num_nodes + 1  # 包含全局节点
        self.global_node = num_nodes      # 全局节点的编号
        self.edges = self._generate_edges()
        self.adj_matrix = self._generate_adj_matrix()

    def _generate_edges(self):
        # 生成所有有自环的边 (i, j) for 原始节点
        idx = torch.arange(self.num_nodes)
        grid_i, grid_j = torch.meshgrid(idx, idx, indexing='ij')
        edges = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)  # [num_nodes*num_nodes, 2]

        # 所有节点指向全局节点 (i, global_node)
        global_node = self.global_node
        from_nodes = torch.arange(self.num_nodes)
        to_global = torch.full((self.num_nodes,), global_node, dtype=torch.long)
        edges_to_global = torch.stack([from_nodes, to_global], dim=1)  # [num_nodes, 2]


        # 合并所有边（不包括全局节点自环，也不包括全局节点指出的边）
        all_edges = torch.cat([edges, edges_to_global], dim=0)
        # 转为list of tuple
        edges_list = [tuple(edge.tolist()) for edge in all_edges]
        return edges_list

    def _generate_adj_matrix(self):
        # 生成邻接矩阵（有自环），并添加全局节点（全局节点无自环且不指出边）
        adj = torch.zeros(self.total_nodes, self.total_nodes, device=self.device)
        # 原始节点之间全连接有自环
        adj[:self.num_nodes, :self.num_nodes] = 1
        # 全局节点自环为0（已初始化为0）
        # 原始节点指向全局节点
        adj[:self.num_nodes, self.global_node] = 1
        # 全局节点不指出边
        # adj[self.global_node, :] = 0  # 已为0
        return adj

    def get_edges(self):
        """返回边列表 [(i, j), ...]"""
        return self.edges

    def get_adj_matrix(self):
        """返回邻接矩阵 (torch.Tensor)"""
        return self.adj_matrix

    def __repr__(self):
        return f"FullyConnectedGraph(num_nodes={self.num_nodes})" 
    
# 测试 FullyConnectedGraph 的功能
if __name__ == "__main__":

    # 测试参数
    num_nodes = 4
    device = torch.device("cpu")

    # 实例化 FullyConnectedGraph
    graph = FullyConnectedGraph(num_nodes, device=device)

    '''
    邻接矩阵
    tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.]])
    '''

    # 打印边列表
    print("Edges:")
    for edge in graph.get_edges():
        print(edge)

    # 打印邻接矩阵
    print("\nAdjacency Matrix:")
    print(graph.get_adj_matrix())

    # 打印对象信息
    print("\nGraph Representation:")
    print(graph)
