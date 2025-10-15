import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix, coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda import amp
from sklearn.manifold import TSNE  # 导入t-SNE
import matplotlib.pyplot as plt  # 导入绘图库
import matplotlib.patches as mpatches  # 用于创建图例


# 1. 加载YelpChi数据集
def load_yelpchi_data(file_path):
    """加载YelpChi.mat文件并提取三种关系矩阵、节点特征和标签"""
    data = sio.loadmat(file_path)

    # 提取三种关系矩阵
    net_rur = data['net_rur']  # R-U-R关系
    net_rsr = data['net_rsr']  # R-S-R关系
    net_rtr = data['net_rtr']  # R-T-R关系

    # 确保矩阵对称（关系是无向的）
    net_rur = (net_rur + net_rur.T) > 0
    net_rsr = (net_rsr + net_rsr.T) > 0
    net_rtr = (net_rtr + net_rtr.T) > 0

    # 提取节点特征和标签
    features = data['features'] if 'features' in data else None
    labels = data['label'] if 'label' in data else None

    # 如果找不到标准名称，尝试其他可能名称
    if features is None:
        for key in data.keys():
            if 'feat' in key.lower() or 'feature' in key.lower():
                features = data[key]
                break

    if labels is None:
        for key in data.keys():
            if 'label' in key.lower() or 'target' in key.lower():
                labels = data[key]
                break

    # 确保特征和标签可用
    if features is None or labels is None:
        raise ValueError("未能在数据集中找到节点特征或标签")

    # 添加特征工程
    from sklearn.decomposition import PCA
    if features.shape[1] > 100:
        pca = PCA(n_components=100)
        features = pca.fit_transform(features)

    # 转换标签为1D数组
    labels = labels.flatten()

    return (net_rur.astype(np.int32), net_rsr.astype(np.int32), net_rtr.astype(np.int32),
            features.astype(np.float32), labels.astype(np.int64))

# 加载Amazon数据集
def load_amazon_data(file_path):
    """加载Amazon.mat文件并提取三种关系矩阵、节点特征和标签"""
    data = sio.loadmat(file_path)

    # 提取三种关系矩阵
    net_upu = data['net_upu']  # U-P-U关系
    net_usu = data['net_usu']  # U-S-U关系
    net_uvu = data['net_uvu']  # U-V-U关系

    # 确保矩阵对称（关系是无向的）
    net_upu = (net_upu + net_upu.T) > 0
    net_usu = (net_usu + net_usu.T) > 0
    net_uvu = (net_uvu + net_uvu.T) > 0

    # 提取节点特征和标签
    features = data['features'] if 'features' in data else None
    labels = data['label'] if 'label' in data else None

    # 如果找不到标准名称，尝试其他可能名称
    if features is None:
        for key in data.keys():
            if 'feat' in key.lower() or 'feature' in key.lower():
                features = data[key]
                break

    if labels is None:
        for key in data.keys():
            if 'label' in key.lower() or 'target' in key.lower():
                labels = data[key]
                break

    # 确保特征和标签可用
    if features is None or labels is None:
        raise ValueError("未能在数据集中找到节点特征或标签")

    # 添加特征工程
    from sklearn.decomposition import PCA
    if features.shape[1] > 100:
        pca = PCA(n_components=100)
        features = pca.fit_transform(features)

    # 转换标签为1D数组
    labels = labels.flatten()

    return (net_upu.astype(np.int32), net_usu.astype(np.int32), net_uvu.astype(np.int32),
            features.astype(np.float32), labels.astype(np.int64))


# 2. 从邻接矩阵中提取超边
def extract_hyperedges_from_adj(adj_matrix):
    """从邻接矩阵中提取连通分量作为超边"""
    if not sp.issparse(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix)

    n_components, labels = connected_components(
        csgraph=adj_matrix,
        directed=False,
        return_labels=True
    )

    hyperedges = []
    for comp_id in range(n_components):
        node_indices = np.where(labels == comp_id)[0]
        if len(node_indices) > 1:
            hyperedges.append(node_indices.tolist())

    return hyperedges


# 3. 构建超图关联矩阵
def build_hypergraph_incidence_matrix(hyperedges, n_nodes):
    n_hyperedges = len(hyperedges)
    H = lil_matrix((n_nodes, n_hyperedges), dtype=np.float32)

    for edge_idx, nodes in enumerate(hyperedges):
        H[nodes, edge_idx] = 1.0

    return H.tocsr()


def sparse_matrix_to_torch(H_norm):
    """将SciPy稀疏矩阵转换为PyTorch稀疏张量"""
    if not isinstance(H_norm, coo_matrix):
        H_norm = H_norm.tocoo()

    row = H_norm.row
    col = H_norm.col
    data = H_norm.data

    # 确保数据是浮点数
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    indices = np.vstack([row, col])
    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(data).float()
    shape = torch.Size(H_norm.shape)

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


class GatedResidualConnection(nn.Module):
    """动态门控残差连接模块"""

    def __init__(self, in_features):
        super().__init__()
        # 门控机制 - 学习控制新旧特征的比例
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()  # 输出在0-1之间
        )
        # 层归一化稳定训练
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, new_features, old_features):
        """
        new_features: 当前层计算的新特征
        old_features: 前一层的特征（或初始特征）
        """
        # 计算门控值 - 决定保留多少旧特征
        gate_value = self.gate(old_features)

        # 应用门控机制：output = gate * new_features + (1 - gate) * old_features
        output = gate_value * new_features + (1 - gate_value) * old_features

        # 层归一化稳定输出
        return self.layer_norm(output)


class HyperEdgeAttention(nn.Module):
    """超边注意力机制 - 学习不同超边的重要性"""

    def __init__(self, in_features, hidden_dim=64):
        super(HyperEdgeAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, hyperedge_features):
        # 计算注意力分数
        Q = self.query(hyperedge_features)
        K = self.key(hyperedge_features)
        V = self.value(hyperedge_features)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权聚合
        weighted_features = torch.matmul(attention_weights, V)
        return weighted_features


class NodeAttention(nn.Module):
    """节点注意力机制 - 在聚合超边信息时关注重要节点"""

    def __init__(self, in_features, hidden_dim=64):
        super(NodeAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, node_features, hyperedge_features):
        # 节点作为query，超边作为key/value
        Q = self.query(node_features)
        K = self.key(hyperedge_features)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.t()) / self.scale.to(Q.device)
        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights


class MultiHeadHyperGraphConv(nn.Module):
    """多头注意力超图卷积层"""

    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1):
        super(MultiHeadHyperGraphConv, self).__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads

        # 多头投影
        self.query_proj = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(n_heads)
        ])
        self.key_proj = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(n_heads)
        ])
        self.value_proj = nn.ModuleList([
            nn.Linear(in_features, self.head_dim) for _ in range(n_heads)
        ])

        # 输出层
        self.out_proj = nn.Linear(n_heads * self.head_dim, out_features)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, X, H_norm):
        # 确保使用float32进行稀疏矩阵运算
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 第一步：节点特征通过超边传播
            if X.dtype != torch.float32:
                X_fp32 = X.to(torch.float32)
            else:
                X_fp32 = X
                
            if H_norm.dtype != torch.float32:
                H_norm_fp32 = H_norm.to(torch.float32)
            else:
                H_norm_fp32 = H_norm
                
            hyperedge_features = torch.sparse.mm(H_norm_fp32.t(), X_fp32)
            
            # 如果原始输入是半精度，将结果转换回半精度
            if X.dtype != torch.float32:
                hyperedge_features = hyperedge_features.to(X.dtype)

        batch_size = X.size(0)
        head_outputs = []

        for i in range(self.n_heads):
            # 投影到子空间
            Q_hyper = self.query_proj[i](hyperedge_features)
            K_hyper = self.key_proj[i](hyperedge_features)
            V_hyper = self.value_proj[i](hyperedge_features)

            # 计算超边间的注意力
            attention_scores = torch.matmul(Q_hyper, K_hyper.t()) / self.scale.to(X.device)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # 加权超边特征
            weighted_hyper = torch.matmul(attention_weights, V_hyper)

            # 确保使用float32进行稀疏矩阵运算
            with torch.amp.autocast(device_type='cuda', enabled=False):
                if weighted_hyper.dtype != torch.float32:
                    weighted_hyper_fp32 = weighted_hyper.to(torch.float32)
                else:
                    weighted_hyper_fp32 = weighted_hyper
                    
                node_agg = torch.sparse.mm(H_norm_fp32, weighted_hyper_fp32)
                
                # 如果原始输入是半精度，将结果转换回半精度
                if X.dtype != torch.float32:
                    node_agg = node_agg.to(X.dtype)
            head_outputs.append(node_agg)

        # 拼接多头输出
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(concatenated)
        return output


class AttentiveHyperGraphConv(nn.Module):
    """带注意力机制的超图卷积层"""

    def __init__(self, in_features, out_features, att_hidden_dim=64, dropout=0.1):
        super(AttentiveHyperGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 超边注意力
        self.hyper_attention = HyperEdgeAttention(in_features, att_hidden_dim)

        # 节点注意力
        self.node_attention = NodeAttention(in_features, att_hidden_dim)

        # 变换层
        self.transform = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H_norm):
        # 确保使用float32进行稀疏矩阵运算
        with torch.cuda.amp.autocast(enabled=False):
            if X.dtype != torch.float32:
                X_fp32 = X.to(torch.float32)
            else:
                X_fp32 = X
                
            if H_norm.dtype != torch.float32:
                H_norm_fp32 = H_norm.to(torch.float32)
            else:
                H_norm_fp32 = H_norm
                
            hyperedge_features = torch.sparse.mm(H_norm_fp32.t(), X_fp32)
            
            # 如果原始输入是半精度，将结果转换回半精度
            if X.dtype != torch.float32:
                hyperedge_features = hyperedge_features.to(X.dtype)

        # 应用超边注意力
        attended_hyper = self.hyper_attention(hyperedge_features)

        # 计算节点-超边注意力权重
        node_hyper_weights = self.node_attention(X, attended_hyper)

        # 创建注意力加权的关联矩阵
        H_att = H_norm.to_dense() * node_hyper_weights

        # 聚合加权超边特征
        aggregated_features = torch.mm(H_att, attended_hyper)

        # 应用变换
        output = self.transform(aggregated_features)
        return output


class ContrastiveLearningModule(nn.Module):
    """对比学习模块 - 使用负采样优化内存"""

    def __init__(self, hidden_dim, temperature=0.5, projection_dim=128, neg_samples=500):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.neg_samples = neg_samples

        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

        # 添加预测头
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, z_i, z_j):
        """计算对比学习损失 - 使用负采样"""
        n_nodes = z_i.size(0)

        # 投影到对比空间
        h_i = self.projection_head(z_i)
        # 使用预测头
        h_i = self.predictor(h_i)
        h_j = self.projection_head(z_j)

        # 归一化
        h_i = F.normalize(h_i, p=2, dim=1)
        h_j = F.normalize(h_j, p=2, dim=1)

        # 随机选择负样本索引
        neg_idx = torch.randint(0, n_nodes, (self.neg_samples,)).to(z_i.device)
        h_neg_i = h_i[neg_idx]
        h_neg_j = h_j[neg_idx]

        # 计算正样本相似度
        pos_sim = torch.sum(h_i * h_j, dim=1) / self.temperature  # [n_nodes]

        # 计算负样本相似度
        neg_sim_i = torch.mm(h_i, h_neg_i.t()) / self.temperature  # [n_nodes, neg_samples]
        neg_sim_j = torch.mm(h_i, h_neg_j.t()) / self.temperature  # [n_nodes, neg_samples]

        # 合并负样本
        neg_sim = torch.cat([neg_sim_i, neg_sim_j], dim=1)  # [n_nodes, 2*neg_samples]

        # 计算InfoNCE损失
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.sum(torch.exp(neg_sim), dim=1)

        loss = -torch.log(numerator / denominator)
        return loss.mean()


class HierarchicalContrastiveLearning(nn.Module):
    """分层对比学习模块 - 结合节点级和超边级对比"""

    def __init__(self, hidden_dim, temperature=0.5, projection_dim=128):
        super(HierarchicalContrastiveLearning, self).__init__()
        # 节点级对比
        self.node_contrastive = ContrastiveLearningModule(
            hidden_dim, temperature, projection_dim
        )

        # 超边级对比
        self.hyper_contrastive = ContrastiveLearningModule(
            hidden_dim, temperature, projection_dim
        )

        # 超边投影头
        self.hyper_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, node_reprs1, node_reprs2, hyper_reprs1, hyper_reprs2):
        """计算分层对比损失"""
        # 节点级对比损失
        node_loss = self.node_contrastive(node_reprs1, node_reprs2)

        # 超边级对比损失
        hyper_proj1 = self.hyper_projector(hyper_reprs1)
        hyper_proj2 = self.hyper_projector(hyper_reprs2)
        hyper_loss = self.hyper_contrastive(hyper_proj1, hyper_proj2)

        return node_loss + hyper_loss, node_loss, hyper_loss


class ContrastiveAugmenter:
    """对比学习数据增强器"""

    def __init__(self, feature_drop_rate=0.2, edge_drop_rate=0.1, feature_noise_scale=0.05):
        self.feature_drop_rate = feature_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.feature_noise_scale = feature_noise_scale

    def augment_features(self, X):
        """特征增强：随机掩码和添加噪声"""
        # 随机掩码
        mask = torch.rand_like(X) > self.feature_drop_rate
        X_aug = X * mask.float()

        # 添加高斯噪声
        noise = torch.randn_like(X) * self.feature_noise_scale
        X_aug += noise

        return X_aug

    def augment_hypergraph(self, H):
        """超图增强：随机丢弃超边"""
        if isinstance(H, sp.spmatrix):
            # 处理SciPy稀疏矩阵
            if not isinstance(H, sp.csr_matrix):
                H = H.tocsr()

            # 随机丢弃超边
            n_edges = H.shape[1]
            keep_indices = np.random.choice([True, False], size=n_edges,
                                            p=[1 - self.edge_drop_rate, self.edge_drop_rate])
            H_aug = H[:, keep_indices]
            return H_aug
        elif isinstance(H, torch.Tensor):
            # 处理PyTorch稀疏张量
            if not H.is_sparse:
                raise ValueError("输入张量必须是稀疏张量")

            # 获取设备信息
            device = H.device

            # 转换为COO格式
            H_coo = H.coalesce()
            indices = H_coo.indices()
            values = H_coo.values()

            # 随机丢弃超边(列)
            n_edges = H.size(1)
            # 在相同设备上创建keep_edges张量
            keep_edges = torch.rand(n_edges, device=device) > self.edge_drop_rate
            keep_mask = keep_edges[indices[1]]

            # 应用掩码
            new_indices = indices[:, keep_mask]
            new_values = values[keep_mask]

            # 创建新的稀疏张量
            H_aug = torch.sparse_coo_tensor(
                new_indices,
                new_values,
                H.size(),
                device=device
            ).coalesce()
            return H_aug
        else:
            raise TypeError("输入必须是SciPy稀疏矩阵或PyTorch稀疏张量")


class EnhancedFraudDetectionModelWithCL(nn.Module):
    """增强的欺诈检测模型 - 包含注意力机制和对比学习"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,
                 n_heads=4, dropout=0.3, use_multihead=True,
                 projection_dim=128, temperature=0.5):
        super(EnhancedFraudDetectionModelWithCL, self).__init__()
        self.layers = nn.ModuleList()
        self.gated_residuals = nn.ModuleList()  # 新增：门控残差模块列
        self.dropout = nn.Dropout(dropout)
        self.use_contrastive = True

        # 存储超边表示的列表
        self.hyperedge_projections = nn.ModuleList()

        # 输入层
        if use_multihead:
            conv_layer = MultiHeadHyperGraphConv(
                input_dim, hidden_dim, n_heads, dropout
            )
        else:
            conv_layer = AttentiveHyperGraphConv(
                input_dim, hidden_dim, hidden_dim // 2, dropout
            )
        self.layers.append(conv_layer)
        # 为输入层添加门控残差
        self.gated_residuals.append(GatedResidualConnection(hidden_dim))

        # 为每一层添加超边投影头
        self.hyperedge_projections.append(
            nn.Sequential(
                nn.Linear(hidden_dim, projection_dim),
                nn.ReLU()
            )
        )

        # 中间层
        for _ in range(num_layers - 2):
            if use_multihead:
                conv_layer = MultiHeadHyperGraphConv(
                    hidden_dim, hidden_dim, n_heads, dropout
                )
            else:
                conv_layer = AttentiveHyperGraphConv(
                    hidden_dim, hidden_dim, hidden_dim // 2, dropout
                )
            self.layers.append(conv_layer)
            # 为中间层添加门控残差
            self.gated_residuals.append(GatedResidualConnection(hidden_dim))
            # 为每一层添加超边投影头
            self.hyperedge_projections.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, projection_dim),
                    nn.ReLU()
                )
            )

        # 输出层
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # 为输出层添加门控残差
        self.gated_residuals.append(GatedResidualConnection(hidden_dim))

        # 添加最后一层的超边投影头
        self.hyperedge_projections.append(
            nn.Sequential(
                nn.Linear(hidden_dim, projection_dim),
                nn.ReLU()
            )
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 残差连接
        self.residual = nn.Linear(input_dim, hidden_dim)

        # 分层对比学习模块
        self.hierarchical_cl = HierarchicalContrastiveLearning(
            hidden_dim, temperature, projection_dim
        )

        # 数据增强器
        self.augmenter = ContrastiveAugmenter()
        # 存储初始特征（用于门控残差）
        self.initial_features = None

    def forward(self, X, H_norm):
        # 保存初始特征用于门控残差
        self.initial_features = self.residual(X)
        residual = self.initial_features.clone()

        # 存储中间节点表示用于对比学习
        node_representations = []
        # 存储超边表示用于对比学习
        hyperedge_representations = []

        # 通过多层超图卷积
        for i, (layer, gated_residual) in enumerate(zip(self.layers, self.gated_residuals)):
            if isinstance(layer, (MultiHeadHyperGraphConv, AttentiveHyperGraphConv)):
                # 超图卷积
                new_X = layer(X, H_norm)

                # 应用门控残差连接（关键修改）
                X = gated_residual(new_X, residual)
                # 计算超边表示
                with torch.no_grad():
                    # 确保使用float32进行稀疏矩阵运算
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        if X.dtype != torch.float32:
                            X_fp32 = X.to(torch.float32)
                        else:
                            X_fp32 = X
                            
                        if H_norm.dtype != torch.float32:
                            H_norm_fp32 = H_norm.to(torch.float32)
                        else:
                            H_norm_fp32 = H_norm
                            
                        hyperedge_feats = torch.sparse.mm(H_norm_fp32.t(), X_fp32)
                        # 如果原始输入是半精度，将结果转换回半精度
                        if X.dtype != torch.float32:
                            hyperedge_feats = hyperedge_feats.to(X.dtype)

                # 应用超边投影
                hyper_repr = self.hyperedge_projections[i](hyperedge_feats)
                hyperedge_representations.append(hyper_repr)

                # 更新残差（用于下一层）
                residual = X.clone()

                X = F.relu(X)
                X = self.dropout(X)

                # 保存当前层的表示用于对比学习
                if self.use_contrastive and i < len(self.layers) - 1:
                    node_representations.append(X)
            else:
                # 线性层处理
                new_X = layer(X)

                # 应用门控残差连接
                X = gated_residual(new_X, residual)
                X = F.relu(X)
                X = self.dropout(X)

                # 更新残差
                residual = X.clone()

                # 计算最后一层的超边表示
                with torch.no_grad():
                    # 确保使用float32进行稀疏矩阵运算
                    with torch.amp.autocast(device_type='cuda', enabled=False):
                        if X.dtype != torch.float32:
                            X_fp32 = X.to(torch.float32)
                        else:
                            X_fp32 = X
                            
                        if H_norm.dtype != torch.float32:
                            H_norm_fp32 = H_norm.to(torch.float32)
                        else:
                            H_norm_fp32 = H_norm
                            
                        hyperedge_feats = torch.sparse.mm(H_norm_fp32.t(), X_fp32)
                        
                        # 如果原始输入是半精度，将结果转换回半精度
                        if X.dtype != torch.float32:
                            hyperedge_feats = hyperedge_feats.to(X.dtype)
                            
                # 应用超边投影
                hyper_repr = self.hyperedge_projections[i](hyperedge_feats)
                hyperedge_representations.append(hyper_repr)

                node_representations.append(X)

        # 分类
        logits = self.classifier(X)

        # 返回分类结果、节点表示和超边表示
        return logits, node_representations, hyperedge_representations

    def compute_contrastive_loss(self, reprs1, reprs2):
        """计算分层对比学习损失"""
        if not self.use_contrastive or len(reprs1[0]) == 0:
            return torch.tensor(0.0).to(reprs1[0][0].device)

        total_loss = 0.0
        num_layers = min(len(reprs1[0]), len(reprs1[1]))

        for i in range(num_layers):
            # 解包表示
            node_reprs1, hyper_reprs1 = reprs1[0][i], reprs1[1][i]
            node_reprs2, hyper_reprs2 = reprs2[0][i], reprs2[1][i]

            # 计算分层对比损失
            layer_loss, node_loss, hyper_loss = self.hierarchical_cl(
                node_reprs1, node_reprs2,
                hyper_reprs1, hyper_reprs2
            )
            total_loss += layer_loss

        return total_loss / num_layers


def calculate_gmean(y_true, y_pred):
    """计算G-mean指标"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        return 0.0  # 如果只有一个类别，返回0

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def train_model(model, optimizer, criterion, X, H_norm, y, train_idx, val_idx,
                epochs=100, patience=15, device='cpu', alpha=0.5):
    """训练模型 - 优化版本确保每次迭代稳定提升"""
    model.to(device)
    X = X.to(device)
    H_norm = H_norm.to(device)
    y = y.to(device)
    # 添加混合精度训练的梯度缩放器
    scaler = amp.GradScaler()  # 新增

    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-5
    )

    # 添加梯度裁剪防止梯度爆炸
    max_grad_norm = 2.0

    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_val_gmean = 0.0
    best_epoch = 0
    no_improve = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': [],
        'val_gmean': [],
        'lr': [],
        'cl_loss': []
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 生成两个增强视图
        X_aug1 = model.augmenter.augment_features(X)
        X_aug2 = model.augmenter.augment_features(X)
        H_aug1 = model.augmenter.augment_hypergraph(H_norm)
        H_aug2 = model.augmenter.augment_hypergraph(H_norm)

        # 如果H_norm是SciPy稀疏矩阵，需要转换为PyTorch张量
        if not isinstance(H_aug1, torch.Tensor) and sp.issparse(H_aug1):
            H_aug1_torch = sparse_matrix_to_torch(H_aug1).to(device)
        else:
            H_aug1_torch = H_aug1  # 如果已经是PyTorch张量

        if not isinstance(H_aug2, torch.Tensor) and sp.issparse(H_aug2):
            H_aug2_torch = sparse_matrix_to_torch(H_aug2).to(device)
        else:
            H_aug2_torch = H_aug2

        # 在原始视图前向传播前，保存当前的初始残差
        with torch.no_grad():
            init_residual_original = model.residual(X)
            model.initial_features = init_residual_original.clone()

         # ===== 使用混合精度训练 =====
        with torch.amp.autocast(device_type='cuda', enabled=False):  # 新增：启用混合精度上下文
            # 前向传播 - 原始视图
            logits, node_reprs, hyper_reprs = model(X, H_norm)
            reprs1 = (node_reprs, hyper_reprs)

            # 前向传播 - 增强视图1
            # 注意：这里需要传递增强后的初始特征
            with torch.no_grad():
                # 为增强视图计算初始残差
                init_residual_aug1 = model.residual(X_aug1)
                model.initial_features = init_residual_aug1
            _, node_reprs_aug1, hyper_reprs_aug1 = model(X_aug1, H_aug1_torch)
            reprs2 = (node_reprs_aug1, hyper_reprs_aug1)

            # 前向传播 - 增强视图2
            with torch.no_grad():
                init_residual_aug2 = model.residual(X_aug2)
                model.initial_features = init_residual_aug2
            _, node_reprs_aug2, hyper_reprs_aug2 = model(X_aug2, H_aug2_torch)
            reprs3 = (node_reprs_aug2, hyper_reprs_aug2)

            # 恢复原始初始特征
            model.initial_features = init_residual_original

            # 计算训练损失
            train_logits = logits[train_idx]
            train_labels = y[train_idx]
            cls_loss = criterion(train_logits, train_labels)

            # 计算分层对比学习损失 (视图1 vs 视图2 和 视图1 vs 视图3)
            cl_loss1 = model.compute_contrastive_loss(reprs1, reprs2)
            cl_loss2 = model.compute_contrastive_loss(reprs1, reprs3)
            cl_loss = (cl_loss1 + cl_loss2) / 2

            # 总损失 = 分类损失 + alpha * 对比学习损失
            total_loss = cls_loss + alpha * cl_loss

        # ===== 使用梯度缩放进行反向传播 =====
        scaler.scale(total_loss).backward()  # 替代原来的 total_loss.backward()
        
        # 梯度裁剪（在缩放后解缩梯度）
        scaler.unscale_(optimizer)  # 新增：在裁剪前解缩梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 使用缩放器更新权重
        scaler.step(optimizer)
        scaler.update()

        # 验证
        model.eval()
        with torch.no_grad():
            # 使用原始视图进行评估
            # 确保使用正确的初始残差
            model.initial_features = init_residual_original
            logits, _, _ = model(X, H_norm)
            val_logits = logits[val_idx]
            val_labels = y[val_idx]
            val_loss = criterion(val_logits, val_labels)

            # 计算验证指标
            val_probs = F.softmax(val_logits, dim=1)
            val_preds = torch.argmax(val_logits, dim=1)

            # 计算各种指标
            val_f1 = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average='macro')
            if val_probs.shape[1] == 2:
                val_auc = roc_auc_score(val_labels.cpu().numpy(), val_probs[:, 1].cpu().numpy())
            else:
                val_auc = roc_auc_score(val_labels.cpu().numpy(), val_probs.cpu().numpy(), multi_class='ovr')
            val_gmean = calculate_gmean(val_labels.cpu().numpy(), val_preds.cpu().numpy())
            cm = confusion_matrix(val_labels.cpu().numpy(), val_preds.cpu().numpy())

            # 记录历史指标
            history['train_loss'].append(total_loss.item())
            history['val_loss'].append(val_loss.item())
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            history['val_gmean'].append(val_gmean)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['cl_loss'].append(cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0)

        # 更新学习率调度器 - 基于F1分数
        scheduler.step()

        # 打印进度
        print(f'Epoch {epoch + 1:03d} | Total Loss: {total_loss.item():.4f} '
              f'(Cls: {cls_loss.item():.4f}, CL: {cl_loss.item():.4f}) | '
              f'Val Loss: {val_loss.item():.4f} | '
              f'Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f} | Val G-mean: {val_gmean:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print("混淆矩阵:")
        print(cm)

        # 复合早停机制 - 考虑F1、AUC和G-mean的综合提升
        current_improvement = (val_f1 + val_auc + val_gmean) - (best_val_f1 + best_val_auc + best_val_gmean)

        # 如果当前epoch比之前所有epoch都好
        if current_improvement > 0:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_val_gmean = val_gmean
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'↳ 保存最佳模型 (综合提升: {current_improvement:.4f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'早停于epoch {epoch + 1}, 最佳epoch: {best_epoch + 1} '
                      f'(F1={best_val_f1:.4f}, AUC={best_val_auc:.4f}, G-mean={best_val_gmean:.4f})')
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pt'))
    print(f'最终模型来自epoch {best_epoch + 1}: '
          f'F1={best_val_f1:.4f}, AUC={best_val_auc:.4f}, G-mean={best_val_gmean:.4f}')

    return model, history


def evaluate_model(model, X, H_norm, y, test_idx, device='cpu', visualize=True):
    """评估模型性能"""
    # 关闭对比学习模式
    model.use_contrastive = False
    model.eval()
    model.to(device)
    X = X.to(device)
    H_norm = H_norm.to(device)
    y = y.to(device)

    with torch.no_grad():
        logits, node_representations, _ = model(X, H_norm)
        test_logits = logits[test_idx]
        test_labels = y[test_idx]
        
        # 获取最后一层节点表示
        last_layer_embeddings = node_representations[-1][test_idx] if node_representations else None

        # 计算预测概率
        probs = F.softmax(test_logits, dim=1)
        preds = torch.argmax(test_logits, dim=1)

        # 转换为numpy数组
        test_labels_np = test_labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # 计算各种指标
        acc = accuracy_score(test_labels_np, preds_np)
        f1 = f1_score(test_labels_np, preds_np, average='macro')

        # 计算AUC
        if probs.shape[1] == 2:
            auc = roc_auc_score(test_labels_np, probs_np[:, 1])
        else:
            auc = roc_auc_score(test_labels_np, probs_np, multi_class='ovr')

        # 计算G-mean
        gmean = calculate_gmean(test_labels_np, preds_np)

        # 计算混淆矩阵
        cm = confusion_matrix(test_labels_np, preds_np)
        recall = recall_score(test_labels_np, preds_np)  # 欺诈类的召回率

        print(f'\n{"=" * 50}')
        print(f'测试结果:')
        print(f'{"-" * 50}')
        print(f'准确率 (Accuracy): {acc:.4f}')
        print(f'F1-macro 分数: {f1:.4f}')
        print(f'AUC 分数: {auc:.4f}')
        print(f'G-mean 分数: {gmean:.4f}')
        print(f'欺诈类召回率 (Recall): {recall:.4f}')
        print(f'{"-" * 50}')
        print(f'混淆矩阵:')
        print(cm)
        print(f'{"=" * 50}\n')

        # 绘制混淆矩阵
        # plot_confusion_matrix(test_labels_np, preds_np, classes=['正常', '欺诈'],
        #                       title='欺诈检测混淆矩阵')

        # 可视化最后一层嵌入
        if visualize and last_layer_embeddings is not None:
            visualize_embeddings(last_layer_embeddings.cpu().numpy(), 
                                 test_labels_np, 
                                 "HCLNet最后一层嵌入")

        return acc, f1, auc, gmean, cm


def visualize_embeddings(embeddings, labels, title="HCLNet嵌入可视化"):
    """可视化模型嵌入使用t-SNE"""
    print("\n生成t-SNE可视化...")
    
    # 确保嵌入是二维数组
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
    
    # 如果节点太多，进行下采样
    sample_size = 10000
    if len(embeddings) > sample_size:
        print(f"从 {len(embeddings)} 个节点中采样 {sample_size} 个点...")
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        embeddings_sample = embeddings
        labels_sample = labels

    # 运行t-SNE
    print(f"在 {len(embeddings_sample)} 个点上运行t-SNE...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_sample)
    except Exception as e:
        print(f"t-SNE失败: {e}")
        return

    # 可视化
    plt.figure(figsize=(10, 8))

    # 创建颜色映射
    colors = ['blue', 'red']  # 0=正常(蓝色), 1=欺诈(红色)
    color_map = [colors[label] for label in labels_sample]

    # 绘制所有点
    plt.scatter(embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=color_map, alpha=0.6, s=20)
    
    # 添加图例
    blue_patch = mpatches.Patch(color='blue', label='正常 (0)')
    red_patch = mpatches.Patch(color='red', label='欺诈 (1)')
    plt.legend(handles=[blue_patch, red_patch])

    plt.title(f't-SNE可视化: {title}')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')

    # 生成文件名
    dataset_name = "YelpChi" if "YelpChi" in title else "Amazon"
    filename = 'figure/tsne_HCLNet_YelpChi.png'
    
    # 保存图像
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"t-SNE可视化已保存至: {filename}")

    # 显示图像
    plt.show()


# 添加Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# 主函数
def main():
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    file_path = 'data/YelpChi.mat'
    if 'YelpChi' in file_path:
        net_rur, net_rsr, net_rtr, features, labels = load_yelpchi_data(file_path)
    elif 'Amazon' in file_path:
        net_rur, net_rsr, net_rtr, features, labels = load_amazon_data(file_path)
    else:
        print('未知数据集')

    # 获取节点总数
    n_nodes = net_rur.shape[0]
    print(f"数据集加载完成，包含 {n_nodes} 个评论节点")
    print(f"特征维度: {features.shape[1]}")
    print(f"标签分布: 正常 {np.sum(labels == 0)}, 欺诈 {np.sum(labels == 1)}")

    # 从三种关系中提取超边
    print("\n正在从R-U-R关系提取超边...")
    hyperedges_rur = extract_hyperedges_from_adj(net_rur)
    print(f"  → 提取到 {len(hyperedges_rur)} 条用户超边")

    print("正在从R-S-R关系提取超边...")
    hyperedges_rsr = extract_hyperedges_from_adj(net_rsr)
    print(f"  → 提取到 {len(hyperedges_rsr)} 条商家超边")

    print("正在从R-T-R关系提取超边...")
    hyperedges_rtr = extract_hyperedges_from_adj(net_rtr)
    print(f"  → 提取到 {len(hyperedges_rtr)} 条时间超边")

    # 合并所有超边
    all_hyperedges = hyperedges_rur + hyperedges_rsr + hyperedges_rtr
    print(f"\n超边总数: {len(all_hyperedges)} (用户: {len(hyperedges_rur)}, "
          f"商家: {len(hyperedges_rsr)}, 时间: {len(hyperedges_rtr)})")

    # 构建超图关联矩阵
    H = build_hypergraph_incidence_matrix(all_hyperedges, n_nodes)
    print(f"超图构建完成，关联矩阵形状: {H.shape}")

    # 计算规范化矩阵
    d_v = np.array(H.sum(axis=1)).flatten()  # 节点度向量
    d_e = np.array(H.sum(axis=0)).flatten()  # 超边度向量

    # 处理零度节点
    d_v = np.maximum(d_v, 1e-10)
    d_e = np.maximum(d_e, 1e-10)

    d_v_inv_sqrt = sp.diags(1.0 / np.sqrt(d_v))
    d_e_inv = sp.diags(1.0 / d_e)

    H_norm = d_v_inv_sqrt @ H @ d_e_inv

    # 确保H_norm是稀疏矩阵
    if not isinstance(H_norm, sp.spmatrix):
        H_norm = sp.coo_matrix(H_norm)

    # 转换为PyTorch张量
    H_norm_torch = sparse_matrix_to_torch(H_norm)

    # 转换特征矩阵为密集张量
    if sp.issparse(features):
        features = features.toarray()
    X_torch = torch.tensor(features, dtype=torch.float32)
    y_torch = torch.tensor(labels, dtype=torch.long)

    # 划分训练集、验证集和测试集
    idx = np.arange(n_nodes)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.5, stratify=labels[train_idx], random_state=42)

    print(f"\n数据集划分:")
    print(f"  训练集大小: {len(train_idx)}")
    print(f"  验证集大小: {len(val_idx)}")
    print(f"  测试集大小: {len(test_idx)}")

    # 转换索引为张量
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    # 初始化模型
    input_dim = features.shape[1]
    hidden_dim = 64
    output_dim = 2  # 二分类
    num_layers = 3
    projection_dim = 64  # 对比学习投影维度

    model = EnhancedFraudDetectionModelWithCL(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        n_heads=2,
        dropout=0.2,
        use_multihead=True,
        projection_dim=projection_dim,
        temperature=0.5
    )
    print("\n模型架构:")
    print(model)
    print(f"总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    # 处理类别不平衡问题
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 将所有张量和模型移动到正确的设备上
    model = model.to(device)
    X_torch = X_torch.to(device)
    H_norm_torch = H_norm_torch.to(device)
    y_torch = y_torch.to(device)

    # 移动类别权重到设备
    class_weights = class_weights.to(device)
    # 替代现有的交叉熵损失
    # criterion = FocalLoss(alpha=class_weights[1].item(), gamma=2.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 将索引张量移动到正确的设备上
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    # 训练模型
    print("\n开始训练...")
    start_time = time.time()
    model, history = train_model(
        model, optimizer, criterion,
        X_torch, H_norm_torch, y_torch,
        train_idx, val_idx,
        epochs=500, patience=50, device=device,
        alpha=0.8  # 对比学习损失权重
    )
    training_time = time.time() - start_time
    print(f"\n训练完成，耗时: {training_time:.2f}秒")

    # 评估模型
    print("\n在测试集上评估模型...")
    acc, f1, auc, gmean, cm = evaluate_model(
        model, X_torch, H_norm_torch, y_torch,
        test_idx, device=device, visualize=True
    )

    # 保存模型
    torch.save(model.state_dict(), 'best_model/fraud_detection_model.pt')
    print("模型已保存为 'best_model/fraud_detection_model.pt'")

    return model, acc, f1, auc, gmean, cm


if __name__ == "__main__":
    model, acc, f1, auc, gmean, cm = main()
