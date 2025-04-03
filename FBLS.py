from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
class FBLS:
    def __init__(self, fuzz_sys, fuzz_rule, enhance_node, reg_param=1e-12):
        self.fuzz_sys = fuzz_sys
        self.fuzz_rule = fuzz_rule
        self.enhance_node = enhance_node
        self.reg_param = reg_param  # 正则化参数
        self.alpha = []
        self.centrals = []
        self.scaler = StandardScaler()
    def get_central(self, X):
        """并行化KMeans聚类"""
        kmeans = KMeans(
            n_clusters=self.fuzz_rule,
            init='k-means++',  # 更优的初始化方法
            n_init='auto'
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_

    def get_rule_activation(self, X, centrals):
        """向量化计算归一化模糊激活度"""
        # 计算所有样本到所有中心的距离 (N, fuzz_rule)

        distances = X[:, np.newaxis] - centrals
        # 高斯激活度 (N, fuzz_rule)
        activation = np.exp(-0.5 * (distances ** 2))
        activation = np.prod(activation,axis=2)
        # 归一化，添加极小值防止除零
        normal_activation = activation / (activation.sum(axis=1, keepdims=True) + 1e-8)
        return normal_activation

    def fuzzy_rule(self, X, alpha_i):
        """向量化模糊规则计算"""
        return X @ alpha_i  # (N, fuzz_rule)

    def parameter_init(self, X):
        """参数初始化，使用Xavier初始化增强层权重"""
        input_dim = self.fuzz_sys * self.fuzz_rule + 1  # 包含偏置项
        # Xavier初始化增强层权重
        # limit = np.sqrt(6 / (input_dim + self.enhance_node))
        self.Wh = np.random.rand(input_dim, self.enhance_node)
        #self.Wh = np.random.normal(input_dim,self.enhance_node)
        #self.Wh = np.random.normal(loc=0.0, scale=1, size=(input_dim, self.enhance_node))
        # 初始化每个模糊子系统的alpha和聚类中心
        for _ in range(self.fuzz_sys):
            self.alpha.append(
                np.random.rand(X.shape[1], self.fuzz_rule) * 0.001)  # 小随机数初始化
            self.centrals.append(self.get_central(X))

    def _compute_Zn(self, X):
        """计算模糊规则输出Zn"""
        N = X.shape[0]
        Zn = np.zeros((N, self.fuzz_sys * self.fuzz_rule))
        for i in range(self.fuzz_sys):
            # 向量化计算激活度和规则输出
            act = self.get_rule_activation(X, self.centrals[i])  # (N, fuzz_rule)
            z_i = self.fuzzy_rule(X, self.alpha[i])              # (N, fuzz_rule)
            Zn[:, i*self.fuzz_rule : (i+1)*self.fuzz_rule] = act * z_i
        return Zn

    def _compute_H(self, Zn):
        """计算增强层特征"""
        # 添加偏置项并做非线性变换
        H_input = np.hstack([Zn, np.ones((Zn.shape[0], 1))])  # (N, fuzz_sys*fuzz_rule + 1)
        return np.tanh(H_input @ self.Wh)  # (N, enhance_node)

    def get_feature(self, X):
        """特征生成"""
        Zn = self._compute_Zn(X)
        H = self._compute_H(Zn)
        self.features = np.hstack([Zn, H])

    def train(self, X, Y):
        """训练模型，添加正则化项提升稳定性"""
        if not isinstance(X, np.ndarray) or X.dtype != np.float16:
            X = np.array(X, dtype=np.float16)
        X = self.scaler.fit_transform(X)
        # 判断 Y 是否已经是 np.float16 类型
        if not isinstance(Y, np.ndarray) or Y.dtype != np.float16:
            Y = np.array(Y, dtype=np.float16)
        self.parameter_init(X)
        self.get_feature(X)
        # 岭回归替代普通伪逆
        I = np.eye(self.features.shape[1])
        #self.W = np.linalg.inv(self.features.T @ self.features + self.reg_param * I) @ self.features.T @ Y
        self.W = np.linalg.pinv(self.features) @ Y

    def predict(self, X):
        """预测"""
        if not isinstance(X, np.ndarray) or X.dtype != np.float16:
            X = np.array(X, dtype=np.float16)
        X = self.scaler.transform(X)
        self.get_feature(X)
        return self.features @ self.W
    