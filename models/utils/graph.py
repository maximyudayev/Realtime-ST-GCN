import numpy as np

class Graph():
    """The Graph to model the node relationships in a skeleton.

    Each row corresponds to a unique node neighborhood: important for normalization and partitioning 
    because each separate matrix is no longer symmetric after these operations.
    
    NOTE: $\alpha$ IS needed because "far" adjacency matrix partition in a skeleton
    will contain 0's for node's at the tip of a limb.

    Attributes:
        A : ``NDArray[float64]`` 
            weighted normalized adjacency matrix.
        
        num_node : ``int``
            number of skeleton joints.

    Methods:
        get_hop_distance()
            Computes the distance between nodes.

        get_adjacency(strategy: str)
            Computes the adjacency matrix based on the edge list and partitioning strategy.
        
        normalize_digraph(A: NDArray[float64])
            Assymetric normalization with the degree matrix.
        
        normalize_undigraph(A: NDArray[float64])
            Symmetric normalization with the degree matrix.
    """

    def __init__(
        self,
        num_node,
        edge,
        center,
        strategy = 'spatial',
        max_hop = 1,
        dilation = 1,
        alpha = 0.001):
        """
        Args:
            num_node : ``int``
                number of joints in the skeleton.
            
            edge : ``list[list[int]]``
                edge list of joint index tuples.
            
            center : ``int``
                index of the center-of-skeleton node.
            
            max_hop : ``int``
                the maximal distance between two connected nodes.
            
            dilation : ``int``
                controls the spacing between the kernel points.

            alpha : ``int``
                stability value to avoid empty rows in partitioned adjacency matrix.
            
            strategy : ``string``
                must be one of the following:
                    ``'uniform'``: Uniform Labeling.
                    ``'distance'``: Distance Partitioning.
                    ``'spatial'``: Spatial Partitioning.
            
            For more information, please refer to the section 'Partition Strategies' in 
                [Yan et al. (2018)](https://arxiv.org/abs/1801.07455).
        """

        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = num_node
        self.edge = edge
        self.center = center
        self.alpha = alpha

        self.hop_dis = self.get_hop_distance()
        self.A = self.get_adjacency(strategy)


    def __str__(self):
        return self.A


    def get_adjacency(self, strategy):
        """Computes the adjacency matrix based on the edge list and partitioning strategy.

        Uses symmetric normalization method.

        TODO:
            ``1.`` Enable selection between normalization methods.

        Args:
            strategy : ``str``
                must be one of the following:
                    ``'uniform'``: Uniform Labeling.
                    ``'distance'``: Distance Partitioning.
                    ``'spatial'``: Spatial Partitioning.
        
        Returns:
            Normalized adjacency matrix of size ``(P, V, V)``, where:
                ``P`` : Number of partitions, according to the partitioning strategy.
                ``V`` : Number of nodes in the skeleton.

        Raises:
            ValueError: If strategy does not match provided options.
        """

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        # adjacency = self.normalize_undigraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = adjacency[self.hop_dis == hop]
        elif strategy == 'spatial':
            # NOTE: original implementation of Yan, et al. (2018) (https://arxiv.org/abs/1801.07455) 
            # did not calculate distance in accordance with their proposed strategy: used hop distance to center node, not average spatial distance
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_far = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[i, j] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[i, j] = adjacency[i, j]
                            elif (self.hop_dis[j, self.center] < self.hop_dis[i, self.center] or 
                                (i == self.center and self.hop_dis[j, self.center] > self.hop_dis[i, self.center])):
                                # COG node and limb ends have only "close" nodes
                                a_close[i, j] = adjacency[i, j]
                            else:
                                a_far[i, j] = adjacency[i, j]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close)
                    A.append(a_far)
            A = np.stack(A)
            # summing together partitions adds up to the original adjacency matrix
            # (np.sum(A,0)==adjacency).all()
        else:
            raise ValueError("Strategy Does Not Exist.")
        
        # normalization of the adjacency matrix must be done after partitioning
        for i in range(A.shape[0]):
            A[i] = self.normalize_undigraph(A[i])
        return A


    def get_hop_distance(self):
        """Computes the distance between nodes.        

        TODO: compute all-pairs shortest paths using the edge list instead of the adjacency matrix

        Returns:
            Distance square matrix of size of the number of joints.
        """

        cost = np.zeros((self.num_node, self.num_node)) + np.inf
        for i, j in self.edge:
            if i == j:
                cost[i, i] = 0
            else:
                cost[j, i] = 1
                cost[i, j] = 1

        # use BFS to compute the distance matrix from the adjacency matrix
        for k in range(self.num_node):
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if cost[i,k] + cost[k,j] < cost [i,j]:
                        cost[i,j] = cost[i,k] + cost[k,j]
        return cost


    def normalize_digraph(self, A):
        """Asymmetric normalization with the degree matrix.

        Args:
            A : ``NDArray[float64]`` 
                Raw adjacency matrix.

        Returns:
            Normalized adjacency matrix of the same size.
        """

        Dl = np.power(np.sum(A, 1),-1)
        Dl[np.isinf(Dl)] = 0
        Dn = np.eye(A.shape[0]) * Dl
        AD = np.dot(A, Dn)
        return AD


    def normalize_undigraph(self, A):
        """Symmetric normalization with the degree matrix.

        Args:
            A : ``NDArray[float64]`` 
                Raw adjacency matrix.

        Returns:
            Normalized adjacency matrix of the same size.
        """

        Dl = np.power(np.sum(A, 1),-0.5)
        Dl[np.isinf(Dl)] = 0
        Dn = np.eye(A.shape[0]) * Dl
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
