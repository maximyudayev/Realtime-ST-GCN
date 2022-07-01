import numpy as np
from numpy.typing import NDArray

class Graph():
    """The Graph to model the node relationships in a skeleton.
    
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
        num_node: int,
        edge: list[list[int]],
        center: int,
        strategy: str ='spatial',
        max_hop: int = 1,
        dilation: int = 1) -> None:
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

        self.hop_dis = self.get_hop_distance()
        self.A = self.get_adjacency(strategy)


    def __str__(self) -> NDArray[np.float64]:
        return self.A


    def get_adjacency(self, strategy: str) -> NDArray[np.float64]:
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
        normalize_adjacency = self.normalize_undigraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            return A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            return A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_far = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_far[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_far)
            A = np.stack(A)
            return A
        else:
            raise ValueError("Strategy Does Not Exist.")


    def get_hop_distance(self) -> NDArray[np.float64]:
        """Computes the distance between nodes.        

        Returns:
            Distance square matrix of size of the number of joints.
        """

        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def normalize_digraph(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Asymmetric normalization with the degree matrix.

        Args:
            A : ``NDArray[float64]`` 
                Raw adjacency matrix.

        Returns:
            Normalized adjacency matrix of the same size.
        """

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


    def normalize_undigraph(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Symmetric normalization with the degree matrix.

        Args:
            A : ``NDArray[float64]`` 
                Raw adjacency matrix.

        Returns:
            Normalized adjacency matrix of the same size.
        """

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
