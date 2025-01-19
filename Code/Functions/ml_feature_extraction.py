# ================================ Presented by: Reza Saadatyar (2023-2024) ====================================
# ================================== E-mail: Reza.Saadatyar@outlook.com ========================================

import numpy as np
from Functions.plot_projection import plot_projection
from sklearn import decomposition, discriminant_analysis, manifold

class FeatureExtraction:
    def __init__(self, dataset: np.ndarray, labels: np.ndarray, display_figure: str = "off") -> None:
        """
        Initialize the FeatureExtraction class.

        Parameters:
        - data: Input feature matrix of shape (n_samples, n_features).
        - labels: Target labels of shape (n_samples,).
        - display_figure: Whether to display the plot ("on" or "off"). Defaults to "off".
        
        Import module:
        - from Functions.ml_feature_extraction import FeatureExtraction
        
        Example:
        - obj_feature = FeatureExtraction(x, y, display_figure="on")
          1. out = obj_feature.PCA(num_features=3, kernel="linear")     # Principal Component Analysis (PCA)
          2. out = obj_feature.LDA(solver="svd")                        # Linear Discriminant Analysis (LDA)
          3. out = obj_feature.ICA(num_features=4, max_iter=200)        # Independent Component Analysis (ICA)
          4. out = obj_feature.TSNE(num_features=3, perplexity=10, learning_rate="auto", max_iter=250) # t-SNE
          5. out = obj_feature.FA(num_features=3, max_iter=200)         # Factor Analysis (FA)
          6. out = obj_feature.Isomap(num_features=3, num_neighbors=20) # Isometric Mapping (Isomap)
        """
        # Initialize class attributes
        self.data = dataset
        self.labels = labels
        self.display_figure = display_figure
        self.title = None
        self.output = None
        
        # Convert data to ndarray if it's not already
        self.data = np.array(self.data) if not isinstance(self.data, np.ndarray) else self.data
    
        # Transpose the data if it has more than one dimension and has fewer rows than columns
        self.data = self.data.T if self.data.ndim > 1 and self.data.shape[0] < self.data.shape[-1] else self.data

    def _plot_results(self, **plot_kwargs) -> None:
        """
        Helper method to plot the results if display_figure is "on".

        Parameters:
        - plot_kwargs: Additional keyword arguments for the plot function.
        """
        if self.display_figure == "on":
            plot_projection(self.output, self.labels, fig_size=(4, 3), title=self.title, **plot_kwargs)

    def PCA(self, num_features: int, kernel: str = "linear", **plot_kwargs) -> np.ndarray:
        """
        Perform Principal Component Analysis (PCA).

        Parameters:
        - num_features: Number of components to extract.
        - kernel: Kernel type for KernelPCA ("linear", "poly", "rbf", "sigmoid", "cosine", "precomputed").
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, num_features).
        """
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'linear', 'poly', 'rbf', 'sigmoid', 'cosine', or 'precomputed'.")

        self.title = "PCA"
        mod = decomposition.KernelPCA(n_components=num_features, kernel=kernel)
        self.output = mod.fit_transform(self.data)

        self._plot_results(**plot_kwargs)
        return self.output

    def LDA(self, solver: str = "svd", **plot_kwargs) -> np.ndarray:
        """
        Perform Linear Discriminant Analysis (LDA).

        Parameters:
        - solver: Solver type ("svd", "lsqr", "eigen").
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, n_classes - 1).
        """
        if solver not in ['svd', 'lsqr', 'eigen']:
            raise ValueError(f"Unknown solver: {solver}. Use 'svd', 'lsqr', or 'eigen'.")

        self.title = "LDA"
        mod = discriminant_analysis.LinearDiscriminantAnalysis(
            n_components=len(np.unique(self.labels)) - 1, solver=solver
        )
        self.output = mod.fit_transform(self.data, self.labels)

        self._plot_results(**plot_kwargs)
        return self.output

    def ICA(self, num_features: int, max_iter: int = 100, **plot_kwargs) -> np.ndarray:
        """
        Perform Independent Component Analysis (ICA).

        Parameters:
        - num_features: Number of components to extract.
        - max_iter: Maximum number of iterations.
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, num_features).
        """
        self.title = "ICA"
        mod = decomposition.FastICA(n_components=num_features, max_iter=max_iter)
        self.output = mod.fit_transform(self.data)

        self._plot_results(**plot_kwargs)
        return self.output

    def SVD(self, num_features: int, **plot_kwargs) -> np.ndarray:
        """
        Perform Singular Value Decomposition (SVD).

        Parameters:
        - num_features: Number of components to extract.
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, num_features).
        """
        self.title = "SVD"
        mod = decomposition.TruncatedSVD(n_components=num_features)
        self.output = mod.fit_transform(self.data)

        self._plot_results(**plot_kwargs)
        return self.output

    def TSNE(self, num_features: int, perplexity: float = 30.0, learning_rate: float = 200.0,
             max_iter: int = 1000, **plot_kwargs) -> np.ndarray:
        """
        Perform t-Distributed Stochastic Neighbor Embedding (t-SNE).

        Parameters:
        - num_features: Number of components to extract.
        - perplexity: Perplexity parameter.
        - learning_rate: Learning rate.
        - max_iter: Maximum number of iterations.
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, num_features).
        """
        self.title = "TSNE"
        mod = manifold.TSNE(
            n_components=num_features, perplexity=perplexity, learning_rate=learning_rate,
            max_iter=max_iter, init='pca', random_state=24
        )
        self.output = mod.fit_transform(self.data)

        self._plot_results(**plot_kwargs)
        return self.output

    def FA(self, num_features: int, max_iter: int = 100, **plot_kwargs) -> np.ndarray:
        """
        Perform Factor Analysis (FA).

        Parameters:
        - num_features: Number of components to extract.
        - max_iter: Maximum number of iterations.
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, num_features).
        """
        self.title = "FA"
        mod = decomposition.FactorAnalysis(n_components=num_features, max_iter=max_iter)
        self.output = mod.fit_transform(self.data)

        self._plot_results(**plot_kwargs)
        return self.output

    def Isomap(self, num_features: int, num_neighbors: int = 5, **plot_kwargs) -> np.ndarray:
        """
        Perform Isometric Feature Mapping (Isomap).

        Parameters:
        - num_features: Number of components to extract.
        - num_neighbors: Number of neighbors.
        - plot_kwargs: Additional keyword arguments for the plot function.

        Returns:
        - Transformed data of shape (n_samples, num_features).
        """
        self.title = "Isomap"
        mod = manifold.Isomap(n_neighbors=num_neighbors, n_components=num_features)
        self.output = mod.fit_transform(self.data)

        self._plot_results(**plot_kwargs)
        return self.output