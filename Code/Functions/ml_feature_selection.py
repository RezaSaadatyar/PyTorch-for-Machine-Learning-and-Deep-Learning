# ================================== Presented by: Reza Saadatyar (2023-2024) ==================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import numpy as np
from typing import Union
from scipy.stats import f_oneway
from Functions.plot_projection import plot_projection
from sklearn import feature_selection, preprocessing, linear_model, ensemble, svm
from skfeature.function.similarity_based import fisher_score

class FeatureSelection:
    def __init__(self, dataset: np.ndarray, label: np.ndarray, display_figure: str = "off", figsize: Union[int,
                 float] = (4, 3)) -> None:
        """
        Initialize the FeatureSelection class.

        Args:
            dataset (Union[np.ndarray, List]): Input dataset (features).
            label (np.ndarray): Target labels.
            display_figure (str, optional): Whether to display figures ("on" or "off"). Defaults to "off".

        The type of feature selection method to use:
            - "VAR": Variance thresholding.
            - "ANOVA": Analysis of Variance (ANOVA).
            - "MI": Mutual information.
            - "UFS": Univariate feature selection.
            - "FS": Fisher score.
            - "RFE": Recursive feature elimination.
            - "FFS": Forward feature selection.
            - "BFS": Backward feature selection.
            - "RF": Random forest feature selection.
            - "L1FS": L1-based feature selection.
            - "TFS": Tree-based feature selection.
        """
        # Initialize class attributes
        self.data = dataset
        self.label = label
        self.figsize = figsize
        self.display_figure = display_figure
        self.title = None
        self.output = None

        # Convert data to ndarray if it's not already
        self.data = np.array(self.data) if not isinstance(self.data, np.ndarray) else self.data
    
        # Transpose the data if it has more than one dimension and has fewer rows than columns
        self.data = self.data.T if self.data.ndim > 1 and self.data.shape[0] < self.data.shape[-1] else self.data
        
    def _plot_features(self) -> None:
        """Helper method to plot features if display_figure is 'on'."""
        if self.display_figure.lower() == "on":
            plot_projection(data=self.output, labels=self.label, title=self.title, fig_size=self.figsize)

class FilterMethods(FeatureSelection): # Define the FilterMethods class, which inherits from FeatureSelection
    def __init__(self, dataset: np.ndarray, label: np.ndarray, display_figure: str = "off", figsize: Union[int,
                 float] = (4, 3)) -> None:
        """
        Args:
            dataset (np.ndarray): Input dataset (features).
            label (np.ndarray): Target labels.
            display_figure (str, optional): Whether to display figures ("on" or "off"). Defaults to "off".
            figsize (Union[int, float], optional): Size of the figure for plotting. Defaults to (4, 3).
            
        Filter Methods for feature selection:
            i. Variance thresholding: Filter features based on variance.
            ii. ANOVA: Compute p-values for each feature using ANOVA and select top features.
            iii. Mutual Information: Select features based on mutual information with class labels.
            iv. Univariate Feature Selection: Select top k features using chi-squared test.
            v. Fisher Score: Select top features based on Fisher score.
        """
        super().__init__(dataset, label, display_figure, figsize) # Call the parent class constructor

    def Variance(self, threshold: Union[int, float]) -> np.ndarray:   # Variance
        """
        Apply variance thresholding to filter features.

        Args:
            threshold (Union[int, float]): Variance threshold.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Apply variance thresholding to filter features
        mod = feature_selection.VarianceThreshold(threshold=threshold)
        mod.fit(self.data)
        self.output = mod.transform(self.data)
        self.title = "Variance"  # Set the title for the plot
        self._plot_features()    # Plot the results if display_figure is "on"
        return self.output

    def ANOVA(self, num_features: int) -> np.ndarray:    # ANOVA
        """
        Select top features using ANOVA F-test.

        Args:
            num_features (int): Number of features to select.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        pvalue = np.zeros(self.data.shape[1])
        for i in range(self.data.shape[1]):   # Compute p-values for each feature using ANOVA
            pvalue[i] = f_oneway(self.data[:, i], self.label)[1]
        ind = np.argsort(pvalue)                         # Sort p-values and get indices of sorted features 
        self.output = self.data[:, ind[:num_features]]   # Select top num_features features
        self.title = "ANOVA"   # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output

    def MutualInformation(self, num_features: int, num_neighbors: int) -> np.ndarray:  # Mutual information
        """
        Select top features using mutual information.

        Args:
            num_features (int): Number of features to select.
            num_neighbors (int, optional): Number of neighbors for mutual information. Defaults to 3.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        # Compute mutual information scores
        mod = feature_selection.mutual_info_classif(self.data, self.label, n_neighbors=num_neighbors)
        self.output = self.data[:, np.argsort(mod)[-num_features:]] # Select top num_features features
        self.title = "Mutual information"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output
    
    def UnivariateFeatureSelection(self, num_features: int) -> np.ndarray:   # Univariate feature selection
        """
        Select top features using univariate feature selection (chi-squared test).

        Args:
            num_features (int): Number of features to select.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        scaler = preprocessing.StandardScaler()      # Perform Min-Max scaling
        data = scaler.fit_transform(self.data)
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        # Select top k features using chi-squared test
        mod = feature_selection.SelectKBest(feature_selection.chi2, k=num_features)
        self.output = mod.fit_transform(self.data, self.label)
        self.title = "Univariate feature selection"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output
    
    def FisherScore(self, num_features: int) -> np.ndarray:   # Fisher_score
        """
        Select top features using Fisher score.

        Args:
            num_features (int): Number of features to select.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        mod = fisher_score.fisher_score(self.data, self.label)  # Compute Fisher scores
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        self.output = self.data[:, mod[-num_features:]] # Select top num_features features
        self.title = "Fisher score"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output

class WrapperMethods(FeatureSelection): # Define the WrapperMethods class, which inherits from FeatureSelection
    def __init__(self, dataset: np.ndarray, label: np.ndarray, display_figure: str = "off", figsize: Union[int,
                 float] = (4, 3)) -> None:
        """
        Args:
            dataset (np.ndarray): Input dataset (features).
            label (np.ndarray): Target labels.
            display_figure (str, optional): Whether to display figures ("on" or "off"). Defaults to "off".
            figsize (Union[int, float], optional): Size of the figure for plotting. Defaults to (4, 3).

        Wrapper Methods:
            i. Recursive Feature Elimination (RFE): Select features recursively using logistic regression.
            ii. Forward Feature Selection (FFS): Select features forwardly based on logistic regression.
            iii. Backward Feature Selection (BFS): Select features backwardly based on logistic regression.
        """
        super().__init__(dataset, label, display_figure, figsize) # Call the parent class constructor

    def RecursiveFeatureElimination(self, num_features: int, max_iter: int = 1000 ) -> np.ndarray: # Recursive Feature Elimination
        """
        Perform Recursive Feature Elimination (RFE) for feature selection.

        Args:
            num_features (int): Number of features to select.
            max_iter (int, optional): Maximum number of iterations for logistic regression. Defaults to 1000.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        # Apply Recursive Feature Elimination (RFE) with logistic regression
        mod = feature_selection.RFE(estimator=linear_model.LogisticRegression(max_iter=max_iter), 
                                n_features_to_select=num_features)
        mod.fit(self.data, self.label)
        self.output = mod.transform(self.data)
        self.title = "Recursive feature elimination"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output
    
    def SequentialFeatureSelection(self, num_features: int, max_iter: int = 1000, direction: str = "forward") -> np.ndarray:
        """
        Perform Forward or Backward Feature Selection for feature selection.

        Args:
            num_features (int): Number of features to select.
            max_iter (int, optional): Maximum number of iterations for logistic regression. Defaults to 1000.
            direction (str, optional): Direction of feature selection ("forward" or "backward"). Defaults to "forward".

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        # Apply Sequential Feature Selection with logistic regression
        mod = linear_model.LogisticRegression(max_iter=max_iter)
        mod = feature_selection.SequentialFeatureSelector(mod, n_features_to_select=num_features, direction=
                                                            direction, cv=5, scoring='accuracy')
        mod.fit(self.data, self.label)
        self.output = self.data[:, mod.support_]    # Select optimal features
        self.title = f"{direction.capitalize()} feature selection"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output
    
class EmbeddedMethods(FeatureSelection): # Define the EmbeddedMethods class, which inherits from FeatureSelection
    def __init__(self, dataset: np.ndarray, label: np.ndarray, display_figure: str = "off", figsize: Union[int,
                 float] = (4, 3)) -> None:
        """
        Args:
            dataset (np.ndarray): Input dataset (features).
            label (np.ndarray): Target labels.
            display_figure (str, optional): Whether to display figures ("on" or "off"). Defaults to "off".
            figsize (Union[int, float], optional): Size of the figure for plotting. Defaults to (4, 3).

        Embedded Methods:
            i. Random Forest: Select top features based on random forest feature importances.
            ii. L1-based Feature Selection: Select features based on linear SVM with L1 regularization.
            iii. Tree-based Feature Selection: Select top features based on extra trees classifier.
        """
        super().__init__(dataset, label, display_figure, figsize) # Call the parent class constructor

    def RandomForest(self, num_features: int, n_estimators: int) -> np.ndarray:   # Random forest
        """
        Perform feature selection using Random Forest feature importances.

        Args:
            num_features (int): Number of features to select.
            n_estimators (int): Number of trees in the random forest.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        # Apply Random Forest feature selection
        mod = ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        mod.fit(self.data, self.label)
        # Select top num_features features based on feature importances
        self.output = self.data[:, np.argsort(mod.feature_importances_)[-num_features:]]
        self.title = "Random forest"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output

    def L1(self, L1_Parameter: float = 0.1, max_iter: int = 1000) -> np.ndarray:
        """
        Perform L1-based feature selection using linear SVM with L1 regularization.

        Args:
            L1_Parameter (float, optional): Regularization parameter (C). Smaller values result in fewer features selected. Defaults to 0.1.
            max_iter (int, optional): Maximum number of iterations for the linear SVM. Defaults to 1000.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # L1-based feature selection; The smaller C the fewer feature selected
        # Apply L1-based feature selection with linear SVM
        mod = svm.LinearSVC(C=L1_Parameter, penalty='l1', dual=False, max_iter=max_iter).fit(self.data, self.label)
        mod = feature_selection.SelectFromModel(mod, prefit=True)
        self.output = mod.transform(self.data)
        self.title = "L 1"     # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output

    def TreeFeatureSelection(self, num_features: int, n_estimators: int):  # Tree-based feature selection
        """
        Perform feature selection using tree-based feature importances (Extra Trees).

        Args:
            num_features (int): Number of features to select.
            n_estimators (int): Number of trees in the Extra Trees classifier.

        Returns:
            np.ndarray: Transformed dataset with selected features.
        """
        # Ensure num_features does not exceed the number of available features
        num_features = self.data.shape[1] if num_features > self.data.shape[1] else num_features
        # Apply Extra Trees feature selection
        mod = ensemble.ExtraTreesClassifier(n_estimators=n_estimators)
        mod.fit(self.data, self.label)
        # Select top num_features features based on feature importances
        self.output = self.data[:, np.argsort(mod.feature_importances_)[-num_features:]]
        self.title = "Tree-based feature selection"  # Set the title for the plot
        self._plot_features()  # Plot the results if display_figure is "on"
        return self.output