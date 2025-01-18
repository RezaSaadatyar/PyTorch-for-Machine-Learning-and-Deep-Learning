import numpy as np
from sklearn.utils import resample

class ImbalanceData:
    """
    A class to handle imbalanced datasets by resampling minority classes to match the size of the majority class.
    """

    def __init__(self, dataset: np.ndarray, labels: np.ndarray) -> None:
        """
        Initialize the ImbalancedDataHandler with the dataset and corresponding labels.

        Args:
            dataset (np.ndarray): The feature matrix (n_samples, n_features).
            labels (np.ndarray): The target labels (n_samples,).
        """
        self.dataset = np.array(dataset) if not isinstance(dataset, np.ndarray) else dataset
        self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

        # Ensure the dataset and labels have compatible shapes
        if self.dataset.shape[0] != self.labels.shape[0]:
            raise ValueError("Number of samples in dataset and labels must match.")

    @property
    def balance_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset by resampling minority classes to match the size of the majority class.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the balanced dataset and corresponding labels.
        """
        # Get counts of each class
        unique_classes, class_counts = np.unique(self.labels, return_counts=True)
        max_samples = max(class_counts)  # Number of samples in the largest class

        # Initialize arrays for balanced dataset
        x_resampled = []
        y_resampled = []

        # Resample each class to match the size of the largest class
        for class_label in unique_classes:
            # Get samples for the current class
            x_class = self.dataset[self.labels == class_label]
            y_class = self.labels[self.labels == class_label]

            if len(y_class) < max_samples:
                # Resample minority class
                x_upsampled, y_upsampled = resample(
                    x_class,
                    y_class,
                    replace=True,  # Sample with replacement
                    n_samples=max_samples,  # Match the size of the majority class
                    random_state=42  # For reproducibility
                )
            else:
                # Keep majority class as is
                x_upsampled = x_class
                y_upsampled = y_class

            x_resampled.append(x_upsampled)
            y_resampled.append(y_upsampled)

        # Combine all classes
        data_balanced = np.vstack(x_resampled)
        labels_balanced = np.hstack(y_resampled)

        # Print the balanced class distribution
        print("\nBalanced class distribution:")
        for class_label in unique_classes:
            if np.sum(self.labels == class_label) != np.sum(labels_balanced == class_label):
                print(f"Class {class_label}: {np.sum(self.labels == class_label)} ---> {np.sum(labels_balanced == class_label)} samples")
            else:
                print(f"Class {class_label}: {np.sum(labels_balanced == class_label)} samples")

        return data_balanced, labels_balanced