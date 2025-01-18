**PyTorch for Machin Learning & Deep Learning**

**PyTorch_Fundamentals.ipynb⏬**<br/>
**1️⃣ CPU (Central Processing Unit) & GPU (Graphics Processing Unit):**
- `CPU`
  - Designed for general-purpose computing.
  - Optimized for sequential tasks.
  - Has a few powerful cores.
  - Excellent at handling complex logic and single-threaded applications.
- `GPU`
  - Designed for parallel processing.
  - Has thousands of smaller, less powerful cores.
  - GPUs offer far faster numerical computing than CPUs.
  - Optimized for tasks that can be divided into many independent calculations.
  - Excellent for tasks like matrix operations, which are common in deep learning.

Putting a tensor on GPU using `to(device)` (e.g. `some_tensor.to(device)`) returns a copy of that tensor, e.g. the same tensor will be on CPU and GPU. `some_tensor = some_tensor.to(device)`  

**2️⃣ N-d Tensor:** A tensor is a multi-dimensional array of numerical values. Tensor computation (like numpy) with strong GPU acceleration.
- `0-dimensional (Scalar):` A single number, e.g., 5, 3.14, -10. A <font color='red'><b>scalar</b></font> is a single number and in tensor-speak it's a zero dimension tensor.
- `1-dimensional (Vector):` A list of numbers, e.g., [1, 2, 3]. A <font color='blue'><b>vector</b></font> is a single dimension tensor but can contain many numbers.<br/>
- `2-dimensional (Matrix):` A table of numbers, e.g., [[1, 2], [3, 4]]. <font color='green'><b>MATRIX</b></font>  has two dimensions.
- `3-dimensional (or higher):` Like a "cube" of numbers or more complex higher-dimensional structures. These are common for representing images, videos, and more.

**3️⃣ Tensor datatypes:**<br/>
There are many different [tensor datatypes available in PyTorch](https://pytorch.org/docs/stable/tensors.html#data-types). Some are specific for CPU and some are better for GPU.<br/>
Generally if you see `torch.cuda` anywhere, the tensor is being used for GPU (since Nvidia GPUs use a computing toolkit called CUDA).<br/>
The most common type (and generally the default) is `torch.float32` or `torch.float`.<br/>

**4️⃣ Getting information from tensors:**<br/>
* `shape` - what shape is the tensor? (some operations require specific shape rules)
* `dtype` - what datatype are the elements within the tensor stored in?
* `device` - what device is the tensor stored on? (usually GPU or CPU)

**5️⃣ Math Operations:**<br/>
* Addition ⇒ `a+b `or `torh.add(a, b)`
* Substraction ⇒ `a-b `or `torh.sub(a, b)`
* Multiplication (element-wise) ⇒ `a*b `
* Division ⇒ `a/b `or `torh.div(a, b)`
* Matrix multiplication ⇒ "`@`" in Python is the symbol for matrix multiplication. [`torch.matmul()`](https://pytorch.org/docs/stable/generated/torch.matmul.html) or [`torch.mm()`](https://pytorch.org/docs/stable/generated/torch.mm.html)
  
**6️⃣ Special Arrays**<br/>
- zeros
- ones
- empty
- eye
- full<br/>

Using [`torch.zeros_like(input)`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html) or [`torch.ones_like(input)`](https://pytorch.org/docs/1.9.1/generated/torch.ones_like.html) which return a tensor filled with zeros or ones in the same shape as the `input` respectively.

**7️⃣ Random Arrays**
- `torch.rand:` Create a n*m tensor filled with random numbers from a uniform distribution on the interval [0, 1)
- `torch.randn:` Create a n*m tensor filled with random numbers from a normal distribution with mean 0 and variance 1. 
- `torch.randint:` Create a n*m tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

`torch.randperm(value):` Create a random permutation of integers from 0 to value.<br/>
`torch.permute(input, dims):` Permute the original tensor to rearrange the axis order.

**8️⃣ Indexing & Slicing**
- `Indexing`
  - Accessing individual elements:  use integer indices to specify the position of the element you want to retrieve.
- `Slicing`
  - Extracting sub-tensors: Slicing allows you to extract a sub-part of your tensor by specifying a range of indices using the colon : operator.
    - `start:end` (exclusive end)
    - `start:` (from start to end of dimension)
    - `:end` (from beginning to end of dimension)
    - `:` (all elements)
    - `start:end:step` (start to end with given step)
  - Slicing with steps: You can include a step to skip elements in the slice. `start:end:step`

**9️⃣ `Unsqueeze & unsqueeze:`**
- The squeeze() method removes all singleton dimensions from a tensor. It will reduce the number of dimensions by removing the ones that have a size of 1.
- The unsqueeze() method adds a singleton dimension at a specified position in a tensor. It will increase the number of dimensions by adding a size of 1 dimension at a specific position.

**🔟 `PyTorch tensors & NumPy:`**
- [`torch.from_numpy(ndarray)`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html)  NumPy array → PyTorch tensor
- [`torch.Tensor.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)  PyTorch tensor → NumPy array.
----

**0️⃣ Dataset: Breast Cancer** | [kaggle](https://www.kaggle.com/datasets/rahmasleam/breast-cancer/data?select=breast-cancer.csv)<br/>
Objective: Predict whether a tumor is malignant or benign.<br/>
**Files:**<br/>
`breast-cancer.csv` - class is the binary target (either m or b)

**1️⃣ Load data:** Load the dataset into the environment from files, databases, or other sources.<br/>
**2️⃣ Data Cleaning:** It involves handling missing values, removing duplicates, correcting errors, and dealing with outliers.<br/>
- `Handle Missing Values`
  - Remove rows or columns with too many missing values.
  - Impute missing values using methods like mean, median, mode, or advanced techniques like KNN or predictive modeling.
- `Handle Outliers`
   - Detect outliers using statistical methods (e.g., Z-score, IQR).
     - The Z-score measures how many standard deviations a data point is from the mean. Typically, a Z-score greater than 3 or less than -3 is considered an outlier.
     - The Interquartile Range (IQR) measures the spread of the middle 50% of a dataset and is less influenced by outliers than the range or standard deviation. It is calculated as the difference between the third quartile (Q3) and the first quartile (Q1).

**⏹ Note:** If the data includes features of string type, handle outliers only after encoding categorical variables into numerical format.<br>

**3️⃣ Encoding Categorical Variables:** Categorical data needs to be converted into a numerical format that machine learning algorithms can understand.<br/>
`Label Encoding:` Convert categorical data into numerical format by assigning a distinct integer to each category.<br/>
`One-Hot Encoding:` Creates binary columns for each category. Only one of these columns is "hot" (1) for a given observation, while the others are "cold" (0).

**4️⃣ Feature Extraction** (optional, depending on the problem)**:** Transforming original features into a new set of features that better represent the problem (Create new features that capture the most important information in the data).<br/>
`Factor Analysis (FA):` Reduce dimensionality by identifying hidden factors that explain correlations among observed variables. Assumes observed variables are linear combinations of underlying factors plus noise, extracting factors that capture maximum variance.<br/>
`Isometric Feature Mapping (Isomap):` Isomap is a nonlinear dimensionality reduction method that maintains the geometric structure of data in a lower-dimensional space. It builds a graph of nearest neighbors in the high-dimensional space, calculates the shortest paths between pairs of points, and utilizes these distances to map the data into a reduced-dimensional space.<br/>
`Principal component analysis (PCA):` PCA minimizes the dimensionality of data while retaining as much variance as possible. It determines new axes, called principal components, that maximize variance in the data, with each axis being orthogonal and uncorrelated to the others.<br/>
`Linear discriminant analysis (LDA):` LDA identifies the linear combination of features that optimally distinguishes between classes. It enhances class separation in the reduced space by maximizing the ratio of between-class variance to within-class variance.<br/>
`Singular value decomposition (SVD):` SVD decomposes a matrix into three other matrices to simplify data representation. It factors a matrix into singular vectors and singular values, capturing important structures in the data.<br/>
`Independent component analysis (ICA):` ICA decomposes a multivariate signal into independent, additive components. It operates under the assumption that the observed data is a combination of independent sources and aims to extract those sources by enhancing their statistical independence.<br/>
`T-distributed Stochastic Neighbor Embedding (T-SNE):` T-SNE is a nonlinear dimensionality reduction method that transforms high-dimensional data into two or three dimensions. It achieves this by minimizing the difference between two probability distributions: one representing pairwise similarities in the high-dimensional space and the other in the low-dimensional space.<br/>

**5️⃣ Feature Selection** (optional, depending on the problem)**:** Choosing the most relevant features from the original dataset to reduce dimensionality, remove irrelevant or redundant features, and improve model performance.<br/>
`Filter Methods:` These techniques evaluate the importance of features through statistical tests, independent of any machine learning model. While they are efficient in terms of computation, they might not account for interactions between features.
  - ***ANOVA:*** Analyzes the variance between groups to select features.
  - ***Variance (Var):*** Chooses features according to their variance, eliminating those with low variance.
  - ***Fisher score (FS):*** Measures the separation between classes.
  - ***Mutual information (MI):*** Measures the dependency between a feature and the target variable.
  - ***Univariate feature selection (UFS):*** Selects the best features based on univariate statistical tests.

`Wrapper Methods:` These methods use a specific machine learning algorithm to evaluate the performance of subsets of features. They are computationally more expensive but can provide better performance.
  - ***Forward feature selection (FFS):*** Starts with no features and adds them one by one, selecting the best at each step.
  - ***Backward feature selection (BFS):*** Starts with all features and removes the least significant one at each step.
  - ***Exhaustive Feature Selection (EFS):*** Evaluates all possible combinations of features.
  - ***Recursive feature elimination (RFE):*** Recursively removes the least important features based on model performance.

`Embedded Methods:` These methods perform feature selection as part of the model training process, combining the efficiency of filter methods and the accuracy of wrapper methods.
  - ***Random forest (RF):*** Uses feature importance scores from a random forest model.
  - ***L1-based feature selection (L1)***: Uses L1 regularization (e.g., Lasso Regression) to shrink less relevant feature coefficients to zero.
  - ***Tree-based feature selection (TFS):*** Uses decision trees to select important features.

**⏹ Note:** *`Feature Extraction Before Feature Selection`* (Use when raw features need transformation before selection (e.g., extracting features from images using CNNs, then selecting important ones)). *`Feature Selection Before Feature Extraction`* (Use when you have many raw features and want to reduce dimensionality first (e.g., selecting columns in tabular data before applying PCA)).<br/>

**6️⃣ Split data into training, Validation and test sets**<br/>
`Training set:` Used to train the model (e.g., 70-80% of the data).<br/>
`Validation set:` Used to tune hyperparameters, Early Stopping and evaluate model performance (monitor overfitting) during training (e.g., 10-15% of the data).<br/>
`Testing set:` Used to evaluate the final model's performance (e.g., 10-15% of the data).<br/>

**7️⃣ Scaling/Normalization:** Scale numerical features to a standard range (e.g., 0 to 1 or -1 to 1).<br/>
`Fit the scaler/normalizer on the training data only:` Calculate scaling parameters (e.g., mean, standard deviation for standardization, or min/max values for Min-Max scaling) using only the training set.<br/>
`Transform all datasets (training, validation, and test):` Apply the scaling parameters learned from the training set to the validation and test sets.<br/>
**Common Scaling/Normalization Techniques:**
- `Min-Max Scaling:` Normalizes or scales the data to a specified range, typically [0, 1] or [a, b]. Best suited for uniformly distributed data with no significant outliers.<br/>
   $X_{\text{scaled}} = \large \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$<br/>
- `Standardization (Z-Score):` Standardizes features by removing the mean and scaling to unit variance. Useful when the data is normally distributed or when the distribution of data varies significantly across features.<br/> 
   $X_{\text{standardized}} = \large \frac{X - \mu}{\sigma}$<br/>
   $\mu$ is the mean and $\sigma$ is the standard deviation of the feature.<br/>
- `Mean:` Similar to standardization but scales data to have a mean of 0 and a range between -1 and 1.<br/>
   $X_{\text{normalized}} = \large \frac{X - \mu}{X_{\text{max}} - X_{\text{min}}}$<br/>
   $\mu$ is the mean.

**Benefits of Normalization in Machine Learning and Deep Learning**<br/>
`Improves Training Stability:` Models, especially neural networks, are sensitive to the scale of input data. Features with large ranges can dominate the learning process, leading to slower convergence or convergence to suboptimal solutions.<br/>
`Faster Convergence:` Normalization helps optimization algorithms converge faster by scaling features to a similar scale.<br/>
`Prevents Numerical Instability:` Neural networks can experience numerical instability when working with features that have very large or very small values.<br/>
`Required by Algorithms:` Some distance-based algorithms (e.g., `k-nearest neighbors`, `SVMs with RBF kernels`) perform better when features are on the same scale.<br/>



