**PyTorch for Machin Learning & Deep Learning**

**01_pytorch_fundamentals.ipynb⏬**<br/>
**1️⃣ CPU (Central Processing Unit) & GPU (Graphics Processing Unit)**
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

**02_data_preprocessing.ipynb⏬**<br/>
**0️⃣ Dataset:**<br/> 
`1.Breast Cancer` [kaggle](https://www.kaggle.com/datasets/rahmasleam/breast-cancer/data?select=breast-cancer.csv)<br/>
File: *breast-cancer.csv* - class is the binary target (either malignant (m) or benign (b))<br/>
`2.Stroke Prediction Dataset` [kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)<br/>
File: *healthcare-dataset-stroke-data.csv* - class is the binary target (1 if the patient had a stroke or 0 if not)<br/>
`3.Titanic Dataset` [kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)<br/>
File: *Titanic-Dataset.csv* - class is the binary target (Survived: 0 , 1)<br/>

**1️⃣ Load data:** Load the dataset into the environment from files, databases, or other sources.<br/>

**2️⃣ Data Cleaning:** It involves handling missing values, removing duplicates, correcting errors, and dealing with outliers.<br/>
`1.Handle Missing Values`<br/>
Remove rows or columns with too many missing values.<br/>
Impute missing values using methods like mean, median, mode, or advanced techniques like KNN or predictive modeling.<br/>
`2.Handle Outliers`<br/>
Detect outliers using statistical methods (e.g., Z-score, IQR).<br/>
The *Z-score* measures how many standard deviations a data point is from the mean. Typically, a Z-score greater than 3 or less than -3 is considered an outlier.<br/>
The *Interquartile Range (IQR)* measures the spread of the middle 50% of a dataset and is less influenced by outliers than the range or standard deviation. It is calculated as the difference between the third quartile (Q3) and the first quartile (Q1).<br/>

**⏹ Note:** If the data includes *features of string type*, *handle outliers* only after encoding categorical variables into numerical format.<br>

**3️⃣ Encoding Categorical Variables:** Categorical data needs to be converted into a numerical format that machine learning algorithms can understand.<br/>
`1.Label Encoding:` Convert categorical data into numerical format by assigning a distinct integer to each category.<br/>
`2.One-Hot Encoding:` Creates binary columns for each category. Only one of these columns is "hot" (1) for a given observation, while the others are "cold" (0).

**4️⃣ Imbalanced Data:** Apply techniques to balance the dataset (e.g., oversampling, undersampling, or using class weights).<br/>
**`Implications:`**<br/>
`1.Model Bias:` Training models on this dataset might result in a bias toward the majority class ("A"), which could negatively impact performance on the minority class ("B").<br/>
`2.Evaluation Metrics:` Accuracy alone may not be a reliable metric. Precision, recall, F1-score, and ROC-AUC should be considered to evaluate model performance effectively.<br/>
**`Suggested Approaches:`**<br/>
`1.Resampling:`<br/>
*Oversampling:* Use techniques like SMOTE to increase the number of minority class samples.<br/>
*Undersampling:* Reduce the number of majority class samples, though this may lead to loss of information.<br/>
`2.Class Weights:` Adjust class weights in the model to give more importance to the minority class.<br/>
`3.Ensemble Methods:` Use algorithms like *Random Forest* or *XGBoost* that can handle class imbalance better.<br/>

**5️⃣ Feature Extraction** (optional, depending on the problem)**:** Transforming original features into a new set of features that better represent the problem (Create new features that capture the most important information in the data).<br/>
**`Factor Analysis (FA, Unsupervise method):`** Reduce dimensionality by identifying hidden factors that explain correlations among observed variables. Assumes observed variables are linear combinations of underlying factors plus noise, extracting factors that capture maximum variance.<br/>
`1.Model Representation:`  $X = LF + \epsilon$<br/>
`2.Covariance Decomposition:` $\Sigma = LL^T + \Psi$<br/>
$\Sigma$: Observed covariance matrix.<br/>
L: Factor loadings.<br/>
$\Psi$: Diagonal matrix of unique variances.<br/>
`3.Objective:` Minimize the difference between the observed and modeled covariance:$\| \Sigma - (LL^T + \Psi) \|$<br/>

**`Isometric Feature Mapping (Isomap, Unsupervise method):`** Isomap is a nonlinear dimensionality reduction method that maintains the geometric structure of data in a lower-dimensional space. It builds a graph of nearest neighbors in the high-dimensional space, calculates the shortest paths between pairs of points, and utilizes these distances to map the data into a reduced-dimensional space.<br/>
`1.Neighborhood Graph:` $\small G = (V, E), \quad \text{where } E \text{ connects } x_i \text{ and } x_j \text{ if } x_j \in N_k(x_i) \text{ or } \|x_i - x_j\| \leq \epsilon$<br/>
`2.Geodesic Distance Matrix:` $D_{ij} = \text{ShortestPath}(x_i, x_j \text{ on } G)$<br/>
`3.MDS Eigenproblem:` $\small B = -\frac{1}{2} H D^2 H, \quad \text{where } \small H = I - \frac{1}{n} \mathbf{1} \mathbf{1}^T$<br/>
   Use eigenvalues and eigenvectors of B to compute embeddings.<br/>

**`Principal component analysis (PCA, Unsupervise method):`** PCA minimizes the dimensionality of data while retaining as much variance as possible. It determines new axes, called principal components, that maximize variance in the data, with each axis being orthogonal and uncorrelated to the others.<br/>
`1.Standardize the d-dimensional dataset.`<br/>
`2.Construct the covariance matrix.`<br/>
`3.Decompose the covariance matrix into its eigenvectors and eigenvalues.`<br/>
`4.Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.`<br/>
`5.Select k eigenvectors`, which correspond to the k largest eigenvalues, where k is the dimensionality of the new feature subspace (𝑘≤d).<br/>
`6.Construct a projection matrix`, W, from the “top” k eigenvectors.<br/>
`7.Transform the d-dimensional input dataset`, X, using the projection matrix, W, to obtain the new k-dimensional feature subspace.<br/>

**`Linear discriminant analysis (LDA, Supervise method):`** LDA identifies the linear combination of features that optimally distinguishes between classes. It enhances class separation in the reduced space by maximizing the ratio of between-class variance to within-class variance.<br/>
`1.Standardize the d-dimensional dataset` (d is the number of features).<br/>
`2.For each class, compute the d-dimensional mean vector`.<br/>
`3.Construct the between-class scatter matrix`, SB, and *the within-class scatter matrix*, SW.<br/>
`4.Compute the eigenvectors and corresponding eigenvalues of the matrix`, $S_W^{-1} S_B$.<br/>
`5.Sort the eigenvalues by decreasing order` to rank the corresponding eigenvectors.<br/>
`6.Choose the k eigenvectors` that correspond to the k largest eigenvalues to construct a d×k-dimensional transformation matrix, W; the eigenvectors are the columns of this matrix.<br/>
`7.Project the examples onto the new feature subspace` using the transformation matrix, W.<br/>

**`Singular value decomposition (SVD, Unsupervise method):`** SVD decomposes a matrix into three other matrices to simplify data representation. It factors a matrix into singular vectors and singular values, capturing important structures in the data.<br/>
`1.Start with a matrix A of size` $m \times n$.<br/>
`2.Compute Covariance matrix`<br/> 
Calculate $\small A^TA$ (size $n\times n$).<br/>
Calculate $\small AA^T$ (size $m\times m$).<br/>
`3.Find Eigenvalues and Eigenvectors`<br/>
Compute eigenvalues and eigenvectors of $\small A^TA$ to form matrix V.<br/>
Compute eigenvalues and eigenvectors of $\small AA^T$ to form matrix U.<br/>
`4.Construct` $\small \Sigma$: Form a diagonal matrix $\small \Sigma$ with singular values (square roots of eigenvalues) in descending order.<br/> 
`5.Express` $\small A$ as $\small U\Sigma V^T$<br/>
U: Columns are eigenvectors of $\small AA^T$.<br/>
$\small \Sigma$: Diagonal matrix of singular values.<br/>
$\small V^T$: Rows are eigenvectors of $\small A^T A$.<br/>
`6.Verify the Decomposition:` Multiply $\small U \Sigma V^T$ to reconstruct A and confirm correctness.<br/>

**`Independent component analysis (ICA, Unsupervise method):`** ICA decomposes a multivariate signal into independent, additive components. It operates under the assumption that the observed data is a combination of independent sources and aims to extract those sources by enhancing their statistical independence.<br/>
`1.Input Data Preprocessing:` Center the data to make it zero-mean. Whiten the data to make it uncorrelated with unit variance.<br/>
`2.Define the ICA Model:` Assume $X = AS$, where X is the observed data, A is the mixing matrix, and S are the independent components.<br/>
`3.Choose an Independence Measure:` Select a measure such as kurtosis or negentropy to quantify statistical independence.<br/>
`4.Optimize to Maximize Independence:` Use algorithms like FastICA or Infomax ICA to estimate A and S.<br/>
`5.Compute Independent Components:` Solve $S = W X$, where W is the demixing matrix.<br/>
`6.Post-Processing:` Normalize or scale the separated components and evaluate their independence.

**`T-distributed Stochastic Neighbor Embedding (T-SNE, Unsupervise method):`** T-SNE is a nonlinear dimensionality reduction method that transforms high-dimensional data into two or three dimensions. It achieves this by minimizing the difference between two probability distributions: one representing pairwise similarities in the high-dimensional space and the other in the low-dimensional space.<br/>
`1.Input Data Preparation:` Start with high-dimensional data X with n points and d-dimensions.<br/>
`2.Compute Pairwise Similarities in High-Dimensional Space:` Calculate conditional probabilities $p_{j|i}$ using a Gaussian kernel. $p_{j|i} = \frac{\exp\left(-\|x_i - x_j\|^2 / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\left(-\|x_i - x_k\|^2 / 2\sigma_i^2\right)}$<br/>
$\|x_i - x_j\|^2$: Squared Euclidean distance between points $x_i$ and $x_j$.<br/>
$\sigma_i$: Perplexity-related bandwidth parameter for point $x_i$.<br/>
The numerator computes the similarity between $x_i$ and $x_j$ using a Gaussian kernel.<br/>
The denominator normalizes the probabilities over all points $\small k \neq i$.<br/>
Symmetrize to compute $p_{ij}$.  $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$<br/>
$p_{j|i}$ and $p_{i|j}$: Conditional probabilities of $x_j$ given $x_i$, and $x_i$ given $x_j$, respectively.<br/>
n: Total number of data points.<br/>
Adjust perplexity parameter $\sigma_i$ to control neighbor influence.<br/>
`3.Define Low-Dimensional Space:` Randomly initialize points $y_1, y_2, \ldots, y_n$ in a low-dimensional space.<br/>
`4.Compute Pairwise Similarities in Low-Dimensional Space:` Calculate similarities $q_{ij}$ using a Student's t-distribution.$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$<br/>
$\|y_i - y_j\|^2$: Squared Euclidean distance between points $y_i$ and $y_j$ in the low-dimensional space.<br/>
The numerator computes similarity using a **Student's t-distribution** with one degree of freedom.<br/>
The denominator normalizes $q_{ij}$ over all pairs of points.<br/>
`5.Minimize KL Divergence`: Optimize $y_i$ by minimizing the KL divergence $\small KL(P || Q)$ using gradient descent. $\small KL(P || Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$<br/>
$p_{ij}$: Pairwise similarity in the high-dimensional space.<br/>
$q_{ij}$: Pairwise similarity in the low-dimensional space.<br/>
The KL divergence quantifies how well the low-dimensional embedding Q preserves the structure of P. The goal in t-SNE is to minimize this divergence.<br/>
`6.Output the Embedded Points:` The $y_i$ represent the data in the low-dimensional space, preserving local similarities.

**6️⃣ Feature Selection** (optional, depending on the problem)**:** Choosing the most relevant features from the original dataset to reduce dimensionality, remove irrelevant or redundant features, and improve model performance.<br/>
**`Filter Methods:`** These techniques evaluate the importance of features through statistical tests, independent of any machine learning model. While they are efficient in terms of computation, they might not account for interactions between features.<br/>
`1.ANOVA:` Analyzes the variance between groups to select features.<br/>
`2.Variance (Var):` Chooses features according to their variance, eliminating those with low variance.<br/>
`3.Fisher score (FS):` Measures the separation between classes.<br/>
`4.Mutual information (MI):` Measures the dependency between a feature and the target variable.<br/>
`5.Univariate feature selection (UFS):` Selects the best features based on univariate statistical tests.<br/>

**`Wrapper Methods:`** These methods use a specific machine learning algorithm to evaluate the performance of subsets of features. They are computationally more expensive but can provide better performance.<br/>
`1.Forward feature selection (FFS):` Starts with no features and adds them one by one, selecting the best at each step.<br/>
`2.Backward feature selection (BFS):` Starts with all features and removes the least significant one at each step.<br/>
`3.Exhaustive Feature Selection (EFS):` Evaluates all possible combinations of features.<br/>
`4.Recursive feature elimination (RFE):` Recursively removes the least important features based on model performance.<br/>

**`Embedded Methods:`** These methods perform feature selection as part of the model training process, combining the efficiency of filter methods and the accuracy of wrapper methods.<br/>
`1.Random forest (RF):` Uses feature importance scores from a random forest model.<br/>
`2.L1-based feature selection (L1)`: Uses L1 regularization (e.g., Lasso Regression) to shrink less relevant feature coefficients to zero.<br/>
`3.Tree-based feature selection (TFS):` Uses decision trees to select important features.<br/>

**⏹ Note:** *`Feature Extraction Before Feature Selection`* (Use when raw features need transformation before selection (e.g., extracting features from images using CNNs, then selecting important ones)). *`Feature Selection Before Feature Extraction`* (Use when you have many raw features and want to reduce dimensionality first (e.g., selecting columns in tabular data before applying PCA)).<br/>

**7️⃣ Split data into training, Validation and test sets**<br/>
`1.Training set:` Used to train the model (e.g., 70-80% of the data).<br/>
`2.Validation set:` Used to tune hyperparameters, Early Stopping and evaluate model performance (monitor overfitting) during training (e.g., 10-15% of the data).<br/>
`3.Testing set:` Used to evaluate the final model's performance (e.g., 10-15% of the data).<br/>

![Split.JPG](attachment:Split.JPG)

**8️⃣ Scaling/Normalization:** Scale numerical features to a standard range (e.g., 0 to 1 or -1 to 1).<br/>
`Fit the scaler/normalizer on the training data only:` Calculate scaling parameters (e.g., mean, standard deviation for standardization, or min/max values for Min-Max scaling) using only the training set.<br/>
`Transform all datasets (training, validation, and test):` Apply the scaling parameters learned from the training set to the validation and test sets.<br/>
**Common Scaling/Normalization Techniques:**<br/>
`1.Min-Max Scaling:` Normalizes or scales the data to a specified range, typically [0, 1] or [a, b]. Best suited for uniformly distributed data with no significant outliers.<br/>
   $X_{\text{scaled}} = \large \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$<br/>
`2.Standardization (Z-Score):` Standardizes features by removing the mean and scaling to unit variance. Useful when the data is normally distributed or when the distribution of data varies significantly across features.<br/> 
   $X_{\text{standardized}} = \large \frac{X - \mu}{\sigma}$<br/>
   $\mu$ is the mean and $\sigma$ is the standard deviation of the feature.<br/>
`3.Mean:` Similar to standardization but scales data to have a mean of 0 and a range between -1 and 1.<br/>
   $X_{\text{normalized}} = \large \frac{X - \mu}{X_{\text{max}} - X_{\text{min}}}$<br/>
   $\mu$ is the mean.

**Benefits of Normalization in Machine Learning and Deep Learning**<br/>
`Improves Training Stability:` Models, especially neural networks, are sensitive to the scale of input data. Features with large ranges can dominate the learning process, leading to slower convergence or convergence to suboptimal solutions.<br/>
`Faster Convergence:` Normalization helps optimization algorithms converge faster by scaling features to a similar scale.<br/>
`Prevents Numerical Instability:` Neural networks can experience numerical instability when working with features that have very large or very small values.<br/>
`Required by Algorithms:` Some distance-based algorithms (e.g., `k-nearest neighbors`, `SVMs with RBF kernels`) perform better when features are on the same scale.<br/>