# The code below aims to perform archetypal analysis on synthetic artificial datasets
# and help me understand how this type of analysis is performed and its significance in
# research. 
# For archetypal analysis, I will use the CVXPY import for convex optimization. 
# The code was adapted from ChatGPT.
# For further information, you may visit https://openai.com/chatgpt

##############################################################

# Step 1: We import the required modules/packages
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Step 2: We set the functions we need to perform archetypal analysis 
# uaing alternating convex optimization.
def fit_archetypes(X, k=3, max_iter=40, tol=1e-4, verbose=True):
    """
    Perform archetypal analysis by using alternating convex optimization.
    
    Args:
        X (np.ndarray): (n_samples, n_features) standardized input matrix
        k (int): Number of archetypes
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        verbose (bool): If True, print iteration losses

    Returns:
        A (np.ndarray): Coefficients (n_samples x k)
        B (np.ndarray): Archetype weights (k x number_samples)
        Z (np.ndarray): Archetypes (k x d)
    """
    n, d = X.shape

# Step 2: We initialize the archetype weights with convex weights
    np.random.seed(40)
    B = np.random.dirichlet(alpha=np.ones(n), size=k)

    prev_loss = np.inf

    for it in range(max_iter):
        # First: We fix B and solve for A
        Z_fixed = B @ X  # shape (k, d)
        A_var = cp.Variable((n, k))
        X_hat = A_var @ Z_fixed
        obj = cp.Minimize(cp.sum_squares(X - X_hat))
        constraints = [A_var >= 0, cp.sum(A_var, axis=1) == 1]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        A = A_var.value

        # Second: We fix A and solve for B
        B_var = cp.Variable((k, n))
        Z_var = B_var @ X
        X_hat = A @ Z_var
        obj = cp.Minimize(cp.sum_squares(X - X_hat))
        constraints = [B_var >= 0, cp.sum(B_var, axis=1) == 1]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        B = B_var.value

        # Third: Computing the reconstruction loss
        X_reconstructed = A @ (B @ X)
        loss = np.linalg.norm(X - X_reconstructed, 'fro')**2

        if verbose:
            print(f"Iteration {it + 1:02d}, Loss: {loss:.4f}")

        if np.abs(prev_loss - loss) < tol:
            if verbose:
                print("Convergence.")
            break
        prev_loss = loss

    Z = B @ X  # these are the final archetypes
    return A, B, Z

# Step 3: We set a function for plotting the archetypes

def plot_archetypes(X, Z, title="Archetypal Analysis"):
    """
    Visualizing data and archetypes (2-D only).
    
    Args:
        X (np.ndarray): Data points (n_samples x 2)
        Z (np.ndarray): Archetypes (k x 2)
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data')
    plt.scatter(Z[:, 0], Z[:, 1], c='red', s=200, marker='X', label='Archetypes')
    plt.title(title)
    plt.xlabel("First Feature")
    plt.ylabel("Second Feature")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    
# Step 4: Running the pipeline

# 1. Generating synthetic data
X_raw, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=40)

# 2. Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 3. Fitting the archetypes
A, B, Z = fit_archetypes(X, k=4)

# 4. Plotting
plot_archetypes(X, Z, title="Archetypal Analysis on Synthetic Data")