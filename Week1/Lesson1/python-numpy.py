import numpy as np

   
def gradientDescent(X : np.ndarray, w : np.ndarray, y : np.ndarray, N : int, maxIter : int = 150, epsilon = 1e-5, lam : float = 0.05, learningRate = 0.15):
   Y_2 = y.T @ y
   YTX = 2 * y.T @ X
   XTY = X.T @ y
   XTX = X.T @ X

   for iter in range(maxIter):
    grad = (XTX@w + XTY) / N + lam * w

    if np.linalg.norm(grad) < epsilon:
      print(f"Gradient descent succesful after {iter} iterations. \n")
      return
    if iter >= maxIter:
        print(f"Gradient descent failed, no convergence found. \n")
        return
    
    print(f"Starting iteration {iter}")
    print(f"Weigts: {w} \n")
    print(f"Gradient: {grad} \n")
    
    w -= learningRate * grad


if __name__ == '__main__':
   N = 1000
   D = 4

   temperatures = np.linspace(273, 1000, num= N)
   pressures = np.linspace(101325, 1013250, num= N)
   c1 = np.linspace(0.0, 1.0, num= N)
   c2 = np.linspace(1.0, 0.0, num= N)

   designMatrix = np.array([temperatures, pressures, c1, c2]).T
   designMatrixMean = np.mean(designMatrix, axis= 0)
   designMatrixSTD = np.std(designMatrix, axis= 0)
   scaledDesignMatrix = (designMatrix - designMatrixMean) / designMatrixSTD

   linearCoeff = np.array([0.001, 0.00007, 0.1, 0.21])
   targetVector =  designMatrix @ linearCoeff + np.random.standard_normal(N)
   targetMean = np.mean(targetVector)
   targetSTD = np.std(targetVector)
   scaledTargetVector = (targetVector - targetMean) / targetSTD
   print(f"Shape of target vector is {targetVector.shape} \n")
   

   weights = np.random.random(D)
   print(f"Shape of weights vector is {weights.shape} \n")

   print(f"Shape of desing matrix is {designMatrix.shape} \n")
   gradientDescent(X= scaledDesignMatrix, w= weights, y= scaledTargetVector, N= N)
