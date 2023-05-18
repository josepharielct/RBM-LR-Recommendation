import numpy as np
import projectLib as lib
import matplotlib.pyplot as plt

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

# some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(trStats):
    # Initializes an all-zero matrix A of shape (n_ratings, n_movies + n_users)
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    # Create the temp_movies variable that stores the list of movie
    temp_movie = trStats['movies']
    # Create temp_users variable that stores the list of user 
    temp_user = trStats['users']
    # For each rating set the corresponding movie index and user index in the matrix A to 1
    row_indices = np.arange(trStats['n_ratings'])
    A[row_indices, temp_movie] = 1
    A[row_indices, np.array(temp_user) + trStats['n_movies']] = 1
    # Output the matrix A
    return A

# we also get c
def getc(rBar, ratings):
    # Convert the ratings list to a numpy array
    ratings_array = np.array(ratings)
    # Calculate the element-wise difference between ratings_array and rBar
    c = ratings_array - rBar
    # Output the vector c
    return c

# apply the functions
A = getA(trStats)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    # Initialize the transpose of matrix A
    A_T = A.transpose()
    # Compute the estimator b by using b = (A^TA)^-1 * A^Tc
    b = np.linalg.pinv(A_T @ A) @ c
    # Output the vector b
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    # Initialize the transpose of matrix A
    A_T = A.transpose()
    # Initialize the identity matrix
    I = np.eye((A_T @ A).shape[0])
    # Compute the regularized estimator b by using b = (A^TA + lI)^-1 * A^Tc
    b = np.linalg.inv(A_T @ A + l * I) @ (A_T @ c)
    # Output the regularized estimator b
    return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)

# Regularised version
x = np.linspace(0,20,101)
y1 = []
y2 = []
for l in x:
    b = param_reg(A, c, l)
    print("Linear regression, l = %f" % l)
    rmse_1 = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
    y1.append(rmse_1)
    print("RMSE for training %f" % rmse_1)
    rmse_2 = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
    print("RMSE for validation %f" % rmse_2)
    y2.append(rmse_2)

# Plot validation and training loss
plt.plot(x, y2, label = 'Validation')
plt.plot(x, y1, label = 'Training')
plt.legend()
plt.show()