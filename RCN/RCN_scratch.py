import numpy as np 
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.sparse.linalg import eigs as Eigens

def init_weight(input_dim,res_size,K_in,K_rec,insca,spra,bisca):
    #input_dim (int): dimension/size of the input
    #res_size (int): resovior size
    #K_in (int): number of input will be connected to the nodes in resirvor (Sparsity of the input weights.)
    #K_rec (int): number of nodes in resirvor will be connected to nodes in resirvor (Sparsity of the recurrent weights.)
    #insca (float): Scaling factor for input->resirvor weights.
    #spra (float): Spectral radius for scaling the recurrent weights.
    #bisca (float): Scaling factor for input weights.

    #---------- Initializing W_in ------------------

    # W_in is connecting from input to resorvior 

    if K_in==-1: #fully connected input-> 
        #weight of 
        W_in=insca*(np.random.rand(res_size,input_dim)*2-1) #random range from [-insca,insca]
    else: #for sparsly connecteded RCN
        #total number of conncections between input and resorivor nodes: 
        nr_entries = np.int32(res_size*K_in)
        #creating zero matrix with (2,nr_entries)
        ij = np.zeros((2,nr_entries))
        #created nr_entries random number from [-insca,insca] 
        datavec = insca *( np.random.rand(nr_entries)*2-1 )

        #initate the counter for the loop
        Ico=0
        #loop through each resorvior nodes
        for en in range(res_size):
        	#Randomly choose K_in unique indices from the range [0, input_dim - 1]
            Per = np.random.choice(input_dim, K_in, replace=False)
            ij[0][Ico:Ico+K_in]=en
            ij[1][Ico:Ico+K_in]=Per
            #add K_in to the counter 
            Ico+=K_in

        #This will create randomly connected input->resior sparce matrix and it will be more computationally efficent to compute full matrix 
        W_in = scipy.sparse.csc_matrix((datavec, np.int_(ij)),dtype='float64')
        #if K_in > input_dim/2, it is more effienet to consider the matrix as dense matrix. 
        if K_in > input_dim/2:
            W_in=W_in.todense()

    #---------- Initializing W_res ---------

    # W_res is connecting from reservoir to reservoir 

    converged = False
    attempts = 50
    while not converged and attempts > 0:  
    	#fully connected reservoir->reservoir
        if K_rec == -1:
            W_res = np.random.randn(res_size, res_size)
        else:
            #similar to W_in init
            nrentries = np.int32(res_size*K_rec)
            ij = np.zeros((2,nrentries))
            datavec =  np.random.randn(nrentries)

            Ico=0
            for en in range(res_size):
                Per=np.random.permutation(res_size)[:K_rec]
                ij[0][Ico:Ico+K_rec]=en
                ij[1][Ico:Ico+K_rec]=Per
                Ico+=K_rec
            W_res = scipy.sparse.csc_matrix((datavec, np.int_(ij)),dtype='float64')
            if K_rec > res_size/2:
                W_res=W_res.todense()

        #get the eigenvalues of the W_res, and only worry about 6 eigenvalues for computational effiecency
        try:
            we =  Eigens(W_res,return_eigenvectors=False,k=6)
            converged = True
        except:
            print ("WARNING: No convergence! Redo %i times ... " % (attempts-1))
            attempts -=1
            pass

    #normalize the W_res to spra/max(abs(we))
    W_res *= (spra / np.amax(np.absolute(we)))

    #---------- Initializing W_bi ---------
    W_bi = bisca * (np.random.rand(res_size) * 2 - 1)

    return W_in,W_res,W_bi


def activation_fun(input_, act_type='tanh'):
    if act_type=='tanh':
        return np.tanh(input_)

# Reservoir execution (online)
def rcn_propagate(w_in,w_res,w_bi,leak,r_prev,u):
    #w_in: Weight matrix for the connections from the input to the reservoir.
    #w_res: Weight matrix for the recurrent connections within the reservoir.
    #w_bi: Bias weights for the reservoir neurons.
    #w_out: Weight matrix for the connections from the     reservoir to the output.
    #leak: The leakage rate of the reservoir neurons.
    #r_prev: The previous state of the reservoir neurons.
    #u: The current input vector.

    #r_now=(1-leak)*r_prev + leak*activation_fun(w_in*u + w_res*r_prev + w_bi)
    #y=[ r_now , 0 ]  * w_out

    if scipy.sparse.issparse(w_in): # applying input weights to the input. Sparse and dense matrix multiplication is different in Python 
        a1 = w_in * u 
    else:
        a1=np.dot(w_in, u)

    # applying recurrent weights to the previous reservoir states
    if scipy.sparse.issparse(W_res):
    	a2 = w_res * r_prev 
    else:
    	a2=np.dot(w_res, r_prev)

    # adding bias and applying activation function
    r_now = activation_fun(a1 + a2 + w_bi, act_type='tanh')

    # applying leak rate
    r_now = (1 - leak) * r_prev + leak * r_now 
    return r_now

def calc_y(r_now,w_out):
	# applying the output weight
    y = np.dot(np.append([1],r_now),w_out) 
    return y

# ### Reservoir execution (offline)
def rcn_recurrent(W_in, W_res, W_bi, leak, U):
    T = U.shape[0]  # Number of time steps
    nres = W_res.shape[0]  # Size of the reservoir
    R = np.zeros((T+1, nres), dtype='float64')  # Reservoir states matrix

    for t in range(T):  # For each time step
        r_prev = R[t, :]  # Previous state of the reservoir
        u = U[t, :]  # Current input

        # Using rcn_propagate to update the reservoir state and compute output
        r_now = rcn_propagate(W_in, W_res, W_bi, leak, r_prev, u)

        R[t+1, :] = r_now  # Storing the updated state

    return R[1:, :]  # Returning the reservoir states, excluding initial state

#calculate W_out using ridge regression
#w_out = (xTx+lambda I)^(-1) (xTy)
#where  x is layer beofore the linerar regression: x=R
#       y is output

def model_fit(X_train, y_train, W_in, W_res, W_bi, leak, regu):
    """
    Trains the RCN on the provided dataset, one set at a time.

    Parameters:
    X_train (list of arrays): List of training input sequences.
    y_train (list of arrays): List of training target outputs.
    W_in, W_res, W_bi: Reservoir weight matrices and biases.
    leak (float): Leakage rate of the reservoir.
    regu (float): Regularization parameter for ridge regression.

    Returns:
    W_out (array): Trained output weight matrix.
    """
    # Initialize matrices for ridge regression
    xTx = 0
    xTy = 0
    xlen = 0

    
    print('RCN is working hard (ง ’̀-‘́  )ง ')

    for U, D in zip(tqdm(X_train), y_train):
        xlen += U.shape[0]

        # Execute reservoir for the current training set
        R = rcn_recurrent(W_in, W_res, W_bi, leak, U)

        # Add bias term to reservoir states
        R_extended = np.concatenate((np.ones((R.shape[0], 1)), R), axis=1)

        
        # Accumulate xTx and xTy for ridge regression
        xTx += R_extended.T @ R_extended # matrix multiplication shape: (nres, n_time) @ (n_time, nres) = (nres,nres)
        xTy += R_extended.T @ D          # matrix multiplication shape: (nres, n_time) @ (n_time, y_features)= (nres, y_features)


    # Apply ridge regression to compute output weights
    lmda = regu ** 2 * xlen
    W_out = np.linalg.inv(xTx + lmda * np.eye(xTx.shape[0]))    @ xTy   # (xTx+lmda*I)^(-1) xTy
            #shape: (nres,nres) @ (nres, y_features) = (nres, y_features)
    return W_out


def model_predict(X_test, W_in, W_res, W_bi, W_out, leak):
    """
    Generates predictions for the given input using the trained RCN.

    Parameters:
    X_test (array): Input data for prediction.
    W_in, W_res, W_bi: Reservoir weight matrices and biases.
    W_out (array): Trained output weight matrix.
    leak (float): Leakage rate of the reservoir.

    Returns:
    predictions (array): Predicted outputs.
    """
    # Execute reservoir
    R = rcn_recurrent(W_in, W_res, W_bi, leak, X_test)

    # Add bias term to reservoir states
    R_extended = np.concatenate((np.ones((R.shape[0], 1)), R), axis=1)

    # Apply output weights to generate predictions
    predictions = np.dot(R_extended, W_out)

    return predictions

def example_X_y(total_len=5000):
    # Generate a combined sine and cosine wave
    x = np.linspace(0, 200, total_len) # 1000 points from 0 to 50
    y = np.sin(x) * np.cos(x * 0.5) + np.sin(x * 0.3)

    # Create sequences
    sequence_length = 20
    predict_length = 5
    X = []
    Y = []
    for i in range(len(y) - sequence_length-predict_length):
        X.append(y[i:i+sequence_length])
        Y.append(y[i+sequence_length:i+sequence_length+predict_length])

    X = np.array(X)
    Y = np.array(Y)


    X=[X[i*400:(i+1)*400,...] for i in range(int(total_len/400))]
    Y=[Y[i*400:(i+1)*400,...] for i in range(int(total_len/400))]


    X = np.array(X)
    Y = np.array(Y)

    print("Input shape:", X.shape)
    print("Output shape:", Y.shape)

    return X, Y

if __name__ == '__main__':
    regu = 0.001       # Regularization parameter
    insca = 1         # Scaling the input weights W_in
    bisca = 0.2       # Scaling the bias weights W_bi
    spra = 0.8        # Scaling the recurrent weights W_res
    leak = 0.2        # Leakage
    k_in = 10         # Number of input connections to each reservoir node
    k_rec = 100       # Number of recurrent connections to each reservoir node

    res_size = 5000   # Reservoir size
    
    train_test_split=0.5

    print('Creating the sample data')
    X, y=example_X_y(total_len=5000)
    
    # Splitting the dataset into training and testing sets
    X_train = X[:int(train_test_split*len(X)),...]
    y_train = y[:int(train_test_split*len(X)),...]
    X_test = X[int(train_test_split*len(X)):,...]
    y_test = y[int(train_test_split*len(X)):,...]


    input_dim = X_train[0].shape[1]  # Define the input dimension
    k_in=min([k_in,input_dim])
    
    print('Initalizing the weight')
    W_in, W_res, W_bi = init_weight(input_dim, res_size, k_in, k_rec, insca, spra, bisca)

    print('Training')
    W_out = model_fit(X_train, y_train, W_in, W_res, W_bi, leak, regu)

    test_index=0
    test_data=X_test[test_index,...]

    print('Making prediction')
    predictions = model_predict(test_data, W_in, W_res, W_bi, W_out, leak)


    print('Plotting')
    print(predictions.shape)
    print(y_test.shape)

    plt.clf()
    fig, ax=plt.subplots(nrows=1,ncols=2,sharex=True) 

    ax[0].plot(test_data[:,0],predictions[:,0],alpha=1, label='predictions')
    ax[1].plot(test_data[:,0],y_test[test_index,:,0],alpha=1,label='true')

    ax[0].legend()
    ax[1].legend()
    plt.show()

    plt.clf()
    fig, ax=plt.subplots(nrows=1,ncols=2,sharex=True) 

    ax[0].plot(predictions[:,0],alpha=1, label='predictions')
    ax[1].plot(y_test[test_index,:,0],alpha=1,label='true')

    ax[0].legend()
    ax[1].legend()
    plt.show()
