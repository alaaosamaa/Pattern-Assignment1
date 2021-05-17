
from matplotlib.colors import PowerNorm
from sklearn.datasets import make_classification
import numpy as np
from numpy.linalg import norm
import sys
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#problem 1 data
X1 = np.array([[50, 55, 70, 80, 130, 150, 155, 160], [1, 1, 1, 1, 1, 1, 1, 1]]).T
Y1 = [1, 1, 1, 1, -1, -1, -1, -1]

#problem 4 data
X2 = np.array([[0, 255, 0, 0, 255, 0, 255, 255],[0, 0, 255, 0, 255, 255, 0, 255],
[0, 0, 0, 255, 0, 255, 255, 255], [1, 1, 1, 1, 1, 1, 1, 1]]).T
Y2 = [1, 1, 1, -1, 1, -1, -1, 1]


#online training function
#returns weights, number of weight updates, nymber of epochs and delta
def online_training(x, y, lr = 1):  
    
    #create random weights ranging from -1 to 1  
    w = np.random.randint(-1, 2, len(x[0]))

    #intialize the variables
    delta = np.ones(len(x[0]))
    e = sys.float_info.epsilon
    w_steps = []
    Delta_online=[]
    w_updates = 0
    epochs = 0

    # Loop until the delta is greater then the epsilon  
    while norm(delta, 1) > e:        
        delta = np.zeros(len(x[0])) 

        # Loop throw x and calculate the weight ,weight updates, nymber of epochs and delta
        for m in range (len(x)):
            a = w.dot(x[m])
            if y[m]*a <= 0:
                delta = delta - y[m]*x[m]
                w = w - delta / len(x)
                w_steps.append(w)
                w_updates += 1
        epochs += 1

        # append delta before reset it 
        Delta_online.append(norm(delta,1))

    return w, w_updates, epochs, Delta_online

#Batch Perceptron training function
#returns weights, number of weight updates, nymber of epochs and delta
def batch_perceptron(x, y, lr = 1):    

    #create random weights ranging from -1 to 1  
    w = np.random.randint(-1, 2, len(x[0]))
    
    #intialize the variables
    delta = np.ones(len(x[0]))
    e = sys.float_info.epsilon
    w_steps = []
    Delta_batch=[]
    w_updates = 0    
    epochs = 0

    # Loop until the delta is greater then the epsilon 
    while norm(delta, 1) > e :
        delta = np.zeros(len(x[0]))

        # Loop throw x and calculate the weight ,weight updates, nymber of epochs and delta
        for m in range(len(x)):
            a = (w.dot(x[m]))
            if y[m] * a <= 0:
                delta = delta - y[m]*x[m]
        delta = delta / len(x)
        w = w - lr * delta
        w_steps.append(w)
        w_updates += 1
        epochs += 1
        # append delta before reset it 
        Delta_batch.append(norm(delta,1))

    return w, w_updates, epochs, Delta_batch

# call online_training and batch_perceptron function for problem 1 and 2 data 
w1_online, wu1_online, e1_online, d1_online = online_training(X1, Y1)
w1_batch, wu1_batch, e1_batch, d1_batch = batch_perceptron(X1, Y1)
w2_online, wu2_online, e2_online, d2_online = online_training(X2, Y2)
w2_batch, wu2_batch, e2_batch, d2_batch = batch_perceptron(X2, Y2)

# Print output to compare between them
print("\t \t \t  weight \t \t \t weight updates \t epochs")

print('Problem 1 ONLINE-Bias: ', w1_online,"\t \t \t",wu1_online,"\t \t \t",e1_online)
print('Problem 1 BATCH-Bias: ',w1_batch,"\t \t \t",wu1_batch,"\t \t \t",e1_batch)

print('Problem 4 ONLINE-Bias: ', w2_online ,"\t",wu2_online,"\t \t \t",e2_online)
print('Problem 4 BATCH-Bias: ',w2_batch,"\t",wu2_batch,"\t \t \t",e2_batch)

# Comparison plot
figure, axis = plt.subplots(2, 2)
# For Online Training Delta Problem 1 Function
axis[0, 0].plot(d1_online)
axis[0, 0].set_title("Online Training Delta Problem 1")
  
# For Online Training Delta Problem 4 Function
axis[0, 1].plot(d2_online)
axis[0, 1].set_title("Online Training Delta Problem 4")
  
# For Batch Perceptron Delta Problem 1 Function
axis[1, 0].plot(d1_batch)
axis[1, 0].set_title("Batch Perceptron Delta Problem 1")
  
# For Batch Perceptron Delta Problem 4 Function
axis[1, 1].plot(d2_batch)
axis[1, 1].set_title("Batch Perceptron Delta Problem 4")
  
# Combine all the operations and display
plt.show()


# /////////// Part 2 /////////

x, y = make_classification(25, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)

print("\t \t \t  weight \t \t \t weight updates \t epochs")

# call the online_training and batch_perceptron functions and pass the trainig data (75% of the original data)
# and print the output
wTraining_online, wuTraining_online, eTraining_online, dTraining_online = online_training(x[0:int(len(x)*(0.75))], y[0:int(len(y)*(0.75))])
print('ONLINE 75%: ', wTraining_online,"\t \t \t",wuTraining_online,"\t \t \t",eTraining_online)

wTraining_batch, wuTraining_batch, eTraining_batch, dTraining_batch = batch_perceptron(x[0:int(len(x)*(0.75))], y[0:int(len(y)*(0.75))])
print('Batch 75%: ', wTraining_batch,"\t \t \t",wuTraining_batch,"\t \t \t",eTraining_batch)

# Comparison plot
figure, axis = plt.subplots(2, 2)
# For Online Training Delta Training Data 
axis[0, 0].plot(dTraining_online)
axis[0, 0].set_title("Online Training Delta Traing Data")
  
# For Batch Perceptron Delta Training Data 
axis[0, 1].plot(dTraining_batch)
axis[0, 1].set_title("Batch Perceptron Delta Training Data")

# Combine all the operations and display
plt.show()

# Assign Xtest an Ytest (25% of data)
xtest=x[(int(len(x)*(0.75))):len(x)]
ytest=y[(int(len(y)*(0.75))):len(y)]

# Assign y predicted for both methods 
yTraining_predicted=[]
yBatch_predicted=[]
for i in xtest:
    yTraining_predicted.append( np.sign(wTraining_online.dot(i)))

for i in xtest:
    yBatch_predicted.append(np.sign(wTraining_batch.dot(i)))


# Calculate the accuracy for both methods
print("Accuracy of Online Training",accuracy_score(ytest,yTraining_predicted))
print("Accuracy of Batch Perceptron",accuracy_score(ytest,yBatch_predicted))

# Model visualization of online training
y_t= -(wTraining_online[0]*x)/wTraining_online[1]
plt.plot(x,y_t)

# Model visualization of Batch Perceptron
y_= -(wTraining_online[0]*x)/wTraining_online[1]
plt.plot(x,y_)

plt.scatter(x[:,0], x[:,1], marker='o', c='y', s=15, edgecolors='k')
plt.show()