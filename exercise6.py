import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
	return 1/(1+np.exp(-z))

def model(param,data):
	values=np.dot(data,param)
	classification=np.zeros(len(values))
	for i in range(len(values)):
		if sigmoid(values[i])>0.5:
			classification[i]=1
	return classification
'''
def loss_function(true_label,prediction):
	return np.sum(-(true_label*np.log(prediction)/np.log(2)+(1-true_label)*np.log(1-prediction)/np.log(2)))
'''
def loss_derivative(features,true_label,predict_label):
	der_loss=np.dot((predict_label-true_label),features)/len(true_label)
	return der_loss

def learning(training_set,true_label,test_set,epoch , learning_param=0.05):
	param=[0,0,0,0,0,0,0]#set all the values to zero
	loss_kop=[]
	for j in range(epoch):
		predict=model(param,training_set)
		loss=loss_derivative(training_set,true_label,predict)
		#loss_kop.append(loss_function(true_label,predict))
		param=param-learning_param*loss
	return param,loss_kop

def end_prediction(param,features):
	values=np.dot(features,param)
	classification=np.zeros(len(values))
	for i in range(len(values)):
		if sigmoid(values[i])>0.5:
			classification[i]=1
	return classification

data=np.loadtxt('GRBs.txt',skiprows=1,usecols=[2,4,5,6,7,8])
class_data=np.loadtxt('GRBs.txt',skiprows=1,usecols=3)
classification=np.zeros(len(class_data))
'''
Make the binary classification
'''
i=0
for j in class_data:
	if j>=10:
		classification[i]=1
	i+=1
'''
set the missing values to zero so they don't influence our model.
'''
data_reworked=np.copy(data)
data_reworked[np.where(data_reworked==-1)]=0

data_bias=np.insert(data_reworked,0,1,axis=1)
param,loss_kop=learning(data_bias,classification,0,10000)
test=end_prediction(param,data_bias)
#loss=loss_function(classification,end_prediction(param,data_bias))

plt.hist(test,label='prediction')
plt.hist(classification,label='true class')
plt.title('classification of GRB')
plt.legend()
plt.xlabel('class')
plt.ylabel('counts')
plt.savefig('plots/GRB.png')

