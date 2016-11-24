# Deep_neural_network

import numpy as np


class dnn():

	def __init__(self, X_train, Y_train, step_size = 0.01, reg = 0.01, h_size = [20,10], niter = 10000):

		self.X = X_train
		self.Y = Y_train

		self.step_size = step_size
		self.reg = reg
		self.h = h_size
		self.h_deep = len(h_size)+1
		self.niter = niter

		self.ndata = np.shape(self.X)[0]
		self.ndim = np.shape(self.X)[1]
		self.nclass = len(np.unique(self.Y))

		self.Y = np.zeros((self.ndata,self.nclass))
		for i in range(self.ndata):
			self.Y[i,Y_train[i]]= 1
		print self.Y

		self.ham_ini = 0.02

		self.ham = {} 
		self.const = {}


		for i in range(self.h_deep):

			if i == 0:
				size_a = self.ndim
				size_b = self.h[i]

			elif i == self.h_deep-1:
				size_a = self.h[i-1]
				size_b = self.nclass

			else:
				size_a = self.h[i-1]
				size_b = self.h[i]

			self.ham[i] = self.ham_ini * np.random.randn(size_a, size_b)
			self.const[i] = np.zeros(size_b)

			print np.shape(self.ham[i]), np.shape(self.const[i])


	def model(self):

		ham = self.ham
		const = self.const
		h_deep = self.h_deep
		loss = 3

		fstate = {}
		fstate[0] = self.X

		bstate = {}
		bstate[h_deep] = self.Y

		for i in range(self.niter):

			for j in range(h_deep):
				fstate[j+1] = np.maximum(0, np.dot(fstate[j], ham[j]) + const[j])

			#calculate loss################:
			scores = (fstate[h_deep])#np.dot(state[h_deep], ham[h_deep]) + const[h_deep]
			exp_scores = np.exp(scores)
			probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)# (np.sum(exp_scores, axis = 1).reshape(self.ndata,1)+np.ones(self.nclass))
			corect_logprobs = -np.log(np.sum(probs*self.Y,axis=1))
			data_loss = np.sum(corect_logprobs)/self.ndata

			reg_loss = 0
			for q in range(h_deep):
				reg_loss -= .5 * self.reg * np.sum((ham[q]*ham[q])) #+ .5 *self.reg*np.sum((wt2*wt2))

			loss_new = data_loss + reg_loss
			if (i%100 == 0 or i == self.niter-1):
				print 'iteration: %d, loss: %f' %(i, loss_new)
			loss = loss_new
			###############################
			###the other option:
			#dscores = probs - self.Y
			#bstate[h_deep] = dscores
			
			for k in range(h_deep-1,-1,-1):
				dham = np.dot(np.array(fstate[k]).T,bstate[k+1]) 
				dconst = np.sum(bstate[k+1], axis=0) 

				self.ham[k]  -= dham / self.reg * self.step_size / self.ndata
				self.const[k] -= dconst / self.reg * self.step_size /self.ndata
				bstate[k] = np.maximum(0,np.dot(bstate[k+1],self.ham[k].T))

			'''
			grads['wt2'] = np.dot(hidden_layer.T, dscores)
			grads['b2'] = np.sum(dscores, axis = 0)
			grads['wt2'] += self.reg * wt2
			wt2 = wt2 - self.step_size * grads['wt2']
			b2 = b2 - self.step_size * grads['b2']

			dhidden = np.dot(dscores, wt2.T)
			dhidden[hidden_layer <=0] = 0

			grads['wt'] = np.dot(self.X.T, dhidden)
			grads['b'] = np.sum(dhidden, axis = 0)
			grads['wt'] += self.reg * wt
			b =  b - self.step_size * grads['b']
			wt = wt - self.step_size * grads['wt']

			self.wt2, self.wt = wt2, wt
			self.b2, self.b = b2, b
			'''
		return loss

	def predict(self, X_test):

		ham = self.ham
		const = self.const
		h_deep = self.h_deep
		fstate = {}
		fstate[0] = X_test
		for j in range(h_deep):
			fstate[j+1] = np.maximum(0, np.dot(fstate[j], ham[j]) + const[j])

		scores = fstate[h_deep]
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
		Y_pred = scores.argmax(axis=1)#probs.argmax(axis = 1)

		return Y_pred


	def accuracy(self, Y_testset, Y_pred):

		accu_array = [1 if Y_testset[i:i+1]==Y_pred[i:i+1] else 0 for i in range(len(Y_testset)) ]
		accu = sum(accu_array)/np.double(len(Y_testset))
		return accu



if __name__=='__main__':

	Xtrain = [[1,0,2],[0,1,3],[3,1,0],[2,2,1]]
	ytrain = np.array([1,0,1,0]).T#[[1,0],[0,1],[1,0],[0,1]]
	dnn1 = dnn(Xtrain,ytrain,niter=10000)
	dnn1.model()
