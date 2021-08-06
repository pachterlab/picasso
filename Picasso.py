import torch
from torch import nn, optim
import numpy as np
import anndata
import pandas as pandas
import matplotlib.pyplot as plt
import random
from torchsummary import summary
from collections import Counter
import itertools

from scipy.optimize import linear_sum_assignment

class autoencoder(nn.Module):
	"""
	Create autoencoder architecture
	Returns: autoencoder object
    """
	def __init__(self,n_input: int, n_hidden: int, n_output: int, dropout_rate = 0.1):
		super(autoencoder,self).__init__()

		#Encoder
		self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
			#Parameter value from scVI original tensorflow implementation
			nn.BatchNorm1d(n_hidden, momentum=0.01, eps=0.001),
			nn.ReLU(True),
			nn.Dropout(p=dropout_rate),
			nn.Linear(n_hidden, n_output))

		#Linear decoder
		self.decoder = nn.Linear(n_output, n_input, bias=False)

	def forward(self, x):
		z = self.encoder(x)
		return self.decoder(z), z


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Picasso():
	"""
	Create object for fitting Picasso model
	Returns: Picasso model object
    """

	def __init__(self, n_latent = 10, n_hidden = 128, epochs = 100,batch_size = 128, lr = 1e-3, weight_decay=1e-5):
		#super(NN_NCA, self).__init__()

		#torch.manual_seed(0)

		self.n_latent = n_latent
		self.epochs = epochs
		self.n_hidden = n_hidden
		self.model = None
		self.batch_size = batch_size
		self.lr = lr
		self.weight_decay = weight_decay

		self.set_weights = False
		self.weights = None
		self.Losses = None
		self.test_losses = None

	def pairwise_dists(self,z1,z2,p=2.0):
		"""
		Parameters:
		z1 : Input matrix 1
		z2 : Input matrix 2
		p : Distance metric (1=manhattan, 2=euclidean)
		Returns :
		Pairwise distance matrix between z1 and z2
		"""
		d1 = z1.clone()
		d2 = z2.clone()
		dist = torch.cdist(d1, d2, p=p)
		#dist = torch.clamp(dist, min=0)
		return dist.clone()


	def softmax(self, p):
		"""
		Parameters:
		p : n_obs x n_obs probability matrix
		Returns :
		Softmax of matrix p
		"""
		#Based on sklearn NCA implementation

		#Subtract max prob from each row for numerical stability
		p = p.clone()
		max_prob, max_indexes = torch.max(p,dim=1,keepdim=True)
		p = p - max_prob.expand_as(p)
		p = torch.exp(p)
		sum_p = torch.sum(p,dim=1,keepdim=True)
		p = p / sum_p.expand_as(p)
		return p


	def lossFunc(self, recon_batch, X_b, z, coord_b, frac):
		"""
		Parameters:
		recon_batch : Reconstruction from decoder for mini-batch
		X_b : Mini-batch of X
		z : Latent space
		coord_b : Coordinates of desired shape
		frac : Fraction of Shape-Aware cost in loss calculation
		Returns :
		Loss value with Shape-Aware and Reconstruction loss
		"""
		#Reconstruction loss
		recon_loss_b = torch.norm(recon_batch-X_b) 


		#Boundary weights (arbitrary shape fitting)
		coord_b = torch.from_numpy(coord_b).float().to(device)
		coord_b = torch.transpose(coord_b,0, 1)

		#Calculate distances
		bound_dists = self.pairwise_dists(z,coord_b) # batch_size x batch_size

		# ---- Test task assignment solution ----

		# Convert dists to numpy
		np_dists = bound_dists.detach().cpu().numpy()

		# Use scipy.optimize.linear_sum_assignment to find matches
		row_ind, col_ind = linear_sum_assignment(np_dists)

		# Make boolean numpy array
		bools = np.full((np_dists.shape[0],np_dists.shape[1]), False)
		bools[row_ind,col_ind] = True

		# Import boolean array to torch

		bools = torch.from_numpy(bools).bool().to(device)


		# Convert to torch

		p_sum_bound = torch.sum(bound_dists*bools)
			
		


		#loss = -1*frac*(p_sum_bound) + (1-frac)*recon_loss_b  
		loss = 1*frac*(p_sum_bound) + (1-frac)*recon_loss_b  
		#loss = 1*(p_sum_bound) + 1*recon_loss_b  

		#return batch_loss
		return p_sum_bound, recon_loss_b, loss

	def getLoadings(self):
		"""
		Returns :
		Weights from the decoder layer, matrix of n_features x n_hidden
		"""
		if self.model != None:
			return self.model.decoder.weight.detach().cpu().numpy()
		else:
			return None

	def plotLosses(self, figsize=(15,4),fname=None,axisFontSize=11,tickFontSize=10):
		"""
		Parameters:
		figsize : Tuple for figure size
		fname : Name for file to save figure to, if None plot is displayed
		axisFontSize : Font size for axis labels
		tickFontSize : Font size for tick labels
		Returns :
		Plot of each loss term over epochs
		"""
		fig, axs = plt.subplots(1, self.Losses.shape[1],figsize=figsize)
		titles = ['Boundary Fit','Reconstruction','Total Loss']
		if(isinstance(self.test_losses, np.ndarray)):

			for i in range(self.Losses.shape[1]):
				axs[i].plot(self.Losses[:,i],label='Train Loss')
				axs[i].plot(self.test_losses[:,i],label='Test Loss')
				axs[i].set_title(titles[i],fontsize=axisFontSize)
				plt.setp(axs[i].get_xticklabels(), fontsize=tickFontSize)
				plt.setp(axs[i].get_yticklabels(), fontsize=tickFontSize)

				axs[i].grid(False)
			plt.legend(prop={'size': axisFontSize})
			plt.xlabel('Epoch',fontsize=axisFontSize)
			plt.ylabel('Loss',fontsize=axisFontSize)

		else:
			for i in range(self.Losses.shape[1]):
				axs[i].plot(self.Losses[:,i])
				axs[i].set_title(titles[i],fontsize=axisFontSize)
				plt.setp(axs[i].get_xticklabels(), fontsize=tickFontSize)
				plt.setp(axs[i].get_yticklabels(), fontsize=tickFontSize)

				axs[i].grid(False)

			plt.xlabel('Epoch',fontsize=axisFontSize)
			plt.ylabel('Loss',fontsize=axisFontSize)


		fig.tight_layout()
		if(fname != None):
			plt.savefig(fname)
		else:
			plt.show()




	def fit(self, X, coords, frac = 0.8, silent = False, ret_loss = False, summ = False):
		"""
		Parameters:
		X : Input data as numpy array (obs x features)
		coords : Shape coordinates (dimension x obs)
		frac : Fraction of Shape-Aware cost in loss calculation (default is 0.8)
		silent : Print average loss per epoch (default is False)
		ret_loss : Boolean to return loss values over epochs
		summ : Boolean to return summary of neural network

		Returns :
		Latent space representation of X
		"""

		iters_per_epoch = int(np.ceil(X.shape[0] / self.batch_size))

		model = autoencoder(X.shape[1], self.n_hidden, self.n_latent).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

		#Print model summary
		if summ:
			print("Num Parameters: "+str(sum([param.nelement() for param in model.parameters()])))
			summary(model, (self.batch_size,X.shape[1]), self.batch_size)

		X = torch.from_numpy(X).float().to(device)
	
		loss_values = []
		for e in range(self.epochs):

			#Shuffle data
			permutation = torch.randperm(X.size()[0])

			model.train()
			allLosses = torch.tensor(0,device=device)

			with torch.autograd.set_detect_anomaly(True):
				for b in range(iters_per_epoch):

					indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
					X_b, coord_b = X[indices], coords 

					#Set grad to zero, compute loss, take gradient step
					optimizer.zero_grad()
					recon_batch, z = model(X_b)


					losses  = self.lossFunc(recon_batch, X_b, z, coord_b, frac) #*****

					
					losses[-1].backward()

					allLosses = allLosses + torch.stack(losses,dim=0)

					optimizer.step()

			if silent != True:
				print('====> Epoch: {} Average loss: {:.4f}'.format(e, allLosses[-1].item() / len(X)))

			loss_values.append([allLosses[i].item() / len(X) for i in range(len(allLosses))])



		model.eval()
		recon_batch, z = model(X)
		self.model = model
		self.Losses = np.array(loss_values)
		if ret_loss:
			return np.array(loss_values), z.detach().cpu().numpy()
		else:
			return z.detach().cpu().numpy()


	def trainTest(self,X,coords, trainFrac = 0.8, frac = 0.8, silent = False):
		"""
		Parameters:
		X : Input data as numpy array (obs x features)
		coords : Shape coordinates (dimension x obs)
		trainFrac : Fraction of X to use for training
		frac : Fraction of Shape-Aware cost in loss calculation (default is 0.8)
		silent : Print average loss per epoch (default is False)

		Returns :
		Loss values from training and validation batches of X
		"""


		trainSize = int(np.floor(trainFrac*X.shape[0]))
		trainInd = random.sample(range(0,X.shape[0]), trainSize) 
		testInd = [i not in trainInd for i in range(0,X.shape[0])]

		X_train = X[trainInd,:]


		X_test = X[testInd,:]


		#print(X.shape)
		iters_per_epoch = int(np.ceil(X_train.shape[0] / self.batch_size))

		model = autoencoder(X_train.shape[1], self.n_hidden, self.n_latent).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


		X_train = torch.from_numpy(X_train).float().to(device)
		X_test = torch.from_numpy(X_test).float().to(device)
		#print(X.size())
		loss_values = []
		test_loss_values = []
		for e in range(self.epochs):

			#Shuffle data
			permutation = torch.randperm(X_train.size()[0])

			model.train()
			allLosses = torch.tensor(0,device=device) 

			with torch.autograd.set_detect_anomaly(True):
				for b in range(iters_per_epoch):

					#Choose batch

					indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
					X_b, coord_b = X_train[indices], coords[:,random.sample(range(0, self.batch_size), len(indices))]
					

					#Set grad to zero, compute loss, take gradient step
					optimizer.zero_grad()
					recon_batch, z = model(X_b)
					losses  = self.lossFunc(recon_batch, X_b, z, coord_b, frac) #*****

					#Get NCA and recons. cost values
					#ncaLoss, reconLoss  = self.getLossParts(loss, recon_batch, X_b, z, masks,weights,cont, lab_weights, frac)

					losses[-1].backward()

					allLosses = allLosses + torch.stack(losses,dim=0)
					optimizer.step()



			test_losses = self.test(model, X_test, coords, frac = frac, silent = silent)
			
			if silent != True:
				print('====> Epoch: {} Average loss: {:.4f}'.format(e, allLosses[-1] / len(X_train)))

			loss_values.append([allLosses[i].item() / len(X_train) for i in range(len(allLosses))])
			test_loss_values.append(test_losses)

		self.Losses = np.array(loss_values)
		self.test_losses = np.array(test_loss_values)
		return np.array(loss_values), np.array(test_loss_values)


	def test(self, model, X, coords, frac = 0.8, silent = False):
			

		#Shuffle data
		permutation = torch.randperm(X.size()[0])
		iters_per_epoch = int(np.ceil(X.size()[0] / self.batch_size))

		model.eval()
		allLosses = torch.tensor(0,device=device) 

		with torch.no_grad():

			for b in range(iters_per_epoch):

				#Choose batch
				indices = permutation[b*self.batch_size:(b+1)*self.batch_size]
				X_b, coord_b = X[indices], coords[:,random.sample(range(0, self.batch_size), len(indices))]

			
				#Set grad to zero, compute loss, take gradient step
				recon_batch, z = model(X_b)
				losses = self.lossFunc(recon_batch, X_b, z, coord_b, frac)

				#Get NCA and recons. cost values
				#ncaLoss, reconLoss  = self.getLossParts(loss, recon_batch, X_b, z, masks, weights, cont, lab_weights, frac)

				
				allLosses = allLosses + torch.stack(losses,dim=0)


		test_loss = allLosses[-1]/len(X)

		if silent != True:
			print('====> Test set loss: {:.4f}'.format(test_loss))


		return [allLosses[i].item() / len(X) for i in range(len(allLosses))]

