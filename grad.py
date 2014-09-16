#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

N = np.array([0,0,0,0])


comm.Bcast([N, MPI.INT], root=0)

num_movies=N[0]
num_users=N[1]
num_features=N[2]
lda = N[3]


Theta = (np.zeros((num_users, num_features)))
comm.Bcast([Theta, MPI.FLOAT], root=0)


if rank !=size-1:
	Tx = (np.zeros((int(num_movies/size), num_features)))
	TY = (np.zeros((int(num_movies/size), num_users)))
	TR = (np.zeros((int(num_movies/size), num_users)))
	size_tx = int(num_movies/size)
else:
	Tx = (np.zeros((int(num_movies/size+num_movies%size), num_features)))
	TY = (np.zeros((int(num_movies/size+num_movies%size), num_users)))
	TR = (np.zeros((int(num_movies/size+num_movies%size), num_users)))
	size_tx = int(num_movies/size+num_movies%size)

comm.Recv(Tx, source=0, tag=0)
comm.Recv(TY, source=0, tag=0)
comm.Recv(TR, source=0, tag=0)

#print np.shape(Tx), ' ', np.shape(TY), ' ', np.shape(TR), ' ', np.shape(Theta)

mat = np.dot(Tx,Theta.T)-TY

mat_sq = mat**2
temp = mat_sq*TR
 
Jprime = np.array([sum(temp.flatten(1))])
reg_Xprime = np.array([sum((Tx**2).flatten(1))])


comm.Reduce([Jprime, MPI.DOUBLE], None, op=MPI.SUM, root=0)
comm.Reduce([reg_Xprime, MPI.DOUBLE], None,op=MPI.SUM, root=0)



temp = mat*TR


lx = Tx * lda 
ltheta = Theta*lda

XX_grad = np.dot(temp, Theta) + lx #420 * 10  So send and concatenate at the master
TTheta_grad = np.dot(temp.T, Tx) +ltheta # 943 * 10 accross each individual processor so we have to accumulate it


comm.Reduce([TTheta_grad, MPI.DOUBLE], None, op=MPI.SUM, root=0)
comm.Gather([XX_grad, MPI.DOUBLE], None, root=0) #GATHER DOES NOT WORK SINCE THE SIZES ARE 420 420 420 422 , so the last two elements are not sent by gather
if num_movies%size >0:
	#print XX_grad
	extra_x,extra_y = np.shape(XX_grad)
	T = XX_grad[extra_x - extra_x%size:extra_x, :]
	comm.Send([T, MPI.DOUBLE], dest=0, tag=1)


comm.Disconnect()
