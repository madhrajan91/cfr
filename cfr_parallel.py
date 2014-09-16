import numpy as np

import csv
from scipy import optimize
from mpi4py import MPI
import mpi4py.rc 
import sys

from normalizeRatings import normalizeRatings

def loadMovieList():
    #GETMOVIELIST reads the fixed movie list in movie_ids.txt and returns a
    #list of the titles
    #   movieList = GETMOVIELIST() reads the fixed movie list in movie_ids.txt
    #   and returns a list of the titles in movieList.

    movieList = []

    ## Read the fixed movieulary list
    with open('movie_ids.txt') as fid:
        for line in fid:
            movieName = line.split(' ', 1)[0].strip()
            movieList.append(movieName)

    return movieList
def lMovieList():
	with open('movie_ids.txt') as fid:
        	for line in fid:
		    #movieName = line.split(' ', 1)[0].strip()
		    movieList.append(line)

    	return movieList
def grad(theta_in, *args):
	y=args[0]
	R=args[1]
	num_movies=args[2]
	num_users=args[3]
	num_features=args[4]
	lda=args[5]


	J=0
	#CHANGE THIS
   	X = (theta_in[0:num_movies*num_features].reshape((num_features, num_movies)).T)
	Theta = (theta_in[num_movies*num_features: num_movies*num_features+num_users*num_features].reshape((num_features, num_users)).T)
        
	Jprime=np.array([0.0])
	reg_Xprime=np.array([0.0])

	nprocs=2

	comm = MPI.COMM_SELF.Spawn(sys.executable, args=['grad.py'], maxprocs=nprocs)
		
	
	N = np.array([num_movies, num_users, num_features, lda])
	

	TTheta_grad = (np.zeros((num_users, num_features)))
	XX_grad = (np.zeros((num_movies-(num_movies%nprocs), num_features)))
	if num_movies%nprocs > 0:
		T_xsize=num_movies-int(num_movies/nprocs)*nprocs;
		T = np.zeros((T_xsize, num_features))
	
	comm.Bcast([N, MPI.INT], root=MPI.ROOT)
	comm.Bcast([np.ascontiguousarray(Theta), MPI.FLOAT], root=MPI.ROOT)

	

	#print Theta
	for i in range(0,nprocs):
		if i != nprocs-1:
			Tx = (np.ascontiguousarray(X[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
			TY = (np.ascontiguousarray(y[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
			TR = (np.ascontiguousarray(R[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
		else:
			extras = num_movies%nprocs
			Tx = (np.ascontiguousarray(X[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
			TY = (np.ascontiguousarray(y[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
			TR = (np.ascontiguousarray(R[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
				
		comm.Send([Tx, MPI.FLOAT], dest=i, tag=0)
		comm.Send([TY, MPI.FLOAT], dest=i, tag=0)
		comm.Send([TR, MPI.INT], dest=i, tag=0)

	comm.Reduce(None, [Jprime, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
	comm.Reduce(None, [reg_Xprime, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
	reg_Xprime = np.array([sum((Tx**2).flatten(1))])

	reg_theta = np.asarray(Theta)**2

    	Jprime = Jprime/2 + lda/2*(np.sum(reg_theta.flatten(1)) + reg_Xprime)
	
		
	comm.Reduce(None, [TTheta_grad, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
	comm.Gather(None, [XX_grad, MPI.DOUBLE], root=MPI.ROOT)
	if num_movies%nprocs > 0:	
		comm.Recv([T, MPI.DOUBLE], source=nprocs-1,tag=1)
		XX_grad = np.concatenate((XX_grad,T))


	

	

	comm.Disconnect()
		
	theta_grad = np.squeeze(np.asarray(np.concatenate((XX_grad.flatten(1), TTheta_grad.flatten(1)))))
    
    
	return Jprime, theta_grad


'''
#Load X
reader = csv.reader(open("movies.csv", "rb"), delimiter=',')
a = list(reader)
X = np.mat(a).astype('float')
size_x = np.shape(X)
num_movies = size_x[0]
print np.shape(X)


#Load Theta
reader = csv.reader(open("users.csv", "rU"), delimiter=',')
a = list(reader)
Theta = np.mat(a).astype('float')
size_theta = np.shape(Theta)
num_users = size_theta[0]
print np.shape(Theta)

num_features = size_x[1]
'''
#Load R
reader = csv.reader(open("R_org.csv", "rU"), delimiter=',')
a = list(reader)
R = np.array(a).astype('float')
size_r = np.shape(R)
print np.shape(R)

#Load Y
reader = csv.reader(open("y_org.csv", "rU"), delimiter=',')
a = list(reader)
y = np.array(a).astype('float')
size_y = np.shape(y)
print np.shape(y)



num_features = 10



#print np.shape(X_grad)
#print X_grad


movieList = loadMovieList()

#  Initialize my ratings
my_ratings = (np.zeros(len(movieList)))
my_ratings = my_ratings[0:1682]
# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[1-1] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[98-1] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[7-1] = 3
my_ratings[12-1]= 5
my_ratings[54-1] = 4
my_ratings[64-1]= 5
my_ratings[66-1]= 3
my_ratings[69-1] = 5
my_ratings[183-1] = 4
my_ratings[226-1] = 5
my_ratings[355-1]= 5

print '\n\nNew user ratings:'
for rating, name in zip(my_ratings, movieList):
    if rating > 0:
        print 'Rated %d for %s' % (rating, name)



#y = np.concatenate((np.asmatrix(my_ratings.T), y),1)
y = np.column_stack((my_ratings, y))
R = np.column_stack(((my_ratings != 0), R))

print y[0,0], y[97,0], y[6,0]
print R[0,0], R[97,0], R[6,0]
size_y= np.shape(y)
num_movies=size_y[0]
num_users = size_y[1]


X = np.asarray(np.random.randn(num_movies, num_features));
Theta = np.asarray(np.random.randn(num_users, num_features));

#print X_grad

initial_parameters =  np.concatenate((X.flatten(1), Theta.flatten(1)))


Ynorm, Ymean = normalizeRatings(y, R)

args = (Ynorm, R, num_movies, num_users, num_features, 10)


def callback(p): sys.stdout.write('.')

#res = optimize.fmin_tnc(func=grad, x0=initial_parameters, args = args, maxfun=200)
res = optimize.minimize(grad, initial_parameters, args, method='CG',jac=True, options={'maxiter':50, 'disp':True}, callback=callback)
theta_grad = res.x
cost = res.fun
X_grad = theta_grad[0:num_movies*num_features].reshape((num_features, num_movies)).T
theta_grad = theta_grad[num_movies*num_features: num_movies*num_features+num_users*num_features].reshape((num_features, num_users)).T



p = np.dot(X_grad, theta_grad.T)
#nr = sum(R,1)
my_predictions = p[:,0] + Ymean

print max(my_predictions)

movieList = lMovieList()

ix = np.argsort(my_predictions)
print '\nTop recommendations for you:'
for j in ix[:-11:-1]:
    print 'Predicting rating {0} for movie {1}' .format(my_predictions[j], movieList[j])

print '\n\nOriginal ratings provided:'
for rating, name in zip(my_ratings, movieList):
    if rating > 0:
        print 'Rated {0} for {1}'.format(rating, name)

