import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scip
import astropy.stats as ast_stat
import scipy
class RNG_class:
	seed = 1923
	state= seed           
	def MWC(self,number):
		number=np.int64(number)
		number=4294957665*(number&(2**32-1))+number>>32
		return number

	def bit_shift_64(self,number):
		number= np.int64(number)
		number= number^(number>>21)
		number = number^(number<<35)
		number = number^(number>>4)
		return number

	def LCG(self, value):
		newvalue = (22695477*value + 1 )% 2**32
		return newvalue
		
	def generator(self):
		state=self.bit_shift_64(self.state)
		state=self.MWC(self.state)
		state=self.LCG(self.state)
		self.state = state
		return (state%2**32)/(2 **32)#only using the lowest 32 bits and divide it by 2**32 to get a number between 0 and 1.
'''
Using a recursive algorithm to split up the sequence in left and right halves, after this it searches to combine the sequences to a sorted one recursively O(NlogN)

'''

def merge_sorting_recursive(sequence):
	if len(sequence)==1:
		return sequence
	elif len(sequence)>1:
		sorted_sequence=[]
		half=int(len(sequence)/2)
		left=sequence[:half]
		right=sequence[half:]
		left=merge_sorting_recursive(left)
		right=merge_sorting_recursive(right)	
		i=0
		j=0
		while i<len(left) and j<len(right):
			if left[i]<right[j]:
				sorted_sequence.append(left[i])
				i+=1
			else:
				sorted_sequence.append(right[j])
				j+=1 
		if i==len(left):
			sorted_sequence += right[j:]			
		else:		
			sorted_sequence += left[i:]
		return sorted_sequence

def Box_Muller(mu,sigma,number=10000):
	random_numbers=[]
	if number>1:
		for j in range(int(number/2)):	
			u_1=RNG.generator()
			u_2=RNG.generator()
			z_1=sigma*np.sqrt(-2*np.log(u_1))*np.cos(2*np.pi*u_2)+mu
			z_2=sigma*np.sqrt(-2*np.log(u_1))*np.sin(2*np.pi*u_2)+mu
			random_numbers.append(z_1)
			random_numbers.append(z_2)
	else:
		u_1=RNG.generator()
		u_2=RNG.generator()
		z_1=sigma*np.sqrt(-2*np.log(u_1))*np.cos(2*np.pi*u_2)+mu
		z_2=sigma*np.sqrt(-2*np.log(u_1))*np.sin(2*np.pi*u_2)+mu
		random_numbers.append(z_1)
		random_numbers.append(z_2)
	return random_numbers
'''
def Box_Muller_2(mu,sigma,random_numbers):
	half=int(len(random_numbers))/2
	u_1=random_numbers[:half]
	u_2=random_numbers[half:]
	z_1=sigma*np.sqrt(-2*np.log(u_1))*np.cos(2*np.pi*u_2)+mu
	z_2=sigma*np.sqrt(-2*np.log(u_1))*np.sin(2*np.pi*u_2)+mu
	return np.concatenate([z_1,z_2])
'''	
def KStest(numbers,CDF_numbers):
	length=len(numbers)
	distances=[]
	for j in range(length):
		distances.append(abs(CDF_numbers[j]-(j+1)/length))
	max_dist=0
	for distance in distances:
		if distance > max_dist:
			max_dist=distance
	z= (np.sqrt(length)+0.12+0.11/np.sqrt(length))*max_dist
	if z<1.18:
		p_value=np.sqrt(2*np.pi)/z*(np.exp(-np.pi**2/(8*z**2))+np.exp(-9*np.pi**2/(8*z**2))+np.exp(-25*np.pi**2/(8*z**2)))
	else:
		p_value=1-2*(np.exp(-2*z**2)-np.exp(-8*z**2)+np.exp(-18*z**2))
	return max_dist,1-p_value

def CDF_order_2(numbers,integrals):
	sort_numbers=merge_sorting_recursive(numbers)
	length=len(sort_numbers)
	CDF_numbers=np.zeros(length)
	for j in range(length):
		CDF_numbers[j]=CDF_interpolation(sort_numbers[j],integrals[:,0],integrals[:,1])
	return sort_numbers,CDF_numbers

def Romberg_integration(function,lower,upper,order=8):
	d=2
	Int_table=np.zeros((order,order))
	h=upper-lower
	Int_table[0,0]=h*(function(upper)+function(lower))/2
	for j in range(1,order):
		h/=d
		Int_table[j,0] = Int_table[j-1,0]/2
		Int_table[j,0] += h * np.sum([function(lower + i * h) for i in range(1, 2 ** j + 1, 2)])

		for k in range(1,j+1):
			Int_table[j,k]=((4**(k))*Int_table[j,k-1]-Int_table[j-1,k-1])/(4**(k)-1)
	return Int_table[order-1,order-1]

def Kuipers_test(numbers,CDF_numbers):
	z=CDF_numbers
	length=len(numbers)
	D_plus=0
	D_min=0
	for j in range(length):
		if j/length-z[j]>D_plus:	
			D_plus=j/length-z[j]
		if z[j]-(j-1)/length>D_min:
			D_min=z[j]-(j-1)/length
	V = D_plus+D_min	
	z=(length**0.5+0.155+0.24/(length**0.5))*V
	if z<0.4:
		return V,1
	else:
		QV=np.zeros(50)
		for j in range(1,50):
			QV[j]=QV[j-1]+2*(4*j**2*z**2-1)*np.exp(-2*j**2*z**2)
			if QV[j]-QV[j-1]<1e-10:
				break
		return V,QV[j]

def KS_test_2(numbers_1,numbers_2):
	'''
	Sort both sequences then do a double for loop to get the empricial CDF distances, if we do not get to the end of 1 of the distributions, we take the length of the missing numbers and that is then 	also added to the distance metric. Finally we compute the p-value with our statistic, this changes now as we have a different definition of length compared to the normal KS-test.
	'''
	sort_numbers_1=merge_sorting_recursive(numbers_1)
	sort_numbers_2=merge_sorting_recursive(list(numbers_2))
	max_dist=0	
	length_1=len(sort_numbers_1)
	length_2=len(sort_numbers_2)
	l=0
	dists=[]
	for ind_1 in range(0,length_1):
		for ind_2 in range(l,length_2):
			if sort_numbers_2[ind_2]>=sort_numbers_1[ind_1]:
				dists.append(np.abs(ind_2+1-(ind_1+1)))
				if ind_2-1>0:
					l=ind_2-1
				else:
					l=ind_2
				break
	if (len(dists))<length_1:
		dists.append(length_1-len(dists))
	max_dist=float(np.max(dists))/length_1
	length=length_1*length_2/(length_1+length_2)
	z= (np.sqrt(length)+0.12+0.11/np.sqrt(length))*max_dist
	if z<1.18:
		p_value=np.sqrt(2*np.pi)/z*(np.exp(-np.pi**2/(8*z**2))+np.exp(-9*np.pi**2/(8*z**2))+np.exp(-25*np.pi**2/(8*z**2)))
	else:
		p_value=1-2*(np.exp(-2*z**2)-np.exp(-8*z**2)+np.exp(-18*z**2))
	return max_dist,1-p_value


def CDF_interpolation(point,points_x,CDF_y):
	if point>=-5 and point<=5:
		j=int((point+5)*1000)
		return CDF_y[j-1]+(point-points_x[j-1])*(CDF_y[j]-CDF_y[j-1])/(points_x[j]-points_x[j-1])
	elif point<-5:
		return  0.5-Romberg_integration(lambda x:np.exp(-x**2),point/np.sqrt(2),0,order=8)/np.sqrt(np.pi)
	else:
		return  0.5+Romberg_integration(lambda x:np.exp(-x**2),0,point/np.sqrt(2),order=8)/np.sqrt(np.pi)

RNG=RNG_class()
print('The seed is set at', RNG.seed)
'''
Generate our RNG class and 1 million random numbers from a uniform distribution
'''
rng=[RNG.generator() for j in range(1000000)]
rng=np.array(rng)
plt.scatter(rng[:1000],rng[1:1001])
plt.xlabel('random number i')
plt.ylabel('random number i+1')
plt.title('distribution of our rng')
plt.savefig('plots/relative_distrng.png')
plt.clf()

plt.scatter(np.linspace(0,999,1000),rng[:1000])
plt.xlabel('index i')
plt.ylabel('random number i')
plt.title('distribution of first 1000 numbers')
plt.savefig('plots/numbers1000.png')
plt.close()

plt.hist(rng,bins=np.linspace(0,1,21))
#plt.axhline(50000-2*np.sqrt(50000),linestyle='dashed')
#plt.axhline(50000+2*np.sqrt(50000),linestyle='dashed')
plt.xlabel('numbers generated')
plt.ylabel('relative counts')
plt.savefig('plots/distribution.png')
plt.clf()



x=np.linspace(3-5*2.4,3+5*2.4,31)
y=np.linspace(3-4*2.4,3+4*2.4,9)
BM=Box_Muller(3,2.4)
BM=np.array(BM)
plt.hist(BM,bins=x,density=True,label='Box_Muller method')
for j in y:
	plt.axvline(j,linestyle='dashed')
plt.plot(x,(1/np.sqrt(2*np.pi*2.4**2))*np.exp(-((x-3)**2)/(2*2.4**2)),label='True Gaussian')
plt.xlabel('generated numbers')
plt.ylabel('relative counts')
plt.legend()
plt.title('Box Muller compared to a Gaussian')
plt.savefig('plots/Box_muller.png')
plt.clf()

BM_2=Box_Muller(0,1,100000)
KS_numbers=[]
KS_test_scip=[]
Kuipers=[]
Kuipers_astr=[]
Kuipers_2=[]
integrals=np.zeros((10001,2))
count=0

'''
Taking the integrals we use for a linear interpolation for our cdf for the error function

'''
for k in np.linspace(-5,5,10001):
	if k<0:
		integrals[count]=k,0.5-Romberg_integration(lambda x:np.exp(-x**2),k/np.sqrt(2),0,order=8)/np.sqrt(np.pi)
	else: 
		integrals[count]=k,0.5+Romberg_integration(lambda x:np.exp(-x**2),0,k/np.sqrt(2),order=8)/np.sqrt(np.pi)
	count+=1

'''
Loop over the numbers and do the Kuipers and the KS-test at the same time, we loose some efficiency here as we calculate the same CDF multiple times, but we do not have enough time to fix this inefficiency.
'''
for j in np.logspace(1,5,41):
	BM=BM_2[:int(j)]
	test,test_2=CDF_order_2(BM,integrals)
	KS_numbers.append(KStest(test,test_2))
	KS_test_scip.append(scip.kstest(test,'norm',N=int(j)))
	Kuipers.append(Kuipers_test(test,test_2))
	Kuipers_astr.append(ast_stat.kuiper(BM,cdf=lambda y: 0.5*(1+scipy.special.erf(y/2**(0.5)))))
	Kuipers_2.append((ast_stat.kuiper(BM,cdf=lambda y:test_2)))
	
Kuipers=np.asarray(Kuipers)
Kuipers_astr=np.asarray(Kuipers_astr)
Kuipers_2=np.asarray(Kuipers_2)
KS_numbers=np.array(KS_numbers)
KS_test_scip=np.array(KS_test_scip)
x_range=np.logspace(1,5,41)
plt.plot(x_range,KS_numbers[:,0],label='own test')
plt.plot(x_range,KS_test_scip[:,0],label='scipy test')
plt.xscale('log')
plt.title('KS-test')
plt.xlabel('#random numbers')
plt.ylabel('statistic')
plt.legend()
plt.savefig('plots/KS_Teststat.png')
plt.clf()
plt.plot(x_range,KS_numbers[:,1],label='own test')
plt.plot(x_range,KS_test_scip[:,1],label='scipy test')
plt.xscale('log')
plt.title('KS-test')
plt.xlabel('#random numbers')
plt.ylabel('p-value')
plt.legend()
plt.savefig('plots/KS_Testp.png')
plt.clf()

plt.plot(x_range,Kuipers[:,0],label='own test')
plt.plot(x_range,Kuipers_2[:,0],label='astropy test')
plt.xscale('log')
plt.title('Kuipers-test')
plt.xlabel('#random numbers')
plt.ylabel('statistic')
plt.legend()
plt.savefig('plots/Kuipers_Teststat.png')
plt.clf()
plt.plot(x_range,Kuipers[:,1],label='own test')
plt.plot(x_range,Kuipers_2[:,1],label='astropy test')
plt.xscale('log')
plt.title('Kuipers-test')
plt.xlabel('#random numbers')
plt.ylabel('p-value')
plt.legend()
plt.savefig('plots/Kuipers_p.png')
plt.clf()

'''
Loading the numbers and then loop over each set of random numbers to make the same plot as before, it is tested against the scipy library function.
'''
Random=np.loadtxt('randomnumbers.txt')
tested_values=np.zeros((10,41,2))
scip_values=np.zeros((10,41,2))
for i in range(0,10):
	l=0
	for j in (np.logspace(1,5,41)):
		k=int(j)
		#tested_values.append(KS_test_2(BM_2[:k],Random[:,i][:k]))
		tested_values[i,l]=KS_test_2(BM_2[:k],Random[:,i][:k])
		scip_values[i,l]=scip.ks_2samp(BM_2[:k],Random[:,i][:k])
		l+=1
'''
Making the 10 plots for the random numbers, compared to the test from scipy. We have a plot for both the p-value and the statistic.
'''
for j in range(0,10):
	plt.plot(x_range,tested_values[j,:,0],label='own test')
	plt.plot(x_range,scip_values[j,:,0],label='scipy test')
	plt.xscale('log')
	plt.ylabel('statistic')
	plt.xlabel('#random numbers')
	plt.title('random set {0}'.format(j))
	plt.legend()
	plt.savefig('plots/two_sided_ks{0}.png'.format(j)) 
	plt.close()
	plt.plot(x_range,tested_values[j,:,1],label='own test')
	plt.plot(x_range,scip_values[j,:,1],label='scipy test')
	plt.xscale('log')
	plt.ylabel('p-value')
	plt.xlabel('#random numbers')
	plt.title('random set {0}'.format(j))
	plt.legend()
	plt.savefig('plots/two_sided_ks_pvalue{0}.png'.format(j)) 
	plt.close()

