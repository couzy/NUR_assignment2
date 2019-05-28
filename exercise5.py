import numpy as np
import matplotlib.pyplot as plt

def NGP(particles,mass=1):
	mass_grid=np.zeros((16,16,16))
	for j in particles:
		grid_points=int(j[0]),int(j[1]),int(j[2])
		mass_grid[grid_points]+=1*mass
	return mass_grid

def CIC(particles,mass=1):
	mass_grid=np.zeros((16,16,16))	
	for j in particles:
		x,y,z=int(j[0]),int(j[1]),int(j[2])
		dists=j[0]-(int(j[0])),j[1]-int(j[1]),j[2]-int(j[2])
		steps=np.array([1,1,1])-dists
		mass_grid[x,y,z]+=mass*steps[0]*steps[1]*steps[2]
		mass_grid[x,(y+1)%16,z]+=mass*steps[0]*steps[2]*dists[1]
		mass_grid[(x+1)%16,(y+1)%16,z]+=mass*dists[0]*steps[2]*dists[1]		
		mass_grid[x,y,(z+1)%16]+=mass*steps[0]*steps[1]*dists[2]
		mass_grid[(x+1)%16,y,(z+1)%16]+=mass*dists[0]*steps[1]*dists[2]		
		mass_grid[x,(y+1)%16,(z+1)%16]+=mass*steps[0]*dists[2]*dists[1]
		mass_grid[(x+1)%16,y,z]+=mass*steps[2]*steps[1]*dists[0]
		mass_grid[(x+1)%16,(y+1)%16,(z+1)%16]+=mass*dists[0]*dists[2]*dists[1]
	return mass_grid
	

def CIC_2(particles,mass=1):
	mass_grid=np.zeros((16,16,16))	
	for j in particles:
		x,y,z=int(j[0]),int(j[1]),int(j[2])
		dists=j[0]-(int(j[0])+0.5),j[1]-int(j[1]+0.5),j[2]-int(j[2]+0.5)
		if j[0]>(int(j[0])+0.5):
			k_x=1
		else:
			k_x=-1
		if j[1]>(int(j[1])+0.5):
			k_y=1
		else:
			k_y=-1
		if j[2]>(int(j[2])+0.5):
			k_z=1
		else:
			k_z=-1
		steps=np.array([0.5,0.5,0.5])-dists
		mass_grid[x,y,z]+=mass*steps[0]*steps[1]*steps[2]
		mass_grid[x,(y+k_y)%16,z]+=mass*steps[0]*steps[2]*dists[1]
		mass_grid[(x+k_x)%16,(y+k_y)%16,z]+=mass*dists[0]*steps[2]*dists[1]		
		mass_grid[x,y,(z+k_z)%16]+=mass*steps[0]*steps[1]*dists[2]
		mass_grid[(x+k_x)%16,y,(z+k_z)%16]+=mass*dists[0]*steps[1]*dists[2]		
		mass_grid[x,(y+k_y)%16,(z+k_z)%16]+=mass*steps[0]*dists[2]*dists[1]
		mass_grid[(x+k_x)%16,y,z]+=mass*steps[2]*steps[1]*dists[0]
		mass_grid[(x+k_x)%16,(y+k_y)%16,(z+k_z)%16]+=mass*dists[0]*dists[2]*dists[1]
	return mass_grid

def make_power_2(number):
	count=0
	while number>1:
		number=number/2
		count+=1
	return 2**count
'''
A recursive FFT-algorithm, which does not make use of bit-reversal. We keep slicing the array in half and then take the FFT of this in an even and odd half of the array.
stop whn we are at a single point.
'''
def FFT_recursive(points,IFT=False):
	"""
	Cooley-Tukkey algorithm
	"""
	if IFT==True:
		ift=-1
	else:
		ift=1
	points=np.asarray(points)
	length=len(points)
	half=int(length/2)
	if length==1:
		return [points[0]]
	else:	
		FFT_even=FFT_recursive(points[::2],IFT)
		FFT_odd=FFT_recursive(points[1::2],IFT)
		transform_factor=np.exp(-2j*ift*np.pi*np.arange(half)/length)*FFT_odd
		FFT_odd=FFT_even-transform_factor
		FFT_even=FFT_even+transform_factor
		return  np.concatenate([FFT_even,FFT_odd])
'''
Makes a 2D-Fourier Transform, first loop over the columns, then loop over the rows, and make use of our 1D-FFT algorithm
'''

def FFT_2D(array_points,IFT=False):
	points=np.asarray(array_points,dtype=np.complex128)
	FFT_2=np.copy(points)
	for i in range(np.shape(points)[0]):
		FFT_2[:,i]=FFT_recursive(points[:,i],IFT)
	for j in range(np.shape(points)[1]):
		FFT_2[j,:]=FFT_recursive(FFT_2[j,:],IFT)
	return FFT_2
	

'''
Creates a 3D-FFT, first loops over all the y-z slices, and then over all x-elements to obtain a 3D-FFT.
'''
def FFT_3D(array_points,IFT=False):
	points=np.asarray(array_points,dtype=np.complex128)
	FFT_3=np.copy(points)
	for i in range(np.shape(points)[0]):
		FFT_3[i,:,:]=FFT_2D(points[i,:,:])
	for j in range(np.shape(points)[1]):
		for k in range(np.shape(points)[2]):
			FFT_3[:,j,k]=FFT_recursive(FFT_3[:,j,k],IFT)
	return FFT_3

'''
Calculate the potential, first take the CIC mass assignment scheme, then normalize the density field, then take the 3D-FFT.
Now we divide each FFT by the length of k-vector squared. After this we take the inverse of the 3D FFT to get back to the potential.
This can be done faster if it would be vectorized, but the triple for loop for a 16x16x16 seems to do the trick now.
'''

def Potential_calculation(positions):
	density=CIC(positions)
	Fourier_density_field=FFT_3D(((density)-np.mean(density))/np.mean(density))
	for i in range(0,16):
		if i<=8:
			k_x=2*np.pi*i/16
		else:
			k_x=-2*np.pi*(-16+i)/16
		for j in range(0,16):
			if j<=8:
				k_y=2*np.pi*j/16
			else:
				k_y=-2*np.pi*(-16+j)/16
			for k in range(0,16):
				if k<=8:
					k_z=2*np.pi*k/16
				else:
					k_z=-2*np.pi*(-16+k)/16
				if i!=0 and j!=0 and k!=0:
					Fourier_density_field[i,j,k]=Fourier_density_field[i,j,k]/(k_x**2+k_y**2+k_z**2)

	potential=FFT_3D(Fourier_density_field,True)
	return potential		
'''
Calculates the gradient at each point of the grid of the potential we use the midpoint rule to calculate the gradient, and use the np.roll function to easily shift our 

'''
def gradient_potential(potential):
	grad=np.zeros((16,16,16,3))
	grad[:,:,:,0]=(np.roll(potential,1,axis=0)-np.roll(potential,-1,axis=0))/2
	grad[:,:,:,1]=(np.roll(potential,1,axis=1)-np.roll(potential,-1,axis=1))/2
	grad[:,:,:,2]=(np.roll(potential,1,axis=2)-np.roll(potential,-1,axis=2))/2
	return grad

def Inverse_CIC(positions,density_field):		
	grad=gradient_potential(density_field)
	grad_x=grad[:,:,:,0]
	grad_y=grad[:,:,:,1]
	grad_z=grad[:,:,:,2]
	grad_particles=[]
	for x in positions:
	
		i,j,k=int(x[0]),int(x[1]),int(x[2])
		dists=x[0]-(int(x[0])),x[1]-int(x[1]),x[2]-int(x[2])
		steps=np.array([1,1,1])-dists

	
		grad_particle_x=grad_x[i,j,k]*steps[0]*steps[1]*steps[2] + grad_x[(i+1)%16,j,k]*dists[0]*steps[1]*steps[2]+grad_x[i,(j+1)%16,k]*steps[0]*dists[1]*steps[2]+grad_x[i,j,(k+1)%16]*steps[0]*steps[1]*dists[2]+grad_x[(i+1)%16,(j+1)%16,k]*dists[0]*dists[1]*steps[2]+ grad_x[(i+1)%16,j,(k+1)%16]*dists[0]*steps[1]*dists[2]+grad_x[i,(j+1)%16,(k+1)%16]*steps[0]*dists[1]*dists[2]+grad_x[(i+1)%16,(j+1)%16,(k+1)%16]*dists[0]*dists[1]*dists[2]
		grad_particle_y=grad_y[i,j,k]*steps[0]*steps[1]*steps[2] +grad_y[(i+1)%16,j,k]*dists[0]*steps[1]*steps[2]+grad_y[i,(j+1)%16,k]*steps[0]*dists[1]*steps[2]+grad_y[i,j,(k+1)%16]*steps[0]*steps[1]*dists[2]+grad_y[(i+1)%16,(j+1)%16,k]*dists[0]*dists[1]*steps[2]+grad_y[(i+1)%16,j,(k+1)%16]*dists[0]*steps[1]*dists[2]+grad_y[i,(j+1)%16,(k+1)%16]*steps[0]*dists[1]*dists[2]+grad_y[(i+1)%16,(j+1)%16,(k+1)%16]*dists[0]*dists[1]*dists[2]
		grad_particle_z=grad_z[i,j,k]*steps[0]*steps[1]*steps[2] +grad_z[(i+1)%16,j,k]*dists[0]*steps[1]*steps[2]+grad_z[i,(j+1)%16,k]*steps[0]*dists[1]*steps[2]+grad_z[i,j,(k+1)%16]*steps[0]*steps[1]*dists[2]+grad_z[(i+1)%16,(j+1)%16,k]*dists[0]*dists[1]*steps[2]+grad_z[(i+1)%16,j,(k+1)%16]*dists[0]*steps[1]*dists[2]+grad_z[i,(j+1)%16,(k+1)%16]*steps[0]*dists[1]*dists[2]+grad_z[(i+1)%16,(j+1)%16,(k+1)%16]*dists[0]*dists[1]*dists[2]
		grad_particles.append([grad_particle_x,grad_particle_y,grad_particle_z])
	return np.asarray(grad_particles)

np.random.seed(121)
positions = np.random.uniform( low=0, high =16, size =(1024 ,3))

grid=NGP(positions)

'''
Making the plots for the NGP method for z=4,9,11,14 
'''

plt.imshow(grid[:,:,4])
plt.title('z=4 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/NGP_4.png')
plt.close()
plt.imshow(grid[:,:,9])
plt.title('z=9 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/NGP_9.png')
plt.close()
plt.imshow(grid[:,:,11])
plt.title('z=11 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/NGP_11.png')
plt.close()
plt.imshow(grid[:,:,14])
plt.title('z=14 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/NGP_14.png')
plt.clf()

'''
Chechking the robustness of our NGP method

'''

x=np.linspace(0,15.9,1601)
NGPoints_4=np.zeros(1601)
NGPoints_0=np.zeros(1601)
j=0
for k in x:
	NGPoints_4[j]=NGP([[k,0,0],[14,13,12]])[4,0,0]
	NGPoints_0[j]=NGP([[k,0,0],[14,13,12]])[0,0,0]
	j+=1

plt.plot(x,NGPoints_4)
plt.xlabel('x_value 1 particle')
plt.ylabel('mass in cell_4')
plt.title('NGP-checker')
plt.savefig('plots/NGP4.png')
plt.close()

plt.plot(x,NGPoints_0)
plt.xlabel('x_value 1 particle')
plt.ylabel('mass in cell_0')
plt.title('NGP-checker')
plt.savefig('plots/NGP0.png')
plt.close()

'''
Making a grid for the CIC mass assignment scheme.
'''
second_grid=CIC(positions)

'''
Making the plots for the mass assignment scheme
'''
plt.imshow(second_grid[:,:,4])
plt.title('z=4 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/CIC_4.png')
plt.close()
plt.imshow(second_grid[:,:,9])
plt.title('z=9 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/CIC_9.png')
plt.close()
plt.imshow(second_grid[:,:,11])
plt.title('z=11 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/CIC_11.png')
plt.close()
plt.imshow(second_grid[:,:,14])
plt.title('z=14 slice')
plt.xlabel('x grid')
plt.ylabel('y grid')
plt.savefig('plots/CIC_14.png')
plt.clf()

CICpoints_4=np.zeros(1601)
CICpoints_0=np.zeros(1601)
j=0
for k in x:
	CICpoints_4[j]=CIC([[k,0,0],[14,13,12]])[4,0,0]
	CICpoints_0[j]=CIC([[k,0,0],[14,13,12]])[0,0,0]
	j+=1

'''
Plot for cell 0 and cell 4 for teh CIC mass assignment scheme
'''

plt.plot(x,CICpoints_4)
plt.xlabel('x_value 1 particle')
plt.ylabel('mass in cell_4')
plt.title('CIC-checker')
plt.savefig('plots/CICcell_4.png')
plt.close()

plt.plot(x,CICpoints_0)
plt.xlabel('x_value 1 particle')
plt.ylabel('mass in cell_0')
plt.title('CIC-checker')
plt.savefig('plots/CICcell0.png')
plt.close()
'''
Plotting our own FFT against the numpy version
'''

x_one=np.ones(64)
k_x=np.linspace(-31*np.pi/64,32*np.pi/64,64)
plt.plot(k_x,np.fft.fftshift(FFT_recursive(x_one).real),label='own FFT')
plt.plot(k_x,np.fft.fftshift(np.fft.fft(x_one).real),label='numpy FFT',linestyle='dashed')
plt.xlabel('k_x')
plt.ylabel('FFT')
plt.title('Real part of our FFT and the numpy one')
plt.legend()
plt.savefig('plots/FFT1D.png')
plt.close()



'''
Compare numpy to our own 2D FFT
'''

x_ones=np.ones((64,64))
plt.imshow(np.fft.fftshift(np.fft.fft2(x_ones).real))
plt.xlabel('k_x')
plt.ylabel('k_y')
plt.title('numpy 2D FFT')
plt.savefig('plots/FFTnumpy_2D.png')
plt.close()
plt.imshow(np.fft.fftshift((FFT_2D(x_ones).real)))
plt.xlabel('k_x')
plt.ylabel('k_y')
plt.title('Our own 2D FFT')
plt.savefig('plots/FFT_2D.png')
plt.close()

x=y=z=np.linspace(-2,2,32)
multivariate_gaussian=np.zeros((32,32,32))
multivariate_gaussian[:,:,:]=1/((2*np.pi)**(3/2))*np.exp(-(x**2+y**2+z**2)/2)

'''
Make the plots of the 3D FFT
'''

plt.imshow(np.fft.fftshift(np.fft.fftn(multivariate_gaussian).real)[15,:,:])
plt.xlabel('k_y')
plt.ylabel('k_z')
plt.title('y-z slice multivariate gaussian')
plt.savefig('plots/FFTnpyz.png')
plt.close()

plt.imshow(np.fft.fftshift(np.fft.fftn(multivariate_gaussian).real)[:,15,:])
plt.xlabel('k_x')
plt.ylabel('k_z')
plt.title('x-z slice multivariate gaussian')
plt.savefig('plots/FFTnpxz.png')
plt.close()

plt.imshow(np.fft.fftshift(np.fft.fftn(multivariate_gaussian).real)[:,:,15])
plt.xlabel('k_x')
plt.ylabel('k_y')
plt.title('x-y slice multivariate gaussian')
plt.savefig('plots/FFTnpxy.png')
plt.close()

'''
Making the numpy 3D FFT plots
'''
plt.imshow(np.fft.fftshift(FFT_3D(multivariate_gaussian).real)[15,:,:])
plt.xlabel('k_y')
plt.ylabel('k_z')
plt.title('y-z slice multivariate gaussian')
plt.savefig('plots/FFTyz.png')
plt.close()

plt.imshow(np.fft.fftshift(FFT_3D(multivariate_gaussian).real)[:,15,:])
plt.xlabel('k_x')
plt.ylabel('k_z')
plt.title('x-z slice multivariate gaussian')
plt.savefig('plots/FFTxz.png')
plt.close()

plt.imshow(np.fft.fftshift(FFT_3D(multivariate_gaussian).real)[:,:,15])
plt.xlabel('k_x')
plt.ylabel('k_y')
plt.title('x-y slice multivariate gaussian')
plt.savefig('plots/FFTxy.png')
plt.close()


'''
Calculate the potential for 5f

'''
potential=Potential_calculation(positions).real

'''
Make the plots of the slices for z=4,9,11,14
'''

plt.imshow(potential[:,:,4])
plt.xlabel('x_grid')
plt.ylabel('y_grid')
plt.title('potential for z=4')
plt.savefig('plots/potential4.png')
plt.clf()
plt.imshow(potential[:,:,9])
plt.xlabel('x_grid')
plt.ylabel('y_grid')
plt.title('potential for z=9')
plt.savefig('plots/potential9.png')
plt.clf()
plt.imshow(potential[:,:,11])
plt.xlabel('x_grid')
plt.ylabel('y_grid')
plt.title('potential for z=11')
plt.savefig('plots/potential11.png')
plt.clf()
plt.imshow(potential[:,:,14])
plt.xlabel('x_grid')
plt.ylabel('y_grid')
plt.title('potential for z=14')
plt.savefig('plots/potential14.png')
plt.clf()

plt.imshow(potential[:,8,:])
plt.xlabel('x_grid')
plt.ylabel('z_grid')
plt.title('potential for y=8')
plt.savefig('plots/potentialcenter_y.png')
plt.clf()
plt.imshow(potential[8,:,:])
plt.xlabel('y_grid')
plt.ylabel('z_grid')
plt.title('potential for x=8')
plt.savefig('plots/potentialcenter_x.png')
plt.clf()


print('gradient of the potential for the first 10 particles',Inverse_CIC(positions[:10],potential))

