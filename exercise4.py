import numpy as np
from scipy.fftpack import ifft2
import matplotlib.pyplot as plt

class RNG_class:
	seed = 23           
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
		firstseed=self.bit_shift_64(self.seed)
		newseed=self.MWC(firstseed)
		newseed=self.LCG(newseed)
		self.seed = newseed
		return (newseed%2**32)/(2 **32)
RNG=RNG_class()

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

def Box_Muller(mu,sigma,number=10000):
	random_numbers=[]
	for j in range(number):	
		u_1=RNG.generator()
		u_2=RNG.generator()
		z_1=sigma*np.sqrt(-2*np.log(u_1))*np.cos(2*np.pi*u_2)+mu
		z_2=sigma*np.sqrt(-2*np.log(u_1))*np.sin(2*np.pi*u_2)+mu
		random_numbers.append(z_1)
		random_numbers.append(z_2)
	return random_numbers

'''
Een array van c_k's maken en die opslaan
'''

def Density_field(scaling=-2,N=1024,axis=0):
	fourier_field=np.zeros((N,N),dtype=np.complex128)
	half=int(N/2+0.5)
	ck_array=np.zeros((N,N),dtype=np.complex128)	
	for i in range(0,half+1):
		k_y=2*np.pi*i/N
		for j in range(0,N):
			if j<= half:
				k_x=2*np.pi*j/N
			else:
				k_x=2*np.pi*(-N+j)/N			
			k_vector=np.sqrt(k_x**2+k_y**2)
			a,b=np.sqrt(k_vector**scaling)*np.asarray(Box_Muller(0,1,1))
			if axis==0:
				fourier_field[i,j]=k_x*1j*complex(a,-b)/(2)
				fourier_field[-i,-j]=fourier_field[i,j].conjugate()
			else:
				fourier_field[i,j]=k_y*1j*complex(a,-b)/(2)
				fourier_field[-i,-j]=fourier_field[i,j].conjugate()
	fourier_field[0,0]=0
	fourier_field[0,half]=(fourier_field[0,half].real)*2
	fourier_field[half,0]=(fourier_field[half,0].real)*2
	fourier_field[half,half]=(fourier_field[half,half].real)*2
	return ifft2(fourier_field)*N**2/10

def Density_field_3(scaling=-2,N=1024,axis=0):
	fourier_field=np.zeros((N,N,N),dtype=np.complex128)
	half=int(N/2+0.5)
	for i in range(0,half+1):
		k_y=2*np.pi*i/N
		for j in range(0,N):
			if j<= half:
				k_x=2*np.pi*j/N
			else:
				k_x=2*np.pi*(-N+j)/N
			for l in range(0,N):
				if l<= half:
					k_z=2*np.pi*l/N
				else:
					k_z=2*np.pi*(-N+l)/N
				k_vector=np.sqrt(k_x**2+k_y**2+k_z**2)
				a,b=Box_Muller(0,np.sqrt(k_vector**scaling),1)
				if axis==0:
					fourier_field[i,j,l]=k_x*1j*complex(a,-b)/(2)
					fourier_field[-i,-j,-l]=fourier_field[i,j,l].conjugate()
				elif axis == 1:
					fourier_field[i,j,l]=k_y*1j*complex(a,-b)/(2)
					fourier_field[-i,-j,-l]=fourier_field[i,j,l].conjugate()
				else:
					fourier_field[i,j,l]=k_z*1j*complex(a,-b)/(2)
					fourier_field[-i,-j,-l]=fourier_field[i,j,l].conjugate()
				
	fourier_field[0,0,0]=0
	fourier_field[0,half,0]=(fourier_field[0,half,0].real)*2
	fourier_field[half,0,0]=(fourier_field[half,0,0].real)*2
	fourier_field[half,half,0]=(fourier_field[half,half,0].real)*2
	fourier_field[half,0,half]=(fourier_field[half,0,half].real)*2
	fourier_field[0,0,half]=(fourier_field[0,0,half].real)*2
	fourier_field[0,half,half]=(fourier_field[0,half,half].real)*2
	fourier_field[half,half,half]=(fourier_field[half,half,half].real)*2
	return np.fft.ifftn(fourier_field)*N**3/100

def scale_integral(s_fac,omega_m=0.3,omega_l=0.7):
	return Romberg_integration(lambda a:1/((omega_m*a**(-1)+omega_l*a**2)**(1.5)),10**(-10),s_fac,order=12)

def scale_derivative(s_fac,H_0=70,omega_m=0.3,omega_l=0.7):
	scale_int=scale_integral(s_fac)
	return (-15/4*omega_m*(s_fac)**(-4)*omega_m/(omega_m*(s_fac)**(-3)+omega_l)**(0.5)*scale_int+2.5*omega_m*(omega_m*s_fac**(-3)+omega_l)**(0.5)*(1/s_fac*omega_m+s_fac**2*omega_l)**(-1.5))*H_0*(omega_m*s_fac**(-3)+omega_l)**(0.5)
	
omega_m=0.3
omega_l=0.7
H_0=70
'''

'''

print('the value of D at a=1/51, or z=50 is:',2.5*omega_m*np.sqrt(omega_m*(1/51)**(-3)+omega_l)*scale_integral(1/51))
print('the value of dD/dt at a=1/51 is:',scale_derivative(1/51))

Scale_integral=Romberg_integration(lambda a:1/((omega_m*a**(-1)+omega_l*a**2)**(1.5)),10**(-14),1/51,order=12)
Scale_integral=Scale_integral*2.5*omega_m*np.sqrt(omega_m*(1/51)**(-3)+omega_l)
s_fac=1/51
Scale_deriv=(2.5*omega_m*-3*(s_fac)**2*omega_m/(omega_m*(s_fac)**(-3)+omega_l)**(0.5)*Scale_integral+2.5*omega_m*(omega_m*s_fac**(-3)+omega_l)**(0.5)*(1/s_fac*omega_m+s_fac**2*omega_l)**(-1.5))*H_0*(omega_m*s_fac**(-3)+omega_l)**(0.5)
'''
Making the initial density fields
'''
S_field_x=Density_field(scaling=-6,N=64,axis=0)
S_field_y=Density_field(scaling=-6,N=64,axis=1)
S_field_x=S_field_x.real
S_field_y=S_field_y.real


x = np.linspace(0, 63, 64)
y = np.linspace(0, 63, 64)
x_coords,y_coords= np.meshgrid(x,y)
particles=[]
particles.append([x_coords,y_coords])
momentum= []
'''
loop for the Zeldovich approx for the particles in a grid
'''
delta_a=0.5*(1-0.0025)/101
for a in np.linspace(0.0025,1,101):
	D_a=2.5*0.3*np.sqrt(0.3*(a)**(-3)+0.7)*scale_integral(a)
	x_coords=(x_coords+D_a*S_field_x)%64
	y_coords=(y_coords+D_a*S_field_y)%64
	particles.append([x_coords,y_coords])
	deriv=scale_derivative(a-delta_a)*-(a-delta_a)**2
	momentum_x=deriv*S_field_x
	momentum_y=deriv*S_field_y
	momentum.append([momentum_x,momentum_y])

'''
Making the movie
'''
a_range=np.linspace(0.0025,1,101)
	
for j in (range(0,100)):
	a=np.float(0.0025+(1-0.0025)/100*j)
	plt.scatter(particles[j][0],particles[j][1],s=1.)
	plt.xlabel('x (Mpc)')
	plt.ylabel('y (Mpc)')
	plt.xlim(-1,64)
	plt.ylim(-1,64)
	plt.title('a={0}'.format(a))
	plt.savefig('./plots/snap%04d.png'%j)
	plt.close()

momentum=np.asarray(momentum)
x=np.linspace(0.0025,1,101)
for i in range(0,10):
	plt.plot(a_range,momentum[:101,0,1,i])
plt.title('momentum of y')
plt.xlabel('a')
plt.ylabel('momentum in y')
plt.savefig('./plots/ymomentum.png')
plt.clf()
particles=np.asarray(particles)
for j in range(0,10):
	plt.plot(a_range,particles[:101,0,0,j])
plt.title('position of 10 particles')
plt.xlabel('a')
plt.ylabel('position in y')
plt.savefig('./plots/ypos.png')
plt.clf()

'''
Making the 3D initial density fields 
'''
S_field_x=Density_field_3(scaling=-6,N=64,axis=0).real
S_field_y=Density_field_3(scaling=-6,N=64,axis=1).real
S_field_z=Density_field_3(scaling=-6,N=64,axis=2).real

x=y=z=np.linspace(0,63,64)
x_coords,y_coords,z_coords=np.meshgrid(x,y,z)
particles_3=[]
delta_a=0.5*(1-1/51)/101
momentum_3D=[]
for a in np.linspace(1/51,1,101):
	D_a=2.5*0.3*np.sqrt(0.3*(a)**(-3)+0.7)*scale_integral(a)
	deriv=scale_derivative(a-delta_a)*-(a-delta_a)**2
	x_coords=(x_coords+D_a*S_field_x)%64
	y_coords=(y_coords+D_a*S_field_y)%64
	z_coords=(z_coords+D_a*S_field_z)%64
	particles_3.append([x_coords,y_coords,z_coords])
	momentum_x=deriv*S_field_x
	momentum_y=deriv*S_field_y
	momentum_z=deriv*S_field_z
	momentum_3D.append([momentum_x,momentum_y,momentum_z])

particles_3=np.asarray(particles_3)
momentum_3D=np.asarray(momentum_3D)	

'''
Loop over the scale-factor for the movie
'''

for j in (range(0,100)):
	a=np.float(1/51+(1-0.0025)/100*j)
	mask=(particles_3[j,2,:,:,:]>=31.5)&(particles_3[j,2,:,:,:]<32.5)
	plt.scatter(particles_3[j,0,mask],(particles_3[j,1,mask]),s=1.)
	plt.xlabel('x (Mpc)')
	plt.ylabel('y (Mpc)')
	plt.xlim(-1,64)
	plt.ylim(-1,64)
	plt.title('z-slice,a={0}'.format(a))
	plt.savefig('./plots/zsnap%04d.png'%j)
	plt.close()
	mask=(particles_3[j,1,:,:,:]>=31.5)&(particles_3[j,1,:,:,:]<32.5)
	plt.scatter(particles_3[j,0,mask],(particles_3[j,2,mask]),s=1.)
	plt.xlabel('x (Mpc)')
	plt.ylabel('z (Mpc)')
	plt.xlim(-1,64)
	plt.ylim(-1,64)
	plt.title('y-slice,a={0}'.format(a))
	plt.savefig('./plots/ysnap%04d.png'%j)
	plt.close()
	mask=(particles_3[j,0,:,:,:]>=31.5)&(particles_3[j,0,:,:,:]<32.5)
	plt.scatter(particles_3[j,1,mask],(particles_3[j,2,mask]),s=1.)
	plt.xlabel('y (Mpc)')
	plt.ylabel('z (Mpc)')
	plt.xlim(-1,64)
	plt.ylim(-1,64)
	plt.title('x-slice,a={0}'.format(a))
	plt.savefig('./plots/xsnap%04d.png'%j)
	plt.close()

for i in range(0,10):
	plt.plot(a_range,momentum_3D[:101,0,0,0,i])
plt.title('momentum of z')
plt.xlabel('a')
plt.ylabel('momentum in z-direction')
plt.savefig('./plots/zmomentum.png')
plt.clf()

for j in range(0,10):
	plt.plot(a_range,particles_3[:101,2,0,0,j])
plt.title('position of z')
plt.xlabel('a')
plt.ylabel('position in z-direction')
plt.savefig('./plots/zpos.png')
plt.clf()
	

