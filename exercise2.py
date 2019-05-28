import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2

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
		return (newseed%2**32)/(2 **32)#only using the lowest 32 bits and divide it by 2**32 to get a number between 0 and 1.

RNG=RNG_class()

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

def Density_field(scaling=-2,N=1024):
	fourier_field=np.zeros((N,N),dtype=np.complex128)
	half=int(N/2+0.5)
	for i in range(0,half+1):
		k_y=2*np.pi*i/N
		for j in range(0,N):
			if j<= half:
				k_x=2*np.pi*j/N
			else:
				k_x=2*np.pi*(-N+j)/N			
			k_vector=np.sqrt(k_x**2+k_y**2)
			a,b=Box_Muller(0,np.sqrt(k_vector**(scaling)),1)
			fourier_field[i,j]=complex(a,b)
			fourier_field[-i,-j]=fourier_field[i,j].conjugate()
	fourier_field[0,0]=0
	fourier_field[0,half]=(fourier_field[0,half].real)*2
	fourier_field[half,0]=(fourier_field[half,0].real)*2
	fourier_field[half,half]=(fourier_field[half,half].real)*2
	return ifft2(fourier_field)*N**2

FFT_1=Density_field(scaling=-1)
FFT_2=Density_field(scaling=-2)
FFT_3=Density_field(scaling=-3)
plt.imshow(np.fft.fftshift(FFT_1.real))
plt.colorbar(fraction=0.2,shrink =0.6)
plt.title('Density field for n=-1')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.savefig('plots/FFT_1.png')
plt.clf()
plt.imshow(np.fft.fftshift(FFT_2.real))
plt.colorbar(fraction=0.2,shrink =0.6)
plt.title('Density field for n=-2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.savefig('plots/FFT_2.png')
plt.clf()
plt.imshow(np.fft.fftshift(FFT_3.real))
plt.colorbar(fraction=0.2,shrink =0.6)
plt.title('Density field for n=-3')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.savefig('plots/FFT_3.png')
