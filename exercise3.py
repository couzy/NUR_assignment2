import numpy as np
import matplotlib.pyplot as plt

'''
Performs the 4th order Runge-Kutta on the second order differential equation, which is split into two first order differential equations which are coupled
'''

def Runge_Kutta_2(function_1,function_2,start_point,start,end,stepsize=0.01):
	steps=int((end-start)/stepsize+0.5)+1
	coordinates=np.zeros((steps,3))
	coordinates[0]=np.array(start_point)
	def ODE_1(coordinate):
		return function_1(coordinate[0],coordinate[1],coordinate[2]) 
	def ODE_2(coordinate):
		return function_2(coordinate[0],coordinate[1],coordinate[2]) 
	for j in range(steps-1):
		k_1=stepsize*ODE_1(coordinates[j])
		l_1=stepsize*ODE_2(coordinates[j])
		k_2=stepsize*ODE_1(coordinates[j]+[0.5*stepsize,k_1/2,l_1/2])
		l_2=stepsize*ODE_2(coordinates[j]+[0.5*stepsize,k_1/2,l_1/2])
		k_3=stepsize*ODE_1(coordinates[j]+[0.5*stepsize,k_2/2,l_2/2])
		l_3=stepsize*ODE_2(coordinates[j]+[0.5*stepsize,k_2/2,l_2/2])
		k_4=stepsize*ODE_1(coordinates[j]+[stepsize,k_3,l_3])
		l_4=stepsize*ODE_2(coordinates[j]+[stepsize,k_3,l_3])
		coordinates[j+1]=coordinates[j]+[stepsize,k_1/6+k_2/3+k_3/3+k_4/6,l_1/6+l_2/3+l_3/3+l_4/6]	
	
	return coordinates

RK_1=Runge_Kutta_2(lambda t,d,z:z,lambda t,d,z: -4*z/(3*t)+2*d/(3*t**2),[1,3,2],1,1000)
RK_2=Runge_Kutta_2(lambda t,d,z:z,lambda t,d,z: -4*z/(3*t)+2*d/(3*t**2),[1,10,-10],1,1000)
RK_3=Runge_Kutta_2(lambda t,d,z:z,lambda t,d,z: -4*z/(3*t)+2*d/(3*t**2),[1,5,0],1,1000)
time_integration=np.linspace(1,1000,999*100+1)

plt.loglog(time_integration,RK_1[:,1],label='Runge-Kutta',linewidth=3)
plt.loglog(time_integration,3*time_integration**(2/3),'--',label='analytical solution',linewidth=3)
plt.xlabel('time')
plt.ylabel('D(t)')
plt.title("Solution for y(1)=3,y'(1)=2")
plt.legend()
plt.savefig('plots/RK_1.png')
plt.clf()

plt.loglog(time_integration,RK_2[:,1],label='Runge-Kutta',linewidth=3)
plt.loglog(time_integration,10/time_integration,'--',label='analytical solution',linewidth=3)
plt.xlabel('time')
plt.ylabel('D(t)')
plt.title("Solution for y(1)=10,y'(1)=-10")
plt.legend()
plt.savefig('plots/RK_2.png')
plt.clf()

plt.loglog(time_integration,RK_3[:,1],label='Runge-Kutta',linewidth=3)
plt.loglog(time_integration,3*time_integration**(2/3)+2/time_integration,'--',label='analytical solution',linewidth=3)
plt.xlabel('time')
plt.ylabel('D(t)')
plt.title("Solution for y(1)=5,y'(1)=0")
plt.legend()
plt.savefig('plots/RK_3.png')


