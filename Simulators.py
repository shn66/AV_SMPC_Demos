import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pdb
from random import sample

class Simulator():
	'''
	For ENV= "TI"::  Simulates a traffic intersection with 1 Ego vehicle (EV) and 2 target vehicles (TVs).
	The TVs can be in one of two modes: Cross intersection or stop
	
	For ENV= "HW"::  Simulates a Highway with 1 Ego vehicle (EV) and 1 target vehicles (TVs).
	The TVs can be in one of two modes: drive straight or change into EV's lane
	
	'''
	
	def __init__(self,
				DT          = 0.1,
				T_FINAL     = 300,
				MODE        = 0,
				S_FINAL     = 50,
				TV_SPEED    = 8.,
				TV_L        = 2.,
				ENV         = "TI",
				EV_POLICY_TYPE = "SMPC_IA"
				):
		
		self.env=ENV
		self.ev_pol=EV_POLICY_TYPE
		
		self.t=0
		self.dt= DT
		self.T=T_FINAL

		# Half length and half width of the cars	
		self.hl=2.9
		self.hw=1.7
		
		# EV states: x,y,v_x
		self.A=np.array([[1., 0., self.dt],[0., 1., 0.], [0., 0., 1.]])
		self.B=np.array([0.,0.,self.dt])
		
		
		
		self.u_prev=0.
		
		self.tv_v=TV_SPEED
		self.tv_l=TV_L
		
		
		
		if self.env=="TI":
			self.ev_init=np.array([-42., 0., 10])
			self.ev_traj=np.zeros((3,self.T+1))
			self.ev_u=np.zeros((1,self.T))
			self.ev_traj[:,0]=self.ev_init
			
			
			self.tv_init= [np.array([26., 35., -15.]), np.array([36., -36., 13.])]
			self.N_TV=len(self.tv_init)
		
			# Control gains for TV: Fast policy, Stop policy
			self.gains=[[.1,0.2], [2,2.5]]
			self.IA_gains=[0.1,1]
		
			self.N_modes=4
			## mode = 0, TV1 fast, TV2 fast
			## mode = 1, TV1 fast, TV2 slow
			## mode = 2, TV1 slow, TV2 fast
			## mode = 3, TV1 slow, TV2 slow
			
			# Final longitudinal position
			self.s_f=50.
			# o_c1, o_c2 are controller parameters for the TVs
			# o_c1 denotes the desired longitudinal distance accross the intersection
			# o_c2 denotes the desired stopping distance from the intersection
			self.o_c1=20.
			self.o_c2=7.5      #Use this for comparing CL vs OL 
			# self.o_c2=6.      #Use this for comparing IA vs no-IA and OBCA vs ellipse
			self.mode=MODE

			
			# TV states: x,y,v_y
			self.Atv=np.array([[1., 0., 0.],[0., 1., self.dt], [0., 0., 1.]])
			self.Btv=self.B
			
			self.noise_std=[0.001, 0.001, 0.01, 0.2]
		else:
			self.ev_init=np.array([-37.5, 0., 11])
			self.ev_traj=np.zeros((3,self.T+1))
			self.ev_u=np.zeros((1,self.T))
			self.ev_traj[:,0]=self.ev_init
			
			
			self.tv_init= [np.array([-20., 8., 0.])]
			self.N_TV=len(self.tv_init)
		
			# Control gains for TV: straight policy, lane change policy
			self.gains=[[0.,0.], [0.02,1]]
			self.IA_gains=[0.01,0.05]
		
			self.N_modes=2
			## mode = 0, TV straight
			## mode = 1, TV changes lane

		
			self.s_f=50.
			# o_c1, o_c2 are controller parameters for the TVs
			# o_c1 is a parameter used in the IA controller
			# o_c2 denotes the desired lateral coordinate
			self.o_c1=10.
			self.o_c2=0.
			self.mode=MODE

	
			# TV states: x,y,psi
			self.Atv=np.array([[1., 0., 0.],[0., 1., self.tv_v*self.dt], [0., 0., 1.]])
			self.Btv=np.array([0.,0.,self.tv_v/2/self.tv_l*self.dt]) 
			
			self.noise_std=[0.001, 0.001, 0.02, 0.01]
			
		
		self.tv_traj=[np.zeros((3,self.T+1)) for k in range(self.N_TV)]
		
		for k in range(self.N_TV):
			self.tv_traj[k][:,0]=self.tv_init[k]
			
		
		
			
		  
	def TV_gen(self):
		'''
		Checks if TVs have reached the end and spawns new TVs for Traffic Intersection Environment
		'''
		if self.tv_traj[0][1,self.t]< -self.o_c1:
			self.tv_traj[0][:,self.t]=self.tv_init[0]
			mode_list=[(self.mode+2)%4, self.mode]
			# self.mode=sample(mode_list,1)[0]
			# self.mode=(self.mode+2)%4

		if self.tv_traj[1][1,self.t]> self.o_c1:
			self.tv_traj[1][:,self.t]=self.tv_init[1]
			mode_list=[self.mode+(-1)**(self.mode), self.mode]
			self.mode=self.mode+(-1)**(self.mode)
			#Uncomment to randomize modes for second TV
#             self.mode=sample(mode_list,1)[0]

		
	def done(self):
		return self.t==self.T or self.s_f-self.ev_traj[0,self.t]<=0.1
		
	def get_update_dict(self, N, *args):
	
		if self.t==0:
			z_lin=[np.zeros((3,N+1))]*self.N_modes
			u_tvs=[[np.zeros(N) for j in range(self.N_modes)] for k in range(self.N_TV)]
			elim_mode=0. 
			ev_cross=0.
		else:
			z_lin, u_tvs, elim_mode, ev_cross =self._get_lin_ref(N, *args)

		update_dict={'x0': self.ev_traj[0,self.t], 'y0': self.ev_traj[1,self.t], 'v0':self.ev_traj[2,self.t], 'u_prev':self.u_prev,
					 'x_tv0': np.array([self.tv_traj[k][0,self.t] for k in range(self.N_TV)]), 'y_tv0': np.array([self.tv_traj[k][1,self.t] for k in range(self.N_TV)]), 'v_tv0': np.array([self.tv_traj[k][2,self.t] for k in range(self.N_TV)]),
					 'z_lin': z_lin, 'u_tvs': u_tvs, 'elim_mode': elim_mode, 'ev_cross': ev_cross}

		if len(args)!=0:
			if "OBCA" not in self.ev_pol:
				[h,M,K,nom_z_tv]=args
				update_dict.update({'ws':[h,M,K]})
			else:
				[h,M,K,lmbd,nu,nom_z_tv]=args
				update_dict.update({'ws':[h,M,K,lmbd,nu]})

		return update_dict
		
	
	def run_step(self, u_ev):

		rng=np.random.default_rng(self.t)

		self.ev_traj[:,self.t+1]=self.A@self.ev_traj[:,self.t]+self.B*u_ev\
									+np.array([rng.normal(0,self.noise_std[0]), 0., rng.normal(0,self.noise_std[1])])
		self.ev_traj[2,self.t+1]=np.max([.0, self.ev_traj[2,self.t+1] ])
		self.ev_u[:,self.t]=u_ev
		self.u_prev=u_ev
		u_tv=[]

		## The following implements the corresponding controller for each TV,
		## depending on the mode and whether they are interaction aware (IA) or not
		if self.env=="TI":

			if "IA" not in self.ev_pol:
				## The following maps the current mode to one of these scenarios and implements the corresponding controller
				
				if int(self.mode/2)==0: # TV1 in fast mode
					# PD control with reference (s_ref,v_ref)=(-o_c1, 0)
					u_tv.append(self.gains[0][0]*( -self.o_c1-self.tv_traj[0][1,self.t])
						        +self.gains[0][1]*( 0.-self.tv_traj[0][2,self.t]))
				else:					# TV1 in slow mode
					# PD control with reference (s_ref,v_ref)=(o_c2, 0)
					u_tv.append(self.gains[1][0]*( self.o_c2-self.tv_traj[0][1,self.t])
						        +self.gains[1][1]*( 0.-self.tv_traj[0][2,self.t]))
				

				if self.mode%2==0:      # TV2 in fast mode
					# PD control with reference (s_ref,v_ref)=(o_c1, 0)
					u_tv.append(self.gains[0][0]*( self.o_c1-self.tv_traj[0][1,self.t])
						        +self.gains[0][1]*( 0.-self.tv_traj[0][2,self.t]))
				else:					# TV2 in slow mode
					# PD control with reference (s_ref,v_ref)=(-o_c2, 0)
					u_tv.append(self.gains[1][0]*( -self.o_c2-self.tv_traj[1][1,self.t])
						        +self.gains[1][1]*( 0.-self.tv_traj[1][2,self.t]))
			
			else:
				
				if int(self.mode/2)==0: # TV1 in fast mode, with same controller as non-IA case

					u_tv.append(self.gains[0][0]*( -self.o_c1-self.tv_traj[0][1,self.t])
						        +self.gains[0][1]*( 0.-self.tv_traj[0][2,self.t]))
				else:					# TV1 in slow mode, with same controller as non-IA case

					u_tv.append(self.gains[1][0]*( self.o_c2-self.tv_traj[0][1,self.t])
						        +self.gains[1][1]*( 0.-self.tv_traj[0][2,self.t]))
				

				if self.mode%2==0:      # TV2 in fast mode, with same controller as non-IA case

					u_tv.append(self.gains[0][0]*( self.o_c1-self.tv_traj[0][1,self.t])
						        +self.gains[0][1]*( 0.-self.tv_traj[0][2,self.t]))
				else:					# TV2 in slow mode
					if self.ev_traj[0,self.t]<=self.tv_traj[1][0,self.t]+2.:
						# If EV is hasn't crossed TV2, a linear term that depends on EV's and TV's positions is added.
						# This term pushes the TV2 back when the EV's x-coordinate is close to TV2's x-coordinate
						u_tv.append(self.gains[1][0]*(-self.IA_gains[0]*(-self.tv_traj[1][0,self.t]+self.o_c1+self.ev_traj[0,self.t])
							                          +self.IA_gains[1]*(-self.o_c2-self.tv_traj[1][1,self.t]))
						            +self.gains[1][1]*( 0.-self.tv_traj[1][2,self.t]))
					else:
						# If EV has crossed the TV, same slow mode control as the non-IA case
						
						u_tv.append(self.gains[1][0]*(-self.o_c2-self.tv_traj[1][1,self.t])
							        +self.gains[1][1]*( 0.-self.tv_traj[1][2,self.t]))
		
			for k in range(self.N_TV):
				self.tv_traj[k][:,self.t+1]=self.Atv@self.tv_traj[k][:,self.t]+self.Btv*u_tv[k]\
												+np.array([0., rng.normal(0,self.noise_std[2]), 0.1*rng.normal(0,self.noise_std[3])])

		else:
			if "IA" not in self.ev_pol:
				## The following maps the current mode to one of these scenarios and implements the corresponding controller
				if self.mode==0: #TV1 keeps current speed
					u_tv.append(0.)
				else:            #TV1 changes into EV's lane
					u_tv.append(self.gains[1][0]*(0.-self.tv_traj[0][1,self.t])
						        +self.gains[1][1]*( 0.-self.tv_traj[0][2,self.t]))
			else:
				if self.mode==0: #TV1 keeps current speed, same as non-IA case
					u_tv.append(0.)
				else:            #TV1 changes into EV's lane
					if self.ev_traj[0,self.t]<=self.tv_traj[0][0,self.t]+4. and self.ev_traj[0,self.t]>=self.tv_traj[0][0,self.t]-10.:
						# If the EV is within [-10m, 4m] of the TV1's longitudinal coordinate, a linear term that depends on the EV's
						# TV1's position is added. This term pushes the TV into it's lane.
						u_tv.append(self.gains[1][0]*(self.IA_gains[0]*(self.o_c1+self.ev_traj[0,self.t]-self.tv_traj[0][0,self.t])
							                          +self.IA_gains[1]*(self.o_c2-self.tv_traj[0][1,self.t]))
						            +self.gains[1][1]*( 0.-self.tv_traj[0][2,self.t]))
					else:
						# Otherwise, changes into EV's lane as in the non-IA case
						u_tv.append(self.gains[1][0]*( self.o_c2-self.tv_traj[0][1,self.t])
							        +self.gains[1][1]*( 0.-self.tv_traj[0][2,self.t]))

		
			for k in range(self.N_TV):
				self.tv_traj[k][:,self.t+1]=self.Atv@self.tv_traj[k][:,self.t]+self.Btv*u_tv[k]\
												+np.array([self.tv_v*self.dt, rng.normal(0,self.noise_std[2]), 0.1*rng.normal(0,self.noise_std[3])])
   
		
		self.t+=1
		if self.env=="TI":
			self.TV_gen() #checks if TVs have reached destination and respawns TVs accordingly
		
		
	def _get_lin_ref(self, N, *args):
		'''
		Getting EV predictions from previous MPC solution.
		This is used for linearizing the collision avoidance constraints
		'''
		w=np.diag(np.array(self.noise_std[0:2])**(-1))@(self.ev_traj[0::2,self.t]-self.A[0::2,:]@self.ev_traj[:,self.t-1]-self.B[0::2]*self.u_prev)
   
		x0=[self.ev_traj[:,self.t] for j in range(self.N_modes)]
		o0=[[self.tv_traj[k][:,self.t] for k in range(self.N_TV)] for j in range(self.N_modes)]
			  
		elim_mode=False
		ev_cross=False
		if self.env=="TI":
			if o0[0][0][1]>0:
				elim_mode=True
			if x0[0][0]>o0[1][1][0]+2.:
				ev_cross=True
		else:
			if o0[0][0][1]<4:
				elim_mode=True
			if x0[0][0]>o0[0][0][0]+4.:
				ev_cross=True
			
  
		x_lin=[np.zeros((3,N+1)) for j in range(self.N_modes)]
		u_tvs=[[np.zeros(N) for j in range(self.N_modes)] for k in range(self.N_TV)]

		for j in range(self.N_modes):
			x=x0[j]
			o=o0[j]
			x_lin[j][:,0]=x   
			w_seq=np.zeros(2*N)
			w_seq[:2]=w

			if "OL" in self.ev_pol or len(args)==0:
				h_opt=np.zeros((N,1))
				M_opt=np.zeros((N,2*N))
				K_opt=[np.zeros((N,2*N))]*self.N_TV
			elif "OBCA" not in self.ev_pol:        
				[h_opt, M_opt, K_opt, nom_x_tv]=args
			else:
				[h_opt, M_opt, K_opt, lmbd, nu, nom_x_tv]=args

				for t in range(1,N):

					u=h_opt[t]+M_opt[j][t,:]@w_seq+np.sum([K_opt[j][l][t,2*t:2*(t+1)]@(o[l][1:]-nom_x_tv[j][l][3*t+1:3*(t+1)]) for l in range(self.N_TV)])
					
					if self.env=="TI":
						if int(j/2)==0 or elim_mode:
							u_tvs[0][j][t-1]=self.gains[0][0]*( -self.o_c1-o[0][1])+self.gains[0][1]*( -0-o[0][2])
						else:
							u_tvs[0][j][t-1]=self.gains[1][0]*( self.o_c2-o[0][1])+self.gains[1][1]*( -0-o[0][2])
						if j%2==0:
							u_tvs[1][j][t-1]=self.gains[0][0]*( self.o_c1-o[1][1])+self.gains[0][1]*( -0-o[1][2])
						else:
							u_tvs[1][j][t-1]=self.gains[1][0]*( -self.o_c2-o[1][1])+self.gains[1][1]*( 0-o[1][2])

						o=[self.Atv@o[k]+self.Btv*u_tvs[k][j][t-1] for k in range(self.N_TV)]
						if t==N-1:
							if int(j/2)==0 or elim_mode:
								u_tvs[0][j][t]=self.gains[0][0]*( -self.o_c1-o[0][1])+self.gains[0][1]*( 0-o[0][2])
							else:
								u_tvs[0][j][t]=self.gains[1][0]*( self.o_c2-o[0][1])+self.gains[1][1]*( 0-o[0][2])
							if j%2==0:
								u_tvs[1][j][t]=self.gains[0][0]*( self.o_c1-o[1][1])+self.gains[0][1]*( 0-o[1][2])
							else:
								u_tvs[1][j][t]=self.gains[1][0]*( -self.o_c2-o[1][1])+self.gains[1][1]*( 0-o[1][2])
					else:
						if j==0 and not elim_mode:
							u_tvs[0][j][t-1]=0.
						else:
							u_tvs[0][j][t-1]=self.gains[1][0]*( self.o_c2-o[0][1])+self.gains[1][1]*( 0.-o[0][2])
					
						o=[self.Atv@o[k]+self.Btv*u_tvs[k][j][t-1]+np.array([self.tv_v*self.dt,0.,0.]) for k in range(self.N_TV)]
						
						if t==N-1:
							if j==0 and not elim_mode:
								u_tvs[0][j][t-1]=0.
							else:
								u_tvs[0][j][t-1]=self.gains[1][0]*( self.o_c2-o[0][1])+self.gains[1][1]*( 0.-o[0][2])
					

					x=self.A@x+self.B*u
					x_lin[j][:,t]=x
					x_lin[j][:,t+1]=self.A@x

			

		return x_lin, u_tvs, float(elim_mode), float(ev_cross)


	def check_collision(self):

		if self.env=="TI":

			x_disjoint_ev_tv1= (self.ev_traj[0,self.t]+self.hl<self.tv_traj[0][0,self.t]-self.hw) or \
								(self.tv_traj[0][0,self.t]+self.hw<self.ev_traj[0,self.t]-self.hl)
			y_disjoint_ev_tv1= (self.ev_traj[1,self.t]+self.hw<self.tv_traj[0][1,self.t]-self.hl) or \
								(self.tv_traj[0][1,self.t]+self.hl<self.ev_traj[1,self.t]-self.hw)

			collision_ev_tv1= (not x_disjoint_ev_tv1) and (not y_disjoint_ev_tv1)

			x_disjoint_ev_tv2= (self.ev_traj[0,self.t]+self.hl<self.tv_traj[1][0,self.t]-self.hw) or \
								(self.tv_traj[1][0,self.t]+self.hw<self.ev_traj[0,self.t]-self.hl)
			y_disjoint_ev_tv2= (self.ev_traj[1,self.t]+self.hw<self.tv_traj[1][1,self.t]-self.hl) or \
								(self.tv_traj[1][1,self.t]+self.hl<self.ev_traj[1,self.t]-self.hw)

			collision_ev_tv2= (not x_disjoint_ev_tv2) and (not y_disjoint_ev_tv2)

			collision= collision_ev_tv1 or collision_ev_tv2

		else:

			x_disjoint_ev_tv1= (self.ev_traj[0,self.t]+self.hl<self.tv_traj[0][0,self.t]-self.hl) or \
								(self.tv_traj[0][0,self.t]+self.hl<self.ev_traj[0,self.t]-self.hl)
			y_disjoint_ev_tv1= (self.ev_traj[1,self.t]+self.hw<self.tv_traj[0][1,self.t]-self.hw) or \
								(self.tv_traj[0][1,self.t]+self.hw<self.ev_traj[1,self.t]-self.hw)

			collision= (not x_disjoint_ev_tv1) and (not y_disjoint_ev_tv1)


		return collision