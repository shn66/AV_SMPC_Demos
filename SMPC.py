import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pdb

class SMPC_MMPreds():

	def __init__(self,
				N            =  11,
				DT           = 0.1,
				V_MIN        = -0.5,      #Speed, acceleration constraints
				V_MAX        = 12.0, 
				A_MIN        = -7.0,
				A_MAX        =  5.0, 
				N_modes      =  4,
				N_TV         =  2,
				D_NOM        =  7,        #minimum distance to maintain between vehicles
				TIGHTENING   =  1.8,
				NOISE_STD    =  [0.001, 0.001, 0.002, 0.002], # process noise standard deviations in order [w_x, w_y, w_theta, w_v, w_TV] 
				O_const1      =  20, 
				O_const2       =  6.,
				TV_GAINS     =  [[.1,0.2], [2,2]],
				IA_GAINS     =  [0.01, 0.1],
				TV_SPEED     =  8.,
				TV_L         =  2.,
				POW          =  1.2, 
				Q = 80.,       # cost for measuring progress: -Q*POW^t*(s_{t+1}-s_t).
				R = 50.,       # cost for penalizing large input rate: POW^t*(u_{t+1}-u_t).T@R@(u_{t+1}-u_t) 
				POLICY     = "SMPC",
				ENV        ="TI"  
				):
		self.N=N
		self.DT=DT
		self.V_MIN=V_MIN
		self.V_MAX=V_MAX
		self.A_MAX=A_MAX
		self.A_MIN=A_MIN
		self.N_modes=N_modes
		self.N_TV=N_TV

		self.d_nom=D_NOM   
		self.tight=TIGHTENING
		self.noise_std=NOISE_STD
		self.o_c1=O_const1
		self.o_c2=O_const2
		
		self.pow=POW
		
		self.gains=[[0.,TV_GAINS[0][0], TV_GAINS[0][1]], [0.,TV_GAINS[1][0], TV_GAINS[1][1]]]
		self.ia_gain=[0., IA_GAINS[0], IA_GAINS[1]]
		
		self.tv_v=TV_SPEED
		self.tv_l=TV_L

		self.Q = ca.diag(Q)
		self.R = ca.diag(R)
		self.pol=POLICY
		self.env=ENV

		self.A=ca.DM([[1., 0., self.DT],[0., 1., 0.], [0., 0., 1.]])
		self.B=ca.DM([0.,0.,self.DT])
				
	   
		self.G=ca.DM([[1., 0.],[-1.,0. ], [0, 1.],[0.,-1. ]])
		self.g=ca.DM([[2.9],[2.9], [1.7],[1.7]])
		
		if self.env=="TI":
			self.Atv=ca.DM([[1., 0., 0.],[0., 1., self.DT], [0., 0., 1.]])
			self.Btv=self.B
			self.Rtv=ca.DM([[0.,-1.],[1.,0.]])
		else:
			self.Atv=np.array([[1., 0., 0.],[0., 1., self.tv_v*self.DT], [0., 0., 1.]])
			self.Btv=np.array([0.,0.,self.tv_v/2/self.tv_l*self.DT])
			self.Ctv=np.array([self.tv_v*self.DT,0.,0.])
			self.Rtv=ca.DM.eye(2)
		
		p_opts = {'expand':False, 'verbose' :False, 'error_on_fail':0}
		s_opts = {'max_cpu_time': 2.0,  'sb': 'no','print_level': 5, 'honor_original_bounds': 'yes'} 
		
		s_opts_grb = {'OutputFlag': 0, 'FeasibilityTol' : 1e-2, 'PSDTol' : 1e-3}
		p_opts_grb = {'error_on_fail':0}
		
		self.opti=ca.Opti()
		self.opti.solver("ipopt", p_opts, s_opts)
#         self.opti.solver("sqpmethod", p_opts, {})
#         self.opti=ca.Opti("conic")
#         self.opti.solver("gurobi", p_opts_grb, s_opts_grb)

		self.z_curr=self.opti.parameter(3)
		self.u_prev=self.opti.parameter(1)
		self.z_tv_curr=self.opti.parameter(3*self.N_TV)
		self.z_lin=[self.opti.parameter(3,self.N+1) for j in range(self.N_modes)]
		self.u_tvs=[[self.opti.parameter(self.N,1) for j in range(self.N_modes)] for k in range(self.N_TV)]
		
		if "OBCA" in self.pol:
			self.h_prev=self.opti.parameter(self.N,1)
			self.M_prev=[self.opti.parameter(self.N,2*self.N) for j in range(self.N_modes)]
			self.K_prev=[[self.opti.parameter(self.N,2*self.N) for k in range(self.N_TV)] for j in range(self.N_modes)]
			self.lmbd_prev=[[self.opti.parameter(4,self.N) for k in range(self.N_TV)] for j in range(self.N_modes)]
			self.nu_prev=[[self.opti.parameter(4,self.N) for k in range(self.N_TV)] for j in range(self.N_modes)]
			self.lmbd_dual_var=[[self.opti.variable(4,self.N) for k in range(self.N_TV)] for j in range(self.N_modes)]
			self.nu_dual_var=[[self.opti.variable(4,self.N) for k in range(self.N_TV)] for j in range(self.N_modes)]

		
		self.elim_mode=self.opti.parameter(1)
		self.ev_cross=self.opti.parameter(1)
				  
		self.policy=self._return_policy_class()
		self._add_constraints_and_cost()
		
		self._update_ev_initial_condition(0., 10., 0., 0. )
		self._update_tv_initial_condition(self.N_TV*[25.], self.N_TV*[10.], self.N_TV*[0.], self.N_TV*[self.N_modes*[np.zeros(self.N)]], 0., 0. )
		self._update_ev_preds(self.N_modes*[np.zeros((3,self.N+1))])            
	
	def _return_policy_class(self):
		"""
		EV Affine disturbance feedback + TV state feedback policies from https://arxiv.org/abs/2109.09792
		""" 
	
		h=[self.opti.variable(1) for t in range(self.N)]
		h_stack=ca.vertcat(*[h[t] for t in range(self.N)])
		
		if "IA" in self.pol:
			if "OL" not in self.pol:
				# Uncomment next line for disturbance feedback when using Gurobi. 
				# Runs slow with Ipopt (default)
				# M=[[[self.opti.variable(1, 2) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
				M=[[[ca.DM(np.zeros((1, 2))) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
				K=[[[[self.opti.variable(1, 2) for n in range(t)] for t in range(self.N)] for k in range(self.N_TV)] for j in range(self.N_modes)]
			else:
				M=[[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
				K=[[[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)] for k in range(self.N_TV)] for j in range(self.N_modes)]
				
			M_stack=[ca.vertcat(*[ca.horzcat(*[M[j][t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)]) for j in range(self.N_modes)]
			K_stack=[[ca.vertcat(*[ca.horzcat(*[K[j][k][t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)]) for k in range(self.N_TV)] for j in range(self.N_modes)] 
		else:
			if "OL" not in self.pol:
				# Uncomment next line for disturbance feedback when using Gurobi. 
				# Runs slow with Ipopt (default)
				# M=[[[self.opti.variable(1, 2) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
				M=[[[ca.DM(np.zeros((1, 2))) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
				K=[[[self.opti.variable(1,2) for t in range(self.N-1)] for k in range(self.N_TV)] for j in range(self.N_modes)]
			else:
				M=[[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)] for j in range(self.N_modes)]
				K=[[[ca.DM(1, 2) for t in range(self.N-1)] for k in range(self.N_TV)] for j in range(self.N_modes)]

			M_stack=[ca.vertcat(*[ca.horzcat(*[M[j][t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)]) for j in range(self.N_modes)]
			K_stack=[[ca.diagcat(ca.DM(1,2),*[K[j][k][t] for t in range(self.N-1)]) for k in range(self.N_TV)] for j in range(self.N_modes)]

		return h_stack,M_stack,K_stack

		
	def _get_ATV_TV_dynamics(self):
		"""
		Constructs system matrices such that for mode j and for TV k,
		O_t=T_tv@o_{t|t}+T_ev@X_t+c_tv+E_tv@N_t   (T_ev=O for interaction-agnostic case)
		where
		O_t=[o_{t|t}, o_{t+1|t},...,o_{t+N|t}].T, (TV state predictions)
		X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
		N_t=[n_{t|t}, n_{t+1|t},...,n_{t+N-1|t}].T,  (TV process noise sequence)
		o_{i|t}= state prediction of kth vehicle at time step i, given current time t
		""" 

		E=ca.DM([[0., 0.],[self.noise_std[2], 0.],[0., self.noise_std[3]]])

		T_tv=[[ca.MX(3*(self.N+1), 3) for k in range(self.N_TV)] for j in range(self.N_modes)]
		T_ev=[[ca.MX(3*(self.N+1), 3*(self.N+1)) for k in range(self.N_TV)] for j in range(self.N_modes)]
		TB_tv=[[ca.MX(3*(self.N+1), self.N) for k in range(self.N_TV)] for j in range(self.N_modes)]
		TC_tv=[[ca.MX(3*(self.N+1), 3) for k in range(self.N_TV)] for j in range(self.N_modes)]
		c_tv=[[ca.MX(3*(self.N+1), 1) for k in range(self.N_TV)] for j in range(self.N_modes)]
		E_tv=[[ca.MX(3*(self.N+1),self.N*2) for k in range(self.N_TV)] for j in range(self.N_modes)]

		u_tvs=self.u_tvs

		for k in range(self.N_TV):
			for j in range(self.N_modes):
				if "IA" not in self.pol:
					for t in range(self.N+1):
						if t==0:
							T_tv[j][k][:3,:]=ca.DM.eye(3)
						else:
							T_tv[j][k][t*3:(t+1)*3,:]=self.Atv@T_tv[j][k][(t-1)*3:t*3,:]
							TB_tv[j][k][t*3:(t+1)*3,:]=self.Atv@TB_tv[j][k][(t-1)*3:t*3,:]
							TB_tv[j][k][t*3:(t+1)*3,t-1:t]=self.Btv
							if self.env=="HW":
								TC_tv[j][k][t*3:(t+1)*3,:]=ca.DM.eye(3)+self.Atv@TC_tv[j][k][(t-1)*3:t*3,:]
							E_tv[j][k][t*3:(t+1)*3,:]=self.Atv@E_tv[j][k][(t-1)*3:t*3,:]    
							E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
					
					if self.env=="TI":
						if k==0:
							if int(j/2)==0:
								c_tv[j][k]=TB_tv[j][k]@u_tvs[k][0]
							else:
								c_tv[j][k]=TB_tv[j][k]@u_tvs[k][1]
						else:
							if j%2==0:
								c_tv[j][k]=TB_tv[j][k]@u_tvs[k][0]
							else:
								c_tv[j][k]=TB_tv[j][k]@u_tvs[k][1]
					else:
						c_tv[j][k]=TB_tv[j][k]@u_tvs[k][j]+TC_tv[j][k]@self.Ctv
							
							
				else:
					for t in range(self.N+1):
						if t==0:
							T_tv[j][k][:3,:]=ca.DM.eye(3)
						else:
							if self.env=="TI":
								if k==0:
									if int(j/2)==0:
										T_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@ca.DM(self.gains[0]).T)@T_tv[j][k][(t-1)*3:t*3,:]
										c_tv[j][k][t*3:(t+1)*3,:]=self.Btv@ca.DM(self.gains[0]).T@ca.DM([0.,-self.o_c1, 0.])+(self.Atv-self.Btv@ca.DM(self.gains[0]).T)@c_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@ca.DM(self.gains[0]).T)@E_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
									else:
										T_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@((1-self.elim_mode)*ca.DM(self.gains[1]).T+self.elim_mode*ca.DM(self.gains[0]).T))@T_tv[j][k][(t-1)*3:t*3,:]
										c_tv[j][k][t*3:(t+1)*3,:]=self.Btv@((1-self.elim_mode)*ca.DM(self.gains[1]).T@ca.DM([0.,self.o_c2, 0.])+self.elim_mode*ca.DM(self.gains[0]).T@ca.DM([0.,-self.o_c1, 0.]))+(self.Atv-self.Btv@((1-self.elim_mode)*ca.DM(self.gains[1]).T+self.elim_mode*ca.DM(self.gains[0]).T))@c_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@((1-self.elim_mode)*ca.DM(self.gains[1]).T+self.elim_mode*ca.DM(self.gains[0]).T))@E_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
								else:
									if j%2==0:
										T_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@ca.DM(self.gains[0]).T)@T_tv[j][k][(t-1)*3:t*3,:]
										c_tv[j][k][t*3:(t+1)*3,:]=self.Btv@ca.DM(self.gains[0]).T@ca.DM([0.,self.o_c1, 0.])+(self.Atv-self.Btv@ca.DM(self.gains[0]).T)@c_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@ca.DM(self.gains[0]).T)@E_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
									else:
										T_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@((1-self.ev_cross)*ca.DM([-self.gains[1][0]*self.ia_gain[0],self.gains[1][0]*self.ia_gain[1], self.gains[1][1]]).T+self.ev_cross*ca.DM(self.gains[1]).T))@T_tv[j][k][(t-1)*3:t*3,:]
										T_ev[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@((1-self.ev_cross)*ca.DM([-self.gains[1][0]*self.ia_gain[0],self.gains[1][0]*self.ia_gain[1], self.gains[1][1]]).T+self.ev_cross*ca.DM(self.gains[1]).T))@T_ev[j][k][(t-1)*3:t*3,:]
										T_ev[j][k][t*3:(t+1)*3,(t-1)*3:t*3]=self.Btv@((1-self.ev_cross)*ca.DM([-self.gains[1][0]*self.ia_gain[0],0.,0.]).T)
										c_tv[j][k][t*3:(t+1)*3,:]=self.Btv@((1-self.ev_cross)*(ca.DM([self.gains[1][0]*self.ia_gain[0],0.,0.]).T+self.ia_gain[1]*ca.DM(self.gains[1]).T)+self.ev_cross*ca.DM(self.gains[1]).T)@ca.DM([-self.o_c1,-self.o_c2, 0.])\
										 +(self.Atv-self.Btv@((1-self.ev_cross)*ca.DM([-self.gains[1][0]*self.ia_gain[0],self.gains[1][0]*self.ia_gain[1], self.gains[1][1]]).T+self.ev_cross*ca.DM(self.gains[1]).T))@c_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@((1-self.ev_cross)*ca.DM([-self.gains[1][0]*self.ia_gain[0],self.gains[1][0]*self.ia_gain[1], self.gains[1][1]]).T+self.ev_cross*ca.DM(self.gains[1]).T))@E_tv[j][k][(t-1)*3:t*3,:]
										E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
							else:
								if j==0:
									gain=(1.-self.elim_mode)*ca.DM(self.gains[0]).T+self.elim_mode*((1-self.ev_cross)*ca.DM([self.gains[1][0]*self.ia_gain[0],self.gains[1][0]*self.ia_gain[1], self.gains[1][1]]).T+self.ev_cross*ca.DM(self.gains[1]).T)
									T_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@gain)@T_tv[j][k][(t-1)*3:t*3,:]
									T_ev[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@gain)@T_ev[j][k][(t-1)*3:t*3,:]
									T_ev[j][k][t*3:(t+1)*3,(t-1)*3:t*3]=self.elim_mode*self.Btv@((1-self.ev_cross)*ca.DM([self.gains[1][0]*self.ia_gain[0],0.,0.]).T)
									c_tv[j][k][t*3:(t+1)*3,:]=self.Ctv+self.elim_mode*self.Btv@((1-self.ev_cross)*self.ia_gain[1]*ca.DM(self.gains[1]).T+self.ev_cross@ca.DM(self.gains[1]).T)@ca.DM([self.o_c1,self.o_c2, 0.])+(self.Atv-self.Btv@gain)@c_tv[j][k][(t-1)*3:t*3,:]
									E_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@gain)@E_tv[j][k][(t-1)*3:t*3,:]
									E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
								else:
									gain=((1-self.ev_cross)*ca.DM([self.gains[1][0]*self.ia_gain[0],self.gains[1][0]*self.ia_gain[1], self.gains[1][1]]).T+self.ev_cross*ca.DM(self.gains[1]).T)
									T_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@gain)@T_tv[j][k][(t-1)*3:t*3,:]
									T_ev[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@gain)@T_ev[j][k][(t-1)*3:t*3,:]
									T_ev[j][k][t*3:(t+1)*3,(t-1)*3:t*3]=self.Btv@((1-self.ev_cross)*ca.DM([self.gains[1][0]*self.ia_gain[0],0.,0.]).T)
									c_tv[j][k][t*3:(t+1)*3,:]=self.Ctv+self.Btv@((1-self.ev_cross)*self.ia_gain[1]*ca.DM(self.gains[1]).T+self.ev_cross@ca.DM(self.gains[1]).T)@ca.DM([self.o_c1,self.o_c2, 0.])+(self.Atv-self.Btv@gain)@c_tv[j][k][(t-1)*3:t*3,:]
									E_tv[j][k][t*3:(t+1)*3,:]=(self.Atv-self.Btv@gain)@E_tv[j][k][(t-1)*3:t*3,:]
									E_tv[j][k][t*3:(t+1)*3,(t-1)*2:t*2]=E
								

		return T_tv, c_tv, E_tv, T_ev
		
	
	def _get_LTV_EV_dynamics(self):
		"""
		Constructs system matrices such for EV,
		X_t=A_pred@x_{t|t}+B_pred@U_t+E_pred@W_t
		where
		X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
		U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence)
		W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T,  (EV process noise sequence)
		x_{i|t}= state prediction of kth vehicle at time step i, given current time t
		""" 
	
		E=ca.DM([[self.noise_std[0], 0.], [0., 0.], [0., self.noise_std[1]]])
				
		A_pred=ca.DM(3*(self.N+1), 3)
		B_pred=ca.DM(3*(self.N+1),self.N)
		E_pred=ca.DM(3*(self.N+1),self.N*2)
		A_pred[:3,:]=ca.DM.eye(3)
		
		for t in range(1,self.N+1):
				A_pred[t*3:(t+1)*3,:]=self.A@A_pred[(t-1)*3:t*3,:]
				
				B_pred[t*3:(t+1)*3,:]=self.A@B_pred[(t-1)*3:t*3,:]
				B_pred[t*3:(t+1)*3,t-1]=self.B
				
				E_pred[t*3:(t+1)*3,:]=self.A@E_pred[(t-1)*3:t*3,:]
				E_pred[t*3:(t+1)*3,(t-1)*2:t*2]=E
					 
		return A_pred,B_pred,E_pred
		
	def _add_constraints_and_cost(self):
		"""
		Constructs obstacle avoidance, state-input constraints for Stochastic MPC, based on https://arxiv.org/abs/2109.09792
		"""   

		[A,B,E]=self._get_LTV_EV_dynamics()
		[T_tv, c_tv, E_tv, T_ev]=self._get_ATV_TV_dynamics()
		[h,M,K]=self.policy
		
		nom_z=A@self.z_curr+B@h
		self.nom_z_tv=[[T_tv[j][k]@self.z_tv_curr[3*k:3*(k+1)]+T_ev[j][k]@nom_z+c_tv[j][k] for k in range(self.N_TV)] for j in range(self.N_modes)]
  
		sel_W=ca.kron(ca.DM.eye(self.N), ca.DM([[0.,1.,0],[0., 0., 1.]]))
		
		cost = 0
   
		for j in range(self.N_modes):
				
			self.opti.subject_to(self.opti.bounded(self.A_MIN, h, self.A_MAX))
	
			for t in range(1,self.N+1):
				
				self.opti.subject_to(self.opti.bounded(self.V_MIN, A[t*3+2,:]@self.z_curr+B[t*3+2,:]@h, self.V_MAX))

				for k in range(self.N_TV):
					# Linearised obstacle avoidance constraints
					if "OBCA" not in self.pol:
#                         oa_ref=self.nom_z_tv[j][k][3*t:3*t+2]+self.d_nom/ca.norm_2(self.z_lin[j][:2,t]-self.nom_z_tv[j][k][3*t:3*t+2])*(self.z_lin[j][:2,t]-self.nom_z_tv[j][k][3*t:3*t+2])
						oa_ref=self.nom_z_tv[j][k][3*t:3*t+2]+self.d_nom/ca.norm_2(self.z_curr[:2]-self.nom_z_tv[j][k][3*t:3*t+2])*(self.z_curr[:2]-self.nom_z_tv[j][k][3*t:3*t+2])
						
						z=(2*(oa_ref-self.nom_z_tv[j][k][3*t:3*t+2]).T@ca.DM([[1, 0, 0], [0, 1, 0]])@\
						  (ca.horzcat(B[t*3:(t+1)*3,:]@M[j]+E[t*3:(t+1)*3,:]-T_ev[j][k][t*3:(t+1)*3,:]@(B@M[j]+E),
						   *[(B[t*3:(t+1)*3,:]-T_ev[j][k][t*3:(t+1)*3,:]@B)@K[j][l]@sel_W@E_tv[j][l][:3*self.N,:]-int(l==k)*(E_tv[j][k][t*3:(t+1)*3,:]) for l in range(self.N_TV)])))

						y=-1.*self.d_nom**2-(oa_ref-self.nom_z_tv[j][k][3*t:3*t+2]).T@(oa_ref-self.nom_z_tv[j][k][3*t:3*t+2])\
						  +2*(oa_ref-self.nom_z_tv[j][k][3*t:3*t+2]).T@ca.DM([[1, 0, 0], [0, 1, 0]])@\
						  (A[t*3:(t+1)*3,:]@self.z_curr+B[t*3:(t+1)*3,:]@h-T_tv[j][k][t*3:(t+1)*3,:]@self.z_tv_curr[3*k:3*(k+1)]-T_ev[j][k][t*3:(t+1)*3,:]@nom_z-c_tv[j][k][t*3:(t+1)*3,:])
					else:
						lmbd=self.lmbd_dual_var[j][k][:,t-1].T
						nu=self.nu_dual_var[j][k][:,t-1].T
						lmbd_prev=self.lmbd_prev[j][k][:,t-1].T
						nu_prev=self.nu_prev[j][k][:,t-1].T
						
						z=lmbd_prev@self.G@ca.DM([[1, 0, 0], [0, 1, 0]])@\
						 (ca.horzcat(B[t*3:(t+1)*3,:]@M[j]+E[t*3:(t+1)*3,:]-T_ev[j][k][t*3:(t+1)*3,:]@(B@M[j]+E),
						 *[(B[t*3:(t+1)*3,:]-T_ev[j][k][t*3:(t+1)*3,:]@B)@K[j][l]@sel_W@E_tv[j][l][:3*self.N,:]-int(l==k)*(E_tv[j][k][t*3:(t+1)*3,:]) for l in range(self.N_TV)]))\
						 +(lmbd-lmbd_prev)@self.G@ca.DM([[1, 0, 0], [0, 1, 0]])@\
						  (ca.horzcat(B[t*3:(t+1)*3,:]@self.M_prev[j]+E[t*3:(t+1)*3,:]-T_ev[j][k][t*3:(t+1)*3,:]@(B@self.M_prev[j]+E),
						 *[(B[t*3:(t+1)*3,:]-T_ev[j][k][t*3:(t+1)*3,:]@B)@self.K_prev[j][l]@sel_W@E_tv[j][l][:3*self.N,:]-int(l==k)*(E_tv[j][k][t*3:(t+1)*3,:]) for l in range(self.N_TV)]))
						 
						y=-lmbd_prev@self.G@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr+B[t*3:(t+1)*3,:]@h-T_tv[j][k][t*3:(t+1)*3,:]@self.z_tv_curr[3*k:3*(k+1)]-T_ev[j][k][t*3:(t+1)*3,:]@nom_z-c_tv[j][k][t*3:(t+1)*3,:])\
						  -(lmbd-lmbd_prev)@self.G@ca.DM([[1, 0, 0], [0, 1, 0]])@(A[t*3:(t+1)*3,:]@self.z_curr+B[t*3:(t+1)*3,:]@self.h_prev-T_tv[j][k][t*3:(t+1)*3,:]@self.z_tv_curr[3*k:3*(k+1)]-T_ev[j][k][t*3:(t+1)*3,:]@nom_z-c_tv[j][k][t*3:(t+1)*3,:])\
						  -(lmbd+nu)@self.g-0.5
						self.opti.subject_to(lmbd>=0)
						self.opti.subject_to(nu>=0)
						self.opti.subject_to(lmbd@self.G+nu@self.G@self.Rtv.T==0)
#                         self.opti.subject_to(ca.norm_2(lmbd@self.G)<=1)
						self.opti.subject_to(lmbd@self.G@self.G.T@lmbd.T<=1)
						
#                     self.opti.subject_to(self.tight*ca.norm_2(z)<=y)
					self.opti.subject_to(self.tight**2*z@z.T<=y**2)
					self.opti.subject_to(0<=y)
					# Use for SOCP solvers: SCS and Gurobi
					# soc_constr=ca.soc(self.tight*z,y)
					# self.opti.subject_to(soc_constr>0)

			upcnt=ca.DM([(self.pow)**(ctr) for ctr in range(self.N)])
			nom_z_diff=ca.diff(nom_z.reshape((3,-1)),1,1).reshape((-1,1))
			nom_s_diff=ca.diag(upcnt)@ca.diff(nom_z.reshape((3,-1)),1,1)[0,:].T
			nom_u_diff=ca.diag(upcnt)@ca.diff(ca.vertcat(self.u_prev,h),1,0)
			
			cost+=-self.Q*ca.sum1(nom_s_diff)+self.R*nom_u_diff.T@nom_u_diff+0.*self.Q*nom_z_diff.T@nom_z_diff 
			
		self.opti.minimize( cost )   
	   

	def solve(self):
		st = time.time()
		try:
			pdb.set_trace()
			sol = self.opti.solve()
			
			# Collect Optimal solution.
			# import pdb; pdb.set_trace()
			u_control  = np.clip(sol.value(self.policy[0][0]), self.A_MIN, self.A_MAX)
			h_opt      = np.clip(sol.value(self.policy[0]), self.A_MIN, self.A_MAX)
			M_opt      = [sol.value(self.policy[1][j]) for j in range(self.N_modes)]
			K_opt      = [[sol.value(self.policy[2][j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
			if "OBCA" in self.pol:
				lmbd_opt    = [[sol.value(self.lmbd_dual_var[j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
				nu_opt     = [[sol.value(self.nu_dual_var[j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]

			nom_z_tv   = [[sol.value(self.nom_z_tv[j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
			solve_time=sol.stats()['t_wall_total']
			is_feas     = True
		except:
			if self.opti.stats()['return_status']!='Infeasible_Problem_Detected':
			  # Suboptimal solution (e.g. timed out)
				is_feas = True
				subsol=self.opti.debug
				solve_time=subsol.stats()['t_wall_total']
				u_control  = np.clip(subsol.value(self.policy[0][0]), self.A_MIN, self.A_MAX)
				h_opt      = np.clip(subsol.value(self.policy[0]), self.A_MIN, self.A_MAX)
				M_opt      = [subsol.value(self.policy[1][j]) for j in range(self.N_modes)] 
				K_opt      = [[subsol.value(self.policy[2][j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
				nom_z_tv   = [[subsol.value(self.nom_z_tv[j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
				if "OBCA" in self.pol:
					lmbd_opt = [[subsol.value(self.lmbd_dual_var[j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]
					nu_opt   = [[subsol.value(self.nu_dual_var[j][k]) for k in range(self.N_TV)] for j in range(self.N_modes)]

			else:
				is_feas = False
				u_control  = self.u_backup
			   
		sol_dict = {}
		sol_dict['u_control']  = u_control  # control input to apply based on solution
		sol_dict['feasible']    = is_feas    # whether the solution is feasible or not
		if is_feas:
				sol_dict['h_opt']=h_opt
				sol_dict['M_opt']=M_opt
				sol_dict['K_opt']=K_opt
				sol_dict['nom_z_tv']=nom_z_tv
				if "OBCA" in self.pol:
					sol_dict['lmbd_opt']=lmbd_opt
					sol_dict['nu_opt']=nu_opt
				sol_dict['solve_time'] = solve_time  # how long the solver took in seconds


		return sol_dict

	def update(self, update_dict):
		self._update_ev_initial_condition(*[update_dict[key] for key in ['x0','y0', 'v0', 'u_prev']] )
		self._update_tv_initial_condition(*[update_dict[key] for key in ['x_tv0', 'y_tv0', 'v_tv0', 'u_tvs', 'elim_mode', 'ev_cross']] )
		self._update_ev_preds(update_dict['z_lin'])
		
		if "OBCA" in self.pol:
			self.opti.set_value(self.h_prev,self.u_backup*ca.DM.ones(self.N,1))
			for j in range(self.N_modes):
				self.opti.set_value(self.M_prev[j],0.1*ca.DM.ones(self.N,2*self.N))
				if 'ws' in update_dict.keys():
					self.opti.set_initial(self.policy[0],  update_dict['ws'][0])
					self.opti.set_initial(self.policy[1][j],  update_dict['ws'][1][j])
					self.opti.set_value(self.h_prev,update_dict['ws'][0])
					self.opti.set_value(self.M_prev[j],update_dict['ws'][1][j])

				for k in range(self.N_TV):

					self.opti.set_initial(self.lmbd_dual_var[j][k],  .1*ca.DM.ones(4,self.N))
					self.opti.set_initial(self.nu_dual_var[j][k],  .1*ca.DM.ones(4,self.N))
					self.opti.set_value(self.lmbd_prev[j][k],  .5*ca.DM.ones(4,self.N))
					self.opti.set_value(self.nu_prev[j][k],  .5*ca.DM.ones(4,self.N))
					self.opti.set_value(self.K_prev[j][k],0.1*ca.DM.ones(self.N,2*self.N))

					if 'ws' in update_dict.keys():
						self.opti.set_initial(self.policy[2][j][k], update_dict['ws'][2][j][k])
						self.opti.set_value(self.K_prev[j][k], update_dict['ws'][2][j][k])
						self.opti.set_initial(self.lmbd_dual_var[j][k], update_dict['ws'][3][j][k])
						self.opti.set_initial(self.nu_dual_var[j][k], update_dict['ws'][4][j][k])
						self.opti.set_value(self.lmbd_prev[j][k], update_dict['ws'][3][j][k])
						self.opti.set_value(self.nu_prev[j][k], update_dict['ws'][4][j][k])
		else:
			if 'ws' in update_dict.keys():
				self.opti.set_initial(self.policy[0],  update_dict['ws'][0])
				for j in range(self.N_modes):
					self.opti.set_initial(self.policy[1][j],  update_dict['ws'][1][j])
					for k in range(self.N_TV):
						self.opti.set_initial(self.policy[2][j][k], update_dict['ws'][2][j][k])

	

	def _update_ev_initial_condition(self, x0, y0,  v0, u_prev):
		self.opti.set_value(self.z_curr, ca.DM([x0, y0, v0]))
		self.opti.set_value(self.u_prev, u_prev)
		self.u_backup=u_prev
				  
	def _update_tv_initial_condition(self, x_tv0, y_tv0, v_tv0, u_tvs, elim_mode, ev_cross):
		self.opti.set_value(self.elim_mode, elim_mode)
		self.opti.set_value(self.ev_cross, ev_cross)
		for k in range(self.N_TV):
			self.opti.set_value(self.z_tv_curr[3*k:3*(k+1)], ca.DM([x_tv0[k], y_tv0[k], v_tv0[k]]))
			for j in range(self.N_modes):
				self.opti.set_value(self.u_tvs[k][j], u_tvs[k][j])


	
	def _update_ev_preds(self, z_lin):
		
		for j in range(self.N_modes):
   
			self.opti.set_value(self.z_lin[j],z_lin[j])
					
		