import numpy as np

def build_matrix(n_r,s,T_r,sphere = True):
    '''
    Builds the matrix for solution of the diffusion equation in 1-D in either spherical or polar coordinates. A derivative boundary condition dT/dr = 0 is used at r = 0 and a constant temperature condition at the other boundary.
    
   Parameters
   ----------
   N_R int : number of steps
   s float : the parameter D*dt/dx**2
   T_r float : temperature at boundary
   sphere bool : if true, solves for a sphere, if false, for a cylinder
   
   Returns
   ----------
   A np.ndarray : the matrix for the forward Euler iteration
  
    '''
    A = np.zeros((n_r,n_r))
    
    if sphere == True:
        A[0,0]= 1-6*s
        A[0,1] = 6*s
        
        for i in range(1,n_r-1):
            A[i,i-1] = s*(1-1/i)
            A[i,i]   = 1 - 2*s
            A[i,i+1] = s*(1+1/i)
    else:
        A[0,0]= 1-4*s
        A[0,1] = 4*s
        
        for i in range(1,n_r-1):
            A[i,i-1] = s*(1-1/i/2)
            A[i,i]   = 1 - 2*s
            A[i,i+1] = s*(1+1/i/2)

    A[n_r-1,n_r-1]= 1
    return A

def forward_euler(T_in,n_t,A):
    '''
    Implements the forward Euler iteration scheme
    
    Parameters
    ----------
    T_in list of floats : the inital conditions for the temperature
    n_t int : number of temporal iterations
    A np.ndarray : the matrix as required for the iteration
    
    Returns
    ----------
    T_list list of np.ndarrays : the tempeartures at each time-step
    '''
    
    T_list = []
    T_list.append(T_in)
    for k in range(n_t-1):
        T_out = np.dot(A,T_in)
        T_in = T_out.copy()
        T_list.append(T_in)
    return T_list



