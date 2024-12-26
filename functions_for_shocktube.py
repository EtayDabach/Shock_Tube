import numpy as np
from typing import Tuple, Union

def HLL_Riemann_Solver(left_params:Tuple[float, float, float], right_params: Tuple[float, float, float], gamma:float) -> np.ndarray: #  params as (rho, u, P)
    """
    Apply the HLL (Harten-Lax-van Leer) approximation to solve the one dimensional Riemann problem in a tube.

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        gamma (float): The adiabatic index.

    Returns:
        np.ndarray: Return the calculated flux in HLL method
    """
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params

    # Calculate the speed of sound at each side
    aL = np.sqrt(gamma * PL / rhoL)
    aR = np.sqrt(gamma * PR / rhoR)

    # Calculate the fastest and slowest signal velocities
    SL = min(uL - aL, uR - aR)
    SR = max(uL + aL, uR + aR)

    # Calculate the internal energy for ideal gas:
    eL = PL/((gamma - 1)*rhoL)
    eR = PR/((gamma - 1)*rhoR)
    
    # Total energy for each side
    E_total_L = rhoL * (0.5 * uL**2 + eL)
    E_total_R = rhoR * (0.5 * uR**2 + eR)

    # Calculate the fluxes
    FL = np.array([rhoL * uL, rhoL* (uL**2) + PL, uL*(E_total_L + PL)]) # FL = F(U_L)
    FR = np.array([rhoR * uR, rhoR* (uR**2) + PR, uR*(E_total_R + PR)]) # FR = F(U_R)

    U_L = np.array([rhoL, rhoL * uL, E_total_L])
    U_R = np.array([rhoR, rhoR * uR, E_total_R])
    F_hll = 1/(SR - SL) * ((SR * FL) - (SL * FR) + SL*SR*(U_R - U_L))

    if 0 <= SL:
        return FL
    elif 0 >= SR:
        return FR
    else:
        return F_hll

#==================================================================================================================================================#

def HLLC_Riemann_Solver(left_params:Tuple[float, float, float], right_params:Tuple[float, float, float], gamma:float) -> np.ndarray:
    """
    Apply the HLLC (Harten-Lax-van Leer-Contact) approximation to solve the one dimensional Riemann problem in a tube.

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        gamma (float): The adiabatic index.

    Returns:
        np.ndarray: Return the calculated flux in HLLC method
    """
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params

    # Calculate the speed of sound at each side
    aL = np.sqrt(gamma * np.abs(PL / rhoL))
    aR = np.sqrt(gamma * np.abs(PR / rhoR))

    # # page 350 in the book
    # step1 # pressure estimation in star region
    # rho_avg = 0.5*(rhoL + rhoR) ; a_avg = 0.5*(aL + aR)
    # P_pvrs = 0.5*(PL + PR) - 0.5*(uR - uL) * rho_avg * a_avg 
    # P_star = max(0, P_pvrs)
  
    # # Adaptive Noniterative Riemann Solver, page 326
    Q_threshold = 2
    TOL = 0
    P_PV = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoL + rhoR)*(aL + aR)
    P_PV = max(TOL, P_PV)
    P_min = min(PL, PR)
    P_max = max(PL, PR)
    Q = P_max/P_min

    if (Q < Q_threshold) and (P_min < P_PV < P_max): # PVRS
        P_star = P_PV
    elif P_PV < P_min: # TRRS
        divL = aL/(PL**((gamma-1)/(2*gamma))) ; divR = aR/(PR**((gamma-1)/(2*gamma)))
        PTR = ((aL + aR - 0.5*(gamma - 1)*(uR - uL))/(divL + divR))**((2*gamma)/(gamma-1))
        P_star = max(TOL, PTR)
    else: # TSRS
        P_hat = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoL + rhoR)*(aL + aR) # estimation, P_hat = P0
        # Data-dependent constants
        AL = 2/((gamma + 1) * rhoL) ; AR = 2/((gamma + 1) * rhoR)
        BL =  (gamma - 1) * PL / (gamma + 1) ; BR =  (gamma - 1) * PR / (gamma + 1)
        gL = np.sqrt(AL/(P_hat + BL)) ; gR = np.sqrt(AR/(P_hat + BR)) # define gK(P)
        PTS = (gL*PL + gR*PR  - (uR - uL))/(gL + gR)
        P_star = max(TOL, PTS)

    # step2 # wave speed estimation
    qK = lambda P_side: (1 + ((gamma + 1)/(2*gamma))*(P_star/P_side - 1))**0.5
    if P_star <= PL:
        qL = 1
    else:
        qL = qK(PL)

    if P_star <= PR:
        qR = 1
    else:
        qR = qK(PR)
    
    SL = uL - aL*qL
    SR = uR + aR*qR

    # Calculate the fastest and slowest signal velocities
    # SL = min(uL - aL, uR - aR)
    # SR = max(uL + aL, uR + aR)

    # Calculate the internal energy for ideal gas:
    eL = PL/((gamma - 1)*rhoL)
    eR = PR/((gamma - 1)*rhoR)
    
    # Total energy for each side
    E_total_L = rhoL * (0.5 * uL**2 + eL)
    E_total_R = rhoR * (0.5 * uR**2 + eR)

    # Calculate the conserved variables vector and fluxes for each side
    FL = np.array([rhoL * uL, rhoL* (uL**2) + PL, uL*(E_total_L + PL)]) # FL = F(U_L)
    FR = np.array([rhoR * uR, rhoR* (uR**2) + PR, uR*(E_total_R + PR)]) # FR = F(U_R)
    U_L = np.array([rhoL, rhoL * uL, E_total_L])
    U_R = np.array([rhoR, rhoR * uR, E_total_R])

    # HLLC variables
    S_star = (PR - PL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) / (rhoL*(SL - uL) - rhoR*(SR - uR))
    UL_star = rhoL*((SL - uL)/(SL - S_star)) * np.array([1, S_star, E_total_L/rhoL + (S_star - uL)*(S_star + PL/(rhoL*(SL - uL)))])
    UR_star = rhoR*((SR - uR)/(SR - S_star)) * np.array([1, S_star, E_total_R/rhoR + (S_star - uR)*(S_star + PR/(rhoR*(SR - uR)))])
    FL_star = FL + SL*(UL_star - U_L)
    FR_star = FR + SR*(UR_star - U_R)


    if 0 <= SL:
        return FL
    elif SL < 0 <= S_star:
        return FL_star
    elif S_star < 0 <= SR:
        return FR_star
    else:
        return FR

#==================================================================================================================================================#

def U_i_nplus1(U:np.ndarray, F:np.ndarray, dx:float, gamma=1.4) -> Tuple[np.ndarray, float]: # U = np.array([[rho0, rho1, rho2...], [rho*u...], [E...]])
    """
    Update the conserved variables, U, in time: U_i_n -> U_i_n+1.

    Args:
        U (np.ndarray): The current conserved variables at each cell as a matrix for the n'th time step, U_i_n
        F (np.ndarray): The Flux F(U).
        dx (float): Size of the i'th cell (constant for eulerian method).
        gamma (float, optional): The adiabatic index. Defaults to 1.4 for ideal gas.

    Returns:
        Tuple[np.ndarray, float]: Return the updated conserved variables as ndarray type and the coressponding time step value as float.
    """
    u_i_n = np.abs(U[1]/U[0]) # particle velocity
    P_i_n = (gamma - 1)*(U[2] - 0.5*(U[1]**2)/U[0]) # P = (gamma - 1)*(E - 0.5*(rho*u**2)), pressure
    a_i_n = np.sqrt(gamma*abs(P_i_n/U[0])) # sound speed
    # print(u_i_n,'\n' ,a_i_n)
    S_max_n = np.max(u_i_n + a_i_n)
    C_cfl = 0.9
    dt = C_cfl*dx/S_max_n # C_cfl = 0.5
    # dt = 0.005
    U[:,1:-1] -= dt/dx * (F[:, 2:] - F[:, 1:-1])
    return U, dt

#==================================================================================================================================================#

def godunov_flux(U:np.ndarray, gamma=1.4, solver=HLLC_Riemann_Solver) -> np.ndarray: # U = np.array([[rho0, rho1, rho2...], [rho*u...], [E...]])
    """
    Using Godunov scheme to calculate the flux F(U).

    Args:
        U (np.ndarray): The conserved variable as a matrix.
        gamma (float, optional): The adiabatic index. Defaults to 1.4 for ideal gas.
        solver (function, optional): Method of approximation (HLL or HLLC). Defaults to HLLC_Riemann_Solver.

    Returns:
        np.ndarray: Return the calculated flux F(U).
    """
    M = U.shape[1]
    F = np.zeros_like(U)
    for i in range(1,M):
        i_minus1_cell = (U[0, i-1], U[1, i-1]/U[0, i-1], (gamma - 1)*(U[2, i-1] - 0.5*(U[1, i-1]**2)/ U[0, i-1])) # left side parameters (rhoL, uL, PL)
        i_cell = (U[0, i], U[1, i]/U[0, i], (gamma - 1)*(U[2, i] - 0.5*(U[1, i]**2)/ U[0, i])) # right side parameters (rhoR, uR, PR)
        F[:,i] = solver(i_minus1_cell, i_cell, gamma)
    return F

#==================================================================================================================================================#

def setting_initial_tube_parameters(left_params:Tuple[float, float, float], 
                                    right_params:Tuple[float, float, float], 
                                    M_x=1000, 
                                    x_0=0.5, 
                                    gamma=1.4, 
                                    allparams=False, 
                                    XandU=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Setting the initial parameters of the tube and the number of equal size cells (Eulerian method).

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        M_x (int, optional): Number of cells. Defaults to 1000.
        x_0 (float, optional): Positon of the divider in the tube. Defaults to 0.5.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.

        allparams (bool, optional): Set to True to return all of the calculated initial values:
        x_vec: vector of the positions of the cells between a normalized tube lengh of 0 to 1.
        rho_vec: vector of the initial density values in each cell.
        u_vec: vector of the initial particle velocity values in each cell.
        P_vec: vector of the initial pressure values in each cell.
        e_vec: vector of the initial internal energy values in each cell.
        E_vec: vector of the initial total energy (internal + kinetic) values in each cell.
        U0_mat: matrix of the initial conserved variables in all cells.
        Defaults to False.

        XandU (bool, optional): Set to True to return only x_vec and U0_mat. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray] | np.ndarray: 
        Retrun all the calculated initial values if allparams = True or return only x_vec and U0_mat if XandU = Ture, else return only U0_mat. 
    """
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params

    x_vec = np.linspace(0, 1, M_x)
    rho_vec = np.where(x_vec < x_0, rhoL, rhoR)
    u_vec = np.where(x_vec < x_0, uL, uR)
    P_vec = np.where(x_vec < x_0, PL, PR)
    e_vec = P_vec/(rho_vec*(gamma -1))
    E_vec = rho_vec*(0.5*(u_vec**2) + e_vec)
    U0_mat = np.vstack((rho_vec, rho_vec*u_vec, E_vec))
    if allparams:
        return x_vec, rho_vec, u_vec, P_vec, e_vec, E_vec, U0_mat
    elif XandU:
        return U0_mat, x_vec
    else:
        return U0_mat

#==================================================================================================================================================#

def tube_time_evolution(initial_left:Tuple[float, float, float], 
                        initial_right:Tuple[float, float, float], 
                        M_x:int,
                        t_final:float, 
                        gamma:float, 
                        solver=HLLC_Riemann_Solver, 
                        x_0=0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve the density, particle velocity and pressure of all the cells of the tube in time.

    Args:
        initial_left (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        initial_right (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        M_x (int): Number of cells.
        t_final (float): The time value to evolve the variables.
        gamma (float): The adiabatic index.
        solver (_type_, optional): Method of approximation (HLL or HLLC). Defaults to HLLC_Riemann_Solver.
        x_0 (float, optional): Positon of the divider in the tube. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Return the evolved U matrix and the positions of the cells.
    """
    U = setting_initial_tube_parameters(initial_left, initial_right, M_x, x_0, gamma)
    x = np.linspace(0, 1, M_x)
    dx = x[1] - x[0]
    t_counter = 0
    while t_counter < t_final:
        F = godunov_flux(U, gamma, solver)
        U, dt = U_i_nplus1(U, F, dx, gamma)
        t_counter += dt
    
        # Rigid wall boundary conditions at x=0 / L
        U[:, 0] = U[:, 1] # U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
    
    return U, x

#==================================================================================================================================================#

def tube_settings(left_params, right_params, t_final, gamma, x_0=0.5, solver_method=HLLC_Riemann_Solver): # left/right params as (rho, u, P), only for dynamic graph
    M = 100
    x = np.linspace(0, 1, M)
    U_final = tube_time_evolution(left_params, right_params, M, t_final, gamma, solver_method, x_0)
    rho = U_final[0]
    u = U_final[1]/rho
    P = (gamma - 1)*(U_final[2] - 0.5*(rho * (u**2)))
    return rho, u, P

#==================================================================================================================================================#
#___________________________________________________________________Exact solution_________________________________________________________________#
#==================================================================================================================================================#
# page 137 for exact solution
def pressure_function_K(WK:Tuple[float, float, float], P:float, gamma=1.4, derivative=False) -> float: # WK =(rhoK, uK, PK) as vector of primitive parameters, K=L or R
    """
    Calculate the pressure function value f_K(P, WK) in K side (Left or Right side) of the contact discontinuity wave in the Star region (the unknown region between the left and right waves).
    The star region is divided by the contact discontinuity in the middle.

    Args:
        WK (Tuple[float, float, float]): Parameters of the K side for the calculations as (density, velociy, pressure).
        P (float): The pressure in the star region
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        derivative (bool, optional): Set to True to return the derivative of the pressure function df_K/dP. Defaults to False.

    Returns:
        float: Returns The calculated pressure function in side K in units of velocity or the derivative df_K/dP in units of velocity/pressure.
    """
    # Primitive parameters
    rhoK, uK, PK = WK

    # Speed of sound
    aK = np.sqrt(gamma * PK/rhoK)

    # Data-dependent constants
    AK = 2/((gamma + 1) * rhoK)
    BK =  (gamma - 1) * PK / (gamma + 1)

    if derivative: # if derivative == True
        if P > PK: # shock wave
            return np.sqrt(AK/(BK + P))*(1-(P - PK)/(2*(BK + P))) # derivative of fK if P > PK
        else:
            return 1/(rhoK*aK) * (P/PK)**(-(gamma + 1)/(2*gamma)) # derivative of fk if P <= PK

    if P > PK: # shock wave
        return (P-PK) * np.sqrt(AK / (P + BK))
    else: # P <= PK , rarefaction wave
        return 2*aK/(gamma - 1) * ((P/PK)**((gamma - 1)/(2*gamma)) - 1)

#==================================================================================================================================================#

def total_pressure_function(left_params:Tuple[float, float, float], right_params:Tuple[float, float, float], P:float, gamma=1.4, derivative=False) -> float:
    """
    Calculate the total pressure function or its derivative.
    The total pressure function is:
    f(P, WL, WR) = f_L(P, WL) + f_R(P, WR) + uR - uL where u is the particle velocity.

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        P (float): The pressure in the star region
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        derivative (bool, optional): Set to True to return the derivative of the total pressure function df/dP. Defaults to False.

    Returns:
        float: Returns The calculated total pressure function in units of velocity or the derivative df/dP in units of velocity/pressure.
    """
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params

    if derivative:
        return pressure_function_K(left_params, P, gamma, derivative=True) + pressure_function_K(right_params,  P, gamma, derivative=True) # f' = fL' + fR'
    else:
        return pressure_function_K(left_params, P, gamma) + pressure_function_K(right_params, P, gamma) + uR - uL # f = fL + fR + uR - uL
    
#==================================================================================================================================================#

def Pstar(left_params:Tuple[float, float, float], 
          right_params:Tuple[float, float, float], 
          gamma=1.4, 
          TOL=1e-9, 
          P0='mean', 
          detective=False) -> Tuple[float, list, float, float, int] | float: # finding Pstar by Newton-Raphson iteration. P0 ='mean'/'TR'/'PV'/'TS'
    """
    Calculate the pressure in the Star region.

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        TOL (_type_, optional): The tolerance for minimum value. Defaults to 1e-9.
        P0 (str, optional): You can ignore this. Defaults to 'mean'.
        detective (bool, optional): Set to True to follow the iteration by returning all the diagnostic parameters to investigate the procedure (like a detective). Defaults to False.
        diagnostic parameters:
        Pk: the pressure in the star value.
        Pstar_list: list of all the pre-updated pressure in the star value.
        CHA: the iteration procedure is stopped whenever the relative pressure change is less than a prescribed small tolerance TOL: while CHA > TOL.
        CHA_first: the first calculated CHA.
        iteration: number of iteration.


    Returns:
        Tuple[float, list, float, float, int] | float: Return a tuple of all the diagnostic parameters if set to True, else return the calculated pressure in the Star region.
    """
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params

    # Calculate the speed of sound at each side
    aL = np.sqrt(gamma * PL / rhoL)
    aR = np.sqrt(gamma * PR / rhoR)

    # Adaptive Noniterative Riemann Solver, page 326
    Q_threshold = 2
    P_PV = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoL + rhoR)*(aL + aR)
    P_PV = max(TOL, P_PV)
    P_min = min(PL, PR)
    P_max = max(PL, PR)
    Q = P_max/P_min

    if (Q < Q_threshold) and (P_min < P_PV < P_max): # PVRS
        P0 = P_PV
    elif P_PV < P_min: # TRRS
        divL = aL/(PL**((gamma-1)/(2*gamma))) ; divR = aR/(PR**((gamma-1)/(2*gamma)))
        PTR = ((aL + aR - 0.5*(gamma - 1)*(uR - uL))/(divL + divR))**((2*gamma)/(gamma-1))
        P0 = max(TOL, PTR)
    else: # TSRS
        P_hat = 0.5*(PL + PR) - 0.125*(uR - uL)*(rhoL + rhoR)*(aL + aR) # estimation, P_hat = P0
        # Data-dependent constants
        AL = 2/((gamma + 1) * rhoL) ; AR = 2/((gamma + 1) * rhoR)
        BL =  (gamma - 1) * PL / (gamma + 1) ; BR =  (gamma - 1) * PR / (gamma + 1)
        gL = np.sqrt(AL/(P_hat + BL)) ; gR = np.sqrt(AR/(P_hat + BR)) # define gK(P)
        PTS = (gL*PL + gR*PR  - (uR - uL))/(gL + gR)
        P0 = max(TOL, PTS)
    
    Pstar_list = [P0]
    Pk_minus1 = Pstar_list[-1]
    f_Pk_minus1 = total_pressure_function(left_params, right_params, Pk_minus1, gamma)
    der_f_Pk_minus1 = total_pressure_function(left_params, right_params, Pk_minus1, gamma, derivative=True)
    Pk = Pk_minus1 - f_Pk_minus1/der_f_Pk_minus1
    CHA = 2*np.abs(Pk - Pk_minus1)/(Pk + Pk_minus1)
    CHA_first = CHA # detective
    iteration = 1 # detective

    while CHA > TOL:
        Pstar_list.append(Pk)
        Pk_minus1 = Pstar_list[-1]
        f_Pk_minus1 = total_pressure_function(left_params, right_params, Pk_minus1, gamma)
        der_f_Pk_minus1 = total_pressure_function(left_params, right_params, Pk_minus1, gamma, derivative=True)
        Pk = Pk_minus1 - f_Pk_minus1/der_f_Pk_minus1
        CHA = 2*np.abs(Pk - Pk_minus1)/(Pk + Pk_minus1)
        iteration += 1

    if detective:
        return Pk, Pstar_list, CHA, CHA_first, iteration
    else:
        return Pk 

#==================================================================================================================================================#

def ustar(left_params:Tuple[float, float, float], right_params:Tuple[float, float, float], P_star:float, gamma=1.4) -> float:
    """
    Calculate the particle velocity in the Star region

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        P_star (float): The pressure in the star region.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.

    Returns:
        float: Returns the calculated particle velocity in the star region.
    """
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params
    fL = pressure_function_K(left_params, P_star, gamma)
    fR = pressure_function_K(right_params, P_star, gamma)

    return 0.5*(uL + uR) + 0.5*(fR - fL)

#==================================================================================================================================================#

def rhostar_K(WK:Tuple[float, float, float], P_star:float, gamma=1.4) -> float:
    """
    Calculate the density in K side (Left or Right side) of the contact discontinuity wave in the Star region (the unknown region between the left and right waves).
    The star region is divided by the contact discontinuity in the middle.

    Args:
        WK (Tuple[float, float, float]): Parameters of the K side for the calculations as (density, velociy, pressure).
        P_star (float): The pressure in the star region.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.

    Returns:
        float: Returns The calculated density in side K.
    """
    # Primitive parameters
    rhoK, uK, PK = WK

    # Speed of sound
    aK = np.sqrt(gamma * PK/rhoK)

    if P_star > PK: # shock wave
        g_const = (gamma - 1)/(gamma + 1)
        return rhoK*(((P_star/PK) + g_const)/(g_const*(P_star/PK) + 1))
    else: # P_star <= PK, rarefaction wave
        return rhoK*(P_star/PK)**(1/gamma)

#==================================================================================================================================================#

def Exact_Riemann_solver(left_params:Tuple[float, float, float], 
                        right_params:Tuple[float, float, float], 
                        x:float, 
                        t:float, 
                        gamma=1.4, 
                        x_center=0.5, 
                        detective=False, **data) -> np.ndarray:
    """
    Calculate the primitive variables (density, velociy, pressure) for exact solution for Riemann problem.

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        x (float): The position of the cell.
        t (float): The given time step.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        x_center (float, optional): The positon of the divider in the tube. Defaults to 0.5.
        detective (bool, optional): Set to True to print all the diagnostic parameters to check the procedure. Defaults to False.

    Returns:
        np.ndarray: Returns the calculated primitive variables.
    """
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params
    # Calculate the speed of sound at each side
    aL = np.sqrt(gamma * PL / rhoL)
    aR = np.sqrt(gamma * PR / rhoR)
    # Calculating star region parameters
    P_star = Pstar(left_params, right_params, gamma, **data)
    u_star = ustar(left_params, right_params, P_star, gamma) # i accidentally used gamma=1.4 as constant here

    if x_center > 0.5: # linear transformation for the x axis
        x_tilde = -(10*x_center - 5)
        S = (x + x_tilde)/t
    else:
        x_tilde = 10*x_center - 5
        S = (x - x_tilde)/t
    
    rhoStar_L = rhostar_K(left_params, P_star, gamma)
    rhoStar_R = rhostar_K(right_params, P_star, gamma)

    if S < u_star: # left side of contact
        # Left side
        if P_star > PL: # left shock wave
            SL = uL - aL*np.sqrt(((gamma + 1)/(2*gamma)*(P_star)/(PL)) + (gamma - 1)/(2*gamma))
            if S < SL:
                W_final = np.array([rhoL, uL, PL]) # left_params as an array
            else:
                W_final = np.array([rhoStar_L, u_star, P_star])
            aStar_L = 'left shock wave -> no aStar_L'
            SHL = 'left shock wave -> no SHL'
            STL = 'left shock wave -> no STL'
        else: # left rarefaction wave
            aStar_L = aL*((P_star/PL)**((gamma-1)/(2*gamma)))
            SHL = uL - aL
            STL = u_star - aStar_L
            if S < SHL:
                W_final = np.array([rhoL, uL, PL]) # left_params as an array
            elif S > STL:
                W_final = np.array([rhoStar_L, u_star, P_star])
            else:
                rhoL_fan = rhoL*(2/(gamma + 1) + (gamma - 1)*(uL - S)/((gamma + 1)*aL))**(2/(gamma - 1))
                uL_fan = 2/(gamma + 1)*(aL + 0.5*(gamma - 1)*uL + S)
                PL_fan = PL*(2/(gamma + 1) + (gamma - 1)*(uL - S)/((gamma + 1)*aL))**(2*gamma/(gamma - 1))
                W_final = np.array([rhoL_fan, uL_fan, PL_fan])
            SL = 'left rarefaction wave -> no SL'
    else: # right side of contact
        # Right side
        if P_star > PR: # right shock wave
            SR = uR + aR*np.sqrt(((gamma + 1)/(2*gamma)*(P_star)/(PR)) + (gamma - 1)/(2*gamma))
            if S > SR:
                W_final = np.array([rhoR, uR, PR]) # right_params as an array
            else:
                W_final = np.array([rhoStar_R, u_star, P_star])
            aStar_R = 'right shock wave -> no aStar_R'
            SHR = 'right shock wave -> no SHR'
            STR = 'right shock wave -> no STR'
        else: # right rarefaction wave
            aStar_R = aR*((P_star/PR)**((gamma-1)/(2*gamma)))
            SHR = uR + aR
            STR = u_star + aStar_R
            if S > SHR:
                W_final = np.array([rhoR, uR, PR]) # right_params as an array
            elif S < STR:
                W_final = np.array([rhoStar_R, u_star, P_star])
            else:
                rhoR_fan = rhoR*(2/(gamma + 1) - (gamma - 1)*(uR - S)/((gamma + 1)*aR))**(2/(gamma - 1))
                uR_fan = 2/(gamma + 1)*(-aR + 0.5*(gamma - 1)*uR + S)
                PR_fan = PR*(2/(gamma + 1) - (gamma - 1)*(uR - S)/((gamma + 1)*aR))**(2*gamma/(gamma - 1))
                W_final = np.array([rhoR_fan, uR_fan, PR_fan])
            SR = 'right rarefaction -> no SR'
    
    if detective:
        if S < u_star: # left side of contact
            SR = 'not on the right side'
            SHR = 'not on the right side'
            STR = 'not on the right side'
            aStar_R = 'not on the right side'
        else: # right side of contact
            SL = 'not on the left side'
            SHL = 'not on the left side'
            STL = 'not on the left side'
            aStar_L = 'not on the left side'
        print(f"Left side parameters as [rhoL, uL, PL] = {left_params}\nRight side parameters as [rhoR, uR, PR] = {right_params}" \
              f"\nP_star = {P_star}\nu_star = {u_star}\nrhoStar_L = {rhoStar_L}\nrhoStar_R = {rhoStar_R}" \
              f"\nS = {S}\nSL = {SL}\naStar_L = {aStar_L}\nSHL = {SHL}\nSTL = {STL}\nSR = {SR}\naStar_R = {aStar_R}\nSHR = {SHR}\nSTR = {STR}")
               
        return W_final 
    else:
        return W_final

#==================================================================================================================================================#

def exact_solution(left_params:Tuple[float, float, float], 
                   right_params:Tuple[float, float, float], 
                   t_final:float, 
                   gamma = 1.4, 
                   M_x=1000, 
                   x_center=0.5, **data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the exact solution for Riemann problem.

    Args:
        left_params (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        right_params (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        t_final (float): The time value to evolve the variables.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        M_x (int, optional): The number of cells Defaults to 1000.
        x_center (float, optional): The positon of the divider in the tube. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Returns the positions of the cells, the density vector, the particle velocity vector and the pressure vector.
    """
    t_final = 10*t_final # just for scale
    # Define the primitive parameters at each side
    rhoL, uL, PL = left_params
    rhoR, uR, PR = right_params
    x_vals_for_calc = np.linspace(-5, 5, M_x) # np.linspace(-5, 5, M_x)
    x_vals_for_plot = np.linspace(0, 1, M_x)
    rho_list = []
    u_list = []
    P_list = []
    for x_i in x_vals_for_calc:
        rho, u, P = Exact_Riemann_solver(left_params, right_params, x_i, t_final, gamma, x_center, **data)
        rho_list.append(rho)
        u_list.append(u)
        P_list.append(P)
    
    return x_vals_for_plot, np.array(rho_list), np.array(u_list), np.array(P_list)

#==================================================================================================================================================#
#____________________________________________________________________second order__________________________________________________________________#
#==================================================================================================================================================#

def slope_limiter(param_iminus1:float, param_i:float, param_iplus1:float, TOL=1e-10, phi_method="minmod") -> float: # dx is the same for eulerian method, [param_i-1, param_i, param_i+1] as parameters inside the cells
    """
    Calculate the slope of a given primitive parameter (density, velociy or pressure) for its change between cells i-1, i, i+1.

    Args:
        param_iminus1 (float): The primitive parameter in the i-1 cell.
        param_i (float): The primitive parameter in the i cell.
        param_iplus1 (float): The primitive parameter in the i+1 cell.
        TOL (_type_, optional): Tolerance for minimum values. Defaults to 1e-10.
        phi_method (str, optional): Choose the method of slope limiter calculation, "minmod" or "van-leer". Defaults to "minmod".

    Returns:
        float: Returns the calculated slope of the given primitive parameter.
    """
    delta_minus = (param_i - param_iminus1) # left
    delta_plus = (param_iplus1 - param_i) # right
    if  0 <= delta_plus < TOL : # r -> inf
        if phi_method=='minmod':
            return (delta_plus * 1) # phi_r =1
        else:
            return (delta_plus * 2) # phi_r =2
    else:
        r = delta_minus/(delta_plus)
        
    if phi_method == 'minmod':
        phi_r = max(0, min(1, r))
    else:
        phi_r = (r + abs(r)) / (1 + abs(r))
    
    return delta_plus * phi_r

#==================================================================================================================================================#

def values_on_sides(param_i:float,  slope:float, side='something') -> float | Tuple[float, float]:
    """
    Calculate given primitive parameter on each side of the i'th cell.

    Args:
        param_i (float): The primitive parameter in the i cell.
        slope (float): The calculated slope of the primitive parameter.
        side (str, optional): For spesific side: 'left' or 'right'. Defaults to 'something'.

    Returns:
        float | Tuple[float, float]: Return the calculated primitive parameter on the specifed side, else return on both sides.
    """
    param_iplus = param_i + 0.5 * slope
    param_iminus = param_i - 0.5 *slope
    if side.upper() in ['LEFT', 'L']:
        return param_iminus
    elif side.upper() in ['RIGHT', 'R']:
        return param_iplus
    else:
        return  param_iminus, param_iplus
#==================================================================================================================================================#

def F_sides(U:np.ndarray, gamma=1.4, detective=False, W_mat_detective = False, at_i=1 ,**data) -> np.ndarray | Tuple[np.ndarray, np.ndarray]: # U = np.array([[rho0, rho1, rho2...], [rho*u...], [E...]])
    """
    Calculate the flux, F(U), on the sides of the i'th cell.

    Args:
        U (np.ndarray): Vector of conserved parameters.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        detective (bool, optional): Set to True for diagnostic at the at_i=i cell. Defaults to False.
        W_mat_detective (bool, optional): Set to True for return the calculated flux and prmitive parameters. Defaults to False.
        at_i (int, optional): Only when detective=True, enable diagnostic mode at the i'th cell. Defaults to 1.

    Returns:
        np.ndarray | Tuple[np.ndarray, np.ndarray]: Return primitive parameters matrix if detective or W_mat_detective is True, else return the calculated flux on the sides of the cell.
    """
    n , m = U.shape
    F_ihalf = np.zeros_like(U)
    W_sides_mat = np.zeros((n, 2*m))
    U = np.c_[U[:,0], U, U[:,-1]] # this CLASS adds an array to the matrix (array) as a column addition. ex: for (3,M) matrix, this will add the matrix of (3,N) to get a matrix of (3, M+N). velocity should be with -
    P_func = lambda j: (gamma - 1)*(U[2][j] - 0.5*((U[1][j])**2)/U[0][j]) # P = (gamma - 1)*(E - 0.5*(rho*u**2))
    for i in range(1, m):
        # Calculating parameters from U matrix. param_iminus1, param_i, param_iplus1
        rho_im1, rho_i, rho_ip1 = U[0][(i-1)], U[0][i], U[0][(i+1)]
        u_im1, u_i, u_ip1 = U[1][(i-1)]/rho_im1, U[1][i]/rho_i, U[1][(i+1)]/rho_ip1
        P_im1, P_i, P_ip1 = P_func(i-1), P_func(i), P_func(i+1)
        # Calculating the slope limiter for each parameter
        rho_slope = slope_limiter(rho_im1, rho_i, rho_ip1,  **data)
        u_slope = slope_limiter(u_im1, u_i, u_ip1,  **data)
        P_slope = slope_limiter(P_im1, P_i, P_ip1,  **data)
        # Calculating parameters at left and right sides of the i cell
        rho_left, rho_right = values_on_sides(rho_i,  rho_slope)
        u_left, u_right = values_on_sides(u_i,  u_slope)
        P_left, P_right = values_on_sides(P_i,  P_slope)
        # Solving Riemann problem using the parameters from the sides of the i cell
        W_sides_mat[:, 2*i-1] = np.array([rho_left, u_left, P_left])
        W_sides_mat[:, 2*i] = np.array([rho_right, u_right, P_right])

        if detective and at_i==i: 
                print(f"rho_{i-1} = {rho_im1}, rho_{i} = {rho_i}, rho_{i+1} = {rho_ip1}\n" \
                      f"u_{i-1} = {u_im1}, u_{i} = {u_i}, u_{i+1} = {u_ip1}\n" \
                      f"P_{i-1} = {P_im1}, P_{i} = {P_i}, P_{i+1} = {P_ip1}\n" \
                      f"left parameters as (rho, u, P): {(rho_left, u_left, P_left)}\n" \
                      f"right parameters as (rho, u, P): {(rho_right, u_right, P_right)}\n")
                return W_sides_mat
    
    W_sides_mat[:,0] = W_sides_mat[:,1]
    W_sides_mat[:,-1] = W_sides_mat[:,-2]
    # W_side_mat as np.array([[0], [-1/2_left_1], [1/2_right_1], [1/2_left_2], [1_1/2_right_2], [1_1/2_left_3], [2_1/2_right_3], ...]) where each [1/2_] is a (3,1) array

    for k in range(2, 2*m, 2):
        left_side_parameters = W_sides_mat[:, k]
        right_side_parameters = W_sides_mat[:, k+1]
        
        F_ihalf[:,int(k/2)] = HLLC_Riemann_Solver(left_side_parameters, right_side_parameters, gamma)
    
    if W_mat_detective:
        return F_ihalf ,W_sides_mat
    else:
        return F_ihalf # as np.array([[0],[F_1/2], [F_1_1/2], [F_2_1/2], [F_3_1/2]....[0]]) where F_i_1/2 is an (3,1) array

#==================================================================================================================================================#

# Almost the same as U_i_nplus1 function but with different sign and without divition with dx.
def U_tilde_func(U_n:np.ndarray, F:np.ndarray, dx:float, gamma=1.4) -> Tuple[np.ndarray, float]: # U = np.array([[rho0, rho1, rho2...], [rho*u...], [E...]]), U_tilde is like U^(n+1/2) -> half time step
    """
    Evolve the conserved parameters matrix ,U, in time. This is the 'half' step in time for the second order approximation.

    Args:
        U_n (np.ndarray): The conserved parameters matrix at the n'th time step.
        F (np.ndarray): The flux.
        dx (float): The size of the cell.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.

    Returns:
        Tuple[np.ndarray, float]: Returns the updated U matrix and the time step value.
    """
    U_n_tilde = np.copy(U_n)
    u_i_n = np.abs(U_n[1]/U_n[0])
    P_i_n = (gamma - 1)*(U_n[2] - 0.5*(U_n[1]**2)/U_n[0]) # P = (gamma - 1)*(E - 0.5*(rho*u**2))
    a_i_n = np.sqrt(gamma* np.abs(P_i_n/U_n[0]))
    S_max_n = np.max(u_i_n + a_i_n)
    C_cfl = 0.9
    dt = C_cfl*dx/S_max_n # C_cfl = 0.5
    U_n_tilde[:,1:-1] -= ((F[:, 2:] - F[:, 1:-1])*dt/dx)
    return U_n_tilde, dt

#==================================================================================================================================================#

def U_nplus1_2nd_order(U_n:np.ndarray, U_til:np.ndarray, F_til:np.ndarray, dt:float, dx:float, detective=False) -> np.ndarray:
    """
    Evolve the conserved parameters matrix ,U, in time. This complete the U_tilde_func time step after evaluating the new flux.

    Args:
        U_n (np.ndarray): The conserved parameters matrix at the n'th time step.
        U_til (np.ndarray): The conserved parameters matrix at the "n+1/2'th" time step.
        F_til (np.ndarray): The flux calculated from U_til.
        dt (float): The time step value.
        dx (float): The size of the cell.
        detective (bool, optional): Set to True for printing the shape of all the given matrices (U_n, U_til and F_til). Defaults to False.

    Returns:
        np.ndarray: Returns the updated U_n matrix after full time step index.
    """
    F_til_copy = np.copy(F_til)
    U_til_copy = np.copy(U_til)
    U_np1 = np.copy(U_n)
    if detective:
        print(f"U_n shape: {U_n.shape}\nU_tilde shape: {U_til.shape}\nF_tilde shape: {F_til.shape}")
        return None
    U_np1[:,1:-1] += (U_til_copy[:,1:-1] - (F_til_copy[:, 2:] - F_til_copy[:, 1:-1])*dt/dx)
    U_np1[:,1:-1] = U_np1[:,1:-1]*0.5
    return U_np1

#==================================================================================================================================================#

def tube_time_evolution_2nd_order(initial_left:Tuple[float, float, float], 
                                  initial_right:Tuple[float, float, float], 
                                  t_final:float, 
                                  M_x=100, 
                                  gamma=1.4, 
                                  x_0=0.5, **data) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve the tube in second order approximation using HLLC Riemann solver and Godunov scheme for second order.

    Args:
        initial_left (Tuple[float, float, float]): Parameters of the left side for the calculations as (density, velociy, pressure).
        initial_right (Tuple[float, float, float]): Parameters of the right side for the calculations as (density, velociy, pressure).
        t_final (float): The time value to evolve the variables.
        M_x (int, optional): The number of cells. Defaults to 100.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.
        x_0 (float, optional): The positon of the divider in the tube. Defaults to 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns the updated U matrix and the cells positions.
    """
    U, x_vec = setting_initial_tube_parameters(initial_left, initial_right, M_x, x_0, gamma, XandU=True) # U_mat, x_vec
    dx = x_vec[1] - x_vec[0]
    iteration = 0
    t_counter = 0
    while t_counter < t_final:
        F_on_sides = F_sides(U, gamma, **data)
        U_tilde, dt = U_tilde_func(U, F_on_sides, dx, gamma) # U_tilde is half time step, 
        F_tilde_on_sides = F_sides(U_tilde, gamma, **data)
        U = U_nplus1_2nd_order(U, U_tilde, F_tilde_on_sides, dt, dx)
        t_counter += dt
        iteration +=1
        # Rigid wall boundary conditions at x=0 / L
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
    # print(f'number of iterations (2nd order): {iteration}')
    return U , x_vec

#==================================================================================================================================================#

def primitive_parameters_vectors(U:np.ndarray, gamma=1.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # U = np.array([[rho0, rho1, rho2...], [rho*u...], [E...]])
    """
    Extract the primitive parameters (density, velociy, pressure) from the conserved parameters matrix U.

    Args:
        U (np.ndarray): The conserved parameters matrix.
        gamma (float, optional): The adiabatic index. Defaults to 1.4.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns the density vector, particle velocity vector and pressure vector.
    """
    rho_vec = U[0]
    u_vec = U[1]/rho_vec
    P_vec = (gamma - 1)*(U[2] - 0.5*(rho_vec * (u_vec**2))) # P = (gamma - 1)*(E - 0.5*(rho*u**2))
    return rho_vec, u_vec, P_vec

