"""
@author: Bárbara Rodrigues, Monse Guedes Ayala
@project: Krakenflex Decomposition Project

Auxiliary module of package 'Build' with functions which define the objective and constraints of the model.
"""
import numpy as np
import pyomo.environ as pyo

def cost_ALR_full(m):
    cost_import_grid   = sum(m.price_import[t] * m.i_G[t] for t in m.T)
    profit_export_grid = sum(m.price_export[t] * m.e_G[t] for t in m.T)
    total_grid         = cost_import_grid - profit_export_grid

    cost_DUoS_import   = sum(m.DUoS_import[t,l] * m.i_S[t,l] for t in m.T for l in m.L)
    cost_DUoS_export   = sum(m.DUoS_export[t,l] * m.e_S[t,l] for t in m.T for l in m.L)
    total_DUoS         = cost_DUoS_import + cost_DUoS_export

    dualized_constraint = sum(m.dual[t]*(m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L)) for t in m.T)
    augmentation = sum((m.dualgamma[t]/2)*((m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L))**2) for t in m.T)
    
    return total_grid + total_DUoS  + augmentation + dualized_constraint


def cost_ALR_location(m):
    cost_DUoS_export   = sum((m.DUoS_export[t,l]  + m.dual[t]) * m.e_S[t,l] for t in m.T for l in m.L)
    cost_DUoS_import   = sum((m.DUoS_import[t,l]  - m.dual[t]) * m.i_S[t,l] for t in m.T for l in m.L)

    homerun = sum((m.dualgamma[t]/2)*((m.commitment_i[t] + m.i_G[t] + sum(m.e_S_prime[t,l] for l in m.L_prime) + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S_prime[t,l] for l in m.L_prime) - sum(m.i_S[t,l] for l in m.L))**2) for t in m.T)
    return cost_DUoS_export + cost_DUoS_import + homerun 
    
def cost_ALR_site(m):
    cost_DUoS_export   = sum((m.DUoS_export[t,l]  + m.dual[t]) * m.e_S[t,l] for t in m.T for l in m.L)
    cost_DUoS_import   = sum((m.DUoS_import[t,l]  - m.dual[t]) * m.i_S[t,l] for t in m.T for l in m.L)
    
    homerun = sum((m.dualgamma[t]/2)*((m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L))**2) for t in m.T)
    

    return cost_DUoS_export + cost_DUoS_import + homerun 


def cost_ALR_site1(m):
    cost_DUoS = sum( m.DUoS_export[t,l]*m.e_S[t,l] + m.DUoS_import[t,l]*m.i_S[t,l] for t in m.T for l in m.L)
    dualss = sum(m.dual[t]*(sum(m.e_S[t,l] - m.i_S[t,l] for l in m.L)) for t in m.T)
    homerun = sum((m.dualgamma[t]/2)*((m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L))**2) for t in m.T)
    
    return cost_DUoS + dualss + homerun

def cost_ALR_portfolio1(m):
    stuff = sum( m.price_import[t]*m.i_G[t] - m.price_export[t]*m.e_G[t] for t in m.T)
    dualss = sum(m.dual[t]*(m.commitment_i[t] + m.i_G[t] - m.commitment_e[t] - m.e_G[t]) for t in m.T)
    homerun = sum((m.dualgamma[t]/2)*((m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L))**2) for t in m.T)
    
    return stuff + dualss + homerun

def cost_ALR_portfolio(m):
    cost_import_grid    = sum((m.price_import[t] + m.dual[t]) * m.i_G[t] for t in m.T)
    profit_export_grid  = sum((m.price_export[t] - m.dual[t]) * m.e_G[t] for t in m.T)
    commitment_constant = sum(m.dual[t] * (m.commitment_i[t] - m.commitment_e[t]) for t in m.T)
    homerun = sum((m.dualgamma[t]/2)*((m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L))**2) for t in m.T)
    
    return cost_import_grid - profit_export_grid + commitment_constant + homerun 

def cost_LR(m):
    cost_DUoS_export   = sum(m.DUoS_export[t,l] * m.e_S[t,l] for t in m.T for l in m.L)
    cost_DUoS_import   = sum(m.DUoS_import[t,l] * m.i_S[t,l] for t in m.T for l in m.L)

    cost_import_grid   = sum(m.price_import[t] * m.i_G[t] for t in m.T)
    profit_export_grid = sum(m.price_export[t] * m.e_G[t] for t in m.T)

    standard_objective = cost_DUoS_export + cost_DUoS_import + cost_import_grid - profit_export_grid

    lagrangian_part = sum((m.dual[t](m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L) - m.commitment_e[t] - m.e_G[t] - sum(m.i_S[t,l] for l in m.L)) for t in m.T))

    return standard_objective + lagrangian_part


def cost(m):
    '''Objecive Function for the general case.'''
    # Cost/Profit from trading with the grid
    cost_import_grid   = sum(m.price_import[t] * m.i_G[t] for t in m.T)
    profit_export_grid = sum(m.price_export[t] * m.e_G[t] for t in m.T)
    total_grid         = cost_import_grid - profit_export_grid
    
    # Cost DUoS
    cost_DUoS_import   = sum(m.DUoS_import[t,l] * m.i_S[t,l] for t in m.T for l in m.L)
    cost_DUoS_export   = sum(m.DUoS_export[t,l] * m.e_S[t,l] for t in m.T for l in m.L)
    total_DUoS         = cost_DUoS_import + cost_DUoS_export
        
    return total_grid + total_DUoS

def cost_sp_portfolio(m):
    '''Objective Function for subproblem portfolio. Cost/Profit from trading with the grid'''
    cost_import_grid    = sum((m.price_import[t] + m.dual[t]) * m.i_G[t] for t in m.T)
    profit_export_grid  = sum((m.price_export[t] + m.dual[t]) * m.e_G[t] for t in m.T)
    commitment_constant = sum(m.dual[t] * (m.commitment_i[t] - m.commitment_e[t]) for t in m.T)
    
    return cost_import_grid - profit_export_grid + commitment_constant
    
def cost_sp_location(m):
    '''Objective function Cost of DUoS'''
    cost_DUoS_import   = sum((m.DUoS_import[t,l]  - m.dual[t]) * m.i_S[t,l] for t in m.T for l in m.L)
    cost_DUoS_export   = sum((m.DUoS_export[t,l]  + m.dual[t]) * m.e_S[t,l] for t in m.T for l in m.L)
        
    return cost_DUoS_import + cost_DUoS_export

def soc_update(m,t,l,b):
    if t==min(m.T):
        return m.b[t,l,b] == m.initial_soc[l,b]
    else:
        charged = m.rte_charge[l,b] * m.c[t-1,l,b] * m.length_t
        discharged = 1/m.rte_discharge[l,b] * m.d[t-1,l,b] * m.length_t
        return m.b[t,l,b] == (1 - m.discharge_rate[l,b]) * m.b[t-1,l,b] + charged - discharged

def soc_final(m,l,b):
    t_max = max(m.T)
    charged = m.rte_charge[l,b] * m.c[t_max,l,b] * m.length_t
    discharged = 1/m.rte_discharge[l,b] * m.d[t_max,l,b] * m.length_t
    final_soc = (1 - m.discharge_rate[l,b]) * m.b[t_max,l,b] + charged - discharged
    return final_soc == m.initial_soc[l,b]

def soc_UB(m,t,l,b):
    return m.b[t,l,b] <= m.max_capacity[l,b]

def soc_LB(m,t,l,b):
    return m.b[t,l,b] >= m.min_capacity[l,b]

def cycle_limit(m,l,b):
    sum_c_d = sum(m.c[t,l,b] + m.d[t,l,b] for t in m.T) * m.length_t
    cycle_limit = m.max_n_cycles[l,b] * (m.max_capacity[l,b] - m.min_capacity[l,b])
    return sum_c_d <= cycle_limit

def charge_UB(m,t,l,b):
    return m.c[t,l,b] + m.d[t,l,b] <= m.max_rate[l,b]

def c_d_compl(m,t,l,b): #NOTE Never used
    return m.c[t,l,b] * m.d[t,l,b] == 0

def c_compl_bound(m,t,l,b):
    '''Battery Charge Complementarity Bound'''
    return m.c[t,l,b] <= m.z[t,l,b] * m.max_rate[l,b]

def d_compl_bound(m,t,l,b):
    '''Battery Discharge Complementarity Bound'''
    return m.d[t,l,b] <= (1-m.z[t,l,b]) * m.max_rate[l,b]

def grid_connect_limit(m,t,l):
    total_grid_usage = m.i_S[t,l] + m.e_S[t,l]
    return total_grid_usage <= m.grid_limit[t,l]

def s_r_compl(m,t,l): #NOTE Never used
    '''Site Import/Export Complementarity: Quadtratic'''
    return m.i_S[t,l] * m.e_S[t,l] == 0

def r_compl_bound(m,t,l):
    '''Site Import Complementarity Bound'''
    return m.i_S[t,l] <= m.delta[t,l] * m.grid_limit[t,l]

def s_compl_bound(m,t,l):
    '''Site Export Complementarity Bound'''
    return m.e_S[t,l] <= (1-m.delta[t,l]) * m.grid_limit[t,l]

def energy_balance_site(m,t,l):
    supply = m.generation[t,l] + sum(m.d[t,l,b]*m.length_t for b in m.B_max if (l,b) in m.LB) + m.i_S[t,l]
    demand = m.demand[t,l] + sum(m.c[t,l,b]*m.length_t for b in m.B_max if (l,b) in m.LB) + m.e_S[t,l]
    return supply == demand

def energy_balance_portfolio(m,t):
    supply = m.commitment_i[t] + m.i_G[t] + sum(m.e_S[t,l] for l in m.L)
    demand = m.commitment_e[t] + m.e_G[t] + sum(m.i_S[t,l] for l in m.L)
    return supply - demand == 0 #>= ???

def i_e_compl(m,t): #NOTE Never used
    return m.i_G[t] * m.e_G[t] == 0

def i_compl_bound(m,t):
    '''Grid Import Complementarity Bound'''
    big_M = m.commitment_e[t] - m.commitment_i[t] + sum(m.grid_limit[t,l] for l in m.L)
    return m.i_G[t] <= m.gamma[t] * big_M

def e_compl_bound(m,t):
    '''Grid Export Complementarity Bound'''
    big_M = m.commitment_i[t] - m.commitment_e[t] + sum(m.grid_limit[t,l] for l in m.L)
    return m.e_G[t] <= (1-m.gamma[t]) * big_M

def i_compl_bound_relax(m,t):
    '''Grid Import Complementarity Bound'''
    big_M = m.commitment_e[t] - m.commitment_i[t] + sum(m.grid_limit[t,l] for l in m.L)
    return m.i_G[t] <=  big_M

def e_compl_bound_relax(m,t):
    '''Grid Export Complementarity Bound'''
    big_M = m.commitment_i[t] - m.commitment_e[t] + sum(m.grid_limit[t,l] for l in m.L)
    return m.e_G[t] <=  big_M

def grid_connect_limit_raf(m,t):
    return m.i_G[t] + m.e_G[t] <= sum(m.grid_limit[t,l] for l in m.L)