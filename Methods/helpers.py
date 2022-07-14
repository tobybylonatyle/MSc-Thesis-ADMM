import pyomo.environ as pyo
from Models.ModelType import ModelType
from Models.LPModel import LPModel
from Models.MILPModel import MILPModel
from Models.SiteModel import SiteModel
from Models.PortfolioModel import PortfolioModel
import numpy as np
from pyomo.environ import value as val


# -- Initialize Solver --
def build_solver(solver_name):
    return pyo.SolverFactory(solver_name)


# -- Build model --
def build_model(formulation):
    if formulation == "LPModel":
        m = LPModel(type=ModelType.LP_Rafal)
    if formulation == "MILPModel":
        m = MILPModel(type=ModelType.MILP)
    if formulation == "SiteModel":
        m = SiteModel(type=ModelType.ALR_Site)
    if formulation == "PortfolioModel":
        m = PortfolioModel(type=ModelType.ALR_Portfolio)
    return m


def extract_from_locations(location_instance):
    """Given a solved location_instance, the function extracts the necessary information to be fed into the portfolio subproblem before solving"""
    exports_S_TL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_l))))
    imports_S_TL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_l))))
    for t in location_instance.T:
        for l in location_instance.L:

            if pyo.value(location_instance.e_S[t,l]) < 0: #BUG prevention
                exports_S_TL[t-1,l-1] = max(0,pyo.value(location_instance.e_S[t,l])) #BUG somtimes this NonNegativeReal is negative (-1E-8)... huh
                print("BUG")

            exports_S_TL[t-1,l-1] = pyo.value(location_instance.e_S[t,l])
            imports_S_TL[t-1,l-1] = pyo.value(location_instance.i_S[t,l])
    
    return exports_S_TL, imports_S_TL


def feed_to_portfolio(portfolio_instnace, exports_S_TL, imports_S_TL):
    """Function feeds the necessary paramter data to portfolio before being solved. """
    for t in portfolio_instnace.T:
        for l in portfolio_instnace.L:
            portfolio_instnace.e_S[t,l] = exports_S_TL[t-1,l-1]
            portfolio_instnace.i_S[t,l] = imports_S_TL[t-1,l-1]

    return portfolio_instnace


def extract_from_portfolio(portfolio_instance):
    exports_G_T = []
    imports_G_T = []

    for t in portfolio_instance.T:
        exports_G_T.append(pyo.value(portfolio_instance.e_G[t]))
        imports_G_T.append(pyo.value(portfolio_instance.i_G[t]))

    return exports_G_T, imports_G_T


def feed_to_locations(locations_instance, exports_G_T, imports_G_TL):
    for t in locations_instance.T:
        locations_instance.e_G[t] = exports_G_T[t-1]
        locations_instance.i_G[t] = imports_G_TL[t-1]
    return locations_instance

def update_the_duals(locations_instance, portfolio_instance):
    new_dualsT = np.zeros(int(pyo.value(locations_instance.N_t)))
    dualgammasT = np.zeros(int(pyo.value(locations_instance.N_t)))
    primal_residualsT = np.zeros(int(pyo.value(locations_instance.N_t))) 
      
    for t in locations_instance.T:
        #Calculate primal residual for a given t
        primal_residual = pyo.value(portfolio_instance.commitment_i[t]) \
                            + pyo.value(portfolio_instance.i_G[t]) \
                            + sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L) \
                            - pyo.value(portfolio_instance.commitment_e[t]) \
                            - pyo.value(portfolio_instance.e_G[t]) \
                            - sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L)
        

        primal_residualsT[t-1] = primal_residual
        dualgammasT[t-1] = pyo.value(locations_instance.dualgamma[t])
        new_dualsT[t-1] =  pyo.value(locations_instance.dual[t]) + pyo.value(locations_instance.dualgamma[t])*primal_residual

        ###NOTE -- Project duals -- ist that a thing still???  -----
        # new_dualsT = project_dual(new_dualsT, portfolio_instance)
        
        #Update the new dual and in locations_instance and portfolio_instance.
        locations_instance.dual[t] = new_dualsT[t-1] #just a misalignment between pyomo iterator and numpy iterator
        portfolio_instance.dual[t] = new_dualsT[t-1]

    return locations_instance, portfolio_instance, new_dualsT, primal_residualsT, dualgammasT


def calculate_obj_cost(locations_instance, portfolio_instance):
    SiteExport = sum( locations_instance.DUoS_export[t,l]*pyo.value(locations_instance.e_S[t,l]) for t in locations_instance.T for l in locations_instance.L)
    SiteImport = sum( locations_instance.DUoS_import[t,l]*pyo.value(locations_instance.i_S[t,l]) for t in locations_instance.T for l in locations_instance.L)
    GridExport = sum( portfolio_instance.price_export[t]*pyo.value(portfolio_instance.e_G[t]) for t in portfolio_instance.T)
    GridImport = sum( portfolio_instance.price_import[t]*pyo.value(portfolio_instance.i_G[t]) for t in portfolio_instance.T)
    objective_cost_original = SiteExport + SiteImport + GridImport - GridExport

    mL = locations_instance
    mP = portfolio_instance
    dualized_constraint_value = sum(pyo.value(mP.dual[t])*(pyo.value(mP.commitment_i[t])+pyo.value(mP.i_G[t])+sum(pyo.value(mL.e_S[t,l]) for l in mL.L) - pyo.value(mP.commitment_e[t]) - pyo.value(mP.e_G[t]) - sum(pyo.value(mL.i_S[t,l]) for l in mL.L)) for t in mP.T)

    augmentation_value = sum((pyo.value(mP.dualgamma[t])/2)*((pyo.value(mP.commitment_i[t])+pyo.value(mP.i_G[t])+sum(pyo.value(mL.e_S[t,l]) for l in mL.L) - pyo.value(mP.commitment_e[t]) - pyo.value(mP.e_G[t]) - sum(pyo.value(mL.i_S[t,l]) for l in mL.L))**2) for t in mP.T)
    
    
    minimization_objective = objective_cost_original + dualized_constraint_value + augmentation_value

    return objective_cost_original, dualized_constraint_value, augmentation_value, minimization_objective
    
    
def calculate_dualized_violation(m):
    '''Just by how much the dualized (complicating constraint: energy balance protfolio) is not actually zero.'''
    violationT = np.zeros(int(pyo.value(m.N_t)))
    for t in m.T:
        violationT[t-1] = pyo.value(m.commitment_i[t])+pyo.value(m.i_G[t])+sum(pyo.value(m.e_S[t,l]) for l in m.L) - pyo.value(m.commitment_e[t]) - pyo.value(m.e_G[t]) - sum(pyo.value(m.i_S[t,l]) for l in m.L)
    
    # violation = sum(pyo.value(m.commitment_i[t])+pyo.value(m.i_G[t])+sum(pyo.value(m.e_S[t,l]) for l in m.L) - pyo.value(m.commitment_e[t]) - pyo.value(m.e_G[t]) - sum(pyo.value(m.i_S[t,l]) for l in m.L) for t in m.T)

    return violationT


def project_dual(dualsT, portfolio_instance):
    """
    Documentation: Project dual multipliers outside the interval [-import price, -export price] back into this interval
    """
    for t in portfolio_instance.T:
        if dualsT[t-1] < -pyo.value(portfolio_instance.price_import[t]):
            dualsT[t-1] = -pyo.value(portfolio_instance.price_import[t])
            print("DONE")
        elif dualsT[t-1] > -pyo.value(portfolio_instance.price_export[t]):
            dualsT[t-1] = -pyo.value(portfolio_instance.price_export[t])
            print("BOOM")
    return dualsT
