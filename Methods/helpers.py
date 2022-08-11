import pyomo.environ as pyo
from Models.LocationsModel import LocationsModel
from Models.ModelType import ModelType
from Models.LPModel import LPModel
from Models.MILPModel import MILPModel
from Models.SiteModel import SiteModel
from Models.PortfolioModel import PortfolioModel
from Models.ALRModel import ALRModel
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
    if formulation == "LocationsModel":
        m = LocationsModel(type=ModelType.LocationsModels)
    if formulation == "ALRModel":
        m = ALRModel(type=ModelType.ALR)
    return m


def extract_from_locations(location_instance):
    """Given a solved location_instance, the function extracts the necessary information to be fed into the portfolio subproblem before solving"""
    exports_S_TL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_l))))
    imports_S_TL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_l))))
    charge_TBL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_b_max)), int(pyo.value(location_instance.N_l))))
    discharge_TBL = np.zeros((int(pyo.value(location_instance.N_t)), int(pyo.value(location_instance.N_b_max)), int(pyo.value(location_instance.N_l))))
    
    for t in location_instance.T:
        for l in location_instance.L:

            if pyo.value(location_instance.e_S[t,l]) < 0: #BUG prevention
                exports_S_TL[t-1,l-1] = max(0,pyo.value(location_instance.e_S[t,l])) #BUG somtimes this NonNegativeReal is negative (-1E-8)... huh
                print("BUG")

            exports_S_TL[t-1,l-1] = pyo.value(location_instance.e_S[t,l])
            imports_S_TL[t-1,l-1] = pyo.value(location_instance.i_S[t,l])


            for b in location_instance.B_max:
                charge_TBL[t-1,b-1,l-1] = pyo.value(location_instance.c[t,l,b])
                discharge_TBL[t-1,b-1,l-1] = pyo.value(location_instance.d[t,l,b])
    
    return exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL


def feed_to_portfolio(portfolio_instnace, exports_S_TL, imports_S_TL):
    """Function feeds the necessary paramter data to portfolio before being solved. """
    for t in portfolio_instnace.T:
        for l in portfolio_instnace.L:
            portfolio_instnace.e_S[t,l] = exports_S_TL[t-1,l-1]
            portfolio_instnace.i_S[t,l] = imports_S_TL[t-1,l-1]

    return portfolio_instnace


def extract_from_portfolio(portfolio_instance):
    exports_G_T = np.zeros(int(pyo.value(portfolio_instance.N_t)))
    imports_G_T = np.zeros(int(pyo.value(portfolio_instance.N_t)))

    for t in portfolio_instance.T:
        exports_G_T[t-1] = pyo.value(portfolio_instance.e_G[t])
        imports_G_T[t-1] = pyo.value(portfolio_instance.i_G[t])
        

    # print(exports_G_T)
    # print(imports_G_T)
    # print(pyo.value(portfolio_instance.Objective_Cost))
    # print(sum(pyo.value(portfolio_instance.dual[t])*(pyo.value(portfolio_instance.commitment_i[t]) - pyo.value(portfolio_instance.commitment_e[t])) for t in portfolio_instance.T))
    # print(sum(calculate_dualized_violation(portfolio_instance)))
    # print('hi')
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
        
        # a = pyo.value(portfolio_instance.commitment_i[t])
        # b = pyo.value(portfolio_instance.i_G[t])
        # c = sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L)
        # d = pyo.value(portfolio_instance.commitment_e[t])
        # e = pyo.value(portfolio_instance.e_G[t])
        # f = sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L)
        primal_residualsT[t-1] = primal_residual
        dualgammasT[t-1] = pyo.value(locations_instance.dualgamma[t])
        new_dualsT[t-1] =  pyo.value(locations_instance.dual[t]) + pyo.value(locations_instance.dualgamma[t])*primal_residual

        
        ###NOTE -- Project duals -- ist that a thing still???  -----
        #new_dualsT = project_dual(new_dualsT, portfolio_instance)
        
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

    obj_cost_locations = pyo.value(mL.Objective_Cost)
    obj_cost_portfolio = pyo.value(mP.Objective_Cost)

    return objective_cost_original, dualized_constraint_value, augmentation_value, minimization_objective, obj_cost_locations, obj_cost_portfolio
    
    
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
            #print("DONE")
        elif dualsT[t-1] > -pyo.value(portfolio_instance.price_export[t]):
            dualsT[t-1] = -pyo.value(portfolio_instance.price_export[t])
            #print("BOOM")
    return dualsT

def extract_solution(locations_instance, portfolio_instance):
    exports_G_T, imports_G_T = extract_from_portfolio(portfolio_instance)
    exports_S_TL, imports_S_TL = extract_from_locations(locations_instance)

    solution ={'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL }
    return solution

def ExchangeUpdateDuals(location_instances, i_G, e_G, imports_S_TL, exports_S_TL):
    new_dualsT = np.zeros(int(pyo.value(location_instances[0].N_t)))
    primal_residualsT = np.zeros(int(pyo.value(location_instances[0].N_t)))
    dualgammasT = np.zeros(int(pyo.value(location_instances[0].N_t)))
    
    m = location_instances[0]
    for t in m.T:
        primal_residualsT[t-1] = (pyo.value(m.commitment_i[t]) + i_G[t-1] + sum(exports_S_TL[t-1,l-1] for l in m.L_prime) - pyo.value(m.commitment_e[t]) - e_G[t-1] - sum(imports_S_TL[t-1,l-1] for l in m.L_prime))
        new_dualsT[t-1] = pyo.value(m.dual[t]) + pyo.value(m.dualgamma[t])*primal_residualsT[t-1]
        dualgammasT[t-1] = pyo.value(m.dualgamma[t])
   

    for l in range(len(location_instances)):
        for t in m.T:
            location_instances[l].dual[t] = new_dualsT[t-1]

            for l_prime in location_instances[l].L_prime:
                location_instances[l].e_S_prime[t,l_prime] = exports_S_TL[t-1,l_prime-1]
                location_instances[l].i_S_prime[t,l_prime] = imports_S_TL[t-1,l_prime-1]
                location_instances[l].e_G[t] = e_G[t-1]
                location_instances[l].i_G[t] = i_G[t-1]

    return new_dualsT, location_instances, primal_residualsT, dualgammasT


def feed_to_exchange_ADMM(location_instances, e_G, i_G):
    for l in location_instances:
        for t in location_instances[l].T:
            location_instances[l].e_G[t] = e_G[t-1]
            location_instances[l].i_G[t] = i_G[t-1]
    return location_instances

def extract_from_non_decomposed(instance):
    exports_G_T = np.zeros(int(pyo.value(instance.N_t)))
    imports_G_T = np.zeros(int(pyo.value(instance.N_t)))

    for t in instance.T:
        exports_G_T[t-1] = pyo.value(instance.e_G[t])
        imports_G_T[t-1] = pyo.value(instance.i_G[t])


    exports_S_TL = np.zeros((int(pyo.value(instance.N_t)), int(pyo.value(instance.N_l))))
    imports_S_TL = np.zeros((int(pyo.value(instance.N_t)), int(pyo.value(instance.N_l))))
    for t in instance.T:
        for l in instance.L:

            if pyo.value(instance.e_S[t,l]) < 0: #BUG prevention
                exports_S_TL[t-1,l-1] = max(0,pyo.value(instance.e_S[t,l])) #BUG somtimes this NonNegativeReal is negative (-1E-8)... huh
                print("BUG")

            exports_S_TL[t-1,l-1] = pyo.value(instance.e_S[t,l])
            imports_S_TL[t-1,l-1] = pyo.value(instance.i_S[t,l])
    
    return exports_G_T, imports_G_T, exports_S_TL, imports_S_TL
       


def extract_from_exchange_ADMM(location_instances):
    exports_S_TL =  np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_l_prime))))
    imports_S_TL = np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_l_prime))))
    charge_TBL = np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_b_max)), int(pyo.value(location_instances[0].N_l_prime))))
    discharge_TBL = np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_b_max)), int(pyo.value(location_instances[0].N_l_prime))))
    
    for l in range(len(location_instances)):
        single_site = pyo.value(location_instances[l].N_l)
        # print("----",l, single_site)
        for t in location_instances[l].T:
            exports_S_TL[t-1,l] = pyo.value(location_instances[l].e_S[t,single_site])
            imports_S_TL[t-1,l] = pyo.value(location_instances[l].i_S[t,single_site])

            for b in location_instances[l].B_max:
                charge_TBL[t-1,b-1,l] = pyo.value(location_instances[l].c[t,single_site,b])
                discharge_TBL[t-1,b-1,l] = pyo.value(location_instances[l].d[t,single_site,b])

    return exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL


def feasibility_heuristic(inst,  exports_S_TL, imports_S_TL):
    inst = inst.clone()
    i_G = np.zeros(int(pyo.value(inst.N_t)))
    e_G = np.zeros(int(pyo.value(inst.N_t)))

    # Heuristic Procedure to compute portfolio variables
    for t in inst.T:
        f = inst.commitment_e[t] - inst.commitment_i[t] \
            + sum(imports_S_TL[t-1,l] for l in range(imports_S_TL.shape[1]) )  \
            - sum(exports_S_TL[t-1,l] for l in range(exports_S_TL.shape[1]) )
        

        i_G[t-1] = max(f,0)
        e_G[t-1] = max(-f,0)


    return i_G, e_G


def feasibility_heuristic_BARBMONSE(inst:pyo.ConcreteModel):

    # Create a new object with a copy of `inst`
    feas_inst = inst.clone()

    # Heuristic Procedure to compute portfolio variables
    for t in inst.T:
        f = inst.commitment_e[t] - inst.commitment_i[t]\
             + sum(pyo.value(inst.i_S[t,l])for l in inst.L)\
             - sum(pyo.value(inst.e_S[t,l])for l in inst.L)
        
        feas_inst.i_G[t] = max(f,0)
        feas_inst.e_G[t] = max(-f,0)
        if f > 0: #Similarly i_G > 0
            feas_inst.gamma[t] = 1
        else:
            feas_inst.gamma[t] = 0

    return feas_inst

def update_the_duals123(locations_instance, e_G_heur, i_G_heur):
    new_dualsT = np.zeros(int(pyo.value(locations_instance.N_t)))
    dualgammasT = np.zeros(int(pyo.value(locations_instance.N_t)))
    primal_residualsT = np.zeros(int(pyo.value(locations_instance.N_t))) 
    for t in locations_instance.T:
        #Calculate primal residual for a given t
        primal_residual = pyo.value(locations_instance.commitment_i[t]) \
                            + pyo.value(i_G_heur[t-1]) \
                            + sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L) \
                            - pyo.value(locations_instance.commitment_e[t]) \
                            - pyo.value(e_G_heur[t]) \
                            - sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L)
        
        # a = pyo.value(portfolio_instance.commitment_i[t])
        # b = pyo.value(portfolio_instance.i_G[t])
        # c = sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L)
        # d = pyo.value(portfolio_instance.commitment_e[t])
        # e = pyo.value(portfolio_instance.e_G[t])
        # f = sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L)
        primal_residualsT[t-1] = primal_residual
        dualgammasT[t-1] = pyo.value(locations_instance.dualgamma[t])
        new_dualsT[t-1] =  pyo.value(locations_instance.dual[t]) + pyo.value(locations_instance.dualgamma[t])*primal_residual

        
        ###NOTE -- Project duals -- ist that a thing still???  -----
        #new_dualsT = project_dual(new_dualsT, portfolio_instance)
        
        #Update the new dual and in locations_instance and portfolio_instance.
        locations_instance.dual[t] = new_dualsT[t-1] #just a misalignment between pyomo iterator and numpy iterator
        # portfolio_instance.dual[t] = new_dualsT[t-1]
    

    return locations_instance, new_dualsT, primal_residualsT, dualgammasT

def update_the_duals1(locations_instance, e_G_heur, i_G_heur):
    new_dualsT = np.zeros(int(pyo.value(locations_instance.N_t)))
    dualgammasT = np.zeros(int(pyo.value(locations_instance.N_t)))
    primal_residualsT = np.zeros(int(pyo.value(locations_instance.N_t))) 
    for t in locations_instance.T:
        #Calculate primal residual for a given t
        primal_residual = pyo.value(locations_instance.commitment_i[t]) \
                            + pyo.value(i_G_heur[t-1]) \
                            + sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L) \
                            - pyo.value(locations_instance.commitment_e[t]) \
                            - pyo.value(e_G_heur[t-1]) \
                            - sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L)
        
        # a = pyo.value(portfolio_instance.commitment_i[t])
        # b = pyo.value(portfolio_instance.i_G[t])
        # c = sum(pyo.value(locations_instance.e_S[t,l]) for l in locations_instance.L)
        # d = pyo.value(portfolio_instance.commitment_e[t])
        # e = pyo.value(portfolio_instance.e_G[t])
        # f = sum(pyo.value(locations_instance.i_S[t,l]) for l in locations_instance.L)
        primal_residualsT[t-1] = primal_residual
        dualgammasT[t-1] = pyo.value(locations_instance.dualgamma[t])
        new_dualsT[t-1] =  pyo.value(locations_instance.dual[t]) + pyo.value(locations_instance.dualgamma[t])*primal_residual

        
        ###NOTE -- Project duals -- ist that a thing still???  -----
        #new_dualsT = project_dual(new_dualsT, portfolio_instance)
        
        #Update the new dual and in locations_instance and portfolio_instance.
        locations_instance.dual[t] = new_dualsT[t-1] #just a misalignment between pyomo iterator and numpy iterator
        # portfolio_instance.dual[t] = new_dualsT[t-1]
    

    return locations_instance, new_dualsT, primal_residualsT, dualgammasT


def calculate_obj_cost_exchange(location_instances, e_G, i_G, portfolio_instance):
    min_obj = np.zeros(int(len(location_instances))+1)
    SiteExport = 0
    SiteImport = 0
    li = location_instances
    for a in li:
        SiteExport += sum( li[a].DUoS_export[t,l]*pyo.value(li[a].e_S[t,l]) for l in li[a].L for t in li[a].T)
        SiteImport += sum( li[a].DUoS_import[t,l]*pyo.value(li[a].i_S[t,l]) for l in li[a].L for t in li[a].T)

    GridRevenue = sum(li[0].price_import[t]*i_G[t-1] - li[0].price_export[t]*e_G[t-1] for t in li[0].T)

    objective_cost_original = SiteExport + SiteImport + GridRevenue

    dualized_constraint_value = sum(li[0].dual[t]*(li[0].commitment_i[t] + i_G[t-1] + sum( pyo.value(li[a].e_S[t,l]) for a in li for l in li[a].L) - li[0].commitment_e[t] - e_G[t-1] - sum( pyo.value(li[a].i_S[t,l]) for a in li for l in li[a].L) ) for t in li[0].T)
    
    augmentation_value = sum((li[0].dualgamma[t]/2)*((li[0].commitment_i[t] + i_G[t-1] + sum( pyo.value(li[a].e_S[t,l]) for a in li for l in li[a].L) - li[0].commitment_e[t] - e_G[t-1] - sum( pyo.value(li[a].i_S[t,l]) for a in li for l in li[a].L) ))**2 for t in li[0].T)
    
    # i = 0 
    # for m in li:
    #     cost_DUoS_export   = sum((li[m].DUoS_export[t,l]  + li[m].dual[t]) * pyo.value(li[m].e_S[t,l]) for t in li[m].T for l in li[m].L)
    #     cost_DUoS_import   = sum((li[m].DUoS_import[t,l]  - li[m].dual[t]) * pyo.value(li[m].i_S[t,l]) for t in li[m].T for l in li[m].L)

    #     homerun = sum((li[m].dualgamma[t]/2)*((li[m].commitment_i[t] + i_G[t-1] + sum(pyo.value(li[m].e_S_prime[t,l]) for l in li[m].L_prime) + sum(pyo.value(li[m].e_S[t,l]) for l in li[m].L) - li[m].commitment_e[t] - e_G[t-1] - sum(pyo.value(li[m].i_S_prime[t,l]) for l in li[m].L_prime) - sum(pyo.value(li[m].i_S[t,l]) for l in li[m].L))**2) for t in li[m].T)
    #     min_obj[i] = cost_DUoS_export + cost_DUoS_import + homerun 
    #     i =+ 1 

    for a in li:
        min_obj[a+1] = pyo.value(li[a].Objective_Cost)

    min_obj[0] = pyo.value(portfolio_instance.Objective_Cost)

    return  objective_cost_original , dualized_constraint_value, augmentation_value, min_obj #, obj_cost_locations, obj_cost_portfolio
     


def feed_to_portfolio_portfolio_vars(portfolio_instance, imports_G_T, exports_G_T):
    for t in portfolio_instance.T:
        portfolio_instance.e_G[t] = exports_G_T[t-1]
        portfolio_instance.i_G[t] = imports_G_T[t-1]
    return portfolio_instance


def from_decision_vars_calculate_General_objective_function(decision_vars, m):
    dv = decision_vars
    original_objective_function = np.zeros(len(dv.keys()))
    try:
        TEMP = m.L_prime
    except:
        TEMP = m.L
    

    for iter in dv:
        local_export =  sum(m.DUoS_export[t,l]*dv[iter]['e_S'][t-1,l-1] for t in m.T for l in TEMP)
        local_import =  sum(m.DUoS_import[t,l]*dv[iter]['i_S'][t-1,l-1] for t in m.T for l in TEMP)

        grid_import = sum(m.price_import[t]*dv[iter]['i_G'][t-1] for t in m.T)
        grid_export = sum(m.price_export[t]*dv[iter]['e_G'][t-1] for t in m.T)

        original_objective_function[iter] = local_export + local_import + grid_import - grid_export

    return original_objective_function