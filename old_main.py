import Methods.helpers as helpers
import time
import pyomo.environ as pyo
import numpy as np


SOLVER_NAME = 'gurobi'#'amplxpress'
INSTANCE_SIZE = 2
EQUAL_PRICES = False
MAX_ITER = 10
TOLERANCE = 1

def solve_MILP():
    print("> Solving MILP")
    solver = helpers.build_solver(SOLVER_NAME)
    model = helpers.build_model('MILPModel')

    instance = model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    start_time = time.perf_counter()
    result = solver.solve(instance)
    print(time.perf_counter()- start_time)


    print(pyo.value(instance.Objective_Cost))

    decision_vars ={}
    exports_G_T, imports_G_T, exports_S_TL, imports_S_TL = helpers.extract_from_non_decomposed(instance)
    decision_vars[0] = {'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL }

    return decision_vars, instance, result

def solve_LP():
    print("> Solving LP")
    solver = helpers.build_solver(SOLVER_NAME)
    model = helpers.build_model('LPModel')

    instance = model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    start_time = time.perf_counter()
    result = solver.solve(instance)
    print(time.perf_counter()- start_time)


    print(pyo.value(instance.Objective_Cost))

    decision_vars ={}
    exports_G_T, imports_G_T, exports_S_TL, imports_S_TL = helpers.extract_from_non_decomposed(instance)
    decision_vars[0] = {'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL }

    return decision_vars, instance, result

def solve_ALR():
    """Do not use, actually we are not maximizing lambda (dual variable) so it has a terrible lower bound."""
    print("> Solving ALR")
    solver = helpers.build_solver('gurobi')
    model = helpers.build_model('ALRModel')

    instance = model.build_instance(instance_size=2, equal_prices=False)

    start_time = time.perf_counter()
    result = solver.solve(instance)
    print(time.perf_counter()- start_time)

    print(pyo.value(instance.Objective_Cost))

    decision_vars ={}
    exports_G_T, imports_G_T, exports_S_TL, imports_S_TL = helpers.extract_from_non_decomposed(instance)
    decision_vars[0] = {'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL }

    return decision_vars, instance, result

def solve_ADMM():
    print("> Solving ADMM")
    solver = helpers.build_solver(SOLVER_NAME)
 
    # -- Create |L| +1 decomposed subproblems -- 
    location_models = {}
    location_instances = {}
    for l in range(INSTANCE_SIZE):
        print(f"   Creating Subproblem for Location {l+1}")
        location_models[l] = helpers.build_model("LocationsModel")
        location_instances[l] = location_models[l].build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES, site_id=l+1)


    
        for k in range(INSTANCE_SIZE):
            for t in location_instances[l].T:
                # location_instances[l].dual[t] = 2000
                location_instances[l].dualgamma[t] = 50
                location_instances[l].e_S_prime[t,k+1] = 0
                location_instances[l].i_S_prime[t,k+1] = 0
                location_instances[l].i_G[t] = 0
                location_instances[l].e_G[t] = 0





    computational_data = {'obj_cost': [], 'dualized_constraint_value' : [], 'augmentation_value': [], 'dualsT': [], 'primal_residualsT': [], 'dualgammasT' : [], 'min_obj' :[] }
    decision_vars = {}
    new_dualsT = np.zeros(int(pyo.value(location_instances[0].N_t)))
    primal_residualsT = np.zeros(int(pyo.value(location_instances[0].N_t)))
    objective_cost_original, dualized_constraint_value, augmentation_value = 0, 0, 0
    i_G = np.zeros(int(pyo.value(location_instances[0].N_t)))
    e_G = np.zeros(int(pyo.value(location_instances[0].N_t)))
    exports_S_TL =  np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_l_prime))))
    imports_S_TL = np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_l_prime))))
    charge_TBL = np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_b_max)), int(pyo.value(location_instances[0].N_l_prime))))
    discharge_TBL = np.zeros((int(pyo.value(location_instances[0].N_t)), int(pyo.value(location_instances[0].N_b_max)), int(pyo.value(location_instances[0].N_l_prime))))
    dualgammasT = np.zeros(int(pyo.value(location_instances[0].N_t)))
    min_obj = np.zeros(int(len(location_instances)))

    #-- ADMM  -- 
    iter = 0
    while(iter < MAX_ITER):
        print(f"iteration {iter}")

        # Store computational Results
        computational_data['obj_cost'].append(objective_cost_original)
        computational_data['dualized_constraint_value'].append(dualized_constraint_value)
        computational_data['augmentation_value'].append(augmentation_value)
        computational_data['dualsT'].append(new_dualsT)
        computational_data['primal_residualsT'].append(primal_residualsT)
        computational_data['dualgammasT'].append(dualgammasT)
        computational_data['min_obj'].append(min_obj)

        decision_vars[iter] = {'e_G': e_G, 'i_G': i_G, 'e_S': exports_S_TL, 'i_S': imports_S_TL, 'charge_TBL': charge_TBL, 'discharge_TBL': discharge_TBL }

        for l in range(INSTANCE_SIZE):
            result_l = solver.solve(location_instances[l])

        exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL = helpers.extract_from_exchange_ADMM(location_instances)

        i_G, e_G = helpers.feasibility_heuristic(location_instances[0], exports_S_TL, imports_S_TL)

        new_dualsT, location_instances, primal_residualsT = helpers.ExchangeUpdateDuals(location_instances, i_G, e_G, imports_S_TL, exports_S_TL)
        objective_cost_original, dualized_constraint_value, augmentation_value, min_obj  = helpers.calculate_obj_cost_exchange(location_instances, e_G, i_G)
        # obj_cost_locations, obj_cost_portfolio 
     
        iter += 1 

    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['min_obj'] = np.stack(computational_data['min_obj'],axis=0)

    return location_instances, computational_data, decision_vars

    
def solve_RFL():
    """Sequential ADMM, first solve 1st subproblem, feed those vars as params into 2nd subproblem, 
    feed 2nd subproblem vars into 1st subproblem as params then update duals, and repeat """

    #TODO understand how updating duals works in this specific problem!!!!
    #TODO Try to make SiteModel and PortfolioModel LPs!!!!!! 

    #IDEA: what am i initializing the duals to? maybe thats the issue. This seems not wanting to work in sovler.
    #IDEA: rewrite the monster, defintiely the lambda cant be taken there. (done, homerun)
    #IDEA: Use persitant solver for better perforamnce


    computational_data = {'obj_cost': [], 'dualized_constraint_value' : [], 'augmentation_value': [], 'dualsT': [], 'primal_residualsT': [], 'dualgammasT' : [], 'min_obj' :[], 'obj_cost_locations' :[], 'obj_cost_portfolio' :[] }
    decision_vars = {}

    print(" > Solving RFL")
    solver = helpers.build_solver(SOLVER_NAME)
    # solver.options['NonConvex'] = 2 #THE BUG! FINALLY FIXED

    locations_model = helpers.build_model("SiteModel")
    locations_instance = locations_model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    portfolio_model = helpers.build_model("PortfolioModel")
    portfolio_instance = portfolio_model.build_instance(instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)

    # Intialize for the very first solve 
    for t in locations_instance.T:
        # locations_instance.dual[t] = 1 #Do not initialize, they have to be within the optimal window
        locations_instance.dualgamma[t] = 250 # Fixed dualgamma for now
        locations_instance.i_G[t] = 0
        locations_instance.e_G[t] = 0

    for t in portfolio_instance.T:
        # portfolio_instance.dual[t] = 1 #Do not initialize, they have to be within the optimal window
        portfolio_instance.dualgamma[t] = 250
        for l in portfolio_instance.L:
            portfolio_instance.i_S[t,l] = 0
            portfolio_instance.e_S[t,l] = 0

    iter = 0
    objective_cost_original, dualized_constraint_value, augmentation_value, min_obj,  obj_cost_locations, obj_cost_portfolio = 0,0,0,0,0,0
    primal_residualsT = np.zeros(int(pyo.value(locations_instance.N_t)))
    dualgammasT = np.zeros(int(pyo.value(locations_instance.N_t)))
    dualsT = np.zeros(int(pyo.value(locations_instance.N_t)))
    exports_G_T, imports_G_T = np.zeros(int(pyo.value(locations_instance.N_t))), np.zeros(int(pyo.value(locations_instance.N_t)))
    exports_S_TL, imports_S_TL = np.zeros((int(pyo.value(locations_instance.N_t)), int(pyo.value(locations_instance.N_l)))), np.zeros((int(pyo.value(locations_instance.N_t)), int(pyo.value(locations_instance.N_l))))
    charge_TBL = np.zeros((int(pyo.value(locations_instance.N_t)), int(pyo.value(locations_instance.N_b_max)), int(pyo.value(locations_instance.N_l))))
    discharge_TBL = np.zeros((int(pyo.value(locations_instance.N_t)), int(pyo.value(locations_instance.N_b_max)), int(pyo.value(locations_instance.N_l))))

    # SEQUENTIAL ADMM 
    while(iter < MAX_ITER):
        print(f"iteration {iter}")

        # Store computational Results
        computational_data['obj_cost'].append(objective_cost_original)
        computational_data['dualized_constraint_value'].append(dualized_constraint_value)
        computational_data['augmentation_value'].append(augmentation_value)
        computational_data['min_obj'].append(min_obj)
        computational_data['primal_residualsT'].append(primal_residualsT)
        computational_data['dualgammasT'].append(dualgammasT)
        computational_data['dualsT'].append(dualsT)
        computational_data['obj_cost_locations'].append(obj_cost_locations)
        computational_data['obj_cost_portfolio'].append(obj_cost_portfolio)

        decision_vars[iter] = {'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL, 'charge_TBL': charge_TBL, 'discharge_TBL': discharge_TBL}

        
        result_p                                                       = solver.solve(locations_instance) # Solve locations subproblem
        # print(result_p)
        exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL          = helpers.extract_from_locations(locations_instance) # Extract data from locations supbroblem 
        portfolio_instance                                             = helpers.feed_to_portfolio(portfolio_instance, exports_S_TL, imports_S_TL) # Feed locations subproblem data into portfolio subproblem
        result_l                                                       = solver.solve(portfolio_instance) # Solve portfolio subproblem
        # print(result_l)
        exports_G_T, imports_G_T                                       = helpers.extract_from_portfolio(portfolio_instance) # Extract data from portfolio subproblem
        locations_instance                                             = helpers.feed_to_locations(locations_instance, exports_G_T, imports_G_T) # Feed portfolio subproblem data into locations subproblem
        

        locations_instance, portfolio_instance, dualsT, primal_residualsT, dualgammasT    = helpers.update_the_duals(locations_instance, portfolio_instance) # Update the dual variables
        objective_cost_original, dualized_constraint_value, augmentation_value, min_obj, obj_cost_locations, obj_cost_portfolio   = helpers.calculate_obj_cost(locations_instance, portfolio_instance)
     

        #TODO Termination Condition
        # num_of_converged_duals = 0
        # for t in portfolio_instance.T:
        #     if ((dualsT[t-1] - computational_data['dualsT'][iter][t-1])/computational_data['dualsT'][iter][t-1] < 1E-2):
        #         num_of_converged_duals += 1 
        # if num_of_converged_duals == pyo.value(portfolio_instance.N_t) -4:
        #     print(f"Termination Condition, iteration {iter}")



        iter += 1 
    # Quickly Process Computational Data: lists of lists are covnerted into a numpy matrix. 
    # Indicies are in order (iteration, time period)
    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['dualgammasT'] = np.stack(computational_data['dualgammasT'],axis=0)
    


    return computational_data, portfolio_instance, locations_instance, decision_vars


if __name__ == '__main__':
    # solve_LP()
    # solve_MILP()
    # solve_ADMM() # This is where each location is also a subproblem ie |L|+1 subproblems
    # solve_RFL() # Just two subproblems, one for portfolio, onr for locations:::: Sequentially 
    # solve_ALR()
    pass