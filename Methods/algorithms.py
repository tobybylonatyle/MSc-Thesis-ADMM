import Methods.helpers as helpers
import time
import pyomo.environ as pyo
import numpy as np

def solve_MILP(solver_name, instance_size, equal_prices):
    """MILP Formulation of the Problem"""
    print("> Solving MILP")
    time_complexity = {'instantiating_single_model':[],'solving_single_model': [], 'instantiating_locations' : [], 'instantiating_portfolio': [], 'solving_locations' : [], 'solving_portfolio': [], 'algorithm_time':[], 'heuristic_time':[]}

    solver = helpers.build_solver(solver_name)

    instantiating_single_model_start = time.perf_counter()
    model = helpers.build_model('MILPModel')
    instance = model.build_instance(instance_size=instance_size, equal_prices=equal_prices)
    time_complexity['instantiating_single_model'] = time.perf_counter() - instantiating_single_model_start

    solving_single_model_start = time.perf_counter()
    result = solver.solve(instance)
    time_complexity['solving_single_model'] = time.perf_counter() - solving_single_model_start

    print(pyo.value(instance.Objective_Cost))

    decision_vars ={}
    exports_G_T, imports_G_T, exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL = helpers.extract_from_non_decomposed(instance)
    decision_vars[0] = {'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL, 'charge_TBL': charge_TBL, 'discharge_TBL': discharge_TBL }

    return decision_vars, instance, result, time_complexity


def solve_LP(solver_name, instance_size, equal_prices):
    print("> Solving LP")
    time_complexity = {'instantiating_single_model':[],'solving_single_model': [], 'instantiating_locations' : [], 'instantiating_portfolio': [], 'solving_locations' : [], 'solving_portfolio': [], 'algorithm_time':[], 'heuristic_time':[]}

    solver = helpers.build_solver(solver_name)
    instantiating_single_model_start = time.perf_counter()
    model = helpers.build_model('LPModel')
    instance = model.build_instance(instance_size=instance_size, equal_prices=equal_prices)
    time_complexity['instantiating_single_model'] = time.perf_counter() - instantiating_single_model_start

    solve_single_model_start = time.perf_counter()
    result = solver.solve(instance)
    time_complexity['solving_single_model'] = time.perf_counter()- solve_single_model_start


    print(pyo.value(instance.Objective_Cost))

    decision_vars ={}
    exports_G_T, imports_G_T, exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL = helpers.extract_from_non_decomposed(instance)
    decision_vars[0] = {'e_G': exports_G_T, 'i_G': imports_G_T, 'e_S': exports_S_TL, 'i_S': imports_S_TL, 'charge_TBL': charge_TBL, 'discharge_TBL': discharge_TBL }

    return decision_vars, instance, result, time_complexity


def solve_two_block_ADMM(solver_name, instance_size, equal_prices, max_iter, dual_gamma:int, lambda_init:str):
    """Sequential 2 block ADMM, SiteModel and PortfolioModel """
    print(" > Solving Sequential 2 block ADMM")

    computational_data = {'obj_cost': [], 'dualized_constraint_value' : [], 'augmentation_value': [], 'dualsT': [], 'primal_residualsT': [], 'dualgammasT' : [], 'min_obj' :[], 'obj_cost_locations' :[], 'obj_cost_portfolio' :[] }
    decision_vars = {}
    time_complexity = {'instantiating_single_model':[],'solving_single_model': [], 'instantiating_locations' : [], 'instantiating_portfolio': [], 'solving_locations' : [], 'solving_portfolio': [], 'algorithm_time':[], 'heuristic_time':[]}


    
    solver = helpers.build_solver(solver_name)
    # solver.options['NonConvex'] = 2 #THE BUG! FINALLY FIXED

    instantiating_locations_start = time.perf_counter()
    locations_model = helpers.build_model("SiteModel")
    locations_instance = locations_model.build_instance(instance_size=instance_size, equal_prices=equal_prices, lambda_init=lambda_init)
    time_complexity['instantiating_locations'] = time.perf_counter() - instantiating_locations_start

    instantiating_portfolio_start = time.perf_counter()
    portfolio_model = helpers.build_model("PortfolioModel")
    portfolio_instance = portfolio_model.build_instance(instance_size=instance_size, equal_prices=equal_prices)
    time_complexity['instantiating_portfolio'] = time.perf_counter() - instantiating_portfolio_start

    # Intialize for the very first solve 
    for t in locations_instance.T:
        # locations_instance.dual[t] = 1 #Do not initialize, they have to be within the optimal window
        locations_instance.dualgamma[t] = dual_gamma # Fixed dualgamma for now
        locations_instance.i_G[t] = 0
        locations_instance.e_G[t] = 0

    for t in portfolio_instance.T:
        # portfolio_instance.dual[t] = 1 #Do not initialize, they have to be within the optimal window
        portfolio_instance.dualgamma[t] = dual_gamma
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
    while(iter < max_iter):
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

        algorithm_time_start = time.perf_counter()

        solving_locations_start = time.perf_counter()
        result_p                                                       = solver.solve(locations_instance) # Solve locations subproblem
        # print(result_p)
        time_complexity['solving_locations'].append(time.perf_counter() - solving_locations_start)

        exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL          = helpers.extract_from_locations(locations_instance) # Extract data from locations supbroblem 
        portfolio_instance                                             = helpers.feed_to_portfolio(portfolio_instance, exports_S_TL, imports_S_TL) # Feed locations subproblem data into portfolio subproblem
        
        solving_portfolio_start = time.perf_counter()
        result_l                                                       = solver.solve(portfolio_instance) # Solve portfolio subproblem
        # print(result_l)
        time_complexity['solving_portfolio'].append(time.perf_counter() - solving_portfolio_start)

        exports_G_T, imports_G_T                                       = helpers.extract_from_portfolio(portfolio_instance) # Extract data from portfolio subproblem
        # print(exports_G_T)
        # print(imports_G_T)
        locations_instance                                             = helpers.feed_to_locations(locations_instance, exports_G_T, imports_G_T) # Feed portfolio subproblem data into locations subproblem
        

        locations_instance, portfolio_instance, dualsT, primal_residualsT, dualgammasT    = helpers.update_the_duals(locations_instance, portfolio_instance) # Update the dual variables
        
        time_complexity['algorithm_time'].append(time.perf_counter() - algorithm_time_start)
        
        objective_cost_original, dualized_constraint_value, augmentation_value, min_obj, obj_cost_locations, obj_cost_portfolio   = helpers.calculate_obj_cost(locations_instance, portfolio_instance)
     


        #NOTE Select Termination Condition



        iter += 1 
    # Quickly Process Computational Data: lists of lists are covnerted into a numpy matrix. 
    # Indicies are in order (iteration, time period)
    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['dualgammasT'] = np.stack(computational_data['dualgammasT'],axis=0)
    


    return computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity

def solve_modified_two_block_ADMM(solver_name, instance_size, equal_prices, max_iter, dual_gamma:int, lambda_init:str):
    """Sequential 2 block ADMM, SiteModel and PortfolioModel(USES HEURISTIC!!!) """
    print(" > Solving Sequential MODIFIED 2 block ADMM")

    computational_data = {'obj_cost': [], 'dualized_constraint_value' : [], 'augmentation_value': [], 'dualsT': [], 'primal_residualsT': [], 'dualgammasT' : [], 'min_obj' :[], 'obj_cost_locations' :[], 'obj_cost_portfolio' :[] }
    decision_vars = {}
    time_complexity = {'instantiating_single_model':[],'solving_single_model': [], 'instantiating_locations' : [], 'instantiating_portfolio': [], 'solving_locations' : [], 'solving_portfolio': [], 'algorithm_time':[], 'heuristic_time':[]}


    
    solver = helpers.build_solver(solver_name)
    # solver.options['NonConvex'] = 2 #THE BUG! FINALLY FIXED

    instantiating_locations_start = time.perf_counter()
    locations_model = helpers.build_model("SiteModel")
    locations_instance = locations_model.build_instance(instance_size=instance_size, equal_prices=equal_prices, lambda_init=lambda_init)
    time_complexity['instantiating_locations'] = time.perf_counter() - instantiating_locations_start

    # portfolio_model = helpers.build_model("PortfolioModel")
    # portfolio_instance = portfolio_model.build_instance(instance_size=instance_size, equal_prices=equal_prices)

    # Intialize for the very first solve 
    for t in locations_instance.T:
        # locations_instance.dual[t] = 1 #Do not initialize, they have to be within the optimal window
        locations_instance.dualgamma[t] = dual_gamma # Fixed dualgamma for now
        locations_instance.i_G[t] = 0
        locations_instance.e_G[t] = 0

    # for t in portfolio_instance.T:
    #     # portfolio_instance.dual[t] = 1 #Do not initialize, they have to be within the optimal window
    #     portfolio_instance.dualgamma[t] = dual_gamma
    #     for l in portfolio_instance.L:
    #         portfolio_instance.i_S[t,l] = 0
    #         portfolio_instance.e_S[t,l] = 0

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
    while(iter < max_iter):
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

        algorithm_time_start = time.perf_counter()

        solving_locations_start = time.perf_counter()
        result_p                                                       = solver.solve(locations_instance) # Solve locations subproblem
        # print(result_p)
        time_complexity['solving_locations'].append( time.perf_counter() - solving_locations_start)

        exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL          = helpers.extract_from_locations(locations_instance) # Extract data from locations supbroblem 
        
        # portfolio_instance                                             = helpers.feed_to_portfolio(portfolio_instance, exports_S_TL, imports_S_TL) # Feed locations subproblem data into portfolio subproblem
        heuristic_start_time = time.perf_counter()
        imports_G_T, exports_G_T                                       = helpers.feasibility_heuristic(locations_instance, exports_S_TL, imports_S_TL)
        time_complexity['heuristic_time'].append(time.perf_counter() - heuristic_start_time)

        # locations_instance = helpers.feasibility_heuristic_BARBMONSE(locations_instance)
        # print(exports_G_T)
        # print(imports_G_T)
        # portfolio_instance                                             = helpers.feed_to_portfolio_portfolio_vars(portfolio_instance, imports_G_T, exports_G_T)
        # result_l                                                       = solver.solve(portfolio_instance) # Solve portfolio subproblem
        # print(result_l)
        # exports_G_T_test, imports_G_T_test                                      = helpers.extract_from_portfolio(portfolio_instance) # Extract data from portfolio subproblem
        # print(exports_G_T_test)
        locations_instance                                             = helpers.feed_to_locations(locations_instance, exports_G_T, imports_G_T) # Feed portfolio subproblem data into locations subproblem
        

        locations_instance, dualsT, primal_residualsT, dualgammasT    = helpers.update_the_duals1(locations_instance, exports_G_T, imports_G_T) # Update the dual variables
        # locations_instance, portfolio_instance, new_dualsT, primal_residualsT, dualgammasT = helpers.update_the_duals(locations_instance, portfolio_instance)
        # objective_cost_original, dualized_constraint_value, augmentation_value, min_obj, obj_cost_locations, obj_cost_portfolio   = helpers.calculate_obj_cost(locations_instance, portfolio_instance)
     
        time_complexity['algorithm_time'].append(time.perf_counter() - algorithm_time_start)
        #TODO Termination Condition
        # Euclidean_Norm_Of_Dual_Convergence =np.linalg.norm((computational_data['dualsT'][iter] - computational_data['dualsT'][iter -1 ])/computational_data['dualsT'][iter -1 ])
        # if(Euclidean_Norm_Of_Dual_Convergence < 1E-4):
        #     break
       


        iter += 1 
    # Quickly Process Computational Data: lists of lists are covnerted into a numpy matrix. 
    # Indicies are in order (iteration, time period)
    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['dualgammasT'] = np.stack(computational_data['dualgammasT'],axis=0)
    
    portfolio_instance= 0
    return computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity

def solve_exchange_ADMM(solver_name, instance_size, equal_prices, max_iter, dual_gamma:int, lambda_init:str):
    """small bug with this one somewhere. non convergant.. but thasts kinda expected so maybe no bug..."""
    print("> Solving Exchange ADMM")
    time_complexity = {'instantiating_single_model':[],'solving_single_model': [], 'instantiating_locations' : [], 'instantiating_portfolio': [], 'solving_locations' : [], 'solving_portfolio': [], 'algorithm_time':[], 'heuristic_time':[]}

    solver = helpers.build_solver(solver_name)
 
    # -- Create |L| +1 decomposed subproblems -- 
    location_models = {}
    location_instances = {}
    instance_size_read_data = instance_size
    if instance_size in [100,200,500,1000]:
        instance_size = int(instance_size/5)

    for l in range(instance_size):
        print(f"   Creating Subproblem for Location {l+1}")

        instantiating_locations_start = time.perf_counter()
        location_models[l] = helpers.build_model("LocationsModel")
        location_instances[l] = location_models[l].build_instance(instance_size=instance_size_read_data, equal_prices=equal_prices, site_id=l+1, lambda_init=lambda_init)
        time_complexity['instantiating_locations'].append(time.perf_counter() - instantiating_locations_start)

    
        for k in range(instance_size):
            for t in location_instances[l].T:
                # location_instances[l].dual[t] = 2000
                location_instances[l].dualgamma[t] = dual_gamma #50
                location_instances[l].e_S_prime[t,k+1] = 0
                location_instances[l].i_S_prime[t,k+1] = 0
                location_instances[l].i_G[t] = 0
                location_instances[l].e_G[t] = 0

    instantiating_portfolio_start = time.perf_counter()
    portfolio_model = helpers.build_model("PortfolioModel")
    portfolio_instance = portfolio_model.build_instance(instance_size=instance_size_read_data, equal_prices=equal_prices) 
    time_complexity['instantiating_portfolio'] = time.perf_counter() - instantiating_locations_start

    for t in portfolio_instance.T:
        portfolio_instance.dualgamma[t] = dual_gamma


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
    min_obj = np.zeros(int(len(location_instances))+1)

    #-- ADMM  -- 
    iter = 0
    while(iter < max_iter):
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


        algorithm_time_start = time.perf_counter()

        iter_solving_loctions = []
        for l in range(instance_size):
            solve_locations_start = time.perf_counter()
            result_l = solver.solve(location_instances[l])
            iter_solving_loctions.append(time.perf_counter() - solve_locations_start)
        time_complexity['solving_locations'].append(iter_solving_loctions)


        

        exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL = helpers.extract_from_exchange_ADMM(location_instances)

        portfolio_instance = helpers.feed_to_portfolio(portfolio_instnace=portfolio_instance, exports_S_TL=exports_S_TL, imports_S_TL=imports_S_TL)
        solve_portfolio_start = time.perf_counter()
        result_l = solver.solve(portfolio_instance)
        time_complexity['solving_portfolio'].append(time.perf_counter() - solve_portfolio_start)

        e_G, i_G = helpers.extract_from_portfolio(portfolio_instance)

        location_instances = helpers.feed_to_exchange_ADMM(location_instances, e_G, i_G)

        new_dualsT, location_instances, primal_residualsT, dualgammasT = helpers.ExchangeUpdateDuals(location_instances, i_G, e_G, imports_S_TL, exports_S_TL)

        time_complexity['algorithm_time'].append( time.perf_counter() - algorithm_time_start)
        objective_cost_original, dualized_constraint_value, augmentation_value, min_obj  = helpers.calculate_obj_cost_exchange(location_instances, e_G, i_G, portfolio_instance)
        # obj_cost_locations, obj_cost_portfolio 
     
        iter += 1 

    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['min_obj'] = np.stack(computational_data['min_obj'],axis=0)
    computational_data['dualgammasT'] = np.stack(computational_data['dualgammasT'],axis=0)

    return computational_data, portfolio_instance, location_instances, decision_vars, time_complexity


def solve_modified_exchange_ADMM(solver_name, instance_size, equal_prices, max_iter, dual_gamma:int, lambda_init:str):
    print("> Solving MODIFIED Exchange ADMM")

    time_complexity = {'instantiating_single_model':[],'solving_single_model': [], 'instantiating_locations' : [], 'instantiating_portfolio': [], 'solving_locations' : [], 'solving_portfolio': [], 'algorithm_time':[], 'heuristic_time':[]}

    solver = helpers.build_solver(solver_name)
 
    # -- Create |L| +1 decomposed subproblems -- 
    location_models = {}
    location_instances = {}
    instance_size_read_data = instance_size
    if instance_size in [100,200,500,1000]:
        instance_size = int(instance_size/5)
    for l in range(instance_size):
        print(f"   Creating Subproblem for Location {l+1}")
        instantiating_locations_start = time.perf_counter()
        location_models[l] = helpers.build_model("LocationsModel")
        location_instances[l] = location_models[l].build_instance(instance_size=instance_size_read_data, equal_prices=equal_prices, site_id=l+1, lambda_init=lambda_init)
        time_complexity['instantiating_locations'].append(time.perf_counter() - instantiating_locations_start)

    
        for k in range(instance_size):
            for t in location_instances[l].T:
                # location_instances[l].dual[t] = 2000
                location_instances[l].dualgamma[t] = dual_gamma #50
                location_instances[l].e_S_prime[t,k+1] = 0
                location_instances[l].i_S_prime[t,k+1] = 0
                location_instances[l].i_G[t] = 0
                location_instances[l].e_G[t] = 0


    portfolio_model = helpers.build_model("PortfolioModel")
    portfolio_instance = portfolio_model.build_instance(instance_size=instance_size_read_data, equal_prices=equal_prices) 
    for t in portfolio_instance.T:
        portfolio_instance.dualgamma[t] = dual_gamma


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
    min_obj = np.zeros(int(len(location_instances))+1)

    #-- ADMM  -- 
    iter = 0
    while(iter < max_iter):
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

        algorithm_time_Start = time.perf_counter()

        iter_solving_locations = []
        for l in range(instance_size):
            solve_locations_start = time.perf_counter()
            result_l = solver.solve(location_instances[l])
            iter_solving_locations.append(time.perf_counter() - solve_locations_start)

        time_complexity['solving_locations'].append(iter_solving_locations)


        

        exports_S_TL, imports_S_TL, charge_TBL, discharge_TBL = helpers.extract_from_exchange_ADMM(location_instances)

        portfolio_instance = helpers.feed_to_portfolio(portfolio_instnace=portfolio_instance, exports_S_TL=exports_S_TL, imports_S_TL=imports_S_TL)
        # result_l = solver.solve(portfolio_instance)

        # e_G, i_G = helpers.extract_from_portfolio(portfolio_instance)
        heuristic_time_start = time.perf_counter()
        i_G, e_G = helpers.feasibility_heuristic(location_instances[0], exports_S_TL, imports_S_TL)
        time_complexity['heuristic_time'].append(time.perf_counter() - heuristic_time_start)

        location_instances = helpers.feed_to_exchange_ADMM(location_instances, e_G, i_G)
        portfolio_instance = helpers.feed_to_portfolio_portfolio_vars(portfolio_instance, i_G, e_G)

        

        new_dualsT, location_instances, primal_residualsT, dualgammasT = helpers.ExchangeUpdateDuals(location_instances, i_G, e_G, imports_S_TL, exports_S_TL)

        time_complexity['algorithm_time'].append(time.perf_counter() - algorithm_time_Start)
        objective_cost_original, dualized_constraint_value, augmentation_value, min_obj  = helpers.calculate_obj_cost_exchange(location_instances, e_G, i_G, portfolio_instance)
        # obj_cost_locations, obj_cost_portfolio 
     
        iter += 1 

    computational_data['dualsT'] = np.stack(computational_data['dualsT'],axis=0)
    computational_data['primal_residualsT'] = np.stack(computational_data['primal_residualsT'],axis=0)
    computational_data['min_obj'] = np.stack(computational_data['min_obj'],axis=0)
    computational_data['dualgammasT'] = np.stack(computational_data['dualgammasT'],axis=0)

    return computational_data, portfolio_instance, location_instances, decision_vars, time_complexity

