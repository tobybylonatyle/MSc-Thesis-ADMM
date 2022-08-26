import Methods.algorithms as alg
import Methods.helpers as helpers
import Results.visualize_and_check as vis_and_check


experiments = [['amplxpress',10,False,40,150,"0"],['gurobi',10,False,40,150,"0"]]
experiment_number = 1
for experiment in experiments:
    print(f"------- {experiment_number} -------")

    SOLVER_NAME = experiment[0]
    INSTANCE_SIZE = experiment[1]
    EQUAL_PRICES = experiment[2]
    MAX_ITER = experiment[3]
    DUAL_GAMMA = experiment[4]
    LAMBDA_INIT = experiment[5]

   
    algorithm = "LP"
    computational_data = "Not Applicable"
    decision_vars, instance, result, time_complexity = alg.solve_LP(solver_name=SOLVER_NAME, instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES)
    run = [computational_data, decision_vars, time_complexity, instance]
    path = f"instance_{INSTANCE_SIZE}/{algorithm}"
    vis_and_check.save_run(run,path)
    decision_vars, instance, result, time_complexity = 0, 0, 0, 0 #clear memory

    algorithm = "MILP"
    computational_data = "Not Applicable"
    decision_vars, instance, result, time_complexity = alg.solve_MILP(solver_name=SOLVER_NAME, instance_size=INSTANCE_SIZE, equal_prices= EQUAL_PRICES)
    run = [computational_data, decision_vars, time_complexity]
    path = f"instance_{INSTANCE_SIZE}/{algorithm}"
    vis_and_check.save_run(run,path)
    decision_vars, instance, result, time_complexity = 0, 0, 0, 0 #clear memory



    algorithm = "ADMM"
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = alg.solve_two_block_ADMM(solver_name=SOLVER_NAME, instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES, max_iter=MAX_ITER, dual_gamma=DUAL_GAMMA, lambda_init = LAMBDA_INIT)
    run = [computational_data, decision_vars, time_complexity]
    path = f"SolverComp/instance_{INSTANCE_SIZE}/{SOLVER_NAME}_{algorithm}_rho_{DUAL_GAMMA}_iter_{MAX_ITER}_lambdaInitial_{LAMBDA_INIT}"
    vis_and_check.save_run(run,path)
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = 0, 0, 0, 0 ,0 #clear memory

    #USELESS
    algorithm ="mADMM"
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = alg.solve_modified_two_block_ADMM(solver_name=SOLVER_NAME, instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES, max_iter=MAX_ITER, dual_gamma=DUAL_GAMMA, lambda_init =LAMBDA_INIT)
    run = [computational_data, decision_vars, time_complexity]
    path = f"instance_{INSTANCE_SIZE}/{algorithm}_rho_{DUAL_GAMMA}_iter_{MAX_ITER}_lambdaInitial_{LAMBDA_INIT}"
    vis_and_check.save_run(run,path)
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = 0, 0, 0, 0 ,0 #clear memory

   
    algorithm ="eADMM"
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = alg.solve_exchange_ADMM(solver_name=SOLVER_NAME, instance_size=INSTANCE_SIZE, equal_prices = EQUAL_PRICES, max_iter=MAX_ITER, dual_gamma=DUAL_GAMMA, lambda_init =LAMBDA_INIT)
    run = [computational_data, decision_vars, time_complexity]
    path = f"instance_{INSTANCE_SIZE}/{algorithm}_rho_{DUAL_GAMMA}_iter_{MAX_ITER}_lambdaInitial_{LAMBDA_INIT}"
    vis_and_check.save_run(run,path)
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = 0, 0, 0, 0 ,0 #clear memory


    #USELESS
    algorithm ="meADMM"
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = alg.solve_modified_exchange_ADMM(solver_name=SOLVER_NAME, instance_size=INSTANCE_SIZE, equal_prices=EQUAL_PRICES, max_iter=MAX_ITER, dual_gamma=DUAL_GAMMA, lambda_init =LAMBDA_INIT)
    run = [computational_data, decision_vars, time_complexity]
    path = f"instance_{INSTANCE_SIZE}/{algorithm}_rho_{DUAL_GAMMA}_iter_{MAX_ITER}_lambdaInitial_{LAMBDA_INIT}"
    vis_and_check.save_run(run,path)
    computational_data, portfolio_instance, locations_instance, decision_vars, time_complexity = 0, 0, 0, 0 ,0 #clear memory


    experiment_number += 1

