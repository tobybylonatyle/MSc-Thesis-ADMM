from os import listdir

import pickle as CPickle
import matplotlib.pyplot as plt
import numpy as np


def save_run(results,name):
    # i = len(listdir("Results/"))
    with open(f'Results/{name}', 'wb') as f:
        CPickle.dump(results, f)
    print("done")

def open_run(path):
    with open(f"Results/{path}", 'rb') as input_file:
        temp = CPickle.load(input_file)
    return temp


def check_complementarities_locations_subproblem(decision_vars):
    total_violations = 0
    for iter in decision_vars.keys():
        T, B, L = decision_vars[iter]['charge_TBL'].shape
        for t in range(T):
            if (decision_vars[iter]['e_G'][t]*decision_vars[iter]['i_G'][t] != 0):
                print(f"Import Export, iteration {iter}, time {t}")
                total_violations += 1

            for l in range(L):
                if (decision_vars[iter]['e_S'][t,l]*decision_vars[iter]['i_S'][t,l] != 0):
                    print(f"send/recieve, iteration {iter},location {l}, time {t} ")
                    total_violations += 1
            
                for b in range(B):
                    if (decision_vars[iter]['charge_TBL'][t,b,l]*decision_vars[iter]['discharge_TBL'][t,b,l] != 0):
                        print(f"charge/discharge, iter {iter}, location {l}, time {t}, battery {b}")
                        total_violations += 1
    return total_violations



def plot_results_behaviour(results):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    tperiods = results['dualsT'].shape[1]
    iters = results['dualsT'].shape[0]
    dualgamma = results['dualgammasT'][1,0]
    
    fig.suptitle(f"Time Perios: {tperiods}, Iterations: {iters}, DualGamma: {dualgamma}")
    ax[0,0].plot(results['obj_cost'][1:])
    ax[0,0].set(xlabel="iteration", ylabel = 'objective value original')
    
    ax[1,1].plot(results['min_obj'][1:])
    ax[1,1].set(xlabel="iteration", ylabel = 'minimization objective')

    for i in range(results['dualsT'].shape[1]):
        ax[0,1].plot(results['dualsT'][1:,i])
        ax[0,1].set(xlabel="iteration",ylabel='dual values')

        ax[1,0].plot(results['primal_residualsT'][1:,i])
        ax[1,0].set(xlabel="iteration", ylabel='primal residual')

    plt.show()

    
def plot_results_behaviour_new(results,iters_to_display):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    tperiods = results['dualsT'].shape[1]
    iters = results['dualsT'].shape[0]
    dualgamma = results['dualgammasT'][1,0]
    
    fig.suptitle(f"Time Perios: {tperiods}, Iterations: {iters}, DualGamma: {dualgamma}")
    ax[0,0].plot(results['obj_cost'][1:iters_to_display])
    ax[0,0].set(xlabel="iteration", ylabel = 'objective value original')
    
    ax[1,1].plot(results['min_obj'][1:iters_to_display])
    ax[1,1].set(xlabel="iteration", ylabel = 'minimization objective')

    for i in range(results['dualsT'].shape[1]):
        ax[0,1].plot(results['dualsT'][1:iters_to_display,i])
        ax[0,1].set(xlabel="iteration",ylabel='dual values')

        ax[1,0].plot(results['primal_residualsT'][1:iters_to_display,i])
        ax[1,0].set(xlabel="iteration", ylabel='primal residual')

    plt.show()

def compare_decision_vars(dv_LP, dv_compare):
    relative_errors = {}
    for iter in dv_compare:
        test = dict.fromkeys(dv_LP[0].keys())
        for var in dv_compare[iter]:
            comparison = (dv_compare[iter][var]-dv_LP[0][var])/dv_LP[0][var]
            test[var] = comparison
        relative_errors[iter] = test

    return relative_errors

def compare_decision_vars_iterations(dv_a, dv_b):
    relative_errors = {}
    for iter in dv_b:
        test = dict.fromkeys(dv_a[iter].keys())
        for var in dv_b[iter]:
            comparison = abs(dv_b[iter][var]-dv_a[iter][var])/abs(dv_a[iter][var])
            test[var] = comparison
        relative_errors[iter] = test

    return relative_errors

def calculate_objective_cost_from_decision_vars(dv, instance_LP):
    virgin_objective_value = np.zeros(int(len(dv.keys())))
    try:
        TEMP = instance_LP.L_prime
    except:
        TEMP = instance_LP.L
    for iter in dv.keys():
        DUoS_export = sum(instance_LP.DUoS_export[t,l]*dv[iter]['e_S'][t-1,l-1] for t in instance_LP.T for l in TEMP)
        DUoS_import = sum(instance_LP.DUoS_import[t,l]*dv[iter]['i_S'][t-1,l-1] for t in instance_LP.T for l in TEMP)
        Grid_export = sum(instance_LP.price_import[t]*dv[iter]['i_G'][t-1] for t in instance_LP.T)
        Grid_import = sum(instance_LP.price_export[t]*dv[iter]['e_G'][t-1] for t in instance_LP.T )

        virgin_objective_value[iter] = DUoS_export + DUoS_import + Grid_import - Grid_export

    return virgin_objective_value


def plot_dual_values(computational_results,iters_to_display):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=100)
    num_of_iters = computational_results['dualsT'].shape[0]
    for i in range(computational_results['dualsT'].shape[1]):
        temp = abs(computational_results['dualsT'][1:,i]-computational_results['dualsT'][num_of_iters-1:,i])/abs(computational_results['dualsT'][num_of_iters-1:,i])
        ax.plot(temp[1:iters_to_display])
    # ax.set_title("Dual Variable Updates")
    fig.gca().set_xlabel(r"Iteration $(k)$", fontsize=11)
    fig.gca().set_ylabel(r'Relative Error of $\lambda_t$', fontsize=11)
#     plt.show()
#     fig.savefig('test.jpg')
    return fig

def plot_primal_residuals(computational_results,iters_to_display):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=100)
    for i in range(computational_results['primal_residualsT'].shape[1]):
        ax.plot(computational_results['primal_residualsT'][1:iters_to_display,i])
    fig.gca().set_xlabel(r"Iteration", fontsize=11)
    fig.gca().set_ylabel(r'$r_t$', fontsize=11)
    return fig

def plot_primal_residuals_new(primal_residuals):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=100)
    for i in range(primal_residuals.shape[1]):
        ax.plot(primal_residuals[1:,i])
    fig.gca().set_xlabel(r"Iteration", fontsize=11)
    fig.gca().set_ylabel(r'$r_t$', fontsize=11)
    return fig
        
        
        
def plot_minimization_objective(computational_results):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=100)
    ax.plot(computational_results['min_obj'])
    fig.gca().set_xlabel(r"Iteration", fontsize=11)
    fig.gca().set_ylabel(r'Minimization Objective', fontsize=11)
    return fig

def plot_duality_gap(computational_results, iters_to_show):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=100)
    ax.plot(np.array(computational_results['min_obj'][1:iters_to_show]))
    ax.plot(computational_results['obj_cost'][1:iters_to_show])
    fig.legend(["dual", "primal"],loc='upper right',borderaxespad=3)
    fig.gca().set_xlabel(r"Iteration", fontsize=11)
    fig.gca().set_ylabel(r'Value', fontsize=11)
    return fig

def plot_relative_duality_gap(computational_results,start_iters_to_show, iters_to_show):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=100)
    a = abs(np.array(computational_results['obj_cost'][start_iters_to_show:iters_to_show]))
    b = abs(np.array(computational_results['min_obj'][start_iters_to_show:iters_to_show]))
    ax.plot(range(start_iters_to_show,iters_to_show),abs(a-b)/b)
    fig.gca().set_xlabel(r"Iteration", fontsize=11)
    fig.gca().set_ylabel(r'Duality Gap %', fontsize=11)
    return fig, abs(a-b)/b
