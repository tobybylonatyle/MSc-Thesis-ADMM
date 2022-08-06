from fileinput import filename
import pyomo.environ as pyo
import Models.auxilary_functions as aux
from Models.ModelType import *
from os import path
import pandas as pd

class LocationsModel(pyo.AbstractModel):
    
    def __init__(self, type:ModelType, **kwds):
        '''Procedure for building the model'''
        super().__init__(**kwds)
        self.__type = type
        self.__build_parameters()
        self.__build_variables()
        self.__build_constraints()
        self.__build_objective()

    @property
    def type(self):
        return self.__type


    def __build_parameters(self):
        ''' Defines all Sets and Parameters for the Model'''

        self.N_t     = pyo.Param(within=pyo.PositiveIntegers) # Number of time periods
        self.T       = pyo.RangeSet(1, self.N_t)              # Set of time periods
        self.N_l     = pyo.Param(within=pyo.PositiveIntegers) # Number of locations
        self.N_l_prime = pyo.Param(within=pyo.PositiveIntegers)

        self.L = pyo.RangeSet(self.N_l, self.N_l) # Set of locations
        self.L_prime = pyo.RangeSet(1, self.N_l_prime)

        # -- Grid parameters --
        # self.price_import = pyo.Param(self.T, within=pyo.NonNegativeReals)   # Market import price
        # self.price_export = pyo.Param(self.T, within=pyo.NonNegativeReals)   # Market export price
        self.commitment_i = pyo.Param(self.T, within=pyo.NonNegativeReals)   # Day-Ahead commitment to import
        self.commitment_e = pyo.Param(self.T, within=pyo.NonNegativeReals)   # Day-Ahead commitment to export

        self.N_b_max = pyo.Param(within=pyo.PositiveIntegers) # Maximum number of batteries a location can have
        self.B_max   = pyo.RangeSet(1, self.N_b_max)          # Maximum set of batteries a location can have
        self.LB      = pyo.Set(within=self.L*self.B_max)      # Set of pairs (Location x Batteries)
        
        # -- Locations parameters --
        self.DUoS_import    = pyo.Param(self.T, self.L, within=pyo.NonNegativeReals) # Import DUoS price
        self.DUoS_export    = pyo.Param(self.T, self.L, within=pyo.NonNegativeReals) # Export DUoS price
        self.generation     = pyo.Param(self.T, self.L, within=pyo.NonNegativeReals) # Generation of locations
        self.demand         = pyo.Param(self.T, self.L, within=pyo.NonNegativeReals) # Demand of locations
        
        # -- Battery parameters --
        self.initial_soc    = pyo.Param(self.LB, within=pyo.NonNegativeReals) # Initial SoC of the battery
        self.min_capacity   = pyo.Param(self.LB, within=pyo.NonNegativeReals) # Minimum battery capacity
        self.max_capacity   = pyo.Param(self.LB, within=pyo.NonNegativeReals) # Maximum battery capacity
        self.max_rate       = pyo.Param(self.LB, within=pyo.NonNegativeReals) # Maximum charge rate of battery
        self.discharge_rate = pyo.Param(self.LB, within=pyo.PercentFraction)  # Self discharge rate of the battery
        self.rte_charge     = pyo.Param(self.LB, within=pyo.PercentFraction)  # Round-trip efficiency of charging
        self.rte_discharge  = pyo.Param(self.LB, within=pyo.PercentFraction)  # Round-trip efficiency of discharging
        self.max_n_cycles   = pyo.Param(self.LB, within=pyo.PositiveIntegers) # Maximum number of cycles
        
        # -- Time parameters --
        self.length_t = pyo.Param(within=pyo.PositiveReals) # Length of time periods
        self.grid_limit = pyo.Param(self.T, self.L, within=pyo.NonNegativeReals)  # Grid connection limit

        self.dual = pyo.Param(self.T, within=pyo.Reals, mutable = True) # Lagrange Multiplier of relaxed constraint
        self.dualgamma = pyo.Param(self.T, within=pyo.Reals, mutable= True) # Additonla penalty term

        self.e_G   = pyo.Param(self.T, within=pyo.NonNegativeReals, mutable=True) # Export to grid
        self.i_G   = pyo.Param(self.T, within=pyo.NonNegativeReals, mutable=True) # Import from grid

        self.e_S_prime = pyo.Param(self.T, self.L_prime , within=pyo.NonNegativeReals, mutable=True) # export from sites
        self.i_S_prime = pyo.Param(self.T, self.L_prime , within=pyo.NonNegativeReals, mutable=True) # import from sites

    def __build_variables(self):
        '''defines and builds all variables of the model'''

        # self.e_G   = pyo.Var(self.T, within=pyo.NonNegativeReals) # Export to grid
        # self.i_G   = pyo.Var(self.T, within=pyo.NonNegativeReals) # Import from grid

        self.e_S   = pyo.Var(self.T, self.L, within=pyo.NonNegativeReals)  # Export to portfolio
        self.i_S   = pyo.Var(self.T, self.L, within=pyo.NonNegativeReals)  # Import from portfolio

        self.c     = pyo.Var(self.T, self.LB, within=pyo.NonNegativeReals) # Charge
        self.d     = pyo.Var(self.T, self.LB, within=pyo.NonNegativeReals) # Discharge
        self.b     = pyo.Var(self.T, self.LB, within=pyo.NonNegativeReals) # Battery SoC



        #self.gamma = pyo.Var(self.T, within=pyo.Binary) # Binary for export/import complementarity

        #--TESTING
        # self.z     = pyo.Var(self.T, self.LB, within=pyo.Binary) # Binary for charging/discharging complementarity
        self.delta = pyo.Var(self.T, self.L, within=pyo.Binary)  # Binary for send/receive complementarity

    def __build_constraints(self):
        '''builds all constraints of the model'''

        self.Cons_soc_update         = pyo.Constraint(self.T, self.LB, rule=aux.soc_update)
        self.Cons_soc_final          = pyo.Constraint(self.LB, rule=aux.soc_final)
        self.Cons_soc_UB             = pyo.Constraint(self.T, self.LB, rule=aux.soc_UB)
        self.Cons_soc_LB             = pyo.Constraint(self.T, self.LB, rule=aux.soc_LB)
        self.Cons_cycle_limit        = pyo.Constraint(self.LB, rule=aux.cycle_limit)
        self.Cons_charge_UB          = pyo.Constraint(self.T, self.LB, rule=aux.charge_UB)
        self.Cons_grid_connect_limit = pyo.Constraint(self.T, self.L, rule=aux.grid_connect_limit)
                
        self.Cons_balance_site       = pyo.Constraint(self.T, self.L, rule=aux.energy_balance_site)
        #Inactive since it was relaxed.
        # self.Cons_balance_portfolio  = pyo.Constraint(self.T, rule=aux.energy_balance_portfolio)

        # -- TESTING
        # self.Cons_c_compl_bound     = pyo.Constraint(self.T, self.LB, rule=aux.c_compl_bound)
        # self.Cons_d_compl_bound     = pyo.Constraint(self.T, self.LB, rule=aux.d_compl_bound)
        self.Cons_r_compl_bound     = pyo.Constraint(self.T, self.L, rule=aux.r_compl_bound)
        self.Cons_s_compl_bound     = pyo.Constraint(self.T, self.L, rule=aux.s_compl_bound)

    def __build_objective(self):
        self.Objective_Cost = pyo.Objective(rule=aux.cost_ALR_location, sense=pyo.minimize) 

    def build_instance(self, instance_size:int, equal_prices:bool, site_id =-1):
        # Check that an site_id is only provided for models of type 'SP_location'
        # assert not((site_id != -1) & (self.type != ModelType.SP_location)), f"Parameter `site_id` is only valid for models of type {ModelType.SP_location}"

        # Create path to folder with instance files and check that it exists
        path_name = "Data/Instances/Size_"+str(instance_size)+"/"
        assert path.exists(path_name), f"Size Instance {instance_size} not available"

        size_csv   = path_name + "instance_size.csv"
        lb_set_csv = path_name + "LB.csv"
        tl_csv     = path_name + "time_location.csv"
        lb_csv     = path_name + "location_battery.csv"

        if equal_prices:
            t_csv = path_name + "SamePrice_time_data.csv"
        else:
            t_csv = path_name + "DiffPrice_time_data.csv"


        # Read common csv data to both subproblems
        df_size   = pd.read_csv(size_csv)
        df_tl     = pd.read_csv(tl_csv)
        dict_data = {'N_t': {None: df_size.iloc[0]['n_timeperiod']}}
        df_t = pd.read_csv(t_csv)
    
        # Check that the site_id is valid
        assert (1 <= site_id <= df_size.iloc[0]['n_locations']), f"Site ID {site_id} is invalid for this instance."


        # Read specific csv to Location subproblem
        df_lb_set = pd.read_csv(lb_set_csv)
        df_lb     = pd.read_csv(lb_csv)

        # Create dictionary with data from data frames
        dict_data['N_l']      = {None: site_id}
        dict_data['N_l_prime'] = {None: df_size.iloc[0]['n_locations']}
        dict_data['N_b_max']  = {None: df_size.iloc[0]['n_maxbatteries']}
        dict_data['length_t'] = {None: df_size.iloc[0]['length_t']}
        dict_data['LB']       = {None: [(l,b) for (l,b) in df_lb_set[['site','battery']].to_records(index=False) if l==site_id]}

        dict_data['price_import'] = {t: df_t.loc[df_t['time'] == t].iloc[0]['price_import'] for t in df_t['time']}
        dict_data['price_export'] = {t: df_t.loc[df_t['time'] == t].iloc[0]['price_export'] for t in df_t['time']}
        dict_data['commitment_i'] = {t: df_t.loc[df_t['time'] == t].iloc[0]['commitment_i'] for t in df_t['time']}
        dict_data['commitment_e'] = {t: df_t.loc[df_t['time'] == t].iloc[0]['commitment_e'] for t in df_t['time']}

        dict_data['grid_limit']   = {(t,l): df_tl[(df_tl['time']==t) & (df_tl['site']==l)].iloc[0]['grid_limit']\
                                            for (t,l) in df_tl[['time','site']].to_records(index=False)}

        # Time_Location Data
        dict_data['demand']      = {(t,l): df_tl[(df_tl['time']==t) & (df_tl['site']==l)].iloc[0]['demand']\
                                            for (t,l) in df_tl[['time','site']].to_records(index=False) if l==site_id}
        dict_data['generation']  = {(t,l): df_tl[(df_tl['time']==t) & (df_tl['site']==l)].iloc[0]['generation']\
                                            for (t,l) in df_tl[['time','site']].to_records(index=False) if l==site_id}
        dict_data['grid_limit']  = {(t,l): df_tl[(df_tl['time']==t) & (df_tl['site']==l)].iloc[0]['grid_limit']\
                                            for (t,l) in df_tl[['time','site']].to_records(index=False) if l==site_id}
        dict_data['DUoS_import'] = {(t,l): df_tl[(df_tl['time']==t) & (df_tl['site']==l)].iloc[0]['DUoS_import']\
                                            for (t,l) in df_tl[['time','site']].to_records(index=False) if l==site_id}
        dict_data['DUoS_export'] = {(t,l): df_tl[(df_tl['time']==t) & (df_tl['site']==l)].iloc[0]['DUoS_export']\
                                            for (t,l) in df_tl[['time','site']].to_records(index=False) if l==site_id}
        # Battery Data                
        dict_data['initial_soc']    = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['initial_soc']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['min_capacity']   = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['min_capacity']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['max_capacity']   = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['max_capacity']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['max_rate']       = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['max_rate']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['discharge_rate'] = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['discharge_rate']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['rte_charge']     = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['rte_charge']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['rte_discharge']  = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['rte_discharge']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}
        dict_data['max_n_cycles']   = {(l,b): df_lb[(df_lb['site']==l) & (df_lb['battery']==b)].iloc[0]['max_n_cycles']\
                                                for (l,b) in df_lb[['site','battery']].to_records(index=False) if l==site_id}

        # Initialize duals to be in their feasible interval [-Pi,-Pe]
        duals_UB = {t: -df_t[df_t['time']==t].iloc[0]['price_export'] for t in df_t['time'].unique()}
        duals_LB = {t: -df_t[df_t['time']==t].iloc[0]['price_import'] for t in df_t['time'].unique()}
        dict_data['dual'] = {t: (duals_UB[t]+duals_LB[t])/2 for t in df_t['time'].unique()}
    
       
        
        # Final dict to create instance from
        data = {None: dict_data}

        # Create Data Instance
        instance = self.create_instance(data, name=self.type)
        
        return instance