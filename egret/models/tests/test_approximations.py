#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

'''
fdf tester vs acopf
    Select case from test_cases
    Set demand = 1.0 * demand
    Solve acopf.py --> base case 'md_basecase'
    Set demand = (0.9 to 1.1) * demand
    Solve acopf.py --> true solution 'md_ac'
    Solve fdf(md_basecase) --> approx solution 'md_fdf'
    ...additional solves: dcopf & dcopf_losses
    Record data: solve time, infeasibility, case attributes --> 'caseSummary'
    Delete md_ac and md_fdf, then back to (4)
    Plot caseSummary: infeasibility vs. demand of case
    Record averages in 'caseSummary' to 'totalSummary'
    Delete 'caseSummary' and repeat from (1)
    Plot totalSummary: infeasbility vs. solve time of all cases
'''
import os, shutil, glob, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math
import unittest
from pyomo.opt import SolverFactory, TerminationCondition
from egret.models.acopf import *
from egret.models.ccm import *
from egret.models.fdf import *
from egret.models.fdf_simplified import *
from egret.models.lccm import *
from egret.models.dcopf_losses import *
from egret.models.dcopf import *
from egret.models.copperplate_dispatch import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
from os import listdir
from os.path import isfile, join

current_dir = os.path.dirname(os.path.abspath(__file__))
#test_cases = [join('../../../download/pglib-opf-master/', f) for f in listdir('../../../download/pglib-opf-master/') if isfile(join('../../../download/pglib-opf-master/', f)) and f.endswith('.m')]
case_names = ['pglib_opf_case3_lmbd',
              'pglib_opf_case5_pjm',
              'pglib_opf_case14_ieee',
              'pglib_opf_case24_ieee_rts',
              'pglib_opf_case30_as',
              'pglib_opf_case30_fsr',
              'pglib_opf_case30_ieee',
              'pglib_opf_case39_epri',
              'pglib_opf_case57_ieee',
              'pglib_opf_case73_ieee_rts',
              'pglib_opf_case89_pegase',
              'pglib_opf_case118_ieee',
              'pglib_opf_case162_ieee_dtc',
              'pglib_opf_case179_goc',
              'pglib_opf_case200_tamu',
              'pglib_opf_case240_pserc',
              'pglib_opf_case300_ieee',
              'pglib_opf_case500_tamu',
              'pglib_opf_case588_sdet',
              ]
test_cases = [os.path.join(current_dir, 'transmission_test_instances', 'pglib-opf-master', '{}.m'.format(i)) for i in case_names]


def set_acopf_basepoint_min_max(md_dict, init_min=0.9, init_max=1.1, **kwargs):
    '''
    returns AC basepoint solution and feasible min/max range
     - new min/max range b/c test case may not be feasible in [init_min to init_max]
    '''
    md = md_dict.clone_in_service()
    loads = dict(md.elements(element_type='load'))

    acopf_model = create_psv_acopf_model

    md_basept, m, results = solve_acopf(md, "ipopt", acopf_model_generator=acopf_model, return_model=True, return_results=True, solver_tee=False)

    # exit if base point does not return optimal
    if not results.solver.termination_condition==TerminationCondition.optimal:
        raise Exception('Base case acopf did not return optimal solution')

    # find feasible min and max demand multipliers
    else:
        mult_min = multiplier_loop(md, loads, init=init_min, steps=10, acopf_model=acopf_model)
        mult_max = multiplier_loop(md, loads, init=init_max, steps=10, acopf_model=acopf_model)

    return md_basept, mult_min, mult_max


def multiplier_loop(md, loads, init=0.9, steps=10, acopf_model=create_psv_acopf_model):
    '''
    init < 1 searches for the lowest demand multiplier >= init that has an optimal acopf solution
    init > 1 searches for the highest demand multiplier <= init that has an optimal acopf solution
    steps determines the increments in [init, 1] where the search is made
    '''

    # step size
    inc = abs(1 - init) / steps

    # initial loads
    init_p_loads = {k: loads[k]['p_load'] for k in loads.keys()}
    init_q_loads = {k: loads[k]['q_load'] for k in loads.keys()}

    # loop
    for step in range(0,steps):

        # for finding minimum
        if init < 1:
            mult = round(init + step * inc, 4)
        # for finding maximum
        elif init > 1:
            mult = round(init - step * inc, 4)

        # adjust load from init_min
        for k in loads.keys():
            loads[k]['p_load'] = init_p_loads[k] * mult
            loads[k]['q_load'] = init_q_loads[k] * mult

        md_, results = solve_acopf(md, "ipopt", acopf_model_generator=acopf_model, return_model=False, return_results=True, solver_tee=False)

        if results.solver.termination_condition==TerminationCondition.optimal:
            final_mult = mult
            break

        if step == steps:
            final_mult = 1
            print('Found no optimal solutions with mult != 1. Try init between 1 and {}.'.format(mult))

    return final_mult


def inner_loop_solves(md_basepoint, mult, test_model_dict):
    '''

    '''
    md = md_basepoint.clone_in_service()

    loads = dict(md.elements(element_type='load'))

    # initial loads
    init_p_loads = {k: loads[k]['p_load'] for k in loads.keys()}
    init_q_loads = {k: loads[k]['q_load'] for k in loads.keys()}

    # multiply loads
    for k in loads.keys():
        loads[k]['p_load'] = init_p_loads[k] * mult
        loads[k]['q_load'] = init_q_loads[k] * mult

    # solve acopf
    md_ac, m, results = solve_acopf(md, "ipopt", return_model=True, return_results=True, solver_tee=False)
    md_ac.data['system']['mult'] = mult
    record_results('acopf', mult, md_ac)

    for idx, val in test_model_dict.items():

        if val and idx == 'ccm':
            md_ccm, m, results = solve_ccm(md, "ipopt", return_model=True, return_results=True, solver_tee=False)

            record_results(idx, mult, md_ccm)

        if val and idx == 'lccm':
            md_lccm, m, results = solve_lccm(md, "gurobi", return_model=True, return_results=True, solver_tee=False)

            record_results(idx, mult, md_lccm)

        if val and idx == 'fdf':
            md_fdf, m, results = solve_fdf(md, "gurobi", return_model=True, return_results=True, solver_tee=False)

            record_results(idx, mult, md_fdf)

        if val and idx == 'fdf_simplified':
            md_fdfs, m, results = solve_fdf_simplified(md, "gurobi", return_model=True, return_results=True, solver_tee=False)

            record_results(idx, mult, md_fdfs)

        if val and idx == 'ptdf_losses':
            md_ptdfl, m, results = solve_dcopf_losses(md, "gurobi", dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                    return_model=True, return_results=True, solver_tee=False)
            record_results(idx, mult, md_ptdfl)

        if val and idx == 'btheta_losses':
            md_bthetal, m, results = solve_dcopf_losses(md, "gurobi", dcopf_losses_model_generator=create_btheta_losses_dcopf_model,
                                                    return_model=True, return_results=True, solver_tee=False)
            record_results(idx, mult, md_bthetal)

        if val and idx == 'ptdf':
            md_ptdf, m, results = solve_dcopf(md, "gurobi", dcopf_model_generator=create_ptdf_dcopf_model,
                                            return_model=True, return_results=True, solver_tee=False)
            record_results(idx, mult, md_ptdf)

        if val and idx == 'btheta':
            md_btheta, m, results = solve_dcopf(md, "gurobi", dcopf_model_generator=create_btheta_dcopf_model,
                                            return_model=True, return_results=True, solver_tee=False)
            record_results(idx, mult, md_btheta)

def record_results(idx, mult, md):
    '''
    writes model data (md) object to .json file
    '''
    filename = md.data['system']['model_name'] + '_' + idx + '_{0:04.0f}'.format(mult*1000)
    md.data['system']['mult'] = mult

    md.write_to_json(filename)
    print(filename)

def create_testcase_directory(test_case):

    # directory locations
    cwd = os.getcwd()
    case_folder, case = os.path.split(test_case)
    case, ext = os.path.splitext(case)
    current_dir, current_file = os.path.split(os.path.realpath(__file__))

    # move to case directory
    source = os.path.join(cwd, case + '_*.json')
    destination = os.path.join(current_dir,'transmission_test_instances','approximation_solution_files',case)

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not glob.glob(source):
        print('No files to move.')
    else:
        print('dest: {}'.format(destination))

        for src in glob.glob(source):
            print('src:  {}'.format(src))
            folder, file = os.path.split(src)
            dest = os.path.join(destination, file) # full destination path will overwrite existing files
            shutil.move(src, dest)

    return destination

def total_cost(md):

    val = md.data['system']['total_cost']

    return val

def ploss(md):

    val = md.data['system']['ploss']

    return val

def qloss(md):

    val = md.data['system']['qloss']

    return val

def pgen(md):

    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['pg']

    return dispatch

def qgen(md):

    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['qg']

    return dispatch

def pflow(md):

    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['pf']

    return flow

def qflow(md):

    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['qf']

    return flow

def vmag(md):

    buses = dict(md.elements(element_type='bus'))
    vm = {}

    for b,bus in buses.items():
        vm[b] = bus['vm']

    return vm


def sum_infeas(md):

    from egret.common.solver_interface import _solve_model
    from pyomo.environ import value

    # calculate AC branch flows from vmag and angle
    #calculate_flows(md)

    # calculate bus power balance from bus injections and AC power flows
    #calculate_balance(md)

    #--- alternate: build ACOPF model
    m_ac, md_ac = create_psv_acopf_model(md, include_feasibility_slack=True)

    # ModelData components
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    load_attrs = md.attributes(element_type='load')
    shunt_attrs = md.attributes(element_type='shunt')

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    # fix variables to the values in modeData object md
    for g, pg in m_ac.pg.items():
        pg.value = gens[g]['pg']
    for g, qg in m_ac.qg.items():
        qg.value = gens[g]['qg']
    for b, va in m_ac.va.items():
        va.value = buses[b]['va']
    for b, vm in m_ac.vm.items():
        vm.value = buses[b]['vm']

    # set objective to sum of infeasibilities (i.e. slacks)
    m_ac.del_component(m_ac.obj)
    penalty_expr = sum(m_ac.p_slack_pos[bus_name] + m_ac.p_slack_neg[bus_name]
                       + m_ac.q_slack_pos[bus_name] + m_ac.q_slack_neg[bus_name]
                       for bus_name in bus_attrs['names'])
    m_ac.obj = pe.Objective(expr=penalty_expr)

    # solve model
    m_ac, results = _solve_model(m_ac, "ipopt", timelimit=None, solver_tee=False, symbolic_solver_labels=False,
                              options=None, return_solver=False)
    sum_infeas = value(m_ac.obj)

    print('{}: infeas={}'.format(md.data['system']['mult'], sum_infeas))

    return sum_infeas


def read_sensitivity_data(case_folder, test_model, data_generator=total_cost):

    parent, case = os.path.split(case_folder)
    filename = case + "_" + test_model + "_*.json"
    file_list = glob.glob(os.path.join(case_folder, filename))

    print("Reading data for " + test_model + ".")

    data = {}
    for file in file_list:
        md_dict = json.load(open(file))
        md = ModelData(md_dict)
        mult = md.data['system']['mult']
        data[mult] = data_generator(md)

    for d in data:

        data_is_vector = hasattr(data[d], "__len__")

        if data_is_vector:
            df_data = pd.DataFrame(data)
            return df_data
            break

    if not data_is_vector:
        df_data = pd.DataFrame(data, index=[0])
        return df_data


def solve_approximation_models(test_case, test_model_dict, init_min=0.9, init_max=1.1, steps=20):
    '''
    1. initialize base case and demand range
    2. loop over demand values
    3. record results to .json files
    '''

    md_dict = create_ModelData(test_case)

    md_basept, min_mult, max_mult = set_acopf_basepoint_min_max(md_dict, init_min, init_max)

    inc = (max_mult - min_mult) / steps

    for step in range(0,steps + 1):

        mult = round(min_mult + step * inc, 4)

        inner_loop_solves(md_basept, mult, test_model_dict)


def generate_sensitivity_plot(test_case, test_model_dict, data_generator=total_cost, vector_norm=2, show_plot=False):

    case_location = create_testcase_directory(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    # acopf comparison
    df_acopf = read_sensitivity_data(case_location, 'acopf', data_generator=data_generator)

    # empty dataframe to add data into
    df_data = pd.DataFrame(data=None)

    # iterate over test_model's
    for test_model, val in test_model_dict.items():
        if val:
            df_approx = read_sensitivity_data(case_location, test_model, data_generator=data_generator)
            df_diff = df_approx - df_acopf

            # calculate norm from df_diff columns
            data = {}
            for col in df_diff:
                colObj = df_diff[col]
                if len(colObj.values) > 1:
                    data_is_vector = True
                    data[col] = np.linalg.norm(colObj.values, vector_norm)
                else:
                    data_is_vector = False
                    data[col] = (colObj.values / df_acopf[col].values) * 100

            # record data in DataFrame
            df_col = pd.DataFrame(data, index=[test_model])
            df_data = pd.concat([df_data, df_col])

    # show data in table
    y_axis_data = data_generator.__name__
    print('Summary data from {} and L-{} norm for non-scalar values.'.format(y_axis_data, vector_norm))
    df_data = df_data.T
    print(df_data)

    # show data in graph
    output = df_data.plot.line()
    output.set_title(y_axis_data)
    output.set_ylim(top=0)
    output.set_xlabel("Demand Multiplier")

    if data_is_vector:
        filename = "sensitivityplot_" + case_name + "_" + y_axis_data + "_L{}_norm.png".format(vector_norm)
        output.set_ylabel('L-{} norm'.format(vector_norm))
    else:
        filename = "sensitivityplot_" + case_name + "_" + y_axis_data + "_pctDiff.png"
        output.set_ylabel('Relative difference (%)')
        output.yaxis.set_major_formatter(mtick.PercentFormatter())

    #save to destination folder
    destination = os.path.join(case_location, 'plots')

    if not os.path.exists(destination):
        os.makedirs(destination)

    plt.savefig(os.path.join(destination, filename))

    # display
    if show_plot is True:
        plt.show()


if __name__ == '__main__':

    test_case = test_cases[0]
    print(test_case)

    test_model_dict = \
        {'ccm' :              False,
         'lccm' :             False,
         'fdf' :              True,
         'simplified_fdf' :   False,
         'ptdf_losses' :      False,
         'ptdf' :             False,
         'btheta_losses' :    False,
         'btheta' :           True
         }

    solve_approximation_models(test_case, test_model_dict, init_min=0.9, init_max=1.1, steps=2)
    generate_sensitivity_plot(test_case, test_model_dict, data_generator=sum_infeas, show_plot=True)
    #generate_sensitivity_plot(test_case, test_model_dict, data_generator=sum_infeas)
    #generate_sensitivity_plot(test_case, test_model_dict, data_generator=total_cost)
    #generate_sensitivity_plot(test_case, test_model_dict, data_generator=ploss)
    #generate_sensitivity_plot(test_case, test_model_dict, data_generator=pgen, vector_norm=2)
    #generate_sensitivity_plot(test_case, test_model_dict, data_generator=pflow, vector_norm=2)
    #generate_sensitivity_plot(test_case, test_model_dict, data_generator=vmag, vector_norm=2)

