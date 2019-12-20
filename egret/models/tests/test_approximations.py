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
import os
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


def inner_loop_solves(md_basepoint, mult, **kwargs):
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

    record_results('test_acopf', mult, md_ac)

    if kwargs:

        for idx, val in kwargs.items():

            if val and idx == 'test_ccm':
                md_ccm, m, results = solve_ccm(md, "ipopt", return_model=True, return_results=True, solver_tee=False)

                record_results(idx, mult, md_ccm)

            if val and idx == 'test_lccm':
                md_lccm, m, results = solve_lccm(md, "gurobi", return_model=True, return_results=True, solver_tee=False)

                record_results(idx, mult, md_lccm)

            if val and idx == 'test_fdf':
                md_fdf, m, results = solve_fdf(md, "gurobi", return_model=True, return_results=True, solver_tee=False)

                record_results(idx, mult, md_fdf)

            if val and idx == 'test_fdf_simplified':
                md_fdfs, m, results = solve_fdf_simplified(md, "gurobi", return_model=True, return_results=True, solver_tee=False)

                record_results(idx, mult, md_fdfs)

            if val and idx == 'test_ptdf_losses':
                md_ptdfl, m, results = solve_dcopf_losses(md, "gurobi", dcopf_losses_model_generator=create_ptdf_losses_dcopf_model,
                                                        return_model=True, return_results=True, solver_tee=False)
                record_results(idx, mult, md_ptdfl)

            if val and idx == 'test_btheta_losses':
                md_bthetal, m, results = solve_dcopf_losses(md, "gurobi", dcopf_losses_model_generator=create_btheta_losses_dcopf_model,
                                                        return_model=True, return_results=True, solver_tee=False)
                record_results(idx, mult, md_bthetal)

            if val and idx == 'test_ptdf':
                md_ptdf, m, results = solve_dcopf(md, "gurobi", dcopf_model_generator=create_ptdf_dcopf_model,
                                                return_model=True, return_results=True, solver_tee=False)
                record_results(idx, mult, md_ptdf)

            if val and idx == 'test_btheta':
                md_btheta, m, results = solve_dcopf(md, "gurobi", dcopf_model_generator=create_btheta_dcopf_model,
                                                return_model=True, return_results=True, solver_tee=False)
                record_results(idx, mult, md_btheta)

def record_results(idx, mult, md):
    '''
    writes model data (md) object to .json file
    '''
    filename = md.data['system']['model_name'] + '_' + idx + '_{0:04.0f}'.format(mult*1000)

    md.write_to_json(filename)

def test_approximation(test_case, init_min=0.9, init_max=1.1, steps=20, **kwargs):
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

        inner_loop_solves(md_basept, mult, **kwargs)


def test_fdf_model(self, test_case, include_kwargs=False):
    acopf_model = create_psv_acopf_model

    md_dict = create_ModelData(test_case)

    kwargs = {}
    if include_kwargs:
        kwargs = {'include_feasibility_slack':True}
    md_acopf, m, results = solve_acopf(md_dict, "ipopt", acopf_model_generator=acopf_model, return_model=True, return_results=True, solver_tee=False, **kwargs)
    filename = md_acopf.data['system']['model_name'] + '_acopf'
    md_acopf.write_to_json(filename)

    md_fdf, m, results = solve_fdf(md_acopf, "gurobi", return_model=True, return_results=True, solver_tee=False, **kwargs)
    filename = md_fdf.data['system']['model_name'] + '_fdf'
    md_fdf.write_to_json(filename)

    self.assertTrue(True)


if __name__ == '__main__':

    test_case = test_cases[0]

    kwargs = {'test_ccm' :              False,
              'test_lccm' :             False,
              'test_fdf' :              True,
              'test_simplified_fdf' :   False,
              'test_ptdf_losses' :      False,
              'test_ptdf' :             False,
              'test_btheta_losses' :    False,
              'test_btheta' :           True
              }

    test_approximation(test_case, init_min=0.9, init_max=1.1, steps=2, **kwargs)

