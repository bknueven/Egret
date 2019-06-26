#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module provides functions that create the modules for typical ACOPF formulations.

#TODO: document this with examples
"""
import pyomo.environ as pe
from math import inf
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
from egret.model_library.defn import ApproximationType, SensitivityCalculationMethod
from egret.data.model_data import zip_items
import egret.data.data_utils as data_utils
import egret.model_library.decl as decl


def _include_system_feasibility_slack(model, gen_attrs, bus_p_loads, bus_q_loads, penalty=1000):
    import egret.model_library.decl as decl
    slack_init = 0
    slack_bounds = (0, sum(bus_p_loads.values()))
    decl.declare_var('p_slack_pos', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    slack_bounds = (0, sum(bus_q_loads.values()))
    decl.declare_var('q_slack_pos', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('q_slack_neg', model=model, index_set=None,
                     initialize=slack_init, bounds=slack_bounds
                     )
    p_rhs_kwargs = {'include_feasibility_slack_pos': 'p_slack_pos', 'include_feasibility_slack_neg': 'p_slack_neg'}
    q_rhs_kwargs = {'include_feasibility_slack_pos': 'q_slack_pos', 'include_feasibility_slack_neg': 'q_slack_neg'}

    p_penalty = penalty * (max([gen_attrs['p_cost'][k]['values'][1] for k in gen_attrs['names']]) + 1)
    q_penalty = penalty * (max(gen_attrs.get('q_cost', gen_attrs['p_cost'])[k]['values'][1] for k in gen_attrs['names']) + 1)

    penalty_expr = p_penalty * (model.p_slack_pos + model.p_slack_neg) + q_penalty * (model.q_slack_pos + model.q_slack_neg)

    return p_rhs_kwargs, q_rhs_kwargs, penalty_expr


def create_fdf_model(model_data, include_feasibility_slack=False, calculation_method=SensitivityCalculationMethod.INVERT):
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    data_utils.create_dicts_of_fdf(md,calculation_method=calculation_method)

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

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    model = pe.ConcreteModel()

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, bus_attrs['names'], initialize=bus_p_loads)
    libbus.declare_var_ql(model, bus_attrs['names'], initialize=bus_q_loads)
    model.pl.fix()
    model.ql.fix()

    ### declare the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the polar voltages
    libbus.declare_var_vm(model, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max'])
                          )

    ### include the feasibility slack for the bus balances
    p_rhs_kwargs = {}
    q_rhs_kwargs = {}
    if include_feasibility_slack:
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_system_feasibility_slack(model, gen_attrs, bus_p_loads, bus_q_loads)

    ### declare the generator real and reactive power
    pg_init = {k: (gen_attrs['p_min'][k] + gen_attrs['p_max'][k]) / 2.0 for k in gen_attrs['pg']}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=pg_init,
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )

    qg_init = {k: (gen_attrs['q_min'][k] + gen_attrs['q_max'][k]) / 2.0 for k in gen_attrs['qg']}
    libgen.declare_var_qg(model, gen_attrs['names'], initialize=qg_init,
                          bounds=zip_items(gen_attrs['q_min'], gen_attrs['q_max'])
                          )

    q_pos_bounds = {k: (0, inf) for k in gen_attrs['qg']}
    decl.declare_var('q_pos', model=model, index_set=gen_attrs['names'], bounds=q_pos_bounds)

    q_neg_bounds = {k: (0, inf) for k in gen_attrs['qg']}
    decl.declare_var('q_neg', model=model, index_set=gen_attrs['names'], bounds=q_neg_bounds)

    ### declare the current flows in the branches
    vr_init = {k: bus_attrs['vm'][k] * pe.cos(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    vj_init = {k: bus_attrs['vm'][k] * pe.sin(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    s_max = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    s_lbub = dict()
    for k in branches.keys():
        if s_max[k] is None:
            s_lbub[k] = (None, None)
        else:
            s_lbub[k] = (-s_max[k],s_max[k])
    pf_bounds = s_lbub
    qf_bounds = s_lbub
    pfl_bounds = s_lbub
    qfl_bounds = s_lbub
    pf_init = dict()
    pfl_init = dict()
    qf_init = dict()
    qfl_init = dict()
    for branch_name, branch in branches.items():
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']
        y_matrix = tx_calc.calculate_y_matrix_from_branch(branch)
        ifr_init = tx_calc.calculate_ifr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        ifj_init = tx_calc.calculate_ifj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        itr_init = tx_calc.calculate_itr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        itj_init = tx_calc.calculate_itj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        pf_init[branch_name] = (tx_calc.calculate_p(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                -tx_calc.calculate_p(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))/2
        pfl_init[branch_name] = (tx_calc.calculate_p(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                +tx_calc.calculate_p(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))
        qf_init[branch_name] = (tx_calc.calculate_q(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                -tx_calc.calculate_q(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))/2
        qfl_init[branch_name] = (tx_calc.calculate_q(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])\
                                +tx_calc.calculate_q(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus]))

    libbranch.declare_var_pf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pf_init)#,
#                             bounds=pf_bounds
#                             )
    libbranch.declare_var_pfl(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pfl_init)#,
#                             bounds=pfl_bounds
#                             )
    libbranch.declare_var_qf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=qf_init)#,
#                             bounds=qf_bounds
#                             )
    decl.declare_var('qfl', model=model, index_set=branch_attrs['names'], initialize=qfl_init)#, bounds=qfl_bounds)

    ### declare the branch real power flow approximation constraints
    libbranch.declare_eq_branch_power_ptdf(model=model,
                                                  index_set=branch_attrs['names'],
                                                  branches=branches,
                                                  buses=buses,
                                                  bus_p_loads=bus_p_loads,
                                                  gens_by_bus=gens_by_bus,
                                                  bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                                  approximation_type=ApproximationType.PTDF_LOSSES,
                                                  include_constant_term=True
                                                  )

    ### declare the branch reactive power flow approximation constraints
    libbranch.declare_eq_branch_power_qtdf(model=model,
                                                  index_set=branch_attrs['names'],
                                                  branches=branches,
                                                  buses=buses,
                                                  bus_q_loads=bus_q_loads,
                                                  gens_by_bus=gens_by_bus,
                                                  bus_bs_fixed_shunts=bus_bs_fixed_shunts
                                                  )

    ### declare the branch real power loss approximation constraints
    libbranch.declare_eq_branch_loss_ptdf(model=model,
                                                  index_set=branch_attrs['names'],
                                                  branches=branches,
                                                  buses=buses,
                                                  bus_p_loads=bus_p_loads,
                                                  gens_by_bus=gens_by_bus,
                                                  bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                                  include_constant_term=True
                                                  )

    ### declare the branch reactive power loss approximation constraints
    # TODO: FIX BUG IN HERE
    libbranch.declare_eq_branch_loss_qtdf(model=model,
                                                  index_set=branch_attrs['names'],
                                                  branches=branches,
                                                  buses=buses,
                                                  bus_q_loads=bus_q_loads,
                                                  gens_by_bus=gens_by_bus,
                                                  bus_bs_fixed_shunts=bus_bs_fixed_shunts
                                                  )

    ### declare the p balance
    libbus.declare_eq_p_balance_fdf(model=model,
                                    index_set=bus_attrs['names'],
                                    buses=buses,
                                    bus_p_loads=bus_p_loads,
                                    gens_by_bus=gens_by_bus,
                                    bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                    include_losses=branch_attrs['names'],
                                    **p_rhs_kwargs
                                    )

    ### declare the q balance
    libbus.declare_eq_q_balance_fdf(model=model,
                                    index_set=bus_attrs['names'],
                                    buses=buses,
                                    bus_q_loads=bus_q_loads,
                                    gens_by_bus=gens_by_bus,
                                    bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                    include_losses=branch_attrs['names'],
                                    **q_rhs_kwargs
                                    )

    ### declare the real power flow limits
    libbranch.declare_fdf_thermal_limit(model=model,
                                        index_set=branch_attrs['names'],
                                        thermal_limits=s_max,
                                        )

    ### declare the voltage min and max inequalities
    libbus.declare_eq_vm_fdf(model=model,
                             index_set=bus_attrs['names'],
                             buses=buses,
                             bus_q_loads=bus_q_loads,
                             gens_by_bus=gens_by_bus,
                             bus_bs_fixed_shunts=bus_bs_fixed_shunts
                             )

    libgen.declare_eq_q_fdf_deviation(model=model,
                                      index_set=gen_attrs['names'],
                                      gens=gens)

    ### declare the generator cost objective
    libgen.declare_expression_pgqg_fdf_cost(model=model,
                                            index_set=gen_attrs['names'],
                                            p_costs=gen_attrs['p_cost']
                                            )

    obj_expr = sum(model.pg_operating_cost[gen_name] for gen_name in model.pg_operating_cost)
    obj_expr += sum(model.qg_operating_cost[gen_name] for gen_name in model.qg_operating_cost)
    if include_feasibility_slack:
        obj_expr += penalty_expr

    model.obj = pe.Objective(expr=obj_expr)

    return model, md


def _load_solution_to_model_data(m, md, results):
    from pyomo.environ import value
    from egret.model_library.transmission.tx_utils import unscale_ModelData_to_pu

    # save results data to ModelData object
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))

    md.data['system']['total_cost'] = value(m.obj)

    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])
        g_dict['qg'] = value(m.qg[g])

    for b,b_dict in buses.items():
        b_dict['pl'] = value(m.pl[b])
        b_dict['ql'] = value(m.ql[b])
        b_dict['vm'] = value(m.vm[b])

        b_dict['lmp'] = value(m.dual[m.eq_p_balance])
        b_dict['qlmp'] = value(m.dual[m.eq_q_balance])
        for k, k_dict in branches.items():
            if k_dict['from_bus'] == b or k_dict['to_bus'] == b:
                ptdf = k_dict['ptdf_r']
                ldf = k_dict['ldf']
                b_dict['lmp'] += ptdf[b]*value(m.dual[m.eq_pf_branch[k]])
                b_dict['lmp'] += ldf[b]*value(m.dual[m.eq_pfl_branch[k]])

                qtdf = k_dict['qtdf_r']
                qldf = k_dict['qldf']
                b_dict['qlmp'] += qtdf[b]*value(m.dual[m.eq_qf_branch[k]])
                b_dict['qlmp'] += qldf[b]*value(m.dual[m.eq_qfl_branch[k]])

    for k, k_dict in branches.items():
        k_dict['pf'] = value(m.pf[k])
        k_dict['qf'] = value(m.qf[k])
        k_dict['pfl'] = value(m.pfl[k])
        k_dict['qfl'] = value(m.qfl[k])

    unscale_ModelData_to_pu(md, inplace=True)

    return


def solve_fdf(model_data,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                fdf_model_generator = create_fdf_model,
                return_model = False,
                return_results = False,
                **kwargs):
    '''
    Create and solve a new acopf model

    Parameters
    ----------
    model_data : egret.data.ModelData
        An egret ModelData object with the appropriate data loaded.
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a string specifying a pyomo solver name, or an instantiated pyomo solver
    timelimit : float (optional)
        Time limit for dcopf run. Default of None results in no time
        limit being set.
    solver_tee : bool (optional)
        Display solver log. Default is True.
    symbolic_solver_labels : bool (optional)
        Use symbolic solver labels. Useful for debugging; default is False.
    options : dict (optional)
        Other options to pass into the solver. Default is dict().
    fdf_model_generator : function (optional)
        Function for generating the fdf model. Default is
        egret.models.acopf.create_fdf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    import pyomo.environ as pe
    from egret.common.solver_interface import _solve_model

    m, md = fdf_model_generator(model_data, **kwargs)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    m, results, flag = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,options=options)

    if not hasattr(md,'results'):
        md.data['results'] = dict()
    md.data['results']['time'] = results.Solver.Time
    md.data['results']['#_cons'] = results.Problem[0]['Number of constraints']
    md.data['results']['#_vars'] = results.Problem[0]['Number of variables']
    md.data['results']['termination'] = results.solver.termination_condition.__str__()

    if flag:
        _load_solution_to_model_data(m, md, results)
        m.pprint()

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md

if __name__ == '__main__':
    import os
    from egret.parsers.matpower_parser import create_ModelData

    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case57_ieee.m'
    matpower_file = os.path.join(path, '../../download/pglib-opf/', filename)
    md = create_ModelData(matpower_file)
    kwargs = {'include_feasibility_slack':False}
    from egret.models.acopf import solve_acopf
    md = solve_acopf(md, "ipopt",**kwargs)
    md = solve_fdf(md, "gurobi",**kwargs)

# not solving pglib_opf_case57_ieee