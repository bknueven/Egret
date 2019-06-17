#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module collects some helper functions useful for performing
different computations for transmission models that require solving
optimization sub-problems
"""

from math import pi, inf, radians, cos, sin
import egret.model_library.transmission.tx_utils as tx_utils
from egret.models.acpf import create_psv_acpf_model
from egret.models.dcopf import solve_dcopf, create_btheta_dcopf_model
from egret.models.acopf import _load_solution_to_model_data, create_psv_acopf_model, solve_acopf
import egret.model_library.decl as decl
from egret.common.solver_interface import _solve_model
import pyomo.environ as pe
from pyomo.environ import value
from egret.data.model_data import zip_items
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
import egret.model_library.transmission.tx_calc as tx_calc
from egret.model_library.defn import BasePointType, ApproximationType


def calculate_ptdf(model_data):
    """
    Calculates the sensitivity of the voltage angle to real power injections
    """
    kwargs = {'return_model':'True', 'return_results':'True', 'include_feasibility_slack':'False'}
    md, m, results = solve_dcopf(model_data, "ipopt", dcopf_model_generator=create_btheta_dcopf_model, **kwargs)

    m, md = create_psv_acpf_model(md)
    _solve_fixed_acpf(m, md)

    m, md = _dual_ptdf_model(md)

    ref_bus = md.data['system']['reference_bus']
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    import numpy as np
    ptdf = np.zeros((_len_branch,_len_bus))
    for _k, branch_name in _mapping_branch.items():
        if hasattr(m,"objective"):
            m.del_component(m.objective)
        m.objective = pe.Objective(expr=m.pf[branch_name])
        m, results = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            ptdf[_k,_b] = m.dual.get(m.eq_p_balance[bus_name])
    print(ptdf)
    ptdf_check = tx_calc.calculate_ptdf(branches, buses, branch_attrs['names'], bus_attrs['names'], ref_bus, base_point=BasePointType.SOLUTION)
    print(ptdf_check)

    return ptdf


def calculate_qtdf(model_data):
    """
    Calculates the sensitivity of the voltage magnitude to reactive power injections
    """
    kwargs = {'return_model':'True', 'return_results':'True', 'include_feasibility_slack':'False'}
    md, m, results = solve_dcopf(model_data, "ipopt", dcopf_model_generator=create_btheta_dcopf_model, **kwargs)

    m, md = create_psv_acpf_model(md)
    _solve_fixed_acpf(m, md)

    ref_bus = md.data['system']['reference_bus']
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')

    for k, k_dict in branches.items():
        k_dict['qfl'] = k_dict['qf'] + k_dict['qt']

    m, md = _dual_qtdf_model(md)

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    import numpy as np
    qtdf = np.zeros((_len_branch,_len_bus))
    for _k, branch_name in _mapping_branch.items():
        if hasattr(m,"objective"):
            m.del_component(m.objective)
        m.objective = pe.Objective(expr=m.qf[branch_name])
        m, results = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            qtdf[_k,_b] = m.dual.get(m.eq_q_balance[bus_name])
    print(qtdf)
    qtdf_check = tx_calc.calculate_qtdf(branches, buses, branch_attrs['names'], bus_attrs['names'], ref_bus, base_point=BasePointType.SOLUTION)
    print(qtdf_check)

    return qtdf


def calculate_ldf(model_data):
    """
    Calculates the sensitivity of the voltage angle to real power losses
    """
    kwargs = {'return_model':'True', 'return_results':'True', 'include_feasibility_slack':'False'}
    md, m, results = solve_dcopf(model_data, "ipopt", dcopf_model_generator=create_btheta_dcopf_model, **kwargs)

    m, md = create_psv_acpf_model(md)
    _solve_fixed_acpf(m, md)

    m, md = _dual_ldf_model(md)

    ref_bus = md.data['system']['reference_bus']
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    import numpy as np
    ldf = np.zeros((_len_branch,_len_bus))
    for _k, branch_name in _mapping_branch.items():
        if hasattr(m,"objective"):
            m.del_component(m.objective)
        m.objective = pe.Objective(expr=m.pfl[branch_name])
        m, results = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            ldf[_k,_b] = m.dual.get(m.eq_p_balance[bus_name])
    print(ldf)
    _, ldf_check, _ = tx_calc.calculate_ptdf_ldf(branches, buses, branch_attrs['names'], bus_attrs['names'], ref_bus, base_point=BasePointType.SOLUTION)
    print(ldf_check)

    return ldf


def calculate_qldf(model_data):
    """
    Calculates the sensitivity of the voltage magnitude to reactive power losses
    """
    kwargs = {'return_model':'True', 'return_results':'True', 'include_feasibility_slack':'False'}
    #md, m, results = solve_dcopf(model_data, "ipopt", dcopf_model_generator=create_btheta_dcopf_model, **kwargs)
    md, m, results = solve_acopf(model_data, "ipopt", acopf_model_generator=create_psv_acopf_model, **kwargs)

    m, md = create_psv_acpf_model(md)
    _solve_fixed_acpf(m, md)

    ref_bus = md.data['system']['reference_bus']
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')

    for k, k_dict in branches.items():
        k_dict['qfl'] = k_dict['qf'] + k_dict['qt']
        k_dict['qf'] = (k_dict['qf'] - k_dict['qt'])/2

    m, md = _dual_qldf_model(md)

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    import numpy as np
    qldf = np.zeros((_len_branch,_len_bus))
    for _k, branch_name in _mapping_branch.items():
        if hasattr(m,"objective"):
            m.del_component(m.objective)
        m.objective = pe.Objective(expr=-m.qfl[branch_name])
        m, results = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            qldf[_k,_b] = m.dual.get(m.eq_q_balance[bus_name])
    print(qldf)
    _, qldf_check, _ = tx_calc.calculate_qtdf_ldf(branches, buses, branch_attrs['names'], bus_attrs['names'], ref_bus, base_point=BasePointType.SOLUTION)
    print(qldf_check)

    return qldf


def _dual_ptdf_model(md):
    m = pe.ConcreteModel()
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    buses = dict(md.elements(element_type='bus'))
    gens = dict(md.elements(element_type='generator'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    gen_attrs = md.attributes(element_type='generator')

    libbus.declare_var_pl(m, bus_attrs['names'], initialize=bus_p_loads)
    m.pl.fix()
    shunts = dict(md.elements(element_type='shunt'))
    _, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    libgen.declare_var_pg(m, gen_attrs['names'], initialize=gen_attrs['pg'],
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )
    m.pg.fix()

    libbus.declare_var_vm(m, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max'])
                          )
    m.vm.fix()

    va_bounds = {k: (-pi, pi) for k in bus_attrs['va']}
    libbus.declare_var_va(m, bus_attrs['names'], initialize=bus_attrs['va'],
                          bounds=va_bounds
                          )

    libbranch.declare_var_pf(model=m,
                             index_set=branch_attrs['names']
                             )

    ref_bus = md.data['system']['reference_bus']
    slack_init = {ref_bus: 0}
    slack_bounds = {ref_bus: (0, inf)}
    decl.declare_var('p_slack_pos', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )

    m.va[ref_bus].fix(0.0)

    con_set = decl.declare_set('_con_eq_p_balance', m, bus_attrs['names'])
    m.eq_p_balance = pe.Constraint(con_set)
    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    for bus_name in con_set:
        p_expr = -sum([m.pf[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        p_expr += sum([m.pf[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        if bus_gs_fixed_shunts[bus_name] != 0.0:
            p_expr -= bus_gs_fixed_shunts[bus_name] * m.vm[bus_name] ** 2

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            p_expr -= m.pl[bus_name]

        if bus_name == ref_bus:
            p_expr -= m.p_slack_pos[bus_name]
            p_expr += m.p_slack_neg[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            p_expr += m.pg[gen_name]

        m.eq_p_balance[bus_name] = \
            p_expr == 0.0

    con_set = decl.declare_set('_con_eq_pf', m, branch_attrs['names'])
    m.eq_pf_branch = pe.Constraint(con_set)

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    J11 = tx_calc._calculate_J11(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION,
                   approximation_type=ApproximationType.PTDF)
    pf_constant = tx_calc._calculate_pf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += J11[_k][_b] * m.va[bus_name]
            expr += pf_constant[_k]

            m.eq_pf_branch[branch_name] = m.pf[branch_name] == expr

    return m, md


def _dual_ldf_model(md):
    m = pe.ConcreteModel()
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    buses = dict(md.elements(element_type='bus'))
    gens = dict(md.elements(element_type='generator'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    gen_attrs = md.attributes(element_type='generator')

    libbus.declare_var_pl(m, bus_attrs['names'], initialize=bus_p_loads)
    m.pl.fix()
    shunts = dict(md.elements(element_type='shunt'))
    _, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    libgen.declare_var_pg(m, gen_attrs['names'], initialize=gen_attrs['pg'],
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )
    m.pg.fix()

    libbus.declare_var_vm(m, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max'])
                          )
    m.vm.fix()

    va_bounds = {k: (-pi, pi) for k in bus_attrs['va']}
    libbus.declare_var_va(m, bus_attrs['names'], initialize=bus_attrs['va'],
                          bounds=va_bounds
                          )

    libbranch.declare_var_pf(model=m,
                             index_set=branch_attrs['names']
                             )

    libbranch.declare_var_pfl(model=m,
                             index_set=branch_attrs['names']
                             )

    ref_bus = md.data['system']['reference_bus']
    slack_init = {ref_bus: 0}
    slack_bounds = {ref_bus: (0, inf)}
    decl.declare_var('p_slack_pos', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )

    m.va[ref_bus].fix(0.0)

    con_set = decl.declare_set('_con_eq_p_balance', m, bus_attrs['names'])
    m.eq_p_balance = pe.Constraint(con_set)
    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    for bus_name in con_set:
        p_expr = -sum([m.pf[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        p_expr += sum([m.pf[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        p_expr += 0.5 * sum(m.pfl[branch_name] for branch_name in outlet_branches_by_bus[bus_name])
        p_expr += 0.5 * sum(m.pfl[branch_name] for branch_name in inlet_branches_by_bus[bus_name])

        if bus_gs_fixed_shunts[bus_name] != 0.0:
            p_expr -= bus_gs_fixed_shunts[bus_name] * m.vm[bus_name] ** 2

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            p_expr -= m.pl[bus_name]

        if bus_name == ref_bus:
            p_expr -= m.p_slack_pos[bus_name]
            p_expr += m.p_slack_neg[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            p_expr += m.pg[gen_name]

        m.eq_p_balance[bus_name] = \
            p_expr == 0.0

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    con_set = decl.declare_set('_con_eq_pf', m, branch_attrs['names'])
    m.eq_pf_branch = pe.Constraint(con_set)

    J11 = tx_calc._calculate_J11(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION,
                   approximation_type=ApproximationType.PTDF_LOSSES)
    pf_constant = tx_calc._calculate_pf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += J11[_k][_b] * m.va[bus_name]
            expr += pf_constant[_k]

            m.eq_pf_branch[branch_name] = m.pf[branch_name] == expr

    con_set = decl.declare_set('_con_eq_pfl', m, branch_attrs['names'])
    m.eq_pfl_branch = pe.Constraint(con_set)

    L11 = tx_calc._calculate_L11(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    pfl_constant = tx_calc._calculate_pfl_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += L11[_k][_b] * m.va[bus_name]
            expr += pfl_constant[_k]

            m.eq_pfl_branch[branch_name] = m.pfl[branch_name] == expr

    return m, md


def _dual_qtdf_model(md):
    m = pe.ConcreteModel()
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    buses = dict(md.elements(element_type='bus'))
    gens = dict(md.elements(element_type='generator'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    gen_attrs = md.attributes(element_type='generator')

    libbus.declare_var_ql(m, bus_attrs['names'], initialize=bus_q_loads)
    m.ql.fix()
    shunts = dict(md.elements(element_type='shunt'))
    bus_bs_fixed_shunts, _ = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    libgen.declare_var_qg(m, gen_attrs['names'], initialize=gen_attrs['qg'],
                          bounds=zip_items(gen_attrs['q_min'], gen_attrs['q_max'])
                          )
    m.qg.fix()

    libbus.declare_var_vm(m, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max'])
                          )

    libbranch.declare_var_qf(model=m,
                             index_set=branch_attrs['names']
                             )

    decl.declare_var('qfl', model=m, index_set=branch_attrs['names'], initialize=branch_attrs['qfl'])
    m.qfl.fix()

    ref_bus = md.data['system']['reference_bus']
    slack_init = {ref_bus: 0}
    slack_bounds = {ref_bus: (0, inf)}
    decl.declare_var('q_slack_pos', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('q_slack_neg', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )

    con_set = decl.declare_set('_con_eq_q_balance', m, bus_attrs['names'])
    m.eq_q_balance = pe.Constraint(con_set)
    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    for bus_name in con_set:
        q_expr = -sum([m.qf[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        q_expr += sum([m.qf[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        q_expr -= 0.5 * sum(m.qfl[branch_name] for branch_name in outlet_branches_by_bus[bus_name])
        q_expr -= 0.5 * sum(m.qfl[branch_name] for branch_name in inlet_branches_by_bus[bus_name])

        if bus_bs_fixed_shunts[bus_name] != 0.0:
            q_expr += bus_bs_fixed_shunts[bus_name] * value(m.vm[bus_name]) ** 2

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            q_expr -= m.ql[bus_name]

        if bus_name == ref_bus:
            q_expr -= m.q_slack_pos[bus_name]
            q_expr += m.q_slack_neg[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            q_expr += m.qg[gen_name]

        m.eq_q_balance[bus_name] = \
            q_expr == 0.0

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    con_set = decl.declare_set('_con_eq_qf', m, branch_attrs['names'])
    m.eq_qf_branch = pe.Constraint(con_set)

    J22 = tx_calc._calculate_J22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qf_constant = tx_calc._calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += J22[_k][_b] * m.vm[bus_name]
            expr += qf_constant[_k]

            m.eq_qf_branch[branch_name] = m.qf[branch_name] == expr

    return m, md


def _dual_qldf_model(md):
    m = pe.ConcreteModel()
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)

    buses = dict(md.elements(element_type='bus'))
    gens = dict(md.elements(element_type='generator'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    gen_attrs = md.attributes(element_type='generator')

    libbus.declare_var_ql(m, bus_attrs['names'], initialize=bus_q_loads)
    m.ql.fix()
    shunts = dict(md.elements(element_type='shunt'))
    bus_bs_fixed_shunts, _ = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    libgen.declare_var_qg(m, gen_attrs['names'], initialize=gen_attrs['qg'],
                          bounds=zip_items(gen_attrs['q_min'], gen_attrs['q_max'])
                          )
    m.qg.fix()

    libbus.declare_var_vm(m, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max'])
                          )

    libbranch.declare_var_qf(model=m, index_set=branch_attrs['names'], initialize=branch_attrs['qf'])

    decl.declare_var('qfl', model=m, index_set=branch_attrs['names'], initialize=branch_attrs['qfl'])

    ref_bus = md.data['system']['reference_bus']
    slack_init = {ref_bus: 0}
    slack_bounds = {ref_bus: (0, inf)}
    decl.declare_var('q_slack_pos', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('q_slack_neg', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )

    con_set = decl.declare_set('_con_eq_q_balance', m, bus_attrs['names'])
    m.eq_q_balance = pe.Constraint(con_set)
    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    for bus_name in con_set:
        q_expr = sum([m.qf[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        q_expr -= sum([m.qf[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        q_expr -= 0.5 * sum(m.qfl[branch_name] for branch_name in outlet_branches_by_bus[bus_name])
        q_expr -= 0.5 * sum(m.qfl[branch_name] for branch_name in inlet_branches_by_bus[bus_name])

        if bus_bs_fixed_shunts[bus_name] != 0.0:
            q_expr += bus_bs_fixed_shunts[bus_name] * value(m.vm[bus_name]) ** 2

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            q_expr -= m.ql[bus_name]

        if bus_name == ref_bus:
            q_expr -= m.q_slack_pos[bus_name]
            q_expr += m.q_slack_neg[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            q_expr += m.qg[gen_name]

        m.eq_q_balance[bus_name] = \
            q_expr == 0.0

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    index_set_branch = branch_attrs['names']
    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    con_set = decl.declare_set('_con_eq_qf', m, branch_attrs['names'])
    m.eq_qf_branch = pe.Constraint(con_set)

    J22 = tx_calc._calculate_J22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qf_constant = tx_calc._calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += J22[_k][_b] * m.vm[bus_name]
            expr += qf_constant[_k]

            m.eq_qf_branch[branch_name] = m.qf[branch_name] == expr

    con_set = decl.declare_set('_con_eq_qfl', m, branch_attrs['names'])
    m.eq_qfl_branch = pe.Constraint(con_set)

    L22 = tx_calc._calculate_L22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qfl_constant = tx_calc._calculate_qfl_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += L22[_k][_b] * m.vm[bus_name]
            expr += qfl_constant[_k]

            m.eq_qfl_branch[branch_name] = m.qfl[branch_name] == expr

    return m, md


def _solve_fixed_acpf(m, md):
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))

    ref_bus = md.data['system']['reference_bus']
    slack_init = {ref_bus: 0}
    slack_bounds = {ref_bus: (0, inf)}
    decl.declare_var('p_slack_pos', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )

    m.eq_p_balance[md.data['system']['reference_bus']]._body += m.p_slack_neg[md.data['system']['reference_bus']] - m.p_slack_pos[md.data['system']['reference_bus']]

    for gen_name in gens:
        m.pg[gen_name].fix(gens[gen_name]['pg'])
    for bus_name in buses:
        m.vm[bus_name].fix(1.0)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    m, results = _solve_model(m,"ipopt")
    _load_solution_to_model_data(m, md)
    tx_utils.scale_ModelData_to_pu(md, inplace=True)
    return

def _solve_fixed_acqf(m, md):
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))

    ref_bus = md.data['system']['reference_bus']
    slack_init = {ref_bus: 0}
    slack_bounds = {ref_bus: (0, inf)}
    decl.declare_var('p_slack_pos', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=m, index_set=[ref_bus],
                     initialize=slack_init, bounds=slack_bounds
                     )

    m.eq_q_balance[md.data['system']['reference_bus']]._body += m.p_slack_neg[md.data['system']['reference_bus']] - m.p_slack_pos[md.data['system']['reference_bus']]

    for gen_name in gens:
        m.qg[gen_name].fix(gens[gen_name]['pg'])
    m.va.fix()

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    m, results = _solve_model(m,"ipopt")
    _load_solution_to_model_data(m, md)
    tx_utils.scale_ModelData_to_pu(md, inplace=True)
    return

if __name__ == '__main__':
    import os
    from egret.parsers.matpower_parser import create_ModelData

    path = os.path.dirname(__file__)
    filename = 'pglib_opf_case3_lmbd.m'
    matpower_file = os.path.join(path, '../../../download/pglib-opf/', filename)
    md = create_ModelData(matpower_file)
    #calculate_ptdf(md)
    #calculate_ldf(md)
    #calculate_qtdf(md)
    calculate_qldf(md)
