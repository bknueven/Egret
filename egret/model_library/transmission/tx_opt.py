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
import numpy as np
from egret.model_library.defn import SensitivityCalculationMethod
import egret.model_library.transmission.tx_utils as tx_utils
from egret.models.acpf import create_psv_acpf_model
#from egret.models.dcopf import solve_dcopf, create_btheta_dcopf_model
from egret.models.acopf import _load_solution_to_model_data, create_psv_acopf_model, solve_acopf
import egret.model_library.decl as decl
from egret.common.solver_interface import _solve_model
import pyomo.environ as pe
from pyomo.environ import value
from egret.data.model_data import zip_items
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
from egret.model_library.transmission.tx_calc import calculate_conductance, calculate_susceptance
from egret.model_library.defn import BasePointType, ApproximationType

def _calculate_J11(branches,buses,index_set_branch,index_set_bus,base_point=BasePointType.FLATSTART,approximation_type=ApproximationType.PTDF):
    """
    Compute the power flow Jacobian for partial derivative of real power flow to voltage angle
    """
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    J11 = np.zeros((_len_branch,_len_bus))

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])

        if approximation_type == ApproximationType.PTDF:
            x = branch['reactance']
            b = -1/(tau*x)
        elif approximation_type == ApproximationType.PTDF_LOSSES:
            b = calculate_susceptance(branch)/tau

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']
        idx_col = [key for key, value in _mapping_bus.items() if value == from_bus][0]
        J11[idx_row][idx_col] = -b * vn * vm * cos(tn - tm - shift)

        idx_col = [key for key, value in _mapping_bus.items() if value == to_bus][0]
        J11[idx_row][idx_col] = b * vn * vm * cos(tn - tm - shift)

    return J11


def _calculate_J22(branches,buses,index_set_branch,index_set_bus,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow Jacobian for partial derivative of reactive power flow to voltage magnitude
    """
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    J22 = np.zeros((_len_branch,_len_bus))

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']
        idx_col = [key for key, value in _mapping_bus.items() if value == from_bus][0]
        J22[idx_row][idx_col] = -(b + bc/2)/tau**2 * vn - g/tau * vm * sin(tn - tm - shift)

        idx_col = [key for key, value in _mapping_bus.items() if value == to_bus][0]
        J22[idx_row][idx_col] = (b + bc/2) * vm - g/tau * vn * sin(tn - tm - shift)

    return J22


def _calculate_L11(branches,buses,index_set_branch,index_set_bus,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow Jacobian for partial derivative of real power losses to voltage angle
    """
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    L11 = np.zeros((_len_branch,_len_bus))

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)/tau

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']

        idx_col = [key for key, value in _mapping_bus.items() if value == from_bus][0]
        L11[idx_row][idx_col] = -2 * g * vn * vm * sin(tn - tm - shift)

        idx_col = [key for key, value in _mapping_bus.items() if value == to_bus][0]
        L11[idx_row][idx_col] = 2 * g * vn * vm * sin(tn - tm - shift)

    return L11


def _calculate_L22(branches,buses,index_set_branch,index_set_bus,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow Jacobian for partial derivative of reactive power losses to voltage magnitude
    """
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    L22 = np.zeros((_len_branch,_len_bus))

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']

        idx_col = [key for key, value in _mapping_bus.items() if value == from_bus][0]
        L22[idx_row][idx_col] = -2 * (b + bc/2)/tau**2 * vn + 2 * b/tau * vm * cos(tn - tm - shift)

        idx_col = [key for key, value in _mapping_bus.items() if value == to_bus][0]
        L22[idx_row][idx_col] = -2 * (b + bc/2) * vm + 2 * b/tau * vn * cos(tn - tm - shift)

    return L22


def _calculate_pf_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow constant for the taylor series expansion of real power flow as
    a convex combination of the from/to directions, i.e.,
    pf = 0.5*g*((tau*vn)^2 - vm^2) - tau*vn*vm*b*sin(tn-tm-shift)
    """

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    pf_constant = np.zeros(_len_branch)

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)
        b = calculate_susceptance(branch)

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']

        pf_constant[idx_row] = 0.5 * g * ((vn/tau) ** 2 - vm ** 2) \
                               - b/tau * vn * vm * (sin(tn - tm - shift) - cos(tn - tm - shift)*(tn - tm))

    return pf_constant


def _calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow constant for the taylor series expansion of reactive power flow as
    a convex combination of the from/to directions, i.e.,
    qf = -0.5*(b+bc/2)*((tau*vn)^2 - vm^2) - tau*vn*vm*g*sin(tn-tm-shift)
    """

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    qf_constant = np.zeros(_len_branch)

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']

        qf_constant[idx_row] = 0.5 * (b+bc/2) * (vn**2/tau**2 - vm**2) \
                               + g/tau * vn * vm * sin(tn - tm - shift)

    return qf_constant


def _calculate_pfl_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power losses constant for the taylor series expansion of real power losses as
    a convex combination of the from/to directions, i.e.,
    pfl = g*((tau*vn)^2 + vm^2) - 2*tau*vn*vm*g*cos(tn-tm-shift)
    """

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    pfl_constant = np.zeros(_len_branch)

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']

        pfl_constant[idx_row] = g * (vn**2/tau**2 + vm**2) \
                              - 2 * g/tau * vn * vm * (sin(tn - tm - shift) * (tn - tm) + cos(tn - tm - shift))

    return pfl_constant


def _calculate_qfl_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow constant for the taylor series expansion of reactive power losses as
    a convex combination of the from/to directions, i.e.,
    qfl = -(b+bc/2)*((tau*vn)^2 + vm^2) + 2*tau*vn*vm*b*cos(tn-tm-shift)
    """

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    qfl_constant = np.zeros(_len_branch)

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = radians(branch['transformer_phase_shift'])
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION: # TODO: check that we are loading the correct values (or results)
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = buses[from_bus]['va']
            tm = buses[to_bus]['va']

        qfl_constant[idx_row] = (b+bc/2) * (vn ** 2/tau**2 + vm**2) \
                               - 2 * b/tau * vn * vm * cos(tn - tm - shift)

    return qfl_constant


def calculate_ptdf(md,base_point=BasePointType.FLATSTART,calculation_method=SensitivityCalculationMethod.INVERT):
    """
    Calculates the sensitivity of voltage angle to real power injections
    """

    if calculation_method == SensitivityCalculationMethod.INVERT:
        branches = dict(md.elements(element_type='branch'))
        buses = dict(md.elements(element_type='bus'))
        branch_attrs = md.attributes(element_type='branch')
        bus_attrs = md.attributes(element_type='bus')
        index_set_branch = branch_attrs['names']
        index_set_bus = bus_attrs['names']
        reference_bus = md.data['system']['reference_bus']

        _len_bus = len(index_set_bus)
        _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}
        _len_branch = len(index_set_branch)
        _ref_bus_idx = [key for key, value in _mapping_bus.items() if value == reference_bus][0]

        J = _calculate_J11(branches,buses,index_set_branch,index_set_bus,base_point,approximation_type=ApproximationType.PTDF)
        A = calculate_adjacency_matrix(branches,index_set_branch,index_set_bus)
        M = np.matmul(A.transpose(),J)

        J0 = np.zeros((_len_bus+1,_len_bus+1))
        J0[:-1,:-1] = M
        J0[-1][_ref_bus_idx] = 1
        J0[_ref_bus_idx][-1] = 1

        try:
            SENSI = np.linalg.inv(J0)
        except np.linalg.LinAlgError:
            print("Matrix not invertible. Calculating pseudo-inverse instead.")
            SENSI = np.linalg.pinv(J0,rcond=1e-7)
            pass
        SENSI = SENSI[:-1,:-1]

        PTDF = np.around(np.matmul(J,SENSI),8)
    elif calculation_method == SensitivityCalculationMethod.DUAL:
        PTDF = solve_ptdf(md)

    return PTDF


def calculate_ptdf_ldf(md,base_point=BasePointType.SOLUTION,calculation_method=SensitivityCalculationMethod.INVERT):
    """
    Calculates the sensitivity of the voltage angle to the real power injections and losses on the lines. Includes the
    calculation of the constant term for the quadratic losses on the lines.
    """

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    index_set_branch = branch_attrs['names']
    index_set_bus = bus_attrs['names']
    reference_bus = md.data['system']['reference_bus']

    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0, _len_bus))}
    _len_branch = len(index_set_branch)
    _ref_bus_idx = [key for key, value in _mapping_bus.items() if value == reference_bus][0]

    Jc = _calculate_pf_constant(branches,buses,index_set_branch,base_point)
    Lc = _calculate_pfl_constant(branches,buses,index_set_branch,base_point)

    A = calculate_adjacency_matrix(branches, index_set_branch, index_set_bus)
    AA = calculate_absolute_adjacency_matrix(A)

    if calculation_method == SensitivityCalculationMethod.INVERT:
        J = _calculate_J11(branches,buses,index_set_branch,index_set_bus,base_point,approximation_type=ApproximationType.PTDF_LOSSES)
        L = _calculate_L11(branches,buses,index_set_branch,index_set_bus,base_point)

        M1 = np.matmul(A.transpose(),J)
        M2 = np.matmul(AA.transpose(),L)
        M = M1 + 0.5 * M2

        J0 = np.zeros((_len_bus+1,_len_bus+1))
        J0[:-1,:-1] = M
        J0[-1][_ref_bus_idx] = 1
        J0[_ref_bus_idx][-1] = 1

        try:
            SENSI = np.linalg.inv(J0)
        except np.linalg.LinAlgError:
            print("Matrix not invertible. Calculating pseudo-inverse instead.")
            SENSI = np.linalg.pinv(J0,rcond=1e-7)
            pass
        SENSI = SENSI[:-1,:-1]

        PTDF = np.around(np.matmul(J, SENSI),8)
        LDF = np.around(np.matmul(L,SENSI),8)
    elif calculation_method == SensitivityCalculationMethod.DUAL:
        PTDF = solve_ptdf(md)
        LDF = solve_ldf(md)

    M1 = np.matmul(A.transpose(), Jc)
    M2 = np.matmul(AA.transpose(), Lc)
    M = M1 + 0.5 * M2
    LDF_constant = -np.matmul(LDF,M) + Lc
    PTDF_constant = -np.matmul(PTDF,M) + Jc

    return PTDF, LDF, PTDF_constant, LDF_constant


def calculate_qtdf_ldf_vdf(md,base_point=BasePointType.SOLUTION,calculation_method=SensitivityCalculationMethod.INVERT):
    """
    Calculates the sensitivity of the voltage magnitude to the reactive power injections and losses on the lines. Includes the
    calculation of the constant term for the quadratic losses on the lines.
    """

    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    shunts = dict(md.elements(element_type='shunt'))

    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    index_set_branch = branch_attrs['names']
    index_set_bus = bus_attrs['names']
    reference_bus = md.data['system']['reference_bus']

    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0, _len_bus))}
    _len_branch = len(index_set_branch)
    _ref_bus_idx = [key for key, value in _mapping_bus.items() if value == reference_bus][0]

    Jc = _calculate_qf_constant(branches,buses,index_set_branch,base_point)
    Lc = _calculate_qfl_constant(branches,buses,index_set_branch,base_point)

    A = calculate_adjacency_matrix(branches,index_set_branch,index_set_bus)
    AA = calculate_absolute_adjacency_matrix(A)

    bus_bs_fixed_shunts, _ = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)
    BS = np.zeros((_len_bus, _len_bus))
    for idx, bus_name in _mapping_bus.items():
        if bus_bs_fixed_shunts[bus_name] != 0.0:
            BS[idx][idx] = -bus_bs_fixed_shunts[bus_name] * buses[bus_name]['vm']

    if calculation_method == SensitivityCalculationMethod.INVERT:
        J = _calculate_J22(branches,buses,index_set_branch,index_set_bus,base_point) # Derive reactive power flow sensitivity to voltage
        L = _calculate_L22(branches,buses,index_set_branch,index_set_bus,base_point) # Derive reactive power loss sensitivity to voltage

        M1 = np.matmul(A.transpose(),J)
        M2 = np.matmul(AA.transpose(),L)
        M = M1 + 0.5 * M2 #+ 2 * BS         # Calculate A’*H + 0.5*absA’*L + 2Bs in paper

        J0 = np.zeros((_len_bus+1,_len_bus+1))
        J0[:-1,:-1] = M
        J0[-1][_ref_bus_idx] = 1
        J0[_ref_bus_idx][-1] = 1

        try:
            # SENSI = np.linalg.inv(J0)
            SENSI = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("Matrix not invertible. Calculating pseudo-inverse instead.")
            # SENSI = np.linalg.inv(J0,rcond=1e-7)
            SENSI = np.linalg.inv(M,rcond=1e-7)
            pass
        # SENSI = SENSI[:-1,:-1]

        QTDF = np.around(np.matmul(J, SENSI),8) # This is H*(A’*H + 0.5*absA’*L + 2Bs)^-1
        LDF = np.around(np.matmul(L,SENSI),8) # This is L*(A’*H + 0.5*absA’*L + 2Bs)^-1

        VDF = np.around(SENSI,8) # This is X = (A’*H + 0.5*absA’*L + 2Bs)^-1
        #VDF = np.around(np.linalg.inv(MV),8)

    elif calculation_method == SensitivityCalculationMethod.DUAL:
        QTDF = solve_qtdf(md)
        LDF = solve_qldf(md)
        VDF = solve_vdf(md)

    M1 = np.matmul(A.transpose(), Jc)
    M2 = np.matmul(AA.transpose(), Lc)
    M = M1 + 0.5 * M2
    LDF_constant = -np.matmul(LDF,M) + Lc
    QTDF_constant = -np.matmul(QTDF,M) + Jc
    VM = np.asarray([value for (key,value) in bus_attrs['vm'].items()])
    BV = np.matmul(BS, VM)
    M = M #- BV
    VDF_constant = -np.matmul(VDF,M)

    gens = dict(md.elements(element_type='generator'))
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    QG = list()
    for bus_name in bus_attrs['names']:
        tmp = 0
        for gen_name in gens_by_bus[bus_name]:
            tmp += gens[gen_name]["qg"]
        QG.append(tmp)
    QG = np.array(QG)
    QL = np.asarray([value for (key,value) in bus_attrs['ql'].items()])
    TMP = - M1 - 0.5 * M2 + np.matmul(BS, VM) + QG - QL
    RESULT = np.matmul(VDF,TMP)

    return QTDF, LDF, VDF, QTDF_constant, LDF_constant, VDF_constant


def calculate_adjacency_matrix(branches,index_set_branch,index_set_bus):
    """
    Calculates the adjacency matrix where (-1) represents flow from the bus and (1) represents flow to the bus
    for a given branch
    """
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    _len_branch = len(index_set_branch)
    _mapping_branch = {i: index_set_branch[i] for i in list(range(0,_len_branch))}

    adjacency_matrix = np.zeros((_len_branch,_len_bus))

    for idx_row, branch_name in _mapping_branch.items():
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        idx_col = [key for key, value in _mapping_bus.items() if value == from_bus][0]
        adjacency_matrix[idx_row,idx_col] = -1

        to_bus = branch['to_bus']
        idx_col = [key for key, value in _mapping_bus.items() if value == to_bus][0]
        adjacency_matrix[idx_row,idx_col] = 1

    return adjacency_matrix


def calculate_absolute_adjacency_matrix(adjacency_matrix):
    """
    Calculates the absolute value of the adjacency matrix
    """
    return np.absolute(adjacency_matrix)


def solve_ptdf(model_data):
    """
    Calculates the sensitivity of the voltage angle to real power injections
    """
    kwargs = {'return_model':'True', 'return_results':'True', 'include_feasibility_slack':'False'}
    #md, m, results = solve_dcopf(model_data, "ipopt", dcopf_model_generator=create_btheta_dcopf_model, **kwargs)
    md, m, results = solve_acopf(model_data, "ipopt", acopf_model_generator=create_psv_acopf_model, **kwargs)

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
        m, results, flag = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            ptdf[_k,_b] = m.dual.get(m.eq_p_balance[bus_name])
    print(ptdf)
    ptdf_check = calculate_ptdf(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT)
    print(ptdf_check)
    print(sum(sum(abs(ptdf-ptdf_check))))
    return ptdf


def solve_qtdf(model_data):
    """
    Calculates the sensitivity of the voltage magnitude to reactive power injections
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

    m, md = _dual_qldf_model(md)
    # m, md = _dual_qtdf_model(md)

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
        m.objective = pe.Objective(expr=-m.qf[branch_name])
        m, results, flag = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            qtdf[_k,_b] = m.dual.get(m.eq_q_balance[bus_name])
    print(qtdf)
    qtdf_check, _, _, _, _, _ = calculate_qtdf_ldf_vdf(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT)
    print(qtdf_check)

    return qtdf


def solve_ldf(model_data):
    """
    Calculates the sensitivity of the voltage angle to real power losses
    """
    kwargs = {'return_model':'True', 'return_results':'True', 'include_feasibility_slack':'False'}
    #md, m, results = solve_dcopf(model_data, "ipopt", dcopf_model_generator=create_btheta_dcopf_model, **kwargs)
    md, m, results = solve_acopf(model_data, "ipopt", acopf_model_generator=create_psv_acopf_model, **kwargs)

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
        m, results, flag = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            ldf[_k,_b] = m.dual.get(m.eq_p_balance[bus_name])
    print(ldf)
    _, ldf_check, _, _ = calculate_ptdf_ldf(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT)
    print(ldf_check)

    return ldf


def solve_qldf(model_data):
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
        m, results, flag = _solve_model(m, "gurobi")
        for _b, bus_name in _mapping_bus.items():
            qldf[_k,_b] = m.dual.get(m.eq_q_balance[bus_name])
    print(qldf)
    _, qldf_check, _, _, _, _ = calculate_qtdf_ldf_vdf(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT)
    print(qldf_check)

    return qldf

def solve_vdf(model_data):
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

    m, md = _dual_vdf_model(md)

    index_set_bus = bus_attrs['names']
    _len_bus = len(index_set_bus)
    _mapping_bus = {i: index_set_bus[i] for i in list(range(0,_len_bus))}

    import numpy as np
    vdf = np.zeros((_len_bus,_len_bus))
    for b, bus_name in _mapping_bus.items():
        if hasattr(m,"objective"):
            m.del_component(m.objective)
        m.objective = pe.Objective(expr=-m.vm[bus_name])
        m, results, flag = _solve_model(m, "gurobi")
        for _b, _bus_name in _mapping_bus.items():
            vdf[b,_b] = m.dual.get(m.eq_q_balance[_bus_name])
    print(vdf)
    _, _, vdf_check, _, _, _ = calculate_qtdf_ldf_vdf(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT)
    print(vdf_check)

    return vdf


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

    J11 = _calculate_J11(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION,
                   approximation_type=ApproximationType.PTDF)
    pf_constant = _calculate_pf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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

    J11 = _calculate_J11(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION,
                   approximation_type=ApproximationType.PTDF_LOSSES)
    pf_constant = _calculate_pf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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

    L11 = _calculate_L11(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    pfl_constant = _calculate_pfl_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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

        # if bus_name == ref_bus:
        #     q_expr -= m.q_slack_pos[bus_name]
        #     q_expr += m.q_slack_neg[bus_name]

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

    J22 = _calculate_J22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qf_constant = _calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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

        # if bus_name == ref_bus:
        #     q_expr -= m.q_slack_pos[bus_name]
        #     q_expr += m.q_slack_neg[bus_name]

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

    J22 = _calculate_J22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qf_constant = _calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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

    L22 = _calculate_L22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qfl_constant = _calculate_qfl_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

    for _k, branch_name in _mapping_branch.items():
        branch = branches[branch_name]
        if branch["in_service"]:
            expr = 0
            for _b, bus_name in _mapping_bus.items():
                expr += L22[_k][_b] * m.vm[bus_name]
            expr += qfl_constant[_k]

            m.eq_qfl_branch[branch_name] = m.qfl[branch_name] == expr

    return m, md


def _dual_vdf_model(md):
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

        # if bus_name == ref_bus:
        #     q_expr -= m.q_slack_pos[bus_name]
        #     q_expr += m.q_slack_neg[bus_name]

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

    J22 = _calculate_J22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qf_constant = _calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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

    L22 = _calculate_L22(branches, buses, index_set_branch, index_set_bus, base_point=BasePointType.SOLUTION)
    qfl_constant = _calculate_qfl_constant(branches,buses,index_set_branch,base_point=BasePointType.SOLUTION)

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
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

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
        if gens_by_bus[bus_name]:
            m.vm[bus_name].fix(1.0)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    m, results, flag = _solve_model(m,"ipopt")
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
    m, results, flag = _solve_model(m,"ipopt")
    _load_solution_to_model_data(m, md)
    tx_utils.scale_ModelData_to_pu(md, inplace=True)
    return

if __name__ == '__main__':
    import os
    from egret.parsers.matpower_parser import create_ModelData

    path = os.path.dirname(__file__)
    # filename = 'pglib_opf_case14_ieee.m'
    filename = 'pglib_opf_case5_pjm.m'
    matpower_file = os.path.join(path, '../../../download/pglib-opf/', filename)
    md = create_ModelData(matpower_file)
    from egret.models.acopf import solve_acopf
    #md = solve_acopf(md, "ipopt")
    solve_ptdf(md)
    #solve_ldf(md)
    #solve_qtdf(md)
    #solve_qldf(md)
    #solve_vdf(md)
