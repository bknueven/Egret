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
different computations for transmission models
"""
import os, time
import math
import numpy as np
import scipy as sp
import pandas as pd
import scipy.sparse
from scipy.sparse.linalg import dsolve,isolve
from math import cos, sin
from egret.model_library.defn import BasePointType, ApproximationType
from pyomo.environ import value
from egret.common.log import logger

def calculate_conductance(branch):
    rs = branch['resistance']
    xs = branch['reactance']
    return rs / (rs**2 + xs**2)


def calculate_susceptance(branch):
    rs = branch['resistance']
    xs = branch['reactance']
    return -xs / (rs**2 + xs**2)


def calculate_y_matrix_from_branch(branch):
    rs = branch['resistance']
    xs = branch['reactance']
    bs = branch['charging_susceptance']
    tau = 1.0
    shift = 0.0
    if branch['branch_type'] == 'transformer':
        tau = branch['transformer_tap_ratio']
        shift = branch['transformer_phase_shift']
    return calculate_y_matrix(rs, xs, bs, tau, shift)


def calculate_y_matrix(rs, xs, bc, tau, shift):
    """
    Compute the y matrix from various branch properties

    Parameters
    ----------
    rs : float
        Branch resistance
    xs : float
        Branch reactance
    bc : float
        Branch charging susceptance
    tau : float
        Branch transformer tap ratio
    shift : float
        Branch transformer phase shift

    Returns
    -------
        list : list of floats representing the y matrix
               [Y(ifr,vfr), Y(ifr,vfj), Y(ifr,vtr), Y(ifr,vtj),
               Y(ifj,vfr), Y(ifj,vfj), Y(ifj,vtr), Y(ifj,vtj),
               Y(itr,vfr), Y(itr,vfj), Y(itr,vtr), Y(itr,vtj),
               Y(itj,vfr), Y(itj,vfj), Y(itj,vtr), Y(itj,vtj)]
    """
    bc = bc/2
    tr = tau * math.cos(math.radians(shift))
    tj = tau * math.sin(math.radians(shift))
    mag = rs**2 + xs**2

    a = rs/(tau**2*mag)                    # c1
    b = (1/tau**2) * (xs/mag - bc)         # c2
    c = (-rs*tr - xs*tj)/(tau**2 * mag)    # c3
    d = (rs*tj - xs*tr)/(tau**2 * mag)     # c4
    e = -b                                 # -c2
    f = a                                  # c1
    g = -d                                 # -c4
    h = c                                  # c3
    i = (xs*tj - rs*tr)/(tau**2 * mag)     # c7
    j = (-rs*tj - xs*tr)/(tau**2 * mag)    # c8
    k = rs/mag                             # c5
    l = xs/mag - bc                        # c6
    m = -j                                 # -c8
    n = i                                  # c7
    o = -l                                 # -c6
    p = k                                  # c5

    # y = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
    y_dict = {}
    y_dict[('ifr', 'vfr')] = a
    y_dict[('ifr', 'vfj')] = b
    y_dict[('ifr', 'vtr')] = c
    y_dict[('ifr', 'vtj')] = d
    
    y_dict[('ifj', 'vfr')] = e
    y_dict[('ifj', 'vfj')] = f
    y_dict[('ifj', 'vtr')] = g
    y_dict[('ifj', 'vtj')] = h
    
    y_dict[('itr', 'vfr')] = i
    y_dict[('itr', 'vfj')] = j
    y_dict[('itr', 'vtr')] = k
    y_dict[('itr', 'vtj')] = l

    y_dict[('itj', 'vfr')] = m
    y_dict[('itj', 'vfj')] = n
    y_dict[('itj', 'vtr')] = o
    y_dict[('itj', 'vtj')] = p

    return y_dict

def calculate_ifr(vfr, vfj, vtr, vtj, y_matrix):
    """
    Compute ifr from voltages and the y_matrix (computed
    from the branch properties using :py:meth:`calculate_branch_y_matrix`)
    """
    ifr = y_matrix['ifr', 'vfr'] * vfr + y_matrix['ifr', 'vfj'] * vfj + \
        y_matrix['ifr', 'vtr'] * vtr + y_matrix['ifr', 'vtj'] * vtj
    return ifr


def calculate_ifj(vfr, vfj, vtr, vtj, y_matrix):
    """
    Compute ify from voltages and the y_matrix (computed
    from the branch properties using :py:meth:`calculate_branch_y_matrix`)
    """
    ifj = y_matrix['ifj', 'vfr'] * vfr + y_matrix['ifj', 'vfj'] * vfj + \
        y_matrix['ifj', 'vtr'] * vtr + y_matrix['ifj', 'vtj'] * vtj
    return ifj


def calculate_itr(vfr, vfj, vtr, vtj, y_matrix):
    """
    Compute itr from voltages and the y_matrix (computed
    from the branch properties using :py:meth:`calculate_branch_y_matrix`)
    """
    itr = y_matrix['itr', 'vfr'] * vfr + y_matrix['itr', 'vfj'] * vfj + \
        y_matrix['itr', 'vtr'] * vtr + y_matrix['itr', 'vtj'] * vtj
    return itr


def calculate_itj(vfr, vfj, vtr, vtj, y_matrix):
    """
    Compute itj from voltages and the y_matrix (computed
    from the branch properties using :py:meth:`calculate_branch_y_matrix`)
    """
    itj = y_matrix['itj', 'vfr'] * vfr + y_matrix['itj', 'vfj'] * vfj + \
        y_matrix['itj', 'vtr'] * vtr + y_matrix['itj', 'vtj'] * vtj
    return itj


def calculate_ir(p, q, vr, vj):
    """
    Compute ir from power flows and voltages
    """
    ir = (q*vj+p*vr)/(vj**2 + vr**2)
    return ir


def calculate_ij(p, q, vr, vj):
    """
    Compute ij from power flows and voltages
    """
    ij = (p*vj-q*vr)/(vj**2 + vr**2)
    return ij


def calculate_p(ir, ij, vr, vj):
    """
    Compute real power flow from currents and voltages
    """
    p = vr * ir + vj * ij
    return p


def calculate_q(ir, ij, vr, vj):
    """
    Compute reactive power flow from currents and voltages
    """
    q = vj * ir - vr * ij
    return q


def calculate_vr_from_vm_va(vm, va):
    """
    Compute the value of vr from vm and va

    Parameters
    ----------
    vm : float
        The value of voltage magnitude (per)
    va : float
        The value of voltage angle (degrees)

    Returns
    -------
        float : the value of vr or None if
           either vm or va (or both) is None
    """
    if vm is not None and va is not None:
        vr = vm * math.cos(va*math.pi/180)
        return vr
    return None


def calculate_vj_from_vm_va(vm, va):
    """
    Compute the value of vj from vm and va

    Parameters
    ----------
    vm : float
        The value of voltage magnitude (per)
    va : float
        The value of voltage angle (degrees)

    Returns
    -------
        float : the value of vj or None if
           either vm or va (or both) is None
    """
    if vm is not None and va is not None:
        vj = vm * math.sin(va*math.pi/180)
        return vj
    return None


def calculate_vm_from_vj_vr(vj,vr):
    """
    Compute the value of vm from vj and vr

    Parameters
    ----------
    vj : float
        The value of the imaginary part of the voltage phasor (per)
    vr : float
        The value of the real part of the voltage phasor (per)

    Returns
    -------
        float : the value of the voltage magnitude vm or None if
           either vj or vr (or both) is None
    """
    if vj is not None and vr is not None:
        vm = math.sqrt(vj**2 + vr**2)
        return vm
    return None


def calculate_va_from_vj_vr(vj, vr):
    """
    Compute the value of va from vj and vr

    Parameters
    ----------
    vj : float
        The value of the imaginary part of the voltage phasor (per)
    vr : float
        The value of the real part of the voltage phasor (per)

    Returns
    -------
        float : the value of the voltage angle va in degrees or None if
           either vj or vr (or both) is None
    """
    if vj is not None and vr is not None:
        va = math.degrees(math.atan(vj/vr))
        return va
    return None

def _calculate_J11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point=BasePointType.FLATSTART,approximation_type=ApproximationType.PTDF):
    """
    Compute the power flow Jacobian for partial derivative of real power flow to voltage angle
    """
    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)

    data = []
    row = []
    col = []

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
            #print('Branch {} has shift {} and tap ratio {}'.format(branch_name,shift,tau))

        if approximation_type == ApproximationType.PTDF:
            x = branch['reactance']
            b = -1/(tau*x)
        elif approximation_type == ApproximationType.PTDF_LOSSES:
            b = calculate_susceptance(branch)/tau

        if base_point == BasePointType.FLATSTART:
            vn = 1
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        val = -b * vn * vm * cos(tn - tm + shift)

        # else somebody might have to check
        if val != 0.0:
            idx_col = mapping_bus_to_idx[from_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(val)

            idx_col = mapping_bus_to_idx[to_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(-val)

    J11 = sp.sparse.coo_matrix( (data, (row,col)), shape=(_len_branch, _len_bus))
    return J11.tocsr()

def _calculate_J22(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow Jacobian for partial derivative of reactive power flow to voltage magnitude
    """
    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)

    data = []
    row = []
    col = []

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        val = -(b + bc/2) * vn / (tau**2) - g * vm * sin(tn - tm + shift)/tau

        # else somebody might have to check
        if val != 0.0:
            idx_col = mapping_bus_to_idx[from_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(val)

        val = (b + bc/2) * vm - g * vn * sin(tn - tm + shift)/tau

        # else somebody might have to check
        if val != 0.0:
            idx_col = mapping_bus_to_idx[to_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(val)

    J22 = sp.sparse.coo_matrix( (data, (row,col)), shape=(_len_branch, _len_bus))
    return J22.tocsr()


def _calculate_L11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow Jacobian for partial derivative of real power losses to voltage angle
    """
    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)

    row = []
    col = []
    data = []

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)/tau

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.

        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        val = 2 * g * vn * vm * sin(tn - tm + shift)

        # else somebody might have to check
        if val != 0.0:
            idx_col = mapping_bus_to_idx[from_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(val)

            idx_col = mapping_bus_to_idx[to_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(-val)

    L11 = sp.sparse.coo_matrix((data,(row,col)),shape=(_len_branch,_len_bus))
    return L11.tocsr()


def _calculate_L22(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow Jacobian for partial derivative of reactive power losses to voltage magnitude
    """
    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)

    row = []
    col = []
    data = []

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        val = -2 * (b + bc/2) * vn / (tau**2) + 2 * b * vm * cos(tn - tm + shift)/tau

        # else somebody might have to check
        if val != 0.0:
            idx_col = mapping_bus_to_idx[from_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(val)

        val = -2 * (b + bc/2) * vm + 2 * b * vn * cos(tn - tm + shift)/tau

        # else somebody might have to check
        if val != 0.0:
            idx_col = mapping_bus_to_idx[to_bus]
            row.append(idx_row)
            col.append(idx_col)
            data.append(val)

    L22 = sp.sparse.coo_matrix((data,(row,col)),shape=(_len_branch,_len_bus))
    return L22.tocsr()


def calculate_phi_constant(branches,index_set_branch,index_set_bus,approximation_type=ApproximationType.PTDF, mapping_bus_to_idx=None):
    """
    Compute the phase shifter constant for fixed phase shift transformers
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    row_from = []
    row_to = []
    col = []
    data = []

    for idx_col, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        b = 0.
        if approximation_type == ApproximationType.PTDF:
            x = branch['reactance']
            b = -(1/x)*(shift/tau)
        elif approximation_type == ApproximationType.PTDF_LOSSES:
            b = calculate_susceptance(branch)*(shift/tau)

        # else somebody might have to check
        if b != 0.0:
            row_from.append(mapping_bus_to_idx[from_bus])
            row_to.append(mapping_bus_to_idx[to_bus])
            col.append(idx_col)
            data.append(b)

    phi_from = sp.sparse.coo_matrix((data,(row_from,col)), shape=(_len_bus,_len_branch))
    phi_to = sp.sparse.coo_matrix((data,(row_to,col)), shape=(_len_bus,_len_branch))

    return phi_from.tocsr(), phi_to.tocsr()


def calculate_phi_q_constant(branches,index_set_branch,index_set_bus,mapping_bus_to_idx=None):
    """
    Compute the phase shifter constant impact on reactive power flow for fixed phase shift transformers
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    row_from = []
    row_to = []
    col = []
    data = []

    for idx_col, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        g = calculate_conductance(branch)*(shift/tau)

        # else somebody might have to check
        if g != 0.0:
            row_from.append(mapping_bus_to_idx[from_bus])
            row_to.append(mapping_bus_to_idx[to_bus])
            col.append(idx_col)
            data.append(g)

    phi_from = sp.sparse.coo_matrix((data,(row_from,col)), shape=(_len_bus,_len_branch))
    phi_to = sp.sparse.coo_matrix((data,(row_to,col)), shape=(_len_bus,_len_branch))

    return phi_from.tocsr(), phi_to.tocsr()


def calculate_phi_loss_constant(branches,index_set_branch,index_set_bus,approximation_type=ApproximationType.PTDF_LOSSES, mapping_bus_to_idx=None):
    """
    Compute the phase shifter constant for fixed phase shift transformers
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    row_from = []
    row_to = []
    col = []
    data = []

    for idx_col, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        g = 0.
        if approximation_type == ApproximationType.PTDF:
            r = branch['resistance']
            g = (1/r)*(1/tau)*shift**2
        elif approximation_type == ApproximationType.PTDF_LOSSES:
            g = calculate_conductance(branch)*(1/tau)*shift**2

        # else somebody might have to check
        if g != 0.0:
            row_from.append(mapping_bus_to_idx[from_bus])
            row_to.append(mapping_bus_to_idx[to_bus])
            col.append(idx_col)
            data.append(g)

    phi_loss_from = sp.sparse.coo_matrix((data,(row_from,col)),shape=(_len_bus,_len_branch))
    phi_loss_to = sp.sparse.coo_matrix((data,(row_to,col)),shape=(_len_bus,_len_branch))

    return phi_loss_from.tocsr(), phi_loss_to.tocsr()


def calculate_phi_loss_q_constant(branches,index_set_branch,index_set_bus,mapping_bus_to_idx=None):
    """
    Compute the phase shifter constant for fixed phase shift transformers
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    row_from = []
    row_to = []
    col = []
    data = []

    for idx_col, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        b = calculate_susceptance(branch)*(1/tau)*shift**2

        # else somebody might have to check
        if b != 0.0:
            row_from.append(mapping_bus_to_idx[from_bus])
            row_to.append(mapping_bus_to_idx[to_bus])
            col.append(idx_col)
            data.append(b)

    phi_loss_from = sp.sparse.coo_matrix((data,(row_from,col)),shape=(_len_bus,_len_branch))
    phi_loss_to = sp.sparse.coo_matrix((data,(row_to,col)),shape=(_len_bus,_len_branch))

    return phi_loss_from.tocsr(), phi_loss_to.tocsr()


def _calculate_pf_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow constant for the taylor series expansion of real power flow as
    a convex combination of the from/to directions, i.e.,
    pf = 0.5*g*((tau*vn)^2 - vm^2) - b*tau*vn*vm*sin(tn-tm-shift) + b*tau*vn*vm*cos(tn-tm-shift)*(tn-tm)
    """

    _len_branch = len(index_set_branch)
    ## this will be fully dense
    pf_constant = np.zeros(_len_branch)

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)
        b = calculate_susceptance(branch)/tau

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        pf_constant[idx_row] = 0.5 * g * ((vn/tau) ** 2 - vm ** 2) \
                               - b * vn * vm * sin(tn - tm + shift) \
                               + b * vn * vm * cos(tn - tm + shift)*(tn - tm)

    return pf_constant


def _calculate_qf_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow constant for the taylor series expansion of reactive power flow as
    a convex combination of the from/to directions, i.e.,
    qf = -0.5*(b+bc/2)*((tau*vn)^2 - vm^2) - tau*vn*vm*g*sin(tn-tm+shift)
    """

    _len_branch = len(index_set_branch)
    ## this will be fully dense
    qf_constant = np.zeros(_len_branch)

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        g = calculate_conductance(branch)
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        qf_constant[idx_row] = 0.5 * (b+bc/2) * (vn**2/tau**2 - vm**2) \
                               + g * vn * vm * sin(tn - tm + shift)/tau

    return qf_constant


def _calculate_pfl_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power losses constant for the taylor series expansion of real power losses as
    a convex combination of the from/to directions, i.e.,
    pfl_constant = g*((tau*vn)^2 + vm^2) - 2*tau*vn*vm*g*cos(tn-tm-shift) - 2*tau*vn*vm*g*sin(tn-tm-shift)(tn-tm)
    """

    _len_branch = len(index_set_branch)

    ## this will be fully dense
    pfl_constant = np.zeros(_len_branch)

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        _g = calculate_conductance(branch)
        g = _g/tau
        g2 = _g/tau**2

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        pfl_constant[idx_row] = g2 * (vn ** 2) + _g * (vm ** 2) \
                                        - 2 * g * vn * vm * cos(tn - tm + shift) \
                                        - 2 * g * vn * vm * sin(tn - tm + shift) * (tn - tm)

    return pfl_constant


def _calculate_qfl_constant(branches,buses,index_set_branch,base_point=BasePointType.FLATSTART):
    """
    Compute the power flow constant for the taylor series expansion of reactive power losses as
    a convex combination of the from/to directions, i.e.,
    qfl = -(b+bc/2)*((tau*vn)^2 + vm^2) + 2*tau*vn*vm*b*cos(tn-tm+shift)
    """

    _len_branch = len(index_set_branch)

    ## this will be fully dense
    qfl_constant = np.zeros(_len_branch)

    for idx_row, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = -math.radians(branch['transformer_phase_shift'])
        b = calculate_susceptance(branch)
        bc = branch['charging_susceptance']

        if base_point == BasePointType.FLATSTART:
            vn = 1.
            vm = 1.
            tn = 0.
            tm = 0.
        elif base_point == BasePointType.SOLUTION:
            vn = buses[from_bus]['vm']
            vm = buses[to_bus]['vm']
            tn = math.radians(buses[from_bus]['va'])
            tm = math.radians(buses[to_bus]['va'])

        qfl_constant[idx_row] = (b+bc/2) * ((vn/tau)**2 + vm**2) \
                               - 2 * b * vn * vm * cos(tn - tm + shift) / tau

    return qfl_constant


def calculate_ptdf(branches,buses,index_set_branch,index_set_bus,reference_bus,base_point=BasePointType.FLATSTART,active_index_set_branch=None,mapping_bus_to_idx=None):
    """
    Calculates the sensitivity of the voltage angle to real power injections
    Parameters
    ----------
    branches: dict{}
        The dictionary of branches for the test case
    buses: dict{}
        The dictionary of buses for the test case
    index_set_branch: list
        The list of keys for branches for the test case
    index_set_bus: list
        The list of keys for buses for the test case
    reference_bus: key value
        The reference bus key value
    base_point: egret.model_library_defn.BasePointType
        The base-point type for calculating the PTDF matrix
    active_index_set_branch: list
        The list of keys for branches needed to compute a partial PTDF matrix
        If this is None, a dense PTDF matrix is returned
    mapping_bus_to_idx: dict
        A map from bus names to indices for matrix construction. If None,
        will be inferred from index_set_bus.
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    _ref_bus_idx = mapping_bus_to_idx[reference_bus]

    J = _calculate_J11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point,approximation_type=ApproximationType.PTDF)
    A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus,mapping_bus_to_idx)
    M = A@J

    ref_bus_row = sp.sparse.coo_matrix(([1],([0],[_ref_bus_idx])), shape=(1,_len_bus))
    ref_bus_col = sp.sparse.coo_matrix(([1],([_ref_bus_idx],[0])), shape=(_len_bus,1))
 
    J0 = sp.sparse.bmat([[M,ref_bus_col],[ref_bus_row,0]], format='coo')

    if active_index_set_branch is None or len(active_index_set_branch) == _len_branch:
        ## the resulting matrix after inversion will be fairly dense,
        ## the scipy documenation recommends using dense for the inversion
        ## as well
        try:
            SENSI = np.linalg.inv(J0.A)
        except np.linalg.LinAlgError:
            print("Matrix not invertible. Calculating pseudo-inverse instead.")
            SENSI = np.linalg.pinv(J0.A,rcond=1e-7)
        SENSI = SENSI[:-1,:-1]
        PTDF = np.matmul(-J.A,SENSI)
    elif len(active_index_set_branch) < _len_branch:
        B = np.array([], dtype=np.int64).reshape(_len_bus + 1,0)
        _active_mapping_branch = {i: branch_n for i, branch_n in enumerate(index_set_branch) if branch_n in active_index_set_branch}

        ## TODO: Maybe just keep the sparse PTDFs as a dict of ndarrays?
        ## Right now the return type depends on the options
        ## passed in
        for idx, branch_name in _active_mapping_branch.items():
            b = np.zeros((_len_branch,1))
            b[idx] = 1
            _tmp = np.matmul(J.transpose(),b)
            _tmp = np.vstack([_tmp,0])
            B = np.concatenate((B,_tmp), axis=1)
        row_idx = list(_active_mapping_branch.keys())
        PTDF = sp.sparse.lil_matrix((_len_branch,_len_bus))
        _ptdf = sp.sparse.linalg.spsolve(J0.transpose().tocsr(), -B).T
        PTDF[row_idx] = _ptdf[:,:-1]

    return PTDF


def calculate_lccm_flow_sensitivies(branches,buses,index_set_branch,index_set_bus,reference_bus,base_point=BasePointType.SOLUTION,mapping_bus_to_idx=None):
    """
    Calculates the following:
        Ft:     real power flow sensitivity to voltage angle theta
        ft_c:   real power flow constant
        Fv:     reactive power flow sensitivity to voltage magnitude
        fv_c:   reactive power flow constant
    Parameters
    ----------
    branches: dict{}
        The dictionary of branches for the test case
    buses: dict{}
        The dictionary of buses for the test case
    index_set_branch: list
        The list of keys for branches for the test case
    index_set_bus: list
        The list of keys for buses for the test case
    reference_bus: key value
        The reference bus key value
    base_point: egret.model_library_defn.BasePointType
        The base-point type for calculating sensitivities
    mapping_bus_to_idx: dict
        A map from bus names to indices for matrix construction. If None,
        will be inferred from index_set_bus.
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    _ref_bus_idx = mapping_bus_to_idx[reference_bus]

    if base_point is BasePointType.SOLUTION:
        approximation_type = ApproximationType.PTDF_LOSSES
    else:
        approximation_type = ApproximationType.PTDF

    Ft = _calculate_J11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point,approximation_type)
    Fv = _calculate_J22(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point)
    ft_c = _calculate_pf_constant(branches,buses,index_set_branch,base_point)
    fv_c = _calculate_qf_constant(branches,buses,index_set_branch,base_point)

    return Ft, ft_c, Fv, fv_c


def calculate_lccm_loss_sensitivies(branches, buses, index_set_branch, index_set_bus, reference_bus,
                                    base_point=BasePointType.SOLUTION, mapping_bus_to_idx=None):
    """
    Calculates the following:
        Lt:     real power loss sensitivity to voltage angle theta
        lt_c:   real power loss constant
        Lv:     reactive power loss sensitivity to voltage magnitude
        lv_c:   reactive power loss constant
    Parameters
    ----------
    branches: dict{}
        The dictionary of branches for the test case
    buses: dict{}
        The dictionary of buses for the test case
    index_set_branch: list
        The list of keys for branches for the test case
    index_set_bus: list
        The list of keys for buses for the test case
    reference_bus: key value
        The reference bus key value
    base_point: egret.model_library_defn.BasePointType
        The base-point type for calculating sensitivities
    mapping_bus_to_idx: dict
        A map from bus names to indices for matrix construction. If None,
        will be inferred from index_set_bus.
    """
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    _ref_bus_idx = mapping_bus_to_idx[reference_bus]

    Lt = _calculate_L11(branches, buses, index_set_branch, index_set_bus, mapping_bus_to_idx, base_point)
    Lv = _calculate_L22(branches, buses, index_set_branch, index_set_bus, mapping_bus_to_idx, base_point)
    lt_c = _calculate_pfl_constant(branches, buses, index_set_branch, base_point)
    lv_c = _calculate_qfl_constant(branches, buses, index_set_branch, base_point)

    return Lt, lt_c, Lv, lv_c


def linsolve_theta_fdf(model, model_data, base_point=BasePointType.SOLUTION,
        mapping_bus_to_idx=None, index_set_bus=None, index_set_branch=None, solve_sparse_system=False):
    '''
    Finds the implied voltage angles from an FDF model solution

    Since sensitivity matrices are recalculated, this MUST be done BEFORE saving model solution to model_data!
    '''
    md = model_data

    if base_point is BasePointType.SOLUTION:
        approximation_type = ApproximationType.PTDF_LOSSES
    else:
        approximation_type = ApproximationType.PTDF

    if index_set_bus is None:
        index_set_bus = md.attributes(element_type='bus')['names']
    if base_point is BasePointType.SOLUTION:
        approximation_type = ApproximationType.PTDF_LOSSES
    else:
        approximation_type = ApproximationType.PTDF

    # Nodal net withdrawal to a Numpy array
    m_p_nw = np.fromiter((value(model.p_nw[b]) for b in index_set_bus), float, count=len(index_set_bus))

    if 'va_SENSI' in md.data['system'] and not solve_sparse_system:

        Z = md.data['system']['va_SENSI']
        c = md.data['system']['va_CONST']

        if Z is not None:
            theta = Z.dot(m_p_nw) + c
            return theta

    print('solving theta with power flow Jacobian.')

    md_sys = md.data['system']
    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}


    if 'nodal_jacobian_p' in list(md_sys.keys()):
        M = md_sys['nodal_jacobian_p']
        m = md_sys['offset_jacobian_p']

    else:
        # Rectangular sensitivity matrices
        J = md_sys['Ft']
        L = md_sys['Lt']
        Jc = md_sys['ft_c']
        Lc = md_sys['lt_c']

        if 'AdjacencyMat' in list(md.data['system'].keys()):
            A = md_sys['AdjacencyMat']
            AA = md_sys['AbsAdj']
        else:
            klist = list(md.data['system'].keys())
            print(klist)
            if index_set_branch is None:
                branch_attrs = md.attributes(element_type='branch')
                index_set_branch = branch_attrs['names']

            branches = dict(md.elements(element_type='branch'))

            A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx)
            AA = calculate_absolute_adjacency_matrix(A)
            logger.warning('WARNING: recalculating nodal matrix in THETA calc.')

        # Nodal power transfer matrix (square)
        M1 = A @ J
        M2 = AA @ L
        M = M1 + 0.5 * M2

        # Transfer matrix linearization constant (vector)
        m1 = A @ Jc
        m2 = AA @ Lc
        m = m1 + 0.5 * m2

    # aggregate constants to solve M*theta = b
    b = -m - m_p_nw

    # Adjust reference bus coefficients to make M full rank and fix reference bus angle
    reference_bus = md.data['system']['reference_bus']
    ref = mapping_bus_to_idx[reference_bus]
    #M = M.toarray()
    M[ref,:] = 0
    M[ref,ref] = 1
    b[ref] = 0

    # Solve linear system
    theta = sp.sparse.linalg.spsolve(M, b)
    #theta = dsolve.spsolve(M,b)

    return theta

def linsolve_vmag_fdf(model, model_data, base_point=BasePointType.SOLUTION,
        mapping_bus_to_idx=None, index_set_bus=None, index_set_branch=None, solve_sparse_system=False):
    '''
    Finds the implied voltage angles from an FDF model solution

    Since sensitivity matrices are recalculated, this MUST be done BEFORE saving model solution to model_data!
    '''
    md = model_data

    if base_point is BasePointType.SOLUTION:
        approximation_type = ApproximationType.PTDF_LOSSES
    else:
        approximation_type = ApproximationType.PTDF

    if index_set_bus is None:
        index_set_bus = md.attributes(element_type='bus')['names']
    # Nodal net withdrawal to a Numpy array
    m_q_nw = np.fromiter((value(model.q_nw[b]) for b in index_set_bus), float, count=len(index_set_bus))

    if 'vdf' in md.data['system'] and not solve_sparse_system:

        Z = md.data['system']['vdf']
        c = md.data['system']['vdf_c']

        vmag = Z.dot(m_q_nw) + c

        return vmag

    print('solving vmag with power flow Jacobian.')

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}
    md_sys = md.data['system']

    if 'nodal_jacobian_q' in list(md_sys.keys()):
        M = md_sys['nodal_jacobian_q']
        m = md_sys['offset_jacobian_q']

    else:
        # Rectangular sensitivity matrices
        J = md_sys['Fv']
        L = md_sys['Lv']
        Jc = md_sys['fv_c']
        Lc = md_sys['lv_c']

        if 'AdjacencyMat' in list(md.data['system'].keys()):
            A = md_sys['AdjacencyMat']
            AA = md_sys['AbsAdj']
        else:
            if index_set_branch is None:
                branch_attrs = md.attributes(element_type='branch')
                index_set_branch = branch_attrs['names']

            branches = dict(md.elements(element_type='branch'))

            A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx)
            AA = calculate_absolute_adjacency_matrix(A)
            logger.warning('WARNING: recalculating adjacency matrix in VMAG calc.')

        # Nodal power transfer matrix (square)
        M1 = A @ J
        M2 = AA @ L
        M = M1 + 0.5 * M2

        # Transfer matrix linearization constant (vector)
        m1 = A @ Jc
        m2 = AA @ Lc
        m = m1 + 0.5 * m2

    # aggregate constants to solve M*theta = b
    b = -m - m_q_nw

    # Solve linear system
    vmag = sp.sparse.linalg.spsolve(M, b)

    return vmag

def remove_reference_bus_row(mat, mapping_bus_to_idx, reference_bus, _len_bus):

    M = mat.copy()
    _ref_bus_idx = mapping_bus_to_idx[reference_bus]
    ref_bus_row = np.zeros([1,_len_bus])
    ref_bus_row[:, _ref_bus_idx] = 1
    M[_ref_bus_idx, :] = ref_bus_row

    return M

def implicit_factor_solve(sens_mat, rhs_mat, index_set, active_index_set=None):
    # solves A^T x = B^T for x, where:
    # sens_mat = A
    # rhs_mat = B
    # index_set defines the names of the rows of B
    # active_index set is the rows of x to be calculated

    # initialize B_J and B_L empty matrices.
    _len_row, _len_col = rhs_mat.transpose().shape
    rhs_act = np.array([], dtype=np.int64).reshape(_len_row, 0)

    # mapping of array indices to branch names
    if active_index_set is None:
        active_index_set = index_set
    _active_mapping = {idx: name for idx, name in enumerate(index_set) if name in active_index_set}

    # fill RHS desired rows from active set
    for idx, name in _active_mapping.items():
        # the 'idx' column of the identity matrix
        b = np.zeros((_len_col, 1))
        b[idx] = 1

        # add the 'idx' column of RHS into active RHS
        _tmp_rhs = np.matmul(rhs_mat.A.transpose(), b)
        rhs_act = np.concatenate((rhs_act, _tmp_rhs), axis=1)

    # solve system ( A^T x^T = B^T ) for selected rows of x
    _sens = sp.sparse.linalg.spsolve(sens_mat.transpose().tocsr(), rhs_act).T
    _sens = truncate(_sens)

    row_idx = list(_active_mapping.keys())
    SENS = sp.sparse.lil_matrix((_len_col, _len_row))
    SENS[row_idx] = _sens[:,:]
    SENS = SENS.A

    return SENS

def truncate(_sens, abs_tol=0, rel_tol=1e-6):

    max_val = abs(_sens).max()
    eps = max(abs_tol, rel_tol * max_val)
    _sens[abs(_sens) < eps] = 0

    return _sens


def save_sens_mat(sens_dict, filename):

    location = os.path.join(os.getcwd(), 'sensitivity_mat')
    if not os.path.exists(location):
        os.mkdir(location)
    file = os.path.join(location, filename)
    np.save(file, sens_dict)

def load_sens_mat(filename):

    location = os.path.join(os.getcwd(), 'sensitivity_mat')
    file = os.path.join(location, filename)
    sens_numpy = np.load(file + '.npy', allow_pickle=True)
    sens_dict = sens_numpy[()]
    return sens_dict


def implicit_calc_p_sens(branches,buses,index_set_branch,index_set_bus,reference_bus,
                       base_point=BasePointType.SOLUTION, active_index_set_branch=None,
                       mapping_bus_to_idx=None, filename=None):

    if filename is not None:
        try:
            sens_dict = load_sens_mat('p_sens_' + filename)
            print('Loading pre-computed p-sensitivity dictionaries.')
            return sens_dict
        except FileNotFoundError:
            pass

    print('Could not find p-sensitivity dictionaries. Calculating...', end =" ")
    start = time.clock()

    # use active branch/bus mapping for large test cases
    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)
    if _len_bus > 1000:
        if base_point==BasePointType.SOLUTION:
            _len_cycle = _len_branch - _len_bus + 1
            active_index_set_branch = reduce_branches(branches, _len_cycle)
        elif base_point==BasePointType.FLATSTART:
            active_index_set_branch = active_index_set_branch
        else:
            pass
    else:
        active_index_set_branch = index_set_branch

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    F = _calculate_J11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point,approximation_type=ApproximationType.PTDF_LOSSES)
    F0 = _calculate_pf_constant(branches,buses,index_set_branch,base_point)
    L = _calculate_L11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point)
    L0 = _calculate_pfl_constant(branches,buses,index_set_branch,base_point)
    A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx)
    AA = calculate_absolute_adjacency_matrix(A)

    _ref_bus_idx = mapping_bus_to_idx[reference_bus]
    e = np.zeros((_len_bus,1))
    e[_ref_bus_idx] = 1

    M = A @ F + 0.5 * AA @ L
    M0 = A @ F0 + 0.5 * AA @ L0
    M_ref = remove_reference_bus_row(M, mapping_bus_to_idx, reference_bus, _len_bus)

    #----- calculate PTDFs by solving M^T * PTDF^T = -F^T  -----#
    PTDF = implicit_factor_solve(M_ref, -F, index_set_branch, active_index_set=active_index_set_branch)
    PT_constant = PTDF @ M0 + F0

    PLDF = implicit_factor_solve(M_ref, -L, index_set_branch, active_index_set=active_index_set_branch)
    PL_constant = PLDF @ M0 + L0

    # set constants of inactive branches to zero
    _inactive_mapping_branch = {name: idx for idx, name in enumerate(index_set_branch) if name not in active_index_set_branch}
    for name, idx in _inactive_mapping_branch.items():
        PT_constant[idx] = 0 # (branches[name]['pf'] - branches[name]['pt']) / 2     # assumed to be zero to help identify unmodeled flows
        PL_constant[idx] = 0 # branches[name]['pf'] + branches[name]['pt']           # added to residual loss function

    #----- calculate LFs by solving M^T * LF = e  -----#
    M_ref = remove_reference_bus_row(M.transpose(), mapping_bus_to_idx, reference_bus, _len_bus)
    U = sp.sparse.linalg.spsolve(M_ref.tocsr(), e)
    LF = U - np.ones(_len_bus)
    LF_const = LF @ M0 + sum(L0)

    #----- calculate branch loss distribution factors -----#
    branch_ploss = [branch['pf'] + branch['pt'] if branch['pt'] is not None else 0 for bn,branch in branches.items() ]
    total_ploss = sum(branch_ploss)
    if total_ploss > 0:
        ploss_dist = [ ploss / total_ploss for ploss in branch_ploss ]
    else:
        ploss_dist = [0 for ploss in branch_ploss]


    sens_dict = {}
    sens_dict['ptdf'] = PTDF.astype(np.float32)
    sens_dict['ptdf_c'] = PT_constant.astype(np.float32)
    sens_dict['pldf'] = PLDF.astype(np.float32)
    sens_dict['pldf_c'] = PL_constant.astype(np.float32)
    sens_dict['ploss_sens'] = LF
    sens_dict['ploss_const'] = LF_const
    sens_dict['ploss_resid_sens'] = LF - sum(PLDF)
    sens_dict['ploss_resid_const'] = LF_const - sum(PL_constant)
    sens_dict['ploss_distribution'] = ploss_dist
    sens_dict['nodal_jacobian_p'] = M.astype(np.float32)
    sens_dict['offset_jacobian_p'] = M0.astype(np.float32)

    elapsed = time.clock() - start
    print('it took {} seconds.'.format(elapsed))
    if filename is not None:
        save_sens_mat(sens_dict, 'p_sens_' + filename)

    return sens_dict

def implicit_calc_q_sens(branches,buses,index_set_branch,index_set_bus,reference_bus,
                       base_point=BasePointType.SOLUTION, active_index_set_branch=None,
                       active_index_set_bus=None, mapping_bus_to_idx=None, filename=None):

    if filename is not None:
        try:
            sens_dict = load_sens_mat('q_sens_' + filename)
            print('Loading pre-computed q-sensitivity dictionaries.')
            return sens_dict
        except FileNotFoundError:
            pass

    print('Could not find q-sensitivity dictionaries. Calculating...', end =" ")

    # use active branch/bus mapping for large test cases
    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)
    if _len_bus > 1000 and base_point==BasePointType.SOLUTION:
        _len_cycle = _len_branch - _len_bus + 1
        active_index_set_branch = reduce_branches(branches, _len_cycle)
        active_index_set_bus = reduce_buses(buses, _len_bus / 4)
    else:
        active_index_set_branch = index_set_branch
        active_index_set_bus = index_set_bus

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    G = _calculate_J22(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point)
    G0 = _calculate_qf_constant(branches,buses,index_set_branch,base_point)
    K = _calculate_L22(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point)
    K0 = _calculate_qfl_constant(branches,buses,index_set_branch,base_point)
    A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx)
    AA = calculate_absolute_adjacency_matrix(A)
    I = sp.sparse.coo_matrix(np.identity(_len_bus))
    e = np.zeros((_len_bus,1))
    #print("H:")
    #print(G.A)
    #print("H0:")
    #print(G0)

    M = A @ G + 0.5 * AA @ K
    M0 = A @ G0 + 0.5 * AA @ K0
    #print("M:")
    #print(M.A)
    #print("M0:")
    #print(M0)
    # M_ref = remove_reference_bus_row(M, mapping_bus_to_idx, reference_bus, _len_bus)

    #----- calculate PTDFs by solving M^T * PTDF^T = -F^T  -----#
    QTDF = implicit_factor_solve(M, -G, index_set_branch, active_index_set=active_index_set_branch)
    QT_constant = QTDF @ M0 + G0

    QLDF = implicit_factor_solve(M, -K, index_set_branch, active_index_set=active_index_set_branch)
    QL_constant = QLDF @ M0 + K0

    VDF = implicit_factor_solve(M, -I, index_set_bus, active_index_set=active_index_set_bus)
    V_constant = VDF @ M0

    # bk -- calculate more directly
    QT_constant = G @ V_constant + G0
    QL_constant = K @ V_constant + K0

    # set constants of inactive branches and buses to basepoint values
    _inactive_mapping_branch = {name: idx for idx, name in enumerate(index_set_branch) if name not in active_index_set_branch}
    _inactive_mapping_bus = {name: idx for idx, name in enumerate(index_set_bus) if name not in active_index_set_bus}
    for name, idx in _inactive_mapping_branch.items():
        QT_constant[idx] = 0 # (branches[name]['qf'] - branches[name]['qt']) / 2     # assumed to be zero to help identify unmodeled flows
        QL_constant[idx] = 0 # branches[name]['qf'] + branches[name]['qt']           # added to residual loss function
    for name, idx in _inactive_mapping_bus.items():
        V_constant[idx] = buses[name]['vm']

    #----- calculate LFs by solving M^T * LF = e  -----#
    U = sp.sparse.linalg.spsolve(M.tocsr(), e)
    QLF = np.ones(_len_bus) - U
    QLF_const = QLF @ M0 + sum(K0)

    #----- calculate branch loss distribution factors -----#
    #branch_qloss = [branch['qf'] + branch['qt'] if branch['pt'] is not None else 0 for bn,branch in branches.items()]
    # BK -- changing for now
    branch_qloss = [0 for bn,branch in branches.items()]
    total_qloss = sum(branch_qloss)
    if total_qloss > 0:
        qloss_dist = [ qloss / total_qloss for qloss in branch_qloss ]
    else:
        qloss_dist = [0 for qloss in branch_qloss]


    sens_dict = {}
    sens_dict['qtdf'] = QTDF
    sens_dict['qtdf_c'] = QT_constant
    sens_dict['qldf'] = QLDF
    sens_dict['qldf_c'] = QL_constant
    sens_dict['vdf'] = VDF
    sens_dict['vdf_c'] = V_constant
    sens_dict['qloss_sens'] = QLF
    sens_dict['qloss_const'] = QLF_const
    sens_dict['qloss_resid_sens'] = QLF - sum(QLDF)
    sens_dict['qloss_resid_const'] = QLF_const - sum(QL_constant)
    sens_dict['qloss_distribution'] = qloss_dist
    sens_dict['nodal_jacobian_q'] = M
    sens_dict['offset_jacobian_q'] = M0

    #print('it took {} seconds.'.format(elapsed))
    #if filename is not None:
    #    save_sens_mat(sens_dict, 'q_sens_' + filename)

    return sens_dict

def implicit_calc_ploss_sens(branches,buses,index_set_branch,index_set_bus,reference_bus,
                       base_point=BasePointType.SOLUTION, mapping_bus_to_idx=None):

    # use active branch/bus mapping for large test cases
    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    F = _calculate_J11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point,approximation_type=ApproximationType.PTDF_LOSSES)
    F0 = _calculate_pf_constant(branches,buses,index_set_branch,base_point)
    L = _calculate_L11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point)
    L0 = _calculate_pfl_constant(branches,buses,index_set_branch,base_point)
    A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx)
    AA = calculate_absolute_adjacency_matrix(A)

    M = A @ F + 0.5 * AA @ L
    M0 = A @ F0 + 0.5 * AA @ L0
    M_ref = remove_reference_bus_row(M.transpose(), mapping_bus_to_idx, reference_bus, _len_bus)
    _ref_bus_idx = mapping_bus_to_idx[reference_bus]
    e = np.zeros((_len_bus,1))
    e[_ref_bus_idx] = 1

    #----- calculate LFs by solving M^T * LF = e  -----#
    U = sp.sparse.linalg.spsolve(M_ref.tocsr(), e)
    LF = np.ones(_len_bus) - U
    LF_offset = LF @ M0 + sum(L0)

    return LF, LF_offset

def calculate_ptdf_pldf(branches,buses,index_set_branch,index_set_bus,reference_bus,
                        base_point=BasePointType.SOLUTION, active_index_set_branch=None,
                        mapping_bus_to_idx=None):
    """
    Calculates the following:
        PTDF:   real power transfer distribution factor
        PT_C:   real power transfer constant
        PLDF:   real power losses distribution factor
        PL_C:   real power losses constant
    Parameters
    ----------
    branches: dict{}
        The dictionary of branches for the test case
    buses: dict{}
        The dictionary of buses for the test case
    index_set_branch: list
        The list of keys for branches for the test case
    index_set_bus: list
        The list of keys for buses for the test case
    reference_bus: key value
        The reference bus key value
    base_point: egret.model_library_defn.BasePointType
        The base-point type for calculating the PTDF and LDF matrix
    active_index_set_branch: list
        The list of keys for branches needed to compute the active PTDF matrix
    mapping_bus_to_idx: dict
        A map from bus names to indices for matrix construction. If None,
        will be inferred from index_set_bus.
    """

    _len_bus = len(index_set_bus)

    if mapping_bus_to_idx is None:
        mapping_bus_to_idx = {bus_n: i for i, bus_n in enumerate(index_set_bus)}

    _len_branch = len(index_set_branch)

    _ref_bus_idx = mapping_bus_to_idx[reference_bus]

    J = _calculate_J11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point,approximation_type=ApproximationType.PTDF_LOSSES)
    L = _calculate_L11(branches,buses,index_set_branch,index_set_bus,mapping_bus_to_idx,base_point)
    Jc = _calculate_pf_constant(branches,buses,index_set_branch,base_point)
    Lc = _calculate_pfl_constant(branches,buses,index_set_branch,base_point)

    A = calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx)
    AA = calculate_absolute_adjacency_matrix(A)
    M1 = A@J
    M2 = AA@L
    M = M1 + 0.5 * M2

    #M = calculate_nodal_matrix_p(branches,buses,index_set_branch,index_set_bus,reference_bus,base_point,mapping_bus_to_idx)

    # Note: "a.A" returns dense ndarray object, same usage as "a.toarray()". Dense array is recommended for matrix inversion.
    # Careful w/ matrix "A" and intrinsic function "a.A"

    ref_bus_row = sp.sparse.coo_matrix(([1],([0],[_ref_bus_idx])), shape=(1,_len_bus))
    ref_bus_col = sp.sparse.coo_matrix(([1],([_ref_bus_idx],[0])), shape=(_len_bus,1))

    J0 = sp.sparse.bmat([[M,ref_bus_col],[ref_bus_row,0]], format='coo')

    # use sparse branch/bus mapping for large test cases
    if _len_bus > 1000:   # change to 1000 after debugging....
        _len_cycle = _len_branch - _len_bus + 1
        active_index_set_branch = reduce_branches(branches, _len_cycle)

    if active_index_set_branch is None or len(active_index_set_branch) == _len_branch:
        ## the resulting matrix after inversion will be fairly dense,
        ## the scipy documenation recommends using dense for the inversion
        ## as well
        try:
            SENSI = np.linalg.inv(J0.A)
        except np.linalg.LinAlgError:
            print("Matrix not invertible. Calculating pseudo-inverse instead.")
            SENSI = np.linalg.pinv(J0.A,rcond=1e-7)
            pass
        SENSI = SENSI[:-1,:-1]

        VA_SENSI = -SENSI
        PTDF = np.matmul(-J.A, SENSI)
        PLDF = np.matmul(-L.A, SENSI)

    elif len(active_index_set_branch) < _len_branch:
        # calculate PTDFs and PLDFs by solving J0^T * PTDF^T = -B_J^T and J0^T * PLDF^T = -B_L^T

        # initialize B_J and B_L empty matrices.
        B_J = np.array([], dtype=np.int64).reshape(_len_bus + 1, 0)
        B_L = np.array([], dtype=np.int64).reshape(_len_bus + 1, 0)

        # mapping of array indices to branch names
        _active_mapping_branch = {i: branch_n for i, branch_n in enumerate(index_set_branch) if branch_n in active_index_set_branch}

        # fill B_J and B_L with desired rows (i.e. partial mapping) of PTDF and PLDF
        for idx, branch_name in _active_mapping_branch.items():
            # the 'idx' column of the identity matrix
            b = np.zeros((_len_branch, 1))
            b[idx] = 1

            # TODO: add the 'idx' (row?/column?) of real power flow Jacobian into B_J
            _tmp_J = np.matmul(J.A.transpose(), b)
            _tmp_J = np.vstack([_tmp_J, 0])
            B_J = np.concatenate((B_J, _tmp_J), axis=1)

            # TODO: add the 'idx' (row?/column?) of real power loss Jacobian into B_L
            _tmp_L = np.matmul(L.A.transpose(), b)
            _tmp_L = np.vstack([_tmp_L, 0])
            B_L = np.concatenate((B_L, _tmp_L), axis=1)

        # solve system ( J0^T PTDF^T = -B_J^T ) for selected rows of PTDF
        _ptdf = sp.sparse.linalg.spsolve(J0.transpose().tocsr(), -B_J).T

        row_idx = list(_active_mapping_branch.keys())
        PTDF = sp.sparse.lil_matrix((_len_branch, _len_bus))
        PTDF[row_idx] = _ptdf[:, :-1]
        PTDF = PTDF.A

        # solve system ( J0^T PLDF^T = -B_L^T ) for selected rows of PLDF
        _pldf = sp.sparse.linalg.spsolve(J0.transpose().tocsr(), -B_L).T

        PLDF = sp.sparse.lil_matrix((_len_branch, _len_bus))
        PLDF[row_idx] = _pldf[:, :-1]
        PLDF = PLDF.A

        VA_SENSI = None

    M1 = A@Jc
    M2 = AA@Lc
    M = M1 + 0.5 * M2
    PT_constant = PTDF@M + Jc
    PL_constant = PLDF@M + Lc
    if VA_SENSI is not None:
        VA_CONST = VA_SENSI@M
    else:
        VA_CONST = None

    return PTDF, PT_constant, PLDF, PL_constant, VA_SENSI, VA_CONST

def reduce_branches(branches, N, min_N=3):
    from heapq import nlargest,nsmallest

    N = int(max(N, min_N))

    s_max = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    sf = {k: math.sqrt(branches[k]['pf']**2 + branches[k]['qf']**2) for k in branches.keys()}
    st = {k: math.sqrt(branches[k]['pt']**2 + branches[k]['qt']**2) for k in branches.keys()}
    rel_room = {k: 1 - max(sf[k], st[k]) / lim for k,lim in s_max.items() if lim is not None}
    abs_room = {k: lim - max(sf[k], st[k]) for k,lim in s_max.items() if lim is not None}
    rel_val = max(nsmallest(N, rel_room.values()))
    abs_val = max(nsmallest(N, abs_room.values()))
    rel_list = [k for k,v in rel_room.items() if v <= rel_val]
    abs_list = [k for k,v in abs_room.items() if v <= abs_val]
    reduced_list = list(set(rel_list + abs_list))

    return reduced_list


def reduce_buses(buses, N, min_N=3):
    from heapq import nsmallest

    N = int(max(N, min_N))

    LB = {k: abs(bus['vm'] - bus['v_min']) for k,bus in buses.items()}
    UB = {k: abs(bus['v_max'] - bus['vm']) for k,bus in buses.items()}
    room = {k: min(LB[k],UB[k]) for k in buses.keys()}
    threshold = max(nsmallest(N, room.values()))
    reduced_list = [k for k,v in room.items() if v <= threshold]

    return reduced_list


def calculate_adjacency_matrix_transpose(branches,index_set_branch,index_set_bus, mapping_bus_to_idx):
    """
    Calculates the adjacency matrix where (-1) represents flow from the bus and (1) represents flow to the bus
    for a given branch
    """
    _len_bus = len(index_set_bus)

    _len_branch = len(index_set_branch)

    row = []
    col = []
    data = []

    for idx_col, branch_name in enumerate(index_set_branch):
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        row.append(mapping_bus_to_idx[from_bus])
        col.append(idx_col)
        data.append(1)

        to_bus = branch['to_bus']
        row.append(mapping_bus_to_idx[to_bus])
        col.append(idx_col)
        data.append(-1)

    adjacency_matrix = sp.sparse.coo_matrix((data,(row,col)), shape=(_len_bus, _len_branch))
    return adjacency_matrix.tocsr()


def calculate_absolute_adjacency_matrix(adjacency_matrix):
    """
    Calculates the absolute value of the adjacency matrix
    """
    return sp.absolute(adjacency_matrix)
