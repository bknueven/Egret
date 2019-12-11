#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains the declarations for the modeling components
typically used for transmission lines
"""
import math
import pyomo.environ as pe
from pyomo.environ import value
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.decl as decl
from egret.model_library.defn import FlowType, CoordinateType, ApproximationType, RelaxationType
from egret.data.model_data import zip_items
from pyomo.core.util import quicksum
from pyomo.core.expr.numeric_expr import LinearExpression

def declare_var_dva(model, index_set, **kwargs):
    """
    Create variable or the angle difference between interconnected bus pairs
    """
    decl.declare_var('dva', model=model, index_set=index_set, **kwargs)


def declare_var_pfl(model, index_set, **kwargs):
    """
    Create variable for the real part of the power loss in the transmission
    line
    """
    decl.declare_var('pfl', model=model, index_set=index_set, **kwargs)


def declare_var_pf(model, index_set, **kwargs):
    """
    Create variable for the real part of the power flow in the "from"
    end of the transmission line
    """
    decl.declare_var('pf', model=model, index_set=index_set, **kwargs)

def declare_expr_pf(model, index_set, **kwargs):
    """
    Create expression for the real part of the power flow in the "from"
    end of the transmission line
    """
    decl.declare_expr('pf', model=model, index_set=index_set, **kwargs)

def declare_var_qf(model, index_set, **kwargs):
    """
    Create variable for the imaginary part of the power flow in the "from"
    end of the transmission line
    """
    decl.declare_var('qf', model=model, index_set=index_set, **kwargs)


def declare_var_pt(model, index_set, **kwargs):
    """
    Create variable for the real part of the power flow in the "to"
    end of the transmission line
    """
    decl.declare_var('pt', model=model, index_set=index_set, **kwargs)


def declare_var_qt(model, index_set, **kwargs):
    """
    Create variable for the imaginary part of the power flow in the "to"
    end of the transmission line
    """
    decl.declare_var('qt', model=model, index_set=index_set, **kwargs)

def declare_var_qfl(model, index_set, **kwargs):
    """
    Create variable for the reactive part of the power loss in the transmission
    line
    """
    decl.declare_var('qfl', model=model, index_set=index_set, **kwargs)


def declare_var_ifr(model, index_set, **kwargs):
    """
    Create variable for the real part of the current flow in the "from"
    end of the transmission line
    """
    decl.declare_var('ifr', model=model, index_set=index_set, **kwargs)


def declare_var_ifj(model, index_set, **kwargs):
    """
    Create variable for the imaginary part of the current flow in the "from"
    end of the transmission line
    """
    decl.declare_var('ifj', model=model, index_set=index_set, **kwargs)


def declare_var_itr(model, index_set, **kwargs):
    """
    Create variable for the real part of the current flow in the "to"
    end of the transmission line
    """
    decl.declare_var('itr', model=model, index_set=index_set, **kwargs)


def declare_var_itj(model, index_set, **kwargs):
    """
    Create variable for the imaginary part of the current flow in the "to"
    end of the transmission line
    """
    decl.declare_var('itj', model=model, index_set=index_set, **kwargs)


def declare_eq_branch_dva(model, index_set, branches):
    """
    Create the equality constraints for the angle difference
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_dva_set", model, index_set)

    m.eq_dva_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        shift = 0.0
        if branch['branch_type'] == 'transformer':
            shift = math.radians(branch['transformer_phase_shift'])

        m.eq_dva_branch[branch_name] = \
            m.dva[branch_name] == \
            m.va[from_bus] - m.va[to_bus] + shift


def declare_expr_c(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create expression for the nonlinear, nonconvex term based on cosine
    of the phase angle difference (polar) or bilinear voltages (rectangular)
    """
    m = model
    expr_set = decl.declare_set('_expr_c', model, index_set)
    m.c = pe.Expression(expr_set)

    if coordinate_type == CoordinateType.RECTANGULAR:
        for from_bus, to_bus in expr_set:
            m.c[(from_bus,to_bus)] = m.vr[from_bus]*m.vr[to_bus] + m.vj[from_bus]*m.vj[to_bus]
    elif coordinate_type == CoordinateType.POLAR:
        for from_bus, to_bus in expr_set:
            m.c[(from_bus,to_bus)] = m.vm[from_bus]*m.vm[to_bus]*pe.cos(m.va[from_bus]-m.va[to_bus])


def declare_expr_s(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create expression for the nonlinear, nonconvex term based on cosine
    of the phase angle difference (polar) or bilinear voltages (rectangular)
    """
    m = model
    expr_set = decl.declare_set('_expr_s', model, index_set)
    m.s = pe.Expression(expr_set)

    if coordinate_type == CoordinateType.RECTANGULAR:
        for from_bus, to_bus in expr_set:
            m.s[(from_bus,to_bus)] = m.vj[from_bus]*m.vr[to_bus] - m.vr[from_bus]*m.vj[to_bus]
    elif coordinate_type == CoordinateType.POLAR:
        for from_bus, to_bus in expr_set:
            m.s[(from_bus,to_bus)] = m.vm[from_bus]*m.vm[to_bus]*pe.sin(m.va[from_bus]-m.va[to_bus])


def declare_eq_branch_current(model, index_set, branches, coordinate_type=CoordinateType.RECTANGULAR):
    """
    Create the equality constraints for the real and imaginary current
    in the branch
    """
    assert(coordinate_type != CoordinateType.POLAR
           and "Branch current in polar coordinates not implemented.")

    m = model
    con_set = decl.declare_set("_con_eq_branch_current_set", model, index_set)

    m.eq_ifr_branch = pe.Constraint(con_set)
    m.eq_ifj_branch = pe.Constraint(con_set)
    m.eq_itr_branch = pe.Constraint(con_set)
    m.eq_itj_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        g = tx_calc.calculate_conductance(branch)
        b = tx_calc.calculate_susceptance(branch)
        bc = branch['charging_susceptance']
        tau = 1.0
        shift = 0.0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        g11 = g / tau**2
        g12 = (g * math.cos(shift) - b * math.sin(shift)) / tau
        g21 = (g * math.cos(shift) + b * math.sin(shift)) / tau
        g22 = g

        b11 = (b + bc / 2) / tau**2
        b12 = (b * math.cos(shift) + g*math.sin(shift)) / tau
        b21 = (b * math.cos(shift) - g*math.sin(shift)) / tau
        b22 = b + bc / 2

        m.eq_ifr_branch[branch_name] = \
            m.ifr[branch_name] == \
            g11 * m.vr[from_bus] - g12 * m.vr[to_bus] - (b11 * m.vj[from_bus] - b12 * m.vj[to_bus])

        m.eq_ifj_branch[branch_name] = \
            m.ifj[branch_name] == \
            g11 * m.vj[from_bus] - g12 * m.vj[to_bus] + (b11 * m.vr[from_bus] - b12 * m.vr[to_bus])

        m.eq_itr_branch[branch_name] = \
            m.itr[branch_name] == \
            -(g21 * m.vr[from_bus] - g22 * m.vr[to_bus] - (b21 * m.vj[from_bus] - b22 * m.vj[to_bus]))

        m.eq_itj_branch[branch_name] = \
            m.itj[branch_name] == \
            -(g21 * m.vj[from_bus] - g22 * m.vj[to_bus] + (b21 * m.vr[from_bus] - b22 * m.vr[to_bus]))


def declare_eq_branch_power(model, index_set, branches, branch_attrs, coordinate_type=CoordinateType.POLAR):
    """
    Create the equality constraints for the real and reactive power
    in the branch
    """
    m = model

    bus_pairs = zip_items(branch_attrs['from_bus'],branch_attrs['to_bus'])
    unique_bus_pairs = list(set([val for idx,val in bus_pairs.items()]))
    declare_expr_c(model,unique_bus_pairs,coordinate_type)
    declare_expr_s(model,unique_bus_pairs,coordinate_type)

    con_set = decl.declare_set("_con_eq_branch_power_set", model, index_set)

    m.eq_pf_branch = pe.Constraint(con_set)
    m.eq_pt_branch = pe.Constraint(con_set)
    m.eq_qf_branch = pe.Constraint(con_set)
    m.eq_qt_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        if coordinate_type == CoordinateType.POLAR:
            vmsq_from_bus = m.vm[from_bus]**2
            vmsq_to_bus = m.vm[to_bus] ** 2
        elif coordinate_type == CoordinateType.RECTANGULAR:
            vmsq_from_bus = m.vr[from_bus]**2 + m.vj[from_bus]**2
            vmsq_to_bus = m.vr[to_bus] ** 2 + m.vj[to_bus] ** 2

        g = tx_calc.calculate_conductance(branch)
        b = tx_calc.calculate_susceptance(branch)
        bc = branch['charging_susceptance']
        tau = 1.0
        shift = 0.0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        g11 = g / tau ** 2
        g12 = g * math.cos(shift) / tau
        g21 = g * math.sin(shift) / tau
        g22 = g

        b11 = (b + bc / 2) / tau ** 2
        b12 = b * math.cos(shift) / tau
        b21 = b * math.sin(shift) / tau
        b22 = b + bc / 2

        m.eq_pf_branch[branch_name] = \
            m.pf[branch_name] == \
            g11 * vmsq_from_bus - \
            (g12 * m.c[(from_bus,to_bus)] +
             g21 * m.s[(from_bus,to_bus)] +
             b12 * m.s[(from_bus,to_bus)] -
             b21 * m.c[(from_bus,to_bus)])

        m.eq_pt_branch[branch_name] = \
            m.pt[branch_name] == \
            g22 * vmsq_to_bus - \
            (g12 * m.c[(from_bus,to_bus)] +
             g21 * m.s[(from_bus,to_bus)] -
             b12 * m.s[(from_bus,to_bus)] +
             b21 * m.c[(from_bus,to_bus)])

        m.eq_qf_branch[branch_name] = \
            m.qf[branch_name] == \
            -b11 * vmsq_from_bus + \
            (b12 * m.c[(from_bus,to_bus)] +
             b21 * m.s[(from_bus,to_bus)] -
             g12 * m.s[(from_bus,to_bus)] +
             g21 * m.c[(from_bus,to_bus)])

        m.eq_qt_branch[branch_name] = \
            m.qt[branch_name] == \
            -b22 * vmsq_to_bus + \
            (b12 * m.c[(from_bus,to_bus)] +
             b21 * m.s[(from_bus,to_bus)] +
             g12 * m.s[(from_bus,to_bus)] -
             g21 * m.c[(from_bus,to_bus)])


def declare_eq_branch_power_btheta_approx(model, index_set, branches, approximation_type=ApproximationType.BTHETA):
    """
    Create the equality constraints for power (from BTHETA approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_power_btheta_approx_set", model, index_set)

    m.eq_pf_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        if approximation_type == ApproximationType.BTHETA:
            x = branch['reactance']
            b = -1/(tau*x)
        elif approximation_type == ApproximationType.BTHETA_LOSSES:
            b = tx_calc.calculate_susceptance(branch)/tau

        m.eq_pf_branch[branch_name] = \
            m.pf[branch_name] == \
            b * (m.va[from_bus] - m.va[to_bus] + shift)


def declare_eq_branch_loss_btheta_approx(model, index_set, branches, relaxation_type = RelaxationType.NONE):
    """
    Create the equality constraints for losses (from BTHETA approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_btheta_approx_set", model, index_set)

    m.eq_pfl_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        tau = 1.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
        g = tx_calc.calculate_conductance(branch)/tau

        if relaxation_type == RelaxationType.NONE:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] == \
                g * (m.dva[branch_name])**2
        elif relaxation_type == RelaxationType.SOC:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] >= \
                g * (m.dva[branch_name])**2

def get_power_flow_expr_ptdf_approx(model, branch_name, ptdf, ptdf_c, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow expression from PTDF matrix
    """

    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    #const = PTDF.get_branch_phase_shift(branch_name) + PTDF.get_branch_phi_adj(branch_name)
    #max_coef = PTDF.get_branch_ptdf_abs_max(branch_name)
    max_coef = 1
    ptdf_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_p_nw = model.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    if isinstance(m_p_nw, pe.Var):
        coef_list = list()
        var_list = list()
        #for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch_name):
        for bus_name, coef in ptdf.items():
            if abs(coef) >= ptdf_tol:
                coef_list.append(coef)
                var_list.append(m_p_nw[bus_name])

        lin_expr_list = [ptdf_c] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_p_nw[bus_name] for bus_name, coef in ptdf.items() if abs(coef) >= ptdf_tol), start=ptdf_c, linear=True)

    return expr

def declare_eq_branch_power_ptdf_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for power (from PTDF 
    approximation) in the branch
    """

    m = model

    con_set = decl.declare_set("_con_eq_branch_power_ptdf_approx_set", model, index_set)

    pf_is_var = isinstance(m.pf, pe.Var)

    if pf_is_var:
        m.eq_pf_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pf, pe.Expression):
            raise Exception("Unrecognized type for m.pf", m.pf.pprint())

    for branch_name in con_set:
        ptdf = sensitivity[branch_name]
        ptdf_c = constant[branch_name]
        expr = \
            get_power_flow_expr_ptdf_approx(m, branch_name, ptdf, ptdf_c, rel_tol=rel_tol, abs_tol=abs_tol)

        if pf_is_var:
            m.eq_pf_branch[branch_name] = \
                m.pf[branch_name] == expr
        else:
            m.pf[branch_name] = expr

def get_branch_loss_expr_pldf_approx(model, branch_name, pldf, pldf_c, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow loss expression from PTDF matrix
    """
    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    #max_coef = PTDF.get_branch_ldf_abs_max(branch_name)
    max_coef = 1
    pldf_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_p_nw = model.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    if isinstance(m_p_nw, pe.Var):
        coef_list = list()
        var_list = list()
        for bus_name, coef in pldf.items():
            if abs(coef) >= pldf_tol:
                coef_list.append(coef)
                var_list.append(m_p_nw[bus_name])

        lin_expr_list = [pldf_c] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_p_nw[bus_name] for bus_name, coef in pldf.items() if abs(coef) >= pldf_tol), start=pldf_c, linear=True)

    return expr

def declare_eq_branch_loss_pldf_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for losses (from PTDF 
    approximation) in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_pldf_approx_set", model, index_set)
    pfl_is_var = isinstance(m.pfl, pe.Var)
    if pfl_is_var:
        m.eq_pfl_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pfl, pe.Expression):
            raise Exception("Unrecognized type for m.pfl", m.pfl.pprint())

    for branch_name in con_set:
        pldf = sensitivity[branch_name]
        pldf_c = constant[branch_name]
        expr = \
            get_branch_loss_expr_pldf_approx(m, branch_name, pldf, pldf_c, rel_tol=rel_tol, abs_tol=abs_tol)

        if pfl_is_var:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] == expr
        else:
            m.pfl[branch_name] = expr

def get_power_flow_expr_qtdf_approx(model, branch_name, qtdf, qtdf_c, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow expression from QTDF matrix (reactive power flows)
    """

    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    #const = QTDF.get_branch_phase_shift(branch_name) + QTDF.get_branch_phi_adj(branch_name)
    #max_coef = QTDF.get_branch_ptdf_abs_max(branch_name)
    max_coef = 1
    qtdf_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_q_nw = model.q_nw
    ## if model.q_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    if isinstance(m_q_nw, pe.Var):
        coef_list = list()
        var_list = list()
        #for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch_name):
        for bus_name, coef in qtdf.items():
            if abs(coef) >= qtdf_tol:
                coef_list.append(coef)
                var_list.append(m_q_nw[bus_name])

        lin_expr_list = [qtdf_c] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_q_nw[bus_name] for bus_name, coef in qtdf.items() if abs(coef) >= qtdf_tol), start=qtdf_c, linear=True)

    return expr

def declare_eq_branch_power_qtdf_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for power (from QTDF
    approximation) in the branch
    """

    m = model

    con_set = decl.declare_set("_con_eq_branch_power_qtdf_approx_set", model, index_set)

    qf_is_var = isinstance(m.qf, pe.Var)

    if qf_is_var:
        m.eq_qf_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.qf, pe.Expression):
            raise Exception("Unrecognized type for m.qf", m.qf.pprint())

    for branch_name in con_set:
        qtdf = sensitivity[branch_name]
        qtdf_c = constant[branch_name]
        expr = \
            get_power_flow_expr_qtdf_approx(m, branch_name, qtdf, qtdf_c, rel_tol=rel_tol, abs_tol=abs_tol)

        if qf_is_var:
            m.eq_qf_branch[branch_name] = \
                m.qf[branch_name] == expr
        else:
            m.qf[branch_name] = expr

def get_branch_loss_expr_qldf_approx(model, branch_name, qldf, qldf_c, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow loss expression from QLDF matrix
    """
    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    #max_coef = QTDF.get_branch_ldf_abs_max(branch_name)
    max_coef = 1
    qldf_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'qldf' dictionary here,
    ##       if we wanted to
    m_q_nw = model.q_nw
    ## if model.q_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    if isinstance(m_q_nw, pe.Var):
        coef_list = list()
        var_list = list()
        for bus_name, coef in qldf.items():
            if abs(coef) >= qldf_tol:
                coef_list.append(coef)
                var_list.append(m_q_nw[bus_name])

        lin_expr_list = [qldf_c] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_q_nw[bus_name] for bus_name, coef in qldf.items() if abs(coef) >= qldf_tol), start=qldf_c, linear=True)

    return expr

def declare_eq_branch_loss_qldf_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for losses (from QLDF
    approximation) in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_qldf_approx_set", model, index_set)
    qfl_is_var = isinstance(m.qfl, pe.Var)
    if qfl_is_var:
        m.eq_qfl_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.qfl, pe.Expression):
            raise Exception("Unrecognized type for m.qfl", m.qfl.pprint())

    for branch_name in con_set:
        qldf = sensitivity[branch_name]
        qldf_c = constant[branch_name]
        expr = \
            get_branch_loss_expr_qldf_approx(m, branch_name, qldf, qldf_c, rel_tol=rel_tol, abs_tol=abs_tol)

        if qfl_is_var:
            m.eq_qfl_branch[branch_name] = \
                m.qfl[branch_name] == expr
        else:
            m.qfl[branch_name] = expr

def get_expr_branch_pf_lccm_approx(model, branch_name, pf_sens, pf_const, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow expression from CCM sensitivity
    """

    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    max_coef = 1
    sens_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_va = model.va
    ## if model.va is Var, we can use LinearExpression
    ## LinearExpression may not have advantage for sparse LCCM constraints, but will be implemented analogously to FDF
    if isinstance(m_va, pe.Var):
        coef_list = list()
        var_list = list()
        for bus_name, coef in pf_sens.items():
            if abs(coef) >= sens_tol:
                coef_list.append(coef)
                var_list.append(m_va[bus_name])

        lin_expr_list = [pf_const] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_va[bus_name] for bus_name, coef in pf_sens.items() if abs(coef) >= sens_tol), start=pf_const, linear=True)

    return expr

def declare_eq_branch_pf_lccm_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for power (from LCCM approximation) in the branch
    """

    m = model

    con_set = decl.declare_set("_con_eq_branch_pf_lccm_approx_set", model, index_set)

    pf_is_var = isinstance(m.pf, pe.Var)

    if pf_is_var:
        m.eq_pf_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pf, pe.Expression):
            raise Exception("Unrecognized type for m.pf", m.pf.pprint())

    for branch_name in con_set:
        pf_sens = sensitivity[branch_name]
        pf_const = constant[branch_name]
        expr = \
            get_expr_branch_pf_lccm_approx(m, branch_name, pf_sens, pf_const, rel_tol=rel_tol, abs_tol=abs_tol)

        if pf_is_var:
            m.eq_pf_branch[branch_name] = \
                m.pf[branch_name] == expr
        else:
            m.pf[branch_name] = expr

def get_expr_branch_pfl_lccm_approx(model, branch_name, pfl_sens, pfl_const, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow loss expression from CCM sensitivity
    """
    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    max_coef = 1
    sens_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_va = model.va
    ## if model.va is Var, we can use LinearExpression
    ## LinearExpression may not have advantage for sparse LCCM constraints, but will be implemented analogously to FDF
    if isinstance(m_va, pe.Var):
        coef_list = list()
        var_list = list()
        for bus_name, coef in pfl_sens.items():
            if abs(coef) >= sens_tol:
                coef_list.append(coef)
                var_list.append(m_va[bus_name])

        lin_expr_list = [pfl_const] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_va[bus_name] for bus_name, coef in pfl_sens.items() if abs(coef) >= sens_tol), start=pfl_const, linear=True)

    return expr

def declare_eq_branch_pfl_lccm_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for losses (from LCCM approximation) in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_pfl_lccm_approx_set", model, index_set)
    pfl_is_var = isinstance(m.pfl, pe.Var)
    if pfl_is_var:
        m.eq_pfl_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pfl, pe.Expression):
            raise Exception("Unrecognized type for m.pfl", m.pfl.pprint())

    for branch_name in con_set:
        pfl_sens = sensitivity[branch_name]
        pfl_const = constant[branch_name]
        expr = \
            get_expr_branch_pfl_lccm_approx(m, branch_name, pfl_sens, pfl_const, rel_tol=rel_tol, abs_tol=abs_tol)

        if pfl_is_var:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] == expr
        else:
            m.pfl[branch_name] = expr

def get_expr_branch_qf_lccm_approx(model, branch_name, qf_sens, qf_const, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow expression from CCM sensitivity
    """

    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    max_coef = 1
    sens_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_vm = model.vm
    ## if model.vm is Var, we can use LinearExpression
    ## LinearExpression may not have advantage for sparse LCCM constraints, but will be implemented analogously to FDF
    if isinstance(m_vm, pe.Var):
        coef_list = list()
        var_list = list()
        #for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch_name):
        for bus_name, coef in qf_sens.items():
            if abs(coef) >= sens_tol:
                coef_list.append(coef)
                var_list.append(m_vm[bus_name])

        lin_expr_list = [qf_const] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_vm[bus_name] for bus_name, coef in qf_sens.items() if abs(coef) >= sens_tol), start=qf_const, linear=True)

    return expr

def declare_eq_branch_qf_lccm_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for power (from LCCM approximation) in the branch
    """

    m = model

    con_set = decl.declare_set("_con_eq_branch_qf_lccm_approx_set", model, index_set)

    qf_is_var = isinstance(m.qf, pe.Var)

    if qf_is_var:
        m.eq_qf_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.qf, pe.Expression):
            raise Exception("Unrecognized type for m.qf", m.qf.pprint())

    for branch_name in con_set:
        qf_sens = sensitivity[branch_name]
        qf_const = constant[branch_name]
        expr = \
            get_expr_branch_qf_lccm_approx(m, branch_name, qf_sens, qf_const, rel_tol=rel_tol, abs_tol=abs_tol)

        if qf_is_var:
            m.eq_qf_branch[branch_name] = \
                m.qf[branch_name] == expr
        else:
            m.qf[branch_name] = expr

def get_expr_branch_qfl_lccm_approx(model, branch_name, qfl_sens, qfl_const, rel_tol=None, abs_tol=None):
    """
    Create a pyomo power flow loss expression from CCM sensitivity
    """
    if rel_tol is None:
        rel_tol = 0.
    if abs_tol is None:
        abs_tol = 0.

    max_coef = 1
    sens_tol = max(abs_tol, rel_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'qldf' dictionary here,
    ##       if we wanted to
    m_vm = model.vm
    ## if model.vm is Var, we can use LinearExpression
    ## LinearExpression may not have advantage for sparse LCCM constraints, but will be implemented analogously to FDF
    if isinstance(m_vm, pe.Var):
        coef_list = list()
        var_list = list()
        for bus_name, coef in qfl_sens.items():
            if abs(coef) >= sens_tol:
                coef_list.append(coef)
                var_list.append(m_vm[bus_name])

        lin_expr_list = [qfl_const] + coef_list + var_list
        expr = LinearExpression(lin_expr_list)
    else:
        expr = quicksum( (coef*m_vm[bus_name] for bus_name, coef in qfl_sens.items() if abs(coef) >= sens_tol), start=qfl_const, linear=True)

    return expr

def declare_eq_branch_qfl_lccm_approx(model, index_set, sensitivity, constant, rel_tol=None, abs_tol=None):
    """
    Create the equality constraints or expressions for losses (from LCCM approximation) in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_qfl_lccm_approx_set", model, index_set)
    qfl_is_var = isinstance(m.qfl, pe.Var)
    if qfl_is_var:
        m.eq_qfl_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.qfl, pe.Expression):
            raise Exception("Unrecognized type for m.qfl", m.qfl.pprint())

    for branch_name in con_set:
        qfl_sens = sensitivity[branch_name]
        qfl_const = constant[branch_name]
        expr = \
            get_expr_branch_qfl_lccm_approx(m, branch_name, qfl_sens, qfl_const, rel_tol=rel_tol, abs_tol=abs_tol)

        if qfl_is_var:
            m.eq_qfl_branch[branch_name] = \
                m.qfl[branch_name] == expr
        else:
            m.qfl[branch_name] = expr


def declare_eq_branch_power_qtdf_approx_depreciated(model, index_set, branches, buses, bus_q_loads, gens_by_bus,
                                        bus_bs_fixed_shunts, qtdf_tol=1e-10):
    """
    Create the equality constraints for reactive power (from QTDF approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_power_qtdf_approx_set", model, index_set)

    m.eq_qf_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]
        expr = 0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])
            g = tx_calc.calculate_conductance(branch)
            expr += (g / tau) * shift

        qtdf = branch['qtdf']
        for bus_name, coef in qtdf.items():
            if qtdf_tol and abs(coef) < qtdf_tol:
                continue
            bus = buses[bus_name]
            phi_q_from = bus['phi_q_from']
            phi_q_to = bus['phi_q_to']

            if bus_bs_fixed_shunts[bus_name] != 0.0:
                expr -= coef * bus_bs_fixed_shunts[bus_name]*(2*buses[bus_name]["vm"]*m.vm[bus_name]-(buses[bus_name]["vm"])**2)

            if bus_q_loads[bus_name] != 0.0:
                expr += coef * m.ql[bus_name]

            for gen_name in gens_by_bus[bus_name]:
                expr -= coef * m.qg[gen_name]

            for _, phi_q in phi_q_from.items():
                expr += coef * phi_q

            for _, phi_q in phi_q_to.items():
                expr -= coef * phi_q

        expr += branch['qtdf_c']

        m.eq_qf_branch[branch_name] = \
            m.qf[branch_name] == expr


def declare_eq_branch_loss_qtdf_approx_depreciated(model, index_set, branches, buses, bus_q_loads, gens_by_bus, bus_bs_fixed_shunts, qldf_tol = 1e-10):
    """
    Create the equality constraints for losses (from QTDF approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_qtdf_approx_set", model, index_set)

    m.eq_qfl_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]
        expr = 0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])
            b = tx_calc.calculate_susceptance(branch)
            expr += (b/tau) * shift**2

        qldf = branch['qldf']
        for bus_name, coef in qldf.items():
            if qldf_tol and abs(coef) < qldf_tol:
                continue
            bus = buses[bus_name]
            phi_loss_q_from = bus['phi_loss_q_from']
            phi_loss_q_to = bus['phi_loss_q_to']

            if bus_bs_fixed_shunts[bus_name] != 0.0:
                expr -= coef * bus_bs_fixed_shunts[bus_name]*(2*buses[bus_name]["vm"]*m.vm[bus_name]-(buses[bus_name]["vm"])**2)

            if bus_q_loads[bus_name] != 0.0:
                expr += coef * m.ql[bus_name]

            for gen_name in gens_by_bus[bus_name]:
                expr -= coef * m.qg[gen_name]

            for _, phi_loss_q in phi_loss_q_from.items():
                expr += coef * phi_loss_q

            for _, phi_loss_q in phi_loss_q_to.items():
                expr -= coef * phi_loss_q

        expr += branch['qldf_c']

        m.eq_qfl_branch[branch_name] = \
            m.qfl[branch_name] == expr


def declare_ineq_s_branch_thermal_limit(model, index_set,
                                        branches, s_thermal_limits,
                                        flow_type=FlowType.POWER):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_s_branch_thermal_limit',
                               model=model, index_set=index_set)

    m.ineq_sf_branch_thermal_limit = pe.Constraint(con_set)
    m.ineq_st_branch_thermal_limit = pe.Constraint(con_set)

    if flow_type == FlowType.CURRENT:
        for branch_name in con_set:
            if s_thermal_limits[branch_name] is None:
                continue

            from_bus = branches[branch_name]['from_bus']
            to_bus = branches[branch_name]['to_bus']
            m.ineq_sf_branch_thermal_limit[branch_name] = \
                (m.vr[from_bus] ** 2 + m.vj[from_bus] ** 2) * (m.ifr[branch_name] ** 2 + m.ifj[branch_name] ** 2) \
                <= s_thermal_limits[branch_name] ** 2
            m.ineq_st_branch_thermal_limit[branch_name] = \
                (m.vr[to_bus] ** 2 + m.vj[to_bus] ** 2) * (m.itr[branch_name] ** 2 + m.itj[branch_name] ** 2) \
                <= s_thermal_limits[branch_name] ** 2
    elif flow_type == FlowType.POWER:
        for branch_name in con_set:
            if s_thermal_limits[branch_name] is None:
                continue

            m.ineq_sf_branch_thermal_limit[branch_name] = \
                m.pf[branch_name] ** 2 + m.qf[branch_name] ** 2 \
                <= s_thermal_limits[branch_name] ** 2
            m.ineq_st_branch_thermal_limit[branch_name] = \
                m.pt[branch_name] ** 2 + m.qt[branch_name] ** 2 \
                <= s_thermal_limits[branch_name] ** 2


def declare_ineq_p_branch_thermal_lbub(model, index_set,
                                        branches, p_thermal_limits,
                                        approximation_type=ApproximationType.BTHETA):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_p_branch_thermal_lbub',
                               model=model, index_set=index_set)

    m.ineq_pf_branch_thermal_lb = pe.Constraint(con_set)
    m.ineq_pf_branch_thermal_ub = pe.Constraint(con_set)

    if approximation_type == ApproximationType.BTHETA or \
            (approximation_type == ApproximationType.PTDF and \
            isinstance(m.pf, pe.Expression)):
        for branch_name in con_set:
            if p_thermal_limits[branch_name] is None:
                continue

            m.ineq_pf_branch_thermal_lb[branch_name] = \
                -p_thermal_limits[branch_name] <= m.pf[branch_name]

            m.ineq_pf_branch_thermal_ub[branch_name] = \
                m.pf[branch_name] <= p_thermal_limits[branch_name]


def declare_ineq_angle_diff_branch_lbub(model, index_set,
                                        branches,
                                        coordinate_type=CoordinateType.POLAR):
    """
    Create the inequality constraints for the angle difference
    bounds between interconnected buses.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_angle_diff_branch_lbub',
                               model=model, index_set=index_set)

    m.ineq_angle_diff_branch_lb = pe.Constraint(con_set)
    m.ineq_angle_diff_branch_ub = pe.Constraint(con_set)

    if coordinate_type == CoordinateType.POLAR:
        for branch_name in con_set:
            from_bus = branches[branch_name]['from_bus']
            to_bus = branches[branch_name]['to_bus']

            m.ineq_angle_diff_branch_lb[branch_name] = \
                branches[branch_name]['angle_diff_min'] <= m.va[from_bus] - m.va[to_bus]
            m.ineq_angle_diff_branch_ub[branch_name] = \
                m.va[from_bus] - m.va[to_bus] <= branches[branch_name]['angle_diff_max']
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for branch_name in con_set:
            from_bus = branches[branch_name]['from_bus']
            to_bus = branches[branch_name]['to_bus']

            m.ineq_angle_diff_branch_lb[branch_name] = \
                branches[branch_name]['angle_diff_min'] <= pe.atan(m.vj[from_bus]/m.vr[from_bus]) \
                - pe.atan(m.vj[to_bus]/m.vr[to_bus])
            m.ineq_angle_diff_branch_ub[branch_name] = \
                pe.atan(m.vj[from_bus] / m.vr[from_bus]) \
                - pe.atan(m.vj[to_bus] / m.vr[to_bus]) <= branches[branch_name]['angle_diff_max']


def declare_fdf_thermal_limit(model, index_set, thermal_limits, cuts=10):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables for the fdf model.
    """
    import cmath
    unit_radius = 1
    points = [(c.real, c.imag) for c in (cmath.rect(unit_radius, math.radians(a)) for a in range(0, 360, 360 // int(cuts)))]
    points_dict = {i: points[i] for i in range(len(points))}
    points_list = list(points_dict.keys())

    index_over = []
    for i in index_set:
        for p in points_list:
            index_over.append((i,p))

    m = model
    con_set = decl.declare_set('_con_ineq_branch_thermal_limit', model=model, index_set=index_over)

    m.ineq_branch_thermal_limit = pe.Constraint(con_set)

    for (branch_name, p) in con_set:
        if thermal_limits[branch_name] is None:
            continue

        x, y = points_dict[p]
        _pf = x * thermal_limits[branch_name]
        _qf = y * thermal_limits[branch_name]

        m.ineq_branch_thermal_limit[branch_name,p] = _pf*m.pf[branch_name] + _qf*m.qf[branch_name]\
                                                     <= thermal_limits[branch_name]**2

        # m.ineq_branch_thermal_limit[branch_name,p] = _pf * (m.pf[branch_name] + 0.5*m.pfl[branch_name]) + _qf * (m.qf[branch_name] + 0.5*m.qfl[branch_name]) \
        #                                              <= thermal_limits[branch_name]**2


def declare_eq_branch_midpoint_power(model, index_set, branches, coordinate_type=CoordinateType.POLAR):
    """
    Create the equality constraints for the real power flow, real power loss, reactive power flow, reactive power loss.
    """
    assert(coordinate_type != CoordinateType.RECTANGULAR
           and "Midpoint branch power in rectangular coordinates not implemented.")

    m = model
    con_set = decl.declare_set("_con_eq_branch_midpoint_power_set", model, index_set)

    m.eq_pf_branch = pe.Constraint(con_set)
    m.eq_pfl_branch = pe.Constraint(con_set)
    m.eq_qf_branch = pe.Constraint(con_set)
    m.eq_qfl_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        g = tx_calc.calculate_conductance(branch)
        b = tx_calc.calculate_susceptance(branch)
        bc = branch['charging_susceptance']
        tau = 1.0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']

        m.eq_pf_branch[branch_name] = \
            m.pf[branch_name] == \
            0.5 * g * ((m.vm[from_bus] / tau) ** 2 - m.vm[to_bus] ** 2) - (b / tau) * m.vm[from_bus] * \
            m.vm[to_bus] * pe.sin(m.dva[branch_name])

        m.eq_pfl_branch[branch_name] = \
            m.pfl[branch_name] == \
            g * ((m.vm[from_bus] / tau) ** 2 + m.vm[to_bus] ** 2) - 2 * (g / tau) * m.vm[from_bus] * \
            m.vm[to_bus] * pe.cos(m.dva[branch_name])

        m.eq_qf_branch[branch_name] = \
            m.qf[branch_name] == \
            -0.5 * (b + bc / 2) * ((m.vm[from_bus] / tau) ** 2 - m.vm[to_bus] ** 2) - (g / tau) * m.vm[from_bus] * \
            m.vm[to_bus] * pe.sin(m.dva[branch_name])

        m.eq_qfl_branch[branch_name] = \
            m.qfl[branch_name] == \
            -(b + bc / 2) * ((m.vm[from_bus] / tau) ** 2 + m.vm[to_bus] ** 2) + 2 * (b / tau) * m.vm[from_bus] * \
            m.vm[to_bus] * pe.cos(m.dva[branch_name])

    #print('~~~~~~~~~~~CCM INITIALIZATION~~~~~~~~~~~')
    #for branch_name in con_set:
    #    print('pf: ', pe.value(m.pf[branch_name]))
    #for branch_name in con_set:
    #    print('pfl: ', pe.value(m.pfl[branch_name]))
    #for branch_name in con_set:
    #    print('qf: ', pe.value(m.qf[branch_name]))
    #for branch_name in con_set:
    #    print('qfl: ', pe.value(m.qfl[branch_name]))
