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
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.decl as decl
from egret.model_library.defn import ApproximationType
from pyomo.core.util import quicksum
from pyomo.core.expr.numeric_expr import LinearExpression


def declare_eq_branch_power_ptdf_approx(model, index_set, branches, buses, bus_p_loads, gens_by_bus,
                                        bus_gs_fixed_shunts, ptdf_tol = 1e-10, include_constant_term = False,
                                        approximation_type = ApproximationType.PTDF):
    """
    Create the equality constraints for power (from PTDF approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_power_ptdf_approx_set", model, index_set)

    # formulate constraints expr == shift + ptdf*(bs + pl - pg + phi_from - phi_to) + constant
    m.eq_pf_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]
        expr = 0

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        # Add phase shifter adjustment
        if approximation_type == ApproximationType.PTDF:
            ptdf = branch['ptdf']
            if shift != 0.:
                b = -(1 / branch['reactance'])
                expr += b * (shift / tau)
        elif approximation_type == ApproximationType.PTDF_LOSSES or approximation_type == ApproximationType.FDF:
            ptdf = branch['ptdf']
            if shift != 0.:
                b = tx_calc.calculate_susceptance(branch)
                expr += b * (shift / tau)

        for bus_name, coef in ptdf.items():
            # Ignore buses that do not affect branch
            if ptdf_tol and abs(coef) < ptdf_tol:
                continue
            bus = buses[bus_name]
            phi_from = bus['phi_from']
            phi_to = bus['phi_to']

            if bus_gs_fixed_shunts[bus_name] != 0.0:
                expr += coef * bus_gs_fixed_shunts[bus_name]

            if bus_p_loads[bus_name] != 0.0:
                expr += coef * m.pl[bus_name]

            for gen_name in gens_by_bus[bus_name]:
                expr -= coef * m.pg[gen_name]

            for _, phi in phi_from.items():
                expr += coef * phi

            for _, phi in phi_to.items():
                expr -= coef * phi

        if include_constant_term:
            expr += branch['ptdf_c']

        m.eq_pf_branch[branch_name] = \
            m.pf[branch_name] == expr


def declare_eq_branch_loss_ptdf_approx(model, index_set, branches, buses, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts, ldf_tol = 1e-10, include_constant_term=False):
    """
    Create the equality constraints for losses (from PTDF approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_ptdf_approx_set", model, index_set)

    m.eq_pfl_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]
        expr = 0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])
            g = tx_calc.calculate_conductance(branch)
            expr += (g/tau) * shift**2

        ldf = branch['ldf']
        for bus_name, coef in ldf.items():
            if ldf_tol and abs(coef) < ldf_tol:
                continue
            bus = buses[bus_name]
            phi_loss_from = bus['phi_loss_from']
            phi_loss_to = bus['phi_loss_to']

            if bus_gs_fixed_shunts[bus_name] != 0.0:
                expr += coef * bus_gs_fixed_shunts[bus_name]

            if bus_p_loads[bus_name] != 0.0:
                expr += coef * m.pl[bus_name]

            for gen_name in gens_by_bus[bus_name]:
                expr -= coef * m.pg[gen_name]

            for _, phi_loss in phi_loss_from.items():
                expr += coef * phi_loss

            for _, phi_loss in phi_loss_to.items():
                expr -= coef * phi_loss

        if include_constant_term:
            expr += branch['ldf_c']

        m.eq_pfl_branch[branch_name] = \
            m.pfl[branch_name] == expr
