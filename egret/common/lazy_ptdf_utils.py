#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## helpers for flow verification across dcopf and unit commitment models
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from egret.model_library.defn import ApproximationType
from egret.common.log import logger
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.tx_calc as tx_calc
import pyomo.environ as pe
import numpy as np

from enum import Enum

class LazyPTDFTerminationCondition(Enum):
    NORMAL = 1
    ITERATION_LIMIT = 2
    FLOW_VIOLATION = 3

def populate_default_ptdf_options(ptdf_options):
    if 'rel_ptdf_tol' not in ptdf_options:
        ptdf_options['rel_ptdf_tol'] = 1.e-6
    if 'abs_ptdf_tol' not in ptdf_options:
        ptdf_options['abs_ptdf_tol'] = 1.e-10
    if 'abs_flow_tol' not in ptdf_options:
        ptdf_options['abs_flow_tol'] = 1.e-3
    if 'rel_flow_tol' not in ptdf_options:
        ptdf_options['rel_flow_tol'] = 1.e-5
    if 'pu_vm_tol' not in ptdf_options:
        ptdf_options['pu_vm_tol'] = 1.e-5
    if 'iteration_limit' not in ptdf_options:
        ptdf_options['iteration_limit'] = 100000
    if 'lp_iteration_limit' not in ptdf_options:
        ptdf_options['lp_iteration_limit'] = 100
    if 'max_violations_per_iteration' not in ptdf_options:
        ptdf_options['max_violations_per_iteration'] = 5
    if 'vm_max_violations_per_iteration' not in ptdf_options:
        ptdf_options['vm_max_violations_per_iteration'] = 15
    if 'lazy' not in ptdf_options:
        ptdf_options['lazy'] = True
    if 'lazy_reactive' not in ptdf_options:
        ptdf_options['lazy_reactive'] = False
    if 'lazy_voltage' not in ptdf_options:
        ptdf_options['lazy_voltage'] = False
    if 'load_from' not in ptdf_options:
        ptdf_options['load_from'] = None
    if 'save_to' not in ptdf_options:
        ptdf_options['save_to'] = None

def check_and_scale_ptdf_options(ptdf_options, baseMVA):
    ## scale to base MVA
    ptdf_options['abs_ptdf_tol'] /= baseMVA
    ptdf_options['abs_flow_tol'] /= baseMVA

    rel_flow_tol = ptdf_options['rel_flow_tol']
    abs_flow_tol = ptdf_options['abs_flow_tol']

    rel_ptdf_tol = ptdf_options['rel_ptdf_tol']
    abs_ptdf_tol = ptdf_options['abs_ptdf_tol']

    max_violations_per_iteration = ptdf_options['max_violations_per_iteration']

    if max_violations_per_iteration < 1 or (not isinstance(max_violations_per_iteration, int)):
        raise Exception("max_violations_per_iteration must be an integer least 1, max_violations_per_iteration={}".format(max_violations_per_iteration))

    if abs_flow_tol < 1e-6:
        logger.warning("WARNING: abs_flow_tol={0}, which is below the numeric threshold of most solvers.".format(abs_flow_tol*baseMVA))
    if abs_flow_tol < rel_ptdf_tol*10:
        logger.warning("WARNING: abs_flow_tol={0}, rel_ptdf_tol={1}, which will likely result in violations. Consider raising abs_flow_tol or lowering rel_ptdf_tol.".format(abs_flow_tol*baseMVA, rel_ptdf_tol))
    if rel_ptdf_tol < 1e-6:
        logger.warning("WARNING: rel_ptdf_tol={0}, which is low enough it may cause numerical issues in the solver. Consider rasing rel_ptdf_tol.".format(rel_ptdf_tol))
    if abs_ptdf_tol < 1e-12:
        logger.warning("WARNING: abs_ptdf_tol={0}, which is low enough it may cause numerical issues in the solver. Consider rasing abs_ptdf_tol.".format(abs_ptdf_tol*baseMVA))

## to hold the indicies of the violations
## in the model or block
def add_monitored_branch_tracker(mb):
    mb._thermal_idx_monitored = list()

def add_monitored_vm_tracker(mb):
    mb._vm_idx_monitored = list()

## transmission violation checker
def check_violations(mb, md, branch_attrs, bus_attrs, max_viol_add, max_viol_add_vm=None, time=None):

    m = mb.model()

    enforced_branch_limits = branch_attrs['rating_long_term']
    index_set_bus = bus_attrs['names']
    index_set_branch = branch_attrs['names']

    _len_bus = len(index_set_bus)
    _len_branch = len(index_set_branch)

    ## Back-solve for theta then calculate real power flows with sparse sensitivity matrix
    THETA = tx_calc.linsolve_theta_fdf(mb, md)
    Ft = md.data['system']['Ft']
    ft_c = md.data['system']['ft_c']
    PFV = Ft.dot(THETA) + ft_c

    ## Back-solve for vmag then calculate reactive power flows with sparse sensitivity matrix
    if hasattr(mb, "qg"):
        if max_viol_add_vm is None:
            max_viol_add_vm = max_viol_add
        VMAG = tx_calc.linsolve_vmag_fdf(mb, md)
        Fv = md.data['system']['Fv']
        fv_c = md.data['system']['fv_c']
        QFV = Fv.dot(VMAG) + fv_c
        SV = np.sqrt(np.square(PFV) + np.square(QFV))
    else:
        VMAG = np.ones(_len_bus)
        QFV = np.zeros(_len_branch)
        SV = PFV

    abs_flow_tol = m._ptdf_options['abs_flow_tol']
    rel_flow_tol = m._ptdf_options['rel_flow_tol']

    ## find thermal violations
    branch_limits = np.array([enforced_branch_limits[k] for k in index_set_branch])
    ## add some wiggle for tolerance 
    branch_limits += np.maximum(branch_limits*rel_flow_tol, abs_flow_tol)
    thermal_idx_monitored = mb._thermal_idx_monitored
    t_viol_num, t_monitored_viol_num, t_viol_lazy = \
        _find_violation_set(mb, md, index_set_branch, SV, -branch_limits, branch_limits, thermal_idx_monitored,
                            max_viol_add, warning_generator=_generate_flow_viol_warning)

    ## find vmag violations
    if hasattr(mb, "qg"):
        pu_vm_tol = m._ptdf_options['pu_vm_tol']
        vmag_lb_limits = np.array([bus_attrs['v_min'][b] for b in index_set_bus])
        vmag_ub_limits = np.array([bus_attrs['v_max'][b] for b in index_set_bus])
        
        ## add some wiggle for tolerance
        vmag_lb_limits -= pu_vm_tol
        vmag_ub_limits += pu_vm_tol

        vm_idx_monitored = mb._vm_idx_monitored
        v_viol_num, v_monitored_viol_num, v_viol_lazy = \
            _find_violation_set(mb, md, index_set_bus, VMAG, vmag_lb_limits, vmag_ub_limits, vm_idx_monitored,
                                max_viol_add_vm, warning_generator=_generate_vmag_viol_warning)
    else:
        v_viol_num, v_monitored_viol_num, v_viol_lazy = (0,0,set())

    viol_num = t_viol_num + v_viol_num
    monitored_viol_num = t_monitored_viol_num + v_monitored_viol_num

    return SV, t_viol_lazy, VMAG, v_viol_lazy, viol_num, monitored_viol_num

def _find_violation_set(mb, md, index_set, actuals, lb_limits, ub_limits, idx_monitored, max_viol_add,
                        warning_generator=None,time=None):

    if warning_generator is None:
        warning_generator = lambda *args, **kwargs: None

    ## we're doing this backwards for argpartition
    ## most negative -> highest violations
    ub_viol_array = ub_limits - actuals
    lb_viol_array = actuals - lb_limits

    ub_viol = set(np.nonzero(ub_viol_array < 0)[0])
    lb_viol = set(np.nonzero(lb_viol_array < 0)[0])

    ## these will hold the violations
    ## we found this iteration
    viol_set = ub_viol.union(lb_viol)

    ## get the lines for which we've found a violation that's
    ## in the model
    viol_in_mb = viol_set.intersection(idx_monitored)

    ## print a warning for these lines
    ## check if the found violations are in the model and print warning
    baseMVA = md.data['system']['baseMVA']
    for i in viol_in_mb:
        bn = index_set[i]
        logger.warning(warning_generator(mb, bn, lb_limits[i], actuals[i], ub_limits[i], baseMVA, time))

    ## thermal_viol_lazy will hold the lines we're adding
    ## this iteration -- don't want to add lines
    ## that are already in the monitored set
    viol_lazy = viol_set.difference(idx_monitored)

    ## limit the number of lines we add in one iteration
    ## if we have too many violations, just take those largest
    ## in absolute value in either direction
    if len(viol_lazy) > max_viol_add:

        viol_slicer = list(viol_lazy)

        ## For sorting, we want the most negative values to be the largest violations
        viol_array = np.minimum(ub_viol_array[viol_slicer], lb_viol_array[viol_slicer])

        ## give the order of the first max_viol_add violations
        measured_viol = np.argpartition(viol_array, range(max_viol_add))
        viol_slicer_indices = measured_viol[0:max_viol_add]

        viol_lazy = set(viol_slicer[i] for i in viol_slicer_indices)

    viol_num = len(viol_set)
    monitored_viol_num = len(viol_in_mb)

    return viol_num, monitored_viol_num, viol_lazy
    
def _generate_flow_viol_warning(mb, bn, LB, actual, UB, baseMVA, time=None):
    ret_str = "WARNING: line {0} is in the  monitored set".format(bn)
    if time is not None:
        ret_str += " at time {}".format(time)
    ret_str += ", but flow exceeds limit!!\n\t apparent={0}, LB={1}, UB={2}".format(actual*baseMVA, LB*baseMVA, UB*baseMVA)
    ret_str += ", model_pf={}".format(pe.value(mb.pf[bn])*baseMVA)
    if hasattr(mb, "qf"):
        ret_str += ", model_qf={}".format(pe.value(mb.qf[bn]) * baseMVA)
    return ret_str


def _generate_vmag_viol_warning(mb, bn, LB, actual, UB, baseMVA=None, time=None):
    ret_str = "WARNING: bus {0} is in the  monitored set".format(bn)
    if time is not None:
        ret_str += " at time {}".format(time)
    ret_str += ", but voltage exceeds limit!!\n\t vmag={0}, LB={1}, UB={2}".format(actual, LB, UB)
    ret_str += ", model_vm={}".format(pe.value(mb.vm[bn]))
    return ret_str


def _generate_flow_monitor_message(bn, flow, limit, baseMVA, time):
    ret_str = "Adding line {0} to monitored set".format(bn)
    if time is not None:
        ret_str += " at time {}".format(time)
    ret_str += ", apparent={0}, limit={1}".format(flow*baseMVA, limit*baseMVA)
    return ret_str

def _generate_vmag_monitor_message(bn, vm, lb_limit, ub_limit, time):
    ret_str = "Adding bus {0} to monitored set".format(bn)
    if time is not None:
        ret_str += " at time {}".format(time)
    ret_str += ", vmag={0}, LB={1}, UB={2}".format(vm, lb_limit, ub_limit)
    return ret_str

## thermal violation adder
def add_thermal_violations(thermal_viol_lazy, SV, mb, md, solver, ptdf_options, branch_attrs,
                    time=None):

    model = mb.model()

    branch_name_list = branch_attrs['names']

    baseMVA = md.data['system']['baseMVA']

    persistent_solver = isinstance(solver, PersistentSolver)

    include_reactive = hasattr(mb, "qg")

    ## static information between runs
    rel_ptdf_tol = ptdf_options['rel_ptdf_tol']
    abs_ptdf_tol = ptdf_options['abs_ptdf_tol']

    pf = mb.pf
    eq_pf_constr = mb.eq_pf_branch
    if include_reactive:
        qf = mb.qf
        eq_qf_constr = mb.eq_qf_branch
        ineq_branch_thermal_constr = mb.ineq_branch_thermal_limit
        _fdf_unitcircle = mb._fdf_unitcircle

    ## helper for generating pf
    def _iter_over_viol_set(viol_set):
        for i in viol_set:
            bn = branch_name_list[i]
            if bn not in eq_pf_constr:
                ## add eq_pf_branch constraint
                ptdf = branch_attrs['ptdf'][bn]
                ptdf_c = branch_attrs['ptdf_c'][bn]
                expr = libbranch.get_expr_branch_pf_fdf_approx(mb, bn, ptdf, ptdf_c, rel_tol=None, abs_tol=None)
                eq_pf_constr[bn] = pf[bn] == expr
                if include_reactive:
                    ## add eq_qf_branch constraint
                    qtdf = branch_attrs['qtdf'][bn]
                    qtdf_c = branch_attrs['qtdf_c'][bn]
                    expr = libbranch.get_expr_branch_qf_fdf_approx(mb, bn, qtdf, qtdf_c, rel_tol=None, abs_tol=None)
                    eq_qf_constr[bn] = qf[bn] == expr
                    ## add ineq_branch_thermal_limit constraint
                    libbranch.add_constr_branch_thermal_limit(mb, bn, branch_attrs['rating_long_term'][bn])
            yield i, bn


    thermal_viol_in_mb = mb._thermal_idx_monitored
    for i, bn in _iter_over_viol_set(thermal_viol_lazy):
        thermal_limit = branch_attrs['rating_long_term'][bn]
        logger.info(_generate_flow_monitor_message(bn, SV[i], thermal_limit, baseMVA, time))
        thermal_viol_in_mb.append(i)
        if persistent_solver:
            solver.add_constraint(eq_pf_constr[bn])
            if include_reactive:
                solver.add_constraint(eq_qf_constr[bn])
                for x,y in _fdf_unitcircle:
                    solver.add_constraint(ineq_branch_thermal_constr[bn,x,y])

## voltage violation adder
def add_vmag_violations(vmag_viol_lazy, VMAG, mb, md, solver, ptdf_options, bus_attrs,
                    time=None):

    model = mb.model()

    bus_name_list = bus_attrs['names']

    baseMVA = md.data['system']['baseMVA']

    persistent_solver = isinstance(solver, PersistentSolver)

    ## static information between runs
    rel_ptdf_tol = ptdf_options['rel_ptdf_tol']
    abs_ptdf_tol = ptdf_options['abs_ptdf_tol']

    vm = mb.vm
    constr = mb.eq_vm_bus

    ## helper for generating pf
    def _iter_over_viol_set(viol_set):
        for i in viol_set:
            bn = bus_name_list[i]
            if bn not in mb.eq_vm_bus:
                ## add eq_pf_branch constraint
                vdf = bus_attrs['vdf'][bn]
                vdf_c = bus_attrs['vdf_c'][bn]
                expr = libbus.get_vm_expr_vdf_approx(mb, bn, vdf, vdf_c, rel_tol=None, abs_tol=None)
                constr[bn] = vm[bn] == expr
            yield i, bn

    vmag_viol_in_mb = mb._vm_idx_monitored
    for i, bn in _iter_over_viol_set(vmag_viol_lazy):
        lb_limit= bus_attrs['v_min'][bn]
        ub_limit= bus_attrs['v_max'][bn]
        logger.info(_generate_vmag_monitor_message(bn, VMAG[i], lb_limit, ub_limit, time))
        vmag_viol_in_mb.append(i)
        if persistent_solver:
            solver.add_constraint(constr[bn])


def _lazy_model_solve_loop(m, md, solver, timelimit, solver_tee=True, symbolic_solver_labels=False, iteration_limit=100000,
                           vars_to_load=None):

    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    ptdf_options = m._ptdf_options

    rel_flow_tol = ptdf_options['rel_flow_tol']
    abs_flow_tol = ptdf_options['abs_flow_tol']

    branch_limits = branch_attrs['rating_long_term']
    #branch_limits = PTDF.branch_limits_array

    ## only enforce the relative and absolute, within tollerance
    #PTDF.enforced_branch_limits = np.maximum(branch_limits*(1+rel_flow_tol), branch_limits+abs_flow_tol)

    persistent_solver = isinstance(solver, PersistentSolver)

    for i in range(iteration_limit):

        ## Check line flow violations
        #SV, viol_num, mon_viol_num, thermal_viol_lazy = check_violations(m, md, branch_attrs, bus_attrs,
        #                                                                 ptdf_options['max_violations_per_iteration'])
        SV, thermal_viol_lazy, VMAG, vmag_viol_lazy, viol_num, mon_viol_num = \
            check_violations(m, md, branch_attrs, bus_attrs,
                    ptdf_options['max_violations_per_iteration'], ptdf_options['vm_max_violations_per_iteration'])

        iter_status_str = "iteration {0}, found {1} violation(s)".format(i,viol_num)
        if mon_viol_num:
            iter_status_str += " ({} already monitored)".format(mon_viol_num)

        print(iter_status_str)

        if viol_num <= 0:
            ## in this case, there are no violations!
            ## load the duals now too, if we're using a persistent solver
            return LazyPTDFTerminationCondition.NORMAL

        elif viol_num == mon_viol_num:
            print('WARNING: Terminating with monitored violations!')
            print('         Result is not transmission feasible.')
            return LazyPTDFTerminationCondition.FLOW_VIOLATION

        add_thermal_violations(thermal_viol_lazy, SV, m, md, solver, ptdf_options, branch_attrs)

        if ptdf_options['lazy_voltage']:
            add_vmag_violations(vmag_viol_lazy, VMAG, m, md, solver, ptdf_options, bus_attrs)

        #m.ineq_pf_branch_thermal_lb.pprint()
        #m.ineq_pf_branch_thermal_ub.pprint()

        if persistent_solver:
            solver.solve(m, tee=solver_tee, load_solutions=False, save_results=False)
            solver.load_vars(vars_to_load=vars_to_load)
        else:
            solver.solve(m, tee=solver_tee, symbolic_solver_labels=symbolic_solver_labels)

    else: # we hit the iteration limit
        print('WARNING: Exiting on maximum iterations for lazy PTDF model.')
        print('         Result is not transmission feasible.')
        return LazyPTDFTerminationCondition.ITERATION_LIMIT



def _binary_var_generator(instance):
    regulation =  hasattr(instance, 'regulation_service')
    if instance.status_vars in ['CA_1bin_vars', 'garver_3bin_vars', 'garver_2bin_vars', 'garver_3bin_relaxed_stop_vars']:
        yield instance.UnitOn
    if instance.status_vars in ['ALS_state_transition_vars']:
        yield instance.UnitStayOn
    if instance.status_vars in ['garver_3bin_vars', 'garver_2bin_vars', 'garver_3bin_relaxed_stop_vars', 'ALS_state_transition_vars']:
        yield instance.UnitStart
    if instance.status_vars in ['garver_3bin_vars', 'ALS_state_transition_vars']:
        yield instance.UnitStop
    if regulation:
        yield instance.RegulationOn

    yield instance.OutputStorage
    yield instance.InputStorage

    if instance.startup_costs in ['KOW_startup_costs']:
        yield instance.StartupIndicator
    elif instance.startup_costs in ['MLR_startup_costs', 'MLR_startup_costs2',]:
        yield instance.delta

def uc_instance_binary_relaxer(model, solver):
    persistent_solver = isinstance(solver, PersistentSolver)
    for ivar in _binary_var_generator(model):
        ivar.domain = pe.UnitInterval
        if persistent_solver:
            for var in ivar.itervalues():
                solver.update_var(var)

def uc_instance_binary_enforcer(model, solver):
    persistent_solver = isinstance(solver, PersistentSolver)
    for ivar in _binary_var_generator(model):
        ivar.domain = pe.Binary
        if persistent_solver:
            for var in ivar.itervalues():
                solver.update_var(var)

