#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import egret.model_library.unit_commitment.thermal_convex_hull as tch
import pyomo.environ as pyo

from egret.models.unit_commitment import create_super_tight_unit_commitment_model,
                                         create_CHP_unit_commitment_model,
                                         _save_uc_results,
                                         _solve_unit_commitment,
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

class RampingPolytopeCutGenerator:
    def __init__(self, solver_name, g):
        self.solver_name = solver_name
        self.g = g
        self.cut_idx = 0
        self._set_up = False

    def setup_chp_subproblem(self, uc_instance):
        print(f"Setting up Ramping Polytope for generator {self.g}")
        uc = uc_instance
        g = self.g
        rp_subproblem = pyo.ConcreteModel()
        timeperiods = list(uc.TimePeriods)

        rp_subproblem.GenTimeIndexSet = pyo.Set(initialize=((g,t) for t in timeperiods))
        rp_subproblem.UnitOn = pyo.Var(rp_subproblem.GenTimeIndexSet)
        rp_subproblem.UnitStart = pyo.Var(rp_subproblem.GenTimeIndexSet)
        rp_subproblem.UnitStop = pyo.Var(rp_subproblem.GenTimeIndexSet)
        rp_subproblem.PowerGeneratedAboveMinimum = pyo.Var(rp_subproblem.GenTimeIndexSet)
        rp_subproblem.ReserveProvided = pyo.Var(rp_subproblem.GenTimeIndexSet)

        ## set-up PiecewiseProduction, build polytope model
        def _piecewise_production_init():
            l_lengths = { t : len(uc.PowerGenerationPiecewisePoints[g,t]) for t in timeperiods }
            l_lengths = { t : l_length-1 for t, l_length in l_lengths.items() if l_length > 2 }
            for t, length in l_lengths.items():
                for l in range(length):
                    yield g,t,l
        rp_subproblem.GenPiecewiseProductionIndexSet = pyo.Set(initialize=_piecewise_production_init())
        rp_subproblem.PiecewiseProduction = pyo.Var(rp_subproblem.GenPiecewiseProductionIndexSet)

        rp_subproblem.z = pyo.Var(within=pyo.NonNegativeReals)

        tch.make_ramping_polytope(rp_subproblem, uc, g, rp_subproblem.UnitOn,
                                   rp_subproblem.UnitStart, rp_subproblem.UnitStop,
                                   rp_subproblem.PowerGeneratedAboveMinimum,
                                   rp_subproblem.ReserveProvided,
                                   rp_subproblem.PiecewiseProduction,
                                   rp_subproblem.z)

        rp_subproblem.obj = pyo.Objective(expr=rp_subproblem.z, sense=pyo.minimize)
        rp_subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        self.solver = pyo.SolverFactory(self.solver_name)
        self.is_persistent = isinstance(self.solver, PersistentSolver)
        if self.is_persistent:
            self.solver.set_instance(rp_subproblem)

            self.duals_to_load = [*rp_subproblem.on_link_y.values(), *rp_subproblem.start_link_y.values(),
                                  *rp_subproblem.stop_link_y.values(), *rp_subproblem.p_link_p_ints.values(),
                                  *rp_subproblem.r_link_r_ints.values(), *rp_subproblem.pl_link.values()]
            self.vars_to_load = [rp_subproblem.z]

        self.rp_subproblem = rp_subproblem
        self._set_up = True

    def set_vars(self, uc_instance):
        value = pyo.value
        rp = self.rp_subproblem
        solver = self.solver
        is_persistent = self.is_persistent
        for idx, var in rp.UnitOn.items():
            var.value = value(uc_instance.UnitOn[idx])
            var.fix()
            if is_persistent:
                solver.update_var(var)
        for idx, var in rp.UnitStart.items():
            var.value = value(uc_instance.UnitStart[idx])
            var.fix()
            if is_persistent:
                solver.update_var(var)
        for idx, var in rp.UnitStop.items():
            var.value = value(uc_instance.UnitStop[idx])
            var.fix()
            if is_persistent:
                solver.update_var(var)
        for idx, var in rp.PowerGeneratedAboveMinimum.items():
            var.value = value(uc_instance.PowerGeneratedAboveMinimum[idx])
            var.fix()
            if is_persistent:
                solver.update_var(var)
        for idx, var in rp.ReserveProvided.items():
            var.value = value(uc_instance.ReserveProvided[idx])
            var.fix()
            if is_persistent:
                solver.update_var(var)
        for idx, var in rp.PiecewiseProduction.items():
            var.value = value(uc_instance.PiecewiseProduction[idx])
            var.fix()
            if is_persistent:
                solver.update_var(var)

    def _get_cut(self, uc_instance, constant):
        value = pyo.value
        rp = self.rp_subproblem
        dual = self.rp_subproblem.dual
        linear_coefs = []
        linear_vars = []
        ## cut is 
        ## \bar{z} + \pi^T ( x - \bar{x} ) <= 0
        ## or
        ## (\bar{z} -  \pi^T\bar{x}) + \pi^T x <= 0
        for g,t in rp.UnitOn:
            coef = dual[rp.on_link_y[t]]
            if coef == 0.:
                continue
            var = uc_instance.UnitOn[g,t]
            linear_coefs.append(coef)
            linear_vars.append(var)
            constant -= coef*var.value
        for g,t in rp.UnitStart:
            coef = dual[rp.start_link_y[t]]
            if coef == 0.:
                continue
            var = uc_instance.UnitStart[g,t]
            linear_coefs.append(coef)
            linear_vars.append(var)
            constant -= coef*var.value
        for g,t in rp.UnitStop:
            coef = dual[rp.stop_link_y[t]]
            if coef == 0.:
                continue
            var = uc_instance.UnitStop[g,t]
            linear_coefs.append(coef)
            linear_vars.append(var)
            constant -= coef*var.value
        for g,t in rp.PowerGeneratedAboveMinimum:
            coef = dual[rp.p_link_p_ints[t]]
            if coef == 0.:
                continue
            var = uc_instance.PowerGeneratedAboveMinimum[g,t]
            linear_coefs.append(coef)
            linear_vars.append(var)
            constant -= coef*var.value
        for g,t in rp.ReserveProvided:
            coef = dual[rp.r_link_r_ints[t]]
            if coef == 0.:
                continue
            var = uc_instance.ReserveProvided[g,t]
            linear_coefs.append(coef)
            linear_vars.append(var)
            constant -= coef*var.value
        for g,t,l in rp.PiecewiseProduction:
            coef = dual[rp.pl_link[l,t]]
            if coef == 0.:
                continue
            var = uc_instance.PiecewiseProduction[g,t,l]
            linear_coefs.append(coef)
            linear_vars.append(var)
            constant -= coef*var.value

        return (None, LinearExpression(linear_vars=linear_vars, linear_coefs=linear_coefs, constant=constant), 0.)

    def generate_cut(self, uc_instance, solver_options):
        g = self.g
        value = pyo.value
        for t in uc_instance.TimePeriods:
            if value(uc_instance.UnitOn[g,t]) not in [0., 1.]:
                break
        else: # no break
            return None

        if not self._set_up:
            self.setup_chp_subproblem(uc_instance)
            self._set_up = True

        print(f"Testing RP feasibility for {g}")
        self.set_vars(uc_instance)
        for k,v in solver_options.items():
            self.solver.options[k] = v
        if is_persistent:
            self.solver.solve(tee=True, load_solutions=False, save_results=False)

            self.solver.load_vars(self.vars_to_load)
        else:
            self.solver.solve(self.rp_subproblem, tee=True)
        constant = value(self.rp_subproblem.z)
        if constant <= 0.:
            print(f"\tFeasible for {g}")
            return None
        ## else we got a cut!
        self.cut_idx += 1
        if self.is_persistent:
            self.solver.load_duals(self.duals_to_load)
        uc_instance.RampingPolytopeCuts[g, self.cut_idx] = self._get_cut(uc_instance, constant)
        print(f"\tAdding RP cut for {g}")
        return uc_instance.RampingPolytopeCuts[g, self.cut_idx]

class RampingPolytopeSolver:
    def __init__(self, uc_instance, subproblem_solver_name):
        '''
        Parameters
        ----------
        uc_instance : unit commitment instance 
        subproblem_solver_name : name of Pyomo persistent solver for subproblems
        '''
        self.solver_name = subproblem_solver_name
        self.setup_chp_subproblems(uc_instance)
        self.add_ramping_polytope_cuts_constr(uc_instance)

    def setup_chp_subproblems(self, uc):
        ramping_gens = tch.get_ramping_gens(uc)

        self._ramping_polytope_cut_generators = {}

        for g in ramping_gens:
            self._ramping_polytope_cut_generators[g] = RampingPolytopeCutGenerator(self.solver_name, g)

    def add_ramping_polytope_cuts_constr(self, uc):
        uc.RampingPolytopeCuts = pyo.Constraint(self._ramping_polytope_cut_generators.keys(),
                                                pyo.PositiveIntegers)

    def add_cuts(self, uc_instance, uc_solver, subproblem_solver_options):
        cuts_added = 0
        is_persistent = isinstance(uc_solver, PersistentSolver)
        for g, cg in self._ramping_polytope_cut_generators.items():
            cut = cg.generate_cut(uc_instance, subproblem_solver_options)
            if cut is not None:
                cuts_added += 1
                if is_persistent:
                    uc_solver.add_constraint(cut)
        return cuts_added

    def solve_ramping_polytope_problem(self, uc_instance, uc_solver, subproblem_solver_options):
        cuts_added = 1
        while cuts_added > 0:
            uc_solver.solve(uc_instance, tee=True)
            cuts_added = self.add_cuts(uc_instance, uc_solver, subproblem_solver_options)

def solve_convex_hull_pricing_problem(model_data,
                                      solver,
                                      timelimit = None,
                                      solver_tee = True,
                                      symbolic_solver_labels = False,
                                      solver_options = None,
                                      solve_method_options = None,
                                      lazy = True,
                                      return_model = False,
                                      return_results = False,
                                      **kwargs):
    '''
    Create and solve a convex hull pricing problem

    ----------
    model_data : egret.data.ModelData
        An egret ModelData object with the appropriate data loaded.
        # TODO: describe the required and optional attributes
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a string specifying a pyomo solver name, or an instanciated pyomo solver
    timelimit : float (optional)
        Time limit for unit commitment run. Default of None results in no time
        limit being set -- runs until mipgap is satisfied
    solver_tee : bool (optional)
        Display solver log. Default is True.
    symbolic_solver_labels : bool (optional)
        Use symbolic solver labels. Useful for debugging; default is False.
    solver_options : dict (optional)
        Other options to pass into the solver. Default is dict().
    solve_method_options : dict (optional)
        Other options to pass into the pyomo solve method. Default is dict().
    lazy : bool (optional)
        If True, uses a cut-generation approach for CHP. If False, constructs
        and solves a single extensive-form problem
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    if lazy:
        m = create_super_tight_unit_commitment_model(model_data, relaxed=True, **kwargs)

        ## TODO: finish lazy implementation
    else:
        m = create_CHP_unit_commitment_model(model_data, relaxed=relaxed, **kwargs)
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        m, results, solver = _solve_unit_commitment(m, solver, mipgap, timelimit, solver_tee, symbolic_solver_labels, solver_options, solve_method_options,relaxed )

    md = _save_uc_results(m, True)

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md
