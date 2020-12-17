#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from egret.models.unit_commitment import create_super_tight_unit_commitment_model,
                                         create_CHP_unit_commitment_model,
                                         _save_uc_results,
                                         _solve_unit_commitment,

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
