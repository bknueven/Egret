#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains several helper functions and classes that are useful when
modifying the data dictionary
"""
import egret.model_library.transmission.tx_calc as tx_calc
from egret.model_library.defn import BasePointType, ApproximationType


def create_dicts_of_fdf(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    reference_bus = md.data['system']['reference_bus']
    ptdf, ptdf_c, pldf, pldf_c, va_sensi, va_const = tx_calc.calculate_ptdf_pldf(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)
    qtdf, qtdf_c, qldf, qldf_c, vm_sensi, vm_const = tx_calc.calculate_qtdf_qldf_vdf(branches, buses, branch_attrs['names'],
                                                    bus_attrs['names'], reference_bus, base_point)

    Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                    reference_bus, base_point)
    Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                    reference_bus, base_point)

    # save sparse sensitivity matrices to system modelData
    md.data['system']['Ft'] = Ft
    md.data['system']['ft_c'] = ft_c
    md.data['system']['Fv'] = Fv
    md.data['system']['fv_c'] = fv_c
    md.data['system']['Lt'] = Lt
    md.data['system']['lt_c'] = lt_c
    md.data['system']['Lv'] = Lv
    md.data['system']['lv_c'] = lv_c
    md.data['system']['va_SENSI'] = va_sensi
    md.data['system']['va_CONST'] = va_const
    md.data['system']['vm_SENSI'] = vm_sensi
    md.data['system']['vm_CONST'] = vm_const

    # TODO: check if phi-constants are used anywhere
    phi_from, phi_to = tx_calc.calculate_phi_constant(branches, branch_attrs['names'], bus_attrs['names'],
                                                      ApproximationType.PTDF_LOSSES)
    phi_loss_from, phi_loss_to = tx_calc.calculate_phi_loss_constant(branches, branch_attrs['names'],
                                                                     bus_attrs['names'], ApproximationType.PTDF_LOSSES)


    phi_q_from, phi_q_to = tx_calc.calculate_phi_q_constant(branches, branch_attrs['names'], bus_attrs['names'])
    phi_loss_q_from, phi_loss_q_to = tx_calc.calculate_phi_loss_q_constant(branches, branch_attrs['names'], bus_attrs['names'])

    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0, _len_branch))}

    _len_bus = len(bus_attrs['names'])
    _mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0, _len_bus))}

    for idx, branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_ptdf = {bus_attrs['names'][i]: ptdf[idx, i] for i in list(range(0, _len_bus))}
        branch['ptdf'] = _row_ptdf

        _row_pldf = {bus_attrs['names'][i]: pldf[idx, i] for i in list(range(0, _len_bus))}
        branch['pldf'] = _row_pldf

        branch['ptdf_c'] = ptdf_c[idx]

        branch['pldf_c'] = pldf_c[idx]

        _row_qtdf = {bus_attrs['names'][i]: qtdf[idx, i] for i in list(range(0, _len_bus))}
        branch['qtdf'] = _row_qtdf

        _row_qldf = {bus_attrs['names'][i]: qldf[idx, i] for i in list(range(0, _len_bus))}
        branch['qldf'] = _row_qldf

        branch['qtdf_c'] = qtdf_c[idx]

        branch['qldf_c'] = qldf_c[idx]


    for idx, bus_name in _mapping_bus.items():
        bus = md.data['elements']['bus'][bus_name]

        _row_vdf = {bus_attrs['names'][i]: vm_sensi[idx, i] for i in list(range(0, _len_bus))}
        bus['vdf'] = _row_vdf

        bus['vdf_c'] = vm_const[idx]

        _row_phi_from = {branch_attrs['names'][i]: phi_from[idx, i] for i in list(range(0, _len_branch)) if
                         phi_from[idx, i] != 0.}
        bus['phi_from'] = _row_phi_from

        _row_phi_to = {branch_attrs['names'][i]: phi_to[idx, i] for i in list(range(0, _len_branch)) if
                       phi_to[idx, i] != 0.}
        bus['phi_to'] = _row_phi_to

        _row_phi_loss_from = {branch_attrs['names'][i]: phi_loss_from[idx, i] for i in list(range(0, _len_branch)) if
                              phi_loss_from[idx, i] != 0.}
        bus['phi_loss_from'] = _row_phi_loss_from

        _row_phi_loss_to = {branch_attrs['names'][i]: phi_loss_to[idx, i] for i in list(range(0, _len_branch)) if
                            phi_loss_to[idx, i] != 0.}
        bus['phi_loss_to'] = _row_phi_loss_to

        _row_phi_q_from = {branch_attrs['names'][i]: phi_q_from[idx, i] for i in list(range(0, _len_branch)) if
                         phi_q_from[idx, i] != 0.}
        bus['phi_q_from'] = _row_phi_q_from

        _row_phi_q_to = {branch_attrs['names'][i]: phi_q_to[idx, i] for i in list(range(0, _len_branch)) if
                       phi_q_to[idx, i] != 0.}
        bus['phi_q_to'] = _row_phi_q_to

        _row_phi_loss_q_from = {branch_attrs['names'][i]: phi_loss_q_from[idx, i] for i in list(range(0, _len_branch)) if
                              phi_loss_q_from[idx, i] != 0.}
        bus['phi_loss_q_from'] = _row_phi_loss_q_from

        _row_phi_loss_q_to = {branch_attrs['names'][i]: phi_loss_q_to[idx, i] for i in list(range(0, _len_branch)) if
                            phi_loss_q_to[idx, i] != 0.}
        bus['phi_loss_q_to'] = _row_phi_loss_q_to


def create_dicts_of_lccm(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    reference_bus = md.data['system']['reference_bus']
    Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)
    Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)

    # save sparse sensitivity matrices to system modelData
    md.data['system']['Ft'] = Ft
    md.data['system']['ft_c'] = ft_c
    md.data['system']['Fv'] = Fv
    md.data['system']['fv_c'] = fv_c
    md.data['system']['Lt'] = Lt
    md.data['system']['lt_c'] = lt_c
    md.data['system']['Lv'] = Lv
    md.data['system']['lv_c'] = lv_c

    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0, _len_branch))}

    _len_bus = len(bus_attrs['names'])
    _mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0, _len_bus))}

    for idx, branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_Ft = {bus_attrs['names'][i]: Ft[idx, i] for i in list(range(0, _len_bus))}
        branch['Ft'] = _row_Ft

        _row_Lt = {bus_attrs['names'][i]: Lt[idx, i] for i in list(range(0, _len_bus))}
        branch['Lt'] = _row_Lt

        branch['ft_c'] = ft_c[idx]

        branch['lt_c'] = lt_c[idx]

        _row_Fv = {bus_attrs['names'][i]: Fv[idx, i] for i in list(range(0, _len_bus))}
        branch['Fv'] = _row_Fv

        _row_Lv = {bus_attrs['names'][i]: Lv[idx, i] for i in list(range(0, _len_bus))}
        branch['Lv'] = _row_Lv

        branch['fv_c'] = fv_c[idx]

        branch['lv_c'] = lv_c[idx]


def create_dicts_of_fdf_simplified(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    reference_bus = md.data['system']['reference_bus']
    ptdf, ptdf_c, pldf, pldf_c, va_sensi, va_const = tx_calc.calculate_ptdf_pldf(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)

    qtdf, qtdf_c, qldf, qldf_c, vm_sensi, vm_const = tx_calc.calculate_qtdf_qldf(branches, buses, branch_attrs['names'],
                                                    bus_attrs['names'], reference_bus, base_point)

    Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)
    Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)

    # save sparse sensitivity matrices to system modelData
    md.data['system']['Ft'] = Ft
    md.data['system']['ft_c'] = ft_c
    md.data['system']['Fv'] = Fv
    md.data['system']['fv_c'] = fv_c
    md.data['system']['Lt'] = Lt
    md.data['system']['lt_c'] = lt_c
    md.data['system']['Lv'] = Lv
    md.data['system']['lv_c'] = lv_c
    md.data['system']['va_SENSI'] = va_sensi
    md.data['system']['va_CONST'] = va_const
    md.data['system']['vm_SENSI'] = vm_sensi
    md.data['system']['vm_CONST'] = vm_const

    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0, _len_branch))}

    _len_bus = len(bus_attrs['names'])
    _mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0, _len_bus))}

    # power transfer factors are the same as in FDF
    for idx, branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_ptdf = {bus_attrs['names'][i]: ptdf[idx, i] for i in list(range(0, _len_bus))}
        branch['ptdf'] = _row_ptdf

        branch['ptdf_c'] = ptdf_c[idx]

        _row_qtdf = {bus_attrs['names'][i]: qtdf[idx, i] for i in list(range(0, _len_bus))}
        branch['qtdf'] = _row_qtdf

        branch['qtdf_c'] = qtdf_c[idx]

    # condensed loss factors are pldf and qldf summed over the set of branches
    md.data['system']['ploss_const'] = sum(pldf_c[idx] for idx in list(range(0, _len_branch)))
    md.data['system']['qloss_const'] = sum(qldf_c[idx] for idx in list(range(0, _len_branch)))


    for idx, bus_name in _mapping_bus.items():
        bus = md.data['elements']['bus'][bus_name]

        # condensed loss factors are pldf and qldf summed over the set of branches
        _ploss_sens = sum(pldf[i, idx] for i in list(range(0, _len_branch)))
        bus['ploss_sens'] = _ploss_sens

        _qloss_sens = sum(qldf[i, idx] for i in list(range(0, _len_branch)))
        bus['qloss_sens'] = _qloss_sens

        # voltage distribution factors are the same as in FDF
        _row_vdf = {bus_attrs['names'][i]: vm_sensi[idx, i] for i in list(range(0, _len_bus))}
        bus['vdf'] = _row_vdf

        bus['vdf_c'] = vm_const[idx]


def create_dicts_of_ptdf_losses(md, base_point=BasePointType.SOLUTION):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    reference_bus = md.data['system']['reference_bus']
    ptdf, ptdf_c, pldf, pldf_c = tx_calc.calculate_ptdf_pldf(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)

    Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                    reference_bus, base_point)
    Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                    reference_bus, base_point)

    # save sparse sensitivity matrices to system modelData
    md.data['system']['Ft'] = Ft
    md.data['system']['ft_c'] = ft_c
    md.data['system']['Fv'] = Fv
    md.data['system']['fv_c'] = fv_c
    md.data['system']['Lt'] = Lt
    md.data['system']['lt_c'] = lt_c
    md.data['system']['Lv'] = Lv
    md.data['system']['lv_c'] = lv_c

    # TODO: check if phi-constants are used anywhere
    phi_from, phi_to = tx_calc.calculate_phi_constant(branches, branch_attrs['names'], bus_attrs['names'],
                                                      ApproximationType.PTDF_LOSSES)
    phi_loss_from, phi_loss_to = tx_calc.calculate_phi_loss_constant(branches, branch_attrs['names'],
                                                                     bus_attrs['names'], ApproximationType.PTDF_LOSSES)


    phi_q_from, phi_q_to = tx_calc.calculate_phi_q_constant(branches, branch_attrs['names'], bus_attrs['names'])
    phi_loss_q_from, phi_loss_q_to = tx_calc.calculate_phi_loss_q_constant(branches, branch_attrs['names'], bus_attrs['names'])

    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0, _len_branch))}

    _len_bus = len(bus_attrs['names'])
    _mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0, _len_bus))}

    for idx, branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_ptdf = {bus_attrs['names'][i]: ptdf[idx, i] for i in list(range(0, _len_bus))}
        branch['ptdf'] = _row_ptdf

        _row_pldf = {bus_attrs['names'][i]: pldf[idx, i] for i in list(range(0, _len_bus))}
        branch['pldf'] = _row_pldf

        branch['ptdf_c'] = ptdf_c[idx]

        branch['pldf_c'] = pldf_c[idx]

    # condensed loss factors are pldf and qldf summed over the set of branches
    md.data['system']['ploss_const'] = sum(pldf_c[idx] for idx in list(range(0, _len_branch)))

    for idx, bus_name in _mapping_bus.items():
        bus = md.data['elements']['bus'][bus_name]

        # condensed loss factors are pldf and qldf summed over the set of branches
        _ploss_sens = sum(pldf[i, idx] for i in list(range(0, _len_branch)))
        bus['ploss_sens'] = _ploss_sens


    for idx, bus_name in _mapping_bus.items():
        bus = md.data['elements']['bus'][bus_name]

        _row_phi_from = {branch_attrs['names'][i]: phi_from[idx, i] for i in list(range(0, _len_branch)) if
                         phi_from[idx, i] != 0.}
        bus['phi_from'] = _row_phi_from

        _row_phi_to = {branch_attrs['names'][i]: phi_to[idx, i] for i in list(range(0, _len_branch)) if
                       phi_to[idx, i] != 0.}
        bus['phi_to'] = _row_phi_to

        _row_phi_loss_from = {branch_attrs['names'][i]: phi_loss_from[idx, i] for i in list(range(0, _len_branch)) if
                              phi_loss_from[idx, i] != 0.}
        bus['phi_loss_from'] = _row_phi_loss_from

        _row_phi_loss_to = {branch_attrs['names'][i]: phi_loss_to[idx, i] for i in list(range(0, _len_branch)) if
                            phi_loss_to[idx, i] != 0.}
        bus['phi_loss_to'] = _row_phi_loss_to

        _row_phi_q_from = {branch_attrs['names'][i]: phi_q_from[idx, i] for i in list(range(0, _len_branch)) if
                         phi_q_from[idx, i] != 0.}
        bus['phi_q_from'] = _row_phi_q_from

        _row_phi_q_to = {branch_attrs['names'][i]: phi_q_to[idx, i] for i in list(range(0, _len_branch)) if
                       phi_q_to[idx, i] != 0.}
        bus['phi_q_to'] = _row_phi_q_to

        _row_phi_loss_q_from = {branch_attrs['names'][i]: phi_loss_q_from[idx, i] for i in list(range(0, _len_branch)) if
                              phi_loss_q_from[idx, i] != 0.}
        bus['phi_loss_q_from'] = _row_phi_loss_q_from

        _row_phi_loss_q_to = {branch_attrs['names'][i]: phi_loss_q_to[idx, i] for i in list(range(0, _len_branch)) if
                            phi_loss_q_to[idx, i] != 0.}
        bus['phi_loss_q_to'] = _row_phi_loss_q_to


def create_dicts_of_ptdf(md, base_point=BasePointType.FLATSTART):
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')

    reference_bus = md.data['system']['reference_bus']
    ptdf, ptdf_c, pldf, pldf_c = tx_calc.calculate_ptdf_pldf(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                                    reference_bus, base_point)

    Ft, ft_c, Fv, fv_c = tx_calc.calculate_lccm_flow_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                    reference_bus, base_point)
    Lt, lt_c, Lv, lv_c = tx_calc.calculate_lccm_loss_sensitivies(branches, buses, branch_attrs['names'], bus_attrs['names'],
                                    reference_bus, base_point)

    # save sparse sensitivity matrices to system modelData
    md.data['system']['Ft'] = Ft
    md.data['system']['ft_c'] = ft_c
    md.data['system']['Fv'] = Fv
    md.data['system']['fv_c'] = fv_c
    md.data['system']['Lt'] = Lt
    md.data['system']['lt_c'] = lt_c
    md.data['system']['Lv'] = Lv
    md.data['system']['lv_c'] = lv_c

    # TODO: check if phi-constants are used anywhere
    phi_from, phi_to = tx_calc.calculate_phi_constant(branches, branch_attrs['names'], bus_attrs['names'],
                                                      ApproximationType.PTDF_LOSSES)
    phi_loss_from, phi_loss_to = tx_calc.calculate_phi_loss_constant(branches, branch_attrs['names'],
                                                                     bus_attrs['names'], ApproximationType.PTDF_LOSSES)


    phi_q_from, phi_q_to = tx_calc.calculate_phi_q_constant(branches, branch_attrs['names'], bus_attrs['names'])
    phi_loss_q_from, phi_loss_q_to = tx_calc.calculate_phi_loss_q_constant(branches, branch_attrs['names'], bus_attrs['names'])

    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0, _len_branch))}

    _len_bus = len(bus_attrs['names'])
    _mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0, _len_bus))}

    for idx, branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_ptdf = {bus_attrs['names'][i]: ptdf[idx, i] for i in list(range(0, _len_bus))}
        branch['ptdf'] = _row_ptdf

        _row_pldf = {bus_attrs['names'][i]: pldf[idx, i] for i in list(range(0, _len_bus))}
        branch['pldf'] = _row_pldf

        branch['ptdf_c'] = ptdf_c[idx]

        branch['pldf_c'] = pldf_c[idx]

        # sparse sensitivities to voltage angle and magnitude
        _row_Ft = {bus_attrs['names'][i]: Ft[idx, i] for i in list(range(0, _len_bus))}
        _row_Fv = {bus_attrs['names'][i]: Fv[idx, i] for i in list(range(0, _len_bus))}
        _row_Lt = {bus_attrs['names'][i]: Lt[idx, i] for i in list(range(0, _len_bus))}
        _row_Lv = {bus_attrs['names'][i]: Lv[idx, i] for i in list(range(0, _len_bus))}
        branch['Ft'] = _row_Ft
        branch['Fv'] = _row_Fv
        branch['Lt'] = _row_Lt
        branch['Lv'] = _row_Lv
        branch['ft_c'] = ft_c[idx]
        branch['fv_c'] = fv_c[idx]
        branch['lt_c'] = lt_c[idx]
        branch['lv_c'] = lv_c[idx]


    for idx, bus_name in _mapping_bus.items():
        bus = md.data['elements']['bus'][bus_name]

        _row_phi_from = {branch_attrs['names'][i]: phi_from[idx, i] for i in list(range(0, _len_branch)) if
                         phi_from[idx, i] != 0.}
        bus['phi_from'] = _row_phi_from

        _row_phi_to = {branch_attrs['names'][i]: phi_to[idx, i] for i in list(range(0, _len_branch)) if
                       phi_to[idx, i] != 0.}
        bus['phi_to'] = _row_phi_to

        _row_phi_loss_from = {branch_attrs['names'][i]: phi_loss_from[idx, i] for i in list(range(0, _len_branch)) if
                              phi_loss_from[idx, i] != 0.}
        bus['phi_loss_from'] = _row_phi_loss_from

        _row_phi_loss_to = {branch_attrs['names'][i]: phi_loss_to[idx, i] for i in list(range(0, _len_branch)) if
                            phi_loss_to[idx, i] != 0.}
        bus['phi_loss_to'] = _row_phi_loss_to

        _row_phi_q_from = {branch_attrs['names'][i]: phi_q_from[idx, i] for i in list(range(0, _len_branch)) if
                         phi_q_from[idx, i] != 0.}
        bus['phi_q_from'] = _row_phi_q_from

        _row_phi_q_to = {branch_attrs['names'][i]: phi_q_to[idx, i] for i in list(range(0, _len_branch)) if
                       phi_q_to[idx, i] != 0.}
        bus['phi_q_to'] = _row_phi_q_to

        _row_phi_loss_q_from = {branch_attrs['names'][i]: phi_loss_q_from[idx, i] for i in list(range(0, _len_branch)) if
                              phi_loss_q_from[idx, i] != 0.}
        bus['phi_loss_q_from'] = _row_phi_loss_q_from

        _row_phi_loss_q_to = {branch_attrs['names'][i]: phi_loss_q_to[idx, i] for i in list(range(0, _len_branch)) if
                            phi_loss_q_to[idx, i] != 0.}
        bus['phi_loss_q_to'] = _row_phi_loss_q_to