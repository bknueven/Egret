#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains several helper functions that are useful when
modifying the data dictionary
"""
from egret.model_library.transmission.tx_opt import calculate_ptdf, calculate_ptdf_ldf, calculate_qtdf_ldf_vdf
from egret.model_library.defn import BasePointType, SensitivityCalculationMethod


def create_dicts_of_fdf(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT):
    create_dicts_of_ptdf_losses(md,base_point,calculation_method)
    create_dicts_of_qtdf_losses(md,base_point,calculation_method)


def create_dicts_of_ptdf(md, base_point=BasePointType.FLATSTART, calculation_method=SensitivityCalculationMethod.INVERT):

    ptdf = calculate_ptdf(md,base_point,calculation_method)

    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    _len_bus = len(bus_attrs['names'])
    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0,_len_branch))}

    for idx,branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_ptdf = {bus_attrs['names'][i]: ptdf[idx,i] for i in list(range(0,_len_bus))}
        branch['ptdf'] = _row_ptdf


def create_dicts_of_ptdf_losses(md, base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT):

    ptdf_r, ldf, ptdf_c, ldf_c = calculate_ptdf_ldf(md,base_point,calculation_method)

    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    _len_bus = len(bus_attrs['names'])
    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0,_len_branch))}

    for idx,branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_ptdf_r = {bus_attrs['names'][i]: ptdf_r[idx,i] for i in list(range(0,_len_bus))}
        branch['ptdf_r'] = _row_ptdf_r

        _row_ldf = {bus_attrs['names'][i]: ldf[idx,i] for i in list(range(0,_len_bus))}
        branch['ldf'] = _row_ldf

        branch['ptdf_c'] = ptdf_c[idx]
        branch['ldf_c'] = ldf_c[idx]


def create_dicts_of_qtdf_losses(md,base_point=BasePointType.SOLUTION, calculation_method=SensitivityCalculationMethod.INVERT):

    qtdf_r, qldf, vdf, qtdf_c, qldf_c, vdf_c = calculate_qtdf_ldf_vdf(md,base_point,calculation_method)

    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    _len_bus = len(bus_attrs['names'])
    _mapping_bus = {i: bus_attrs['names'][i] for i in list(range(0,_len_bus))}
    _len_branch = len(branch_attrs['names'])
    _mapping_branch = {i: branch_attrs['names'][i] for i in list(range(0,_len_branch))}

    for idx,branch_name in _mapping_branch.items():
        branch = md.data['elements']['branch'][branch_name]
        _row_qtdf_r = {bus_attrs['names'][i]: qtdf_r[idx,i] for i in list(range(0,_len_bus))}
        branch['qtdf_r'] = _row_qtdf_r

        _row_qldf = {bus_attrs['names'][i]: qldf[idx,i] for i in list(range(0,_len_bus))}
        branch['qldf'] = _row_qldf

        branch['qtdf_c'] = qtdf_c[idx]

        branch['qldf_c'] = qldf_c[idx]

    for idx,bus_name in _mapping_bus.items():
        bus = md.data['elements']['bus'][bus_name]
        _row_vdf = {bus_attrs['names'][i]: vdf[idx,i] for i in list(range(0,_len_bus))}
        bus['vdf'] = _row_vdf

        bus['vdf_c'] = vdf_c[idx]
