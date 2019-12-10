#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

'''
fdf tester
'''
import os
import math
import unittest
from pyomo.opt import SolverFactory, TerminationCondition
from egret.models.acopf import *
from egret.models.fdf import *
from egret.data.model_data import ModelData
from parameterized import parameterized
from egret.parsers.matpower_parser import create_ModelData
from os import listdir
from os.path import isfile, join

current_dir = os.path.dirname(os.path.abspath(__file__))
#test_cases = [join('../../../download/pglib-opf-master/', f) for f in listdir('../../../download/pglib-opf-master/') if isfile(join('../../../download/pglib-opf-master/', f)) and f.endswith('.m')]
case_names = ['pglib_opf_case3_lmbd.m',
              'pglib_opf_case5_pjm.m',
              'pglib_opf_case14_ieee.m',
              'pglib_opf_case24_ieee_rts.m',
              'pglib_opf_case30_as.m',
              'pglib_opf_case30_fsr,m',
              'pglib_opf_case30_ieee.m',
              'pglib_opf_case39_epri.m',
              'pglib_opf_case57_ieee.m',
              'pglib_opf_case73_ieee_rts.m',
              'pglib_opf_case89_pegase.m',
              'pglib_opf_case118_ieee.m',
              'pglib_opf_case162_ieee_dtc.m',
              'pglib_opf_case179_goc.m',
              'pglib_opf_case200_tamu.m',
              'pglib_opf_case240_pserc.m',
              'pglib_opf_case300_ieee.m',
              'pglib_opf_case500_tamu.m',
              'pglib_opf_case588_sdet.m',
              ]
test_cases = [os.path.join(current_dir, 'transmission_test_instances', 'pglib-opf-master', '{}.m'.format(i)) for i in case_names]



class TestFDF(unittest.TestCase):
    show_output = True

    @classmethod
    def setUpClass(self):
        download_dir = os.path.join(current_dir, 'transmission_test_instances')
        if not os.path.exists(os.path.join(download_dir, 'pglib-opf-master')):
            from egret.thirdparty.get_pglib import get_pglib
            get_pglib(download_dir)

    @parameterized.expand(test_cases)
    def test_fdf_model(self, test_case, include_kwargs=False):
        acopf_model = create_psv_acopf_model

        md_dict = create_ModelData(test_case)

        kwargs = {}
        if include_kwargs:
            kwargs = {'include_feasibility_slack':True}
        md_acopf, m, results = solve_acopf(md_dict, "ipopt", acopf_model_generator=acopf_model, return_model=True, return_results=True, solver_tee=False, **kwargs)
        filename = md_acopf.data['system']['model_name'] + '_acopf'
        md_acopf.write_to_json(filename)

        md_fdf, m, results = solve_fdf(md_acopf, "gurobi", return_model=True, return_results=True, solver_tee=False, **kwargs)
        filename = md_fdf.data['system']['model_name'] + '_fdf'
        md_fdf.write_to_json(filename)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
