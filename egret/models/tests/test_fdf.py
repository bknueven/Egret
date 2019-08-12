#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

'''
acopf tester
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
test_cases = [join('../../../download/pglib-opf-master/', f) for f in listdir('../../../download/pglib-opf-master/') if isfile(join('../../../download/pglib-opf-master/', f)) and f.endswith('.m')]


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
        filename = md.data['system']['model_name'] + '_fdf'
        md_fdf.write_to_json(filename)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
