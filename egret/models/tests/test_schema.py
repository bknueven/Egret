import os
import json
import unittest
import copy

from egret.models.unit_commitment import solve_unit_commitment, create_tight_unit_commitment_model
from egret.models.schema_demo import UnitCommitmentDataModel

current_dir = os.path.dirname(os.path.abspath(__file__))

class TestUnitCommitmentModelSchema(unittest.TestCase):
    """Test class for unit commitment model schema using Pydantic."""
    @classmethod
    def setUpClass(cls):
        input_json_file_name = os.path.join(current_dir, 'uc_test_instances', 'schema_demo_tiny_uc_2.json')

        with open(input_json_file_name, 'r') as f:
            input_json = json.load(f)

        cls.base_case = input_json
    
    def test_schema_pprint(self):
        """Tests that the schema definition can be pretty-printed in JSON format."""
        schema = UnitCommitmentDataModel.schema_json(indent=2)
        print(schema)
    
    def test_base_case_read_and_validate(self):
        """Tests that the base case template can be read as JSON and validated by using the schema."""
        input_json = copy.deepcopy(self.base_case)
        
        uc_data = uc_data = UnitCommitmentDataModel(
            elements = input_json["elements"],
            system = input_json["system"]
            )
    
    def test_validation_error_for_required_field(self):
        """Tests that validation fails if a required field is not provided."""
        input_json = copy.deepcopy(self.base_case)

        # Delete the required "baseMVA" field from the system field.
        del input_json['system']['baseMVA']

        with self.assertRaises(ValueError):
            uc_data = UnitCommitmentDataModel(
                elements = input_json["elements"],
                system = input_json["system"]
                )
    
    def test_validation_error_for_wrong_unit_numeric_quantity(self):
        """Tests that validation fails if a NumericQuantity's unit is incorrect."""
        input_json = copy.deepcopy(self.base_case)

        # Give the units of load_mismatch_cost as # instead of the expected $
        input_json['system']['load_mismatch_cost']['units'] = '#'

        with self.assertRaises(ValueError):
            uc_data = UnitCommitmentDataModel(
                elements = input_json["elements"],
                system = input_json["system"]
                )
    
    def test_validation_error_for_numeric_quantity_missing_units(self):
        """Tests that validation fails if a NumericQuantity's unit is incorrect."""
        input_json = copy.deepcopy(self.base_case)

        # Delete the units field from load_mismatch_cost
        del input_json['system']['load_mismatch_cost']['units']

        with self.assertRaises(ValueError):
            uc_data = UnitCommitmentDataModel(
                elements = input_json["elements"],
                system = input_json["system"]
                )
    
    def test_validation_error_for_invalid_choice_enumerated_field(self):
        """Tests that validation fails if an enumerated field has an unexpected choice."""
        input_json = copy.deepcopy(self.base_case)

        # If 'matpower_bustype' is provided to a Bus object, it must be one of 'PV', 'PQ', 'ref', or 'isolated'
        input_json['elements']['bus']['Bus1']['matpower_bustype'] = 'banana'

        with self.assertRaises(ValueError):
            uc_data = UnitCommitmentDataModel(
                elements = input_json["elements"],
                system = input_json["system"]
                )
    
    def test_validation_error_for_custom_criterion(self):
        """Tests that validation fails if a RenewableGenerator object has a power factor outside of [-1, 1]."""
        input_json = copy.deepcopy(self.base_case)

        # Add a renewable generator and set its power factor outside the appropriate range
        input_json['elements']['renewable_generator'] = {}
        input_json['elements']['renewable_generator']['RGEN1'] = {}
        input_json['elements']['renewable_generator']['RGEN1']['bus'] = 'Bus1'
        input_json['elements']['renewable_generator']['RGEN1']['power_factor'] = 37

        with self.assertRaises(ValueError):
            uc_data = UnitCommitmentDataModel(
                elements = input_json["elements"],
                system = input_json["system"]
                )


if __name__ == '__main__':
    unittest.main()
