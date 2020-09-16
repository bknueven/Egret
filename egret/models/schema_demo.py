import json
import os
import math
from datetime import datetime
from typing import List, Dict, Union, Tuple
from enum import Enum

from pydantic import Schema, BaseModel, Json, validator, ValidationError
from pydantic.dataclasses import dataclass

current_dir = os.path.dirname(os.path.abspath(__file__))


class NumericQuantity(BaseModel):
    """Numeric quantity with a specified unit."""
    value: Union[int, float] = None
    units: str


class TimeSeries(BaseModel):
    """Time series dictionary-like objects."""
    values: Dict[Union[int, datetime], Union[float, int]] = None
    units: str


class CostCurveType(str, Enum):
    """Cost curve type: one of 'piecewise' or 'polynomial'."""
    piecewise = 'piecewise'
    polynomial = 'polynomial'


class CostCurve(BaseModel):
    """Cost curve dictionary-like objects."""
    data_type: str = 'cost_curve'
    cost_curve_type: CostCurveType
    values: Union[Dict[str, float], List]


class MatpowerBusType(str, Enum):
    """Matpower bus type: one of 'PV', 'PQ', 'ref', or 'isolated'."""
    pv = 'PV'
    pq = 'PQ'
    ref = 'ref'
    isolated = 'isolated'


class Bus(BaseModel):
    """Bus object."""
    matpower_bustype: MatpowerBusType = None
    base_kv: NumericQuantity = None
    v_min: NumericQuantity = None
    v_max: NumericQuantity = None

    vm: Union[float, TimeSeries] = None
    va: Union[float, TimeSeries] = None
    p_balance_violation: Union[float, TimeSeries] = None
    q_balance_violation: Union[float, TimeSeries] = None
    lmp: Union[float, TimeSeries] = None
    q_lmp: Union[float, TimeSeries] = None


class Load(BaseModel):
    """Load object."""
    bus: Union[int, str]
    area: Union[int, str] = None
    zone: Union[int, str] = None
    owner: Union[int, str] = None
    in_service: bool

    p_load: Union[float, TimeSeries] = None
    q_load: Union[float, TimeSeries] = None
    p_load_shed: Union[float, TimeSeries] = None
    q_load_shed: Union[float, TimeSeries] = None


class Branch(BaseModel):
    """Base class for branch objects."""
    from_bus: Union[int, str]
    to_bus: Union[int, str]
    resistance: NumericQuantity
    reactance: NumericQuantity
    charging_susceptance: NumericQuantity
    rating_long_term: NumericQuantity
    rating_short_term: NumericQuantity
    rating_emergency: NumericQuantity
    pf: Union[NumericQuantity, TimeSeries] = None
    qf: Union[NumericQuantity, TimeSeries] = None
    pt: Union[NumericQuantity, TimeSeries] = None
    qt: Union[NumericQuantity, TimeSeries] = None

    in_service: bool
    planned_outage: Union[bool, TimeSeries] = None
    angle_diff_min: NumericQuantity = None
    angle_diff_max: NumericQuantity = None
    circuit: str
    owner: str
    owner_fraction: float = None


class Line(Branch):
    """Line object."""
    pass


class Transformer(Branch):
    """Transformer object."""
    transformer_tap_ratio: float
    transformer_magnetizing_conductance: NumericQuantity
    transformer_magnetizing_susceptance: NumericQuantity
    transformer_phase_shift: NumericQuantity


class FuelSupply(BaseModel):
    """Fuel supply object."""
    fuel_supply_type: str
    fuel_available: TimeSeries


class Generator(BaseModel):
    """Base class for generator objects."""
    p_min: Union[NumericQuantity, TimeSeries] = None
    p_max: Union[NumericQuantity, TimeSeries] = None
    pg: Union[float, TimeSeries] = None
    qg: Union[float, TimeSeries] = None

    in_service: bool = None
    fuel: str = None

    bus: Union[int, str]
    area: Union[int, str] = None
    zone: Union[int, str] = None
    owner: Union[int, str] = None


class RenewableGenerator(Generator):
    """Renewable generator object."""
    power_factor: float = None

    @validator('power_factor')
    def power_factor_range(cls, v):
        if not (v >= -1 and v <= 1):
            raise ValueError('The power factor must be in the range [-1, 1].')
        return v


class ThermalGenerator(Generator):
    """Thermal generator object."""
    failure_rate: float = 0.0
    q_min: NumericQuantity = None
    q_max: NumericQuantity = None
    p_cost: CostCurve = None
    q_cost: CostCurve = None
    startup_cost: List[Tuple[int, float]] = None
    shutdown_cost: NumericQuantity = None
    startup_capacity: NumericQuantity = None
    shutdown_capacity: NumericQuantity = None
    ramp_up_60min: NumericQuantity
    ramp_down_60min: NumericQuantity
    min_up_time: NumericQuantity
    min_down_time: NumericQuantity
    initial_status: NumericQuantity
    initial_p_output: NumericQuantity
    initial_q_output: NumericQuantity = None

    rg: Union[float, TimeSeries] = None

    fast_start: bool = False
    supplemental_start: bool = False
    agc_capable: bool = False
    ramp_agc: NumericQuantity = None
    p_min_agc: NumericQuantity = None
    p_max_agc: NumericQuantity = None
    agc_fixed_cost: NumericQuantity = None
    agc_marginal_cost: NumericQuantity = None
    spinning_cost: NumericQuantity = None
    spinning_capacity: NumericQuantity = None
    non_spinning_cost: NumericQuantity = None
    non_spinning_capacity: NumericQuantity = None
    supplemental_cost: NumericQuantity = None
    supplemental_spinning_capacity: NumericQuantity = None
    supplemental_non_spinning_capacity: NumericQuantity = None
    reg_provider: Union[float, TimeSeries] = None
    reg_up_supplied: Union[float, TimeSeries] = None
    reg_down_supplied: Union[float, TimeSeries] = None
    spinning_supplied: Union[float, TimeSeries] = None
    flex_up_supplied: Union[float, TimeSeries] = None
    flex_down_supplied: Union[float, TimeSeries] = None
    non_spinning_supplied: Union[float, TimeSeries] = None
    supplemental_supplied: Union[float, TimeSeries] = None
    fixed_commitment: Union[int, None, TimeSeries] = None
    fixed_regulation: Union[int, None, TimeSeries] = None
    commitment: Union[float, TimeSeries] = None
    commitment_cost: Union[float, TimeSeries] = None
    production_cost: Union[float, TimeSeries] = None
    fuel_supply: str = ''

    @validator('initial_status')
    def initial_status_cannot_be_zero(cls, v):
        if math.isclose(0, v.value):
            raise ValueError('The number of time periods the generator has been or off initially cannot be zero.')

        return v
    
    @validator('fixed_commitment')
    def fixed_commitment_binary_value(cls, v):
        if v:
            if isinstance(v, int) not in {0, 1}:
                raise ValueError('The fixed commitment status of the generator must be 0, 1, or None.')
            elif isinstance(v, TimeSeries):
                fixed_commitment_status_series = v.values

                if not all([uc_status in {0, 1, None} for uc_status in fixed_commitment_status_series.values()]):
                    raise ValueError('The fixed commitment status of the generator must be 0, 1, or None.')

        return v
    
    @validator('fixed_regulation')
    def fixed_regulation_binary_value(cls, v):
        if v:
            if isinstance(v, int) not in {0, 1}:
                raise ValueError('The fixed regulation status of the generator must be 0, 1, or None.')
            elif isinstance(v, TimeSeries):
                fixed_regulation_status_series = v.values

                if not all([uc_status in {0, 1, None} for uc_status in fixed_regulation_status_series.values()]):
                    raise ValueError('The fixed regulation status of the generator must be 0, 1, or None.')

        return v
    
    # TODO: Implement more data validation...


class Shunt(BaseModel):
    """Base class for shunt objects."""
    bus: Union[int, str]
    bs: NumericQuantity
    gs: NumericQuantity


class ControllableShunt(Shunt):
    """Shunt with controllable susceptance and conductance."""
    bs_min: NumericQuantity
    bs_max: NumericQuantity
    gs_min: NumericQuantity
    gs_max: NumericQuantity
    step_count: Union[int, float]


class FixedShunt(Shunt):
    """Shunt with fixed susceptance and conductance."""
    pass


class Region(BaseModel):
    """Base class for region-like objects."""
    spinning_reserve_requirement: Union[float, TimeSeries] = None
    non_spinning_reserve_requirement: Union[float, TimeSeries] = None
    regulation_up_requirement: Union[float, TimeSeries] = None
    regulation_down_requirement: Union[float, TimeSeries] = None
    flexible_ramp_up_requirement: Union[float, TimeSeries] = None
    flexible_ramp_down_requirement: Union[float, TimeSeries] = None
    supplemental_reserve_requirement: Union[float, TimeSeries] = None

    spinning_reserve_shortfall: Union[float, TimeSeries] = None
    non_spinning_reserve_shortfall: Union[float, TimeSeries] = None
    regulation_up_shortfall: Union[float, TimeSeries] = None
    regulation_down_shortfall: Union[float, TimeSeries] = None
    flexible_ramp_up_shortfall: Union[float, TimeSeries] = None
    flexible_ramp_down_shortfall: Union[float, TimeSeries] = None
    supplemental_shortfall: Union[float, TimeSeries] = None

    spinning_reserve_price: Union[float, TimeSeries] = None
    non_spinning_reserve_price: Union[float, TimeSeries] = None
    regulation_up_price: Union[float, TimeSeries] = None
    regulation_down_price: Union[float, TimeSeries] = None
    flexible_ramp_up_price: Union[float, TimeSeries] = None
    flexible_ramp_down_price: Union[float, TimeSeries] = None
    supplemental_price: Union[float, TimeSeries] = None


class Area(Region):
    """Area object."""
    pass


class Zone(Region):
    """Zone object."""
    pass


class Interface(BaseModel):
    """Interface object."""
    interface_lines: List[Line]
    interface_from_limit: NumericQuantity = None
    interface_to_limit: NumericQuantity = None
    pf: Union[float, TimeSeries] = None
    qf: Union[float, TimeSeries] = None
    pt: Union[float, TimeSeries] = None
    qt: Union[float, TimeSeries] = None


class Owner(BaseModel):
    """Owner object."""
    pass


class Storage(BaseModel):
    """Storage object."""
    energy_capacity: NumericQuantity
    initial_state_of_charge: float
    minimum_state_of_charge: float
    charge_efficiency: float
    discharge_efficiency: float
    max_discharge_rate: NumericQuantity
    min_discharge_rate: NumericQuantity
    max_charge_rate: NumericQuantity
    min_charge_rate: NumericQuantity
    initial_charge_rate: NumericQuantity
    initial_discharge_rate: NumericQuantity
    
    charge_cost: NumericQuantity = None
    discharge_cost: NumericQuantity = None

    retention_rate_60min: float
    ramp_up_input_60min: NumericQuantity
    ramp_down_input_60min: NumericQuantity
    ramp_up_output_60min: NumericQuantity
    ramp_down_output_60min: NumericQuantity

    in_service: bool = False
    state_of_charge: Union[float, TimeSeries] = None
    p_discharge: Union[float, TimeSeries] = None
    p_charge: Union[float, TimeSeries] = None
    operational_cost: Union[float, TimeSeries] = None

    bus: Union[int, str]
    area: Union[int, str] = None
    owner: Union[int, str] = None
    zone: Union[int, str] = None

    @validator('initial_discharge_rate')
    def initial_discharge_rate_exclusivity(cls, v, values, **kwargs):
        """Enforces that the initial_charge_rate and initial_discharge_rate cannot simultaneously be zero."""
        if 'initial_charge_rate' in values:
            # Should be always True if initial_charge_rate is required.
            initial_charge_rate = values['initial_charge_rate'].value
        else:
            return v

        initial_discharge_rate = v.value

        if not math.isclose(0, initial_discharge_rate) and not math.isclose(0, initial_charge_rate):
            raise ValueError('The initial charge rate and initial discharge rate cannot be simultaneously non-zero.')

        return v


class ElementsDataModel(BaseModel):
    """Element data for unit commitment model."""
    bus: Dict[str, Bus]
    load: Dict[str, Load]
    line: Dict[str, Line] = None
    transformer: Dict[str, Transformer] = None
    interface : Dict[str, Interface] = None
    zone: Dict[str, Zone] = None
    area: Dict[str, Area] = None
    owner: Dict[str, Owner] = None
    thermal_generator: Dict[str, ThermalGenerator] = None
    renewable_generator: Dict[str, RenewableGenerator] = None
    fixed_shunt: Dict[str, FixedShunt] = None
    controllable_shunt: Dict[str, ControllableShunt] = None
    storage : Dict[str, Storage] = None
    fuel_supply: Dict[str, FuelSupply] = None


class SystemDataModel(BaseModel):
    """System data for unit commitment model."""
    time_indices : List[Union[int, datetime]] = None

    time_period_length_minutes : NumericQuantity

    baseMVA             : NumericQuantity
    reference_bus       : str = None
    reference_bus_angle : float = None

    load_mismatch_cost     : NumericQuantity = None
    reserve_shortfall_cost : NumericQuantity = None
    total_cost             : NumericQuantity = None

    reserve_requirement              : TimeSeries = None
    spinning_reserve_requirement     : TimeSeries = None
    non_spinning_reserve_requirement : TimeSeries = None
    regulation_up_requirement        : TimeSeries = None
    regulation_down_requirement      : TimeSeries = None
    flexible_ramp_up_requirement     : TimeSeries = None
    flexible_ramp_down_requirement   : TimeSeries = None
    supplmental_reserve_requirement  : TimeSeries = None

    reserve_shortfall                : TimeSeries = None
    spinning_reserve_shortfall       : TimeSeries = None
    non_spinning_reserve_shortfall   : TimeSeries = None
    regulation_up_shortfall          : TimeSeries = None
    regulation_down_shortfall        : TimeSeries = None
    flexible_ramp_up_shortfall       : TimeSeries = None
    flexible_ramp_down_shortfall     : TimeSeries = None
    supplemental_shortfall           : TimeSeries = None

    @validator('time_period_length_minutes')
    def time_period_length_minutes_units(cls, v):
        if v.units != 'min':
            raise ValueError('time_period_length_minutes must have units of min.')
        
        return v
    
    @validator('load_mismatch_cost', 'reserve_shortfall_cost', 'total_cost')
    def cost_units(cls, v, field):
        if v.units != '$':
            raise ValueError('{0} must have units of $.'.format(field.name))

        return v

    @validator('baseMVA')
    def baseMVA_units(cls, v):
        if v.units != 'MVA':
            raise ValueError('baseMVA must have units of MVA.')

        return v

    
class UnitCommitmentDataModel(BaseModel):
    """"""
    elements : ElementsDataModel
    system   : SystemDataModel


if __name__ == '__main__':
    print("\n\nSCHEMA IS...")
    print(UnitCommitmentDataModel.schema_json(indent=2))

    with open('unit_commitment_data_model_schema.json', 'w') as f:
        print(UnitCommitmentDataModel.schema_json(indent=2), file=f)

    input_json_file_name = os.path.join(current_dir, 'tests', 'uc_test_instances', 'schema_demo_tiny_uc_2.json')

    # Read in JSON file for test case.
    with open(input_json_file_name, 'r') as f:
        input_json = json.load(f)

    # Validate JSON file against UnitCommitmentDataModel schema.
    print("\n\nCALLING VALIDATOR...")
    try:
        uc_data = UnitCommitmentDataModel(
            elements = input_json["elements"],
            system = input_json["system"]
            )
    except ValidationError as e:
        print(e)
    else:
        print("SUCCESS!")
    
    # Pass the validated model object (back as dictionary) to the egret ModelData parser.
    from egret.data.model_data import ModelData
    from egret.models.unit_commitment import solve_unit_commitment, create_tight_unit_commitment_model

    md = ModelData(uc_data.dict())
    print(md)

    # solved_md = solve_unit_commitment(md,
    #                         'cbc',
    #                         mipgap = 0.001,
    #                         timelimit = None,
    #                         solver_tee = True,
    #                         symbolic_solver_labels = False,
    #                         options = None,
    #                         uc_model_generator=create_tight_unit_commitment_model,
    #                         relaxed=False,
    #                         return_model=False)

    # TODO: The NumericQuantity and TimeSeries objects need to be parsed correctly by downstream egret functions. 
    # (Need to get the value(s) field now.)
    # Some others like CostCurve may need adjustment as well.
    # e.g., 
    # ModelData.data['system']['baseMVA'] ==> ModelData.data['system']['baseMVA']['value']
    # Actually, I think the TimeSeries-like objects might be OK since the "values" fields are functionally identical.
