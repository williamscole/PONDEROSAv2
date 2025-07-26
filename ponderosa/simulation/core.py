"""
Script for executing the simulation
"""

from simulate import simulate
from config import SimulationConfig
from ..core import run_ponderosa

def run_simulation(config: SimulationConfig):

    ponderosa_config = simulate(config)

    run_ponderosa(ponderosa_config)