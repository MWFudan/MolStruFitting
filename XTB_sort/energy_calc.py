from xtb.libxtb import VERBOSITY_MUTED
from xtb.interface import Calculator, Param, XTBException
from ase import Atoms
from ase.calculators.emt import EMT
import os



def calculate_energy_xtb(numbers, positions):
    positions *= 1.8897259886
    calc = Calculator(Param.GFN1xTB, numbers, positions)
    calc.set_verbosity(VERBOSITY_MUTED)
    try:
        res = calc.singlepoint()  # energy printed is only the electronic part
    except XTBException:
        return 999.999
    return res.get_energy()



def calculate_energy_ase(numbers, positions):


    atom = Atoms(numbers=numbers, positions=positions)
    atom.calc = EMT()
    energy = atom.get_potential_energy()

    return energy
