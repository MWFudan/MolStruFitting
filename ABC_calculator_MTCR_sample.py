import numpy as np
import re
from filter_coor_file_with_atom_distance import verify_coor_text
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
from scipy import optimize
from scipy.stats import qmc
from tqdm import tqdm
import datetime


# http://www.nist.gov/pml/data/comp.cfm
mass_table = {
    'H': 1.00782503207,
    'D': 2.0141017778,
    '2H': 2.0141017778,

    'He': 4.0026032541,

    'C': 12.0,
    '13C': 13.0033548378,

    'O': 15.99491461956,
    '16O': 15.99491461956,
    '18O': 17.9991610,

    'F': 18.99840316,

    'P': 30.973761998,

    'S': 31.97207100,
    '34S': 33.96786690,

    'Cl': 34.96885268,
    '35Cl': 34.96885268,
    '37Cl': 36.96590259
}
re_coor = re.compile(r'\s*(\d{0,3}[A-Z][a-z]?)\s+(\S+)\s+(\S+)\s+(\S+)\s*')
re_double_end_line = re.compile(r'\n\s*\n')
# re_file_split = re.compile(r'\n\s*\d+\n')  # crest
re_file_split = re_double_end_line

error_percent_discard_threshold = 0.01
ftol = 1E-6  # ftol
jac = '2-point'
rot_method = 'euler'  # euler, quat
sample_space_trans = 8.0
sample_space_rot = 360.0
process_num = 12
save_batch_num = 100

add_transform_array_info_flag = False


class Molecule:
    # must translate first then rotate
    @staticmethod
    def from_text(text, is_active=True):
        mass, coor = text_converter(text)
        return Molecule(mass, coor, 0, is_active)

    # def __init__(self, mass, coor, center_idx, active_flag):
    #     self.mass = mass
    #     self.coor = self.coor0 = coor
    #     self.center_idx = center_idx
    #     self.v0 = coor - coor[center_idx]  # vector from center to other atom, initial value
    #     self.active_flag = active_flag
    #
    # def translate(self, x):
    #     self.coor = self.coor0 + x
    #
    # def rotate(self, q):
    #     if len(q) == 4:
    #         r = Rotation.from_quat(q)
    #     else:
    #         r = Rotation.from_euler('xyz', q)
    #
    #     new_v = r.apply(self.v0)
    #     center_coor = self.coor[self.center_idx]
    #     self.coor = new_v + center_coor
    def __init__(self, mass, coor, _, active_flag):
        self.mass = mass
        self.coor = self.coor0 = coor
        mass_sum = np.sum(mass)
        self.mass_center = self.mass_center0 = np.sum(mass.reshape(-1, 1) * coor, 0) / mass_sum
        self.v0 = coor - self.mass_center0
        self.active_flag = active_flag

    def translate(self, x):
        self.coor = self.coor0 + x
        self.mass_center = self.mass_center0 + x

    def rotate(self, q):
        if len(q) == 4:
            r = Rotation.from_quat(q)
        else:
            r = Rotation.from_euler('xyz', q)

        new_v = r.apply(self.v0)
        self.coor = new_v + self.mass_center


def get_mass_and_coor_vector(motion_inf_list, molecules, uncertainty_analysis_flag=False):
    mass = np.array([])
    coor = np.empty([0, 3])
    rot_param_num = 4 if len(motion_inf_list) % 7 == 0 else 3
    n = 0
    for molecule in molecules:
        if molecule.active_flag:
            motion_trans = motion_inf_list[n:n + 3]
            motion_rot = motion_inf_list[n + 3:n + rot_param_num + 3]
            n += rot_param_num + 3
            molecule.translate(motion_trans)
            molecule.rotate(motion_rot)
        mass = np.append(mass, molecule.mass)
        coor = np.append(coor, molecule.coor, axis=0)
    principle_axis_coor = convert_to_main_axis(mass, coor)
    if uncertainty_analysis_flag:
        return principle_axis_coor.reshape(-1)
    return mass, principle_axis_coor


class RandomEngine(qmc.QMCEngine):
    def __init__(self, d, seed=None):
        super().__init__(d=d, seed=seed)

    def random(self, n=1):
        self.num_generated += n
        return self.rng.random((n, self.d))

    def reset(self):
        super().__init__(d=self.d, seed=self.rng_seed)
        return self

    def fast_forward(self, n):
        self.random(n)
        return self
def optimize_coor_Quasi_Monte_Carlo_sample(coor_text, target_ABC, sample_space=10, loop_num=100, acceptable_error=10, save_file_path=None, in_time_save=True):
    # coor_text: molecule should be separated by two \n
    # X ...
    # X ...
    #
    # X ...
    # first molecule is not active
    molecules = re.split(re_double_end_line, coor_text)
    molecule_list = []
    flag_first_molecule = True
    for molecule in molecules:
        if not molecule:
            continue

        molecule_list.append(Molecule.from_text(molecule, is_active=not flag_first_molecule))
        if flag_first_molecule:
            flag_first_molecule = False

    engine = RandomEngine(6)
    # can also use [0, 0, 0, 0, 0, 0, 1] as quat, seems no diff
    output_list = []
    for i in range(loop_num):
        random_input = engine.random(len(molecule_list) - 1)
        norm_input = ((random_input - 0.5) * np.array([sample_space, sample_space, sample_space, 360, 360, 360])).reshape(-1)
        if (error := calc_error(norm_input, target_ABC, molecule_list)) <= acceptable_error:
            output_list.append([norm_input, error])
            if in_time_save:
                output_list.sort(key=lambda x: x[1])
                with open(save_file_path, 'w') as f:
                    for j in output_list:
                        f.write('{:g}\n{}\n\n'.format(j[1], coordination_generator(j[0], molecule_list)))

    if not output_list:
        print('no result')
        return

    if not in_time_save:
        output_list.sort(key=lambda x: x[1])
        if not save_file_path:
            for i in output_list[:20]:
                print('{:g}\n{}\n\n'.format(i[1], coordination_generator(i[0], molecule_list)))
            return

        with open(save_file_path, 'w') as f:
            for i in output_list:
                f.write('{:g}\n{}\n\n'.format(i[1], coordination_generator(i[0], molecule_list)))


def optimize_coor(coor_text, target_ABC, ignore_error=False, MTCR=False, idx=-1, random_input=None, sep_idx=None):
    # coor_text: molecule should be separated by two \n
    # X ...
    # X ...
    #
    # X ...
    # first molecule is not active
    molecules = re.split(re_double_end_line, coor_text)
    molecule_list = []
    flag_first_molecule = True
    for molecule in molecules:
        if not molecule:
            continue

        molecule_list.append(Molecule.from_text(molecule, is_active=not flag_first_molecule))
        if flag_first_molecule:
            flag_first_molecule = False

    if MTCR:
        if rot_method == 'quat':
            initial_input = (
                    (random_input - 0.5) * np.array([sample_space, sample_space, sample_space, 1, 1, 1, 1])).reshape(
                -1)
        elif rot_method == 'euler':
            initial_input = (
                    (random_input - 0.5) * np.array([sample_space_trans, sample_space_trans, sample_space_trans, sample_space_rot, sample_space_rot, sample_space_rot])).reshape(-1)
        else:
            raise ValueError('bad rot method')
    else:
        # can also use [0, 0, 0, 0, 0, 0, 1] as quat, seems no diff
        if rot_method == 'quat':
            initial_input = [0, 0, 0, 0, 0, 0, 1] * (len(molecule_list) - 1)
        elif rot_method == 'euler':
            initial_input = [0, 0, 0, 0, 0, 0] * (len(molecule_list) - 1)
        else:
            raise ValueError('bad rot method')
    # least square
    if not verify_coor_text(coordination_generator(initial_input, molecule_list), sid=sep_idx):
        # print('pri not passed')
        return
    result = optimize.least_squares(calc_error, initial_input, args=(target_ABC, molecule_list), jac=jac, ftol=ftol)
    if ignore_error:
        opt_coor = coordination_generator(result.x, molecule_list)
        if result.success:
            if add_transform_array_info_flag:
                return opt_coor, '\n' + ','.join(map('{:.2e}'.format, result.x)), idx
            else:
                return opt_coor, '', idx,
        else:
            return opt_coor, result.message, idx
    else:
        print(result.message)
        print(result.fun)
        opt_coor = coordination_generator(result.x, molecule_list)

        derivative = optimize.approx_fprime(result.x, get_mass_and_coor_vector, np.sqrt(np.finfo(float).eps), molecule_list, True)
        uncertainty_for_coor = []
        for i in derivative:
            uc = np.sqrt(np.sum((i * result.fun)**2))
            uncertainty_for_coor.append(uc)
        uncertainty_for_coor = np.array(uncertainty_for_coor).reshape(-1, 3)

        print_line = []
        for i in uncertainty_for_coor:
            print_line.append('{:>14.8f}\t{:>14.8f}\t{:>14.8f}'.format(*i))

        return opt_coor, '\n'.join(print_line)


def calc_error(motion_inf_list, target_ABC, molecules):
    # motion_inf_list = [*7,*7,..]
    mass, coor = get_mass_and_coor_vector(motion_inf_list, molecules)

    ABC = calc_ABC(mass, coor)
    # scale of error effect gtol, large get more precise result, but slow
    error = (target_ABC - ABC) / target_ABC * 1000
    # error = target_ABC - ABC
    error = np.sum(error**2)
    return error

def calc_ABC(mass, coor):
    # coor is array,[[][][]]
    mass_sum = np.sum(mass)
    mass_center = np.sum(mass.reshape(-1, 1) * coor, 0) / mass_sum
    coor_centered = coor - mass_center

    Ixx = np.sum(mass * (coor_centered[:, 1] ** 2 + coor_centered[:, 2] ** 2))
    Iyy = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 2] ** 2))
    Izz = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 1] ** 2))
    Ixy = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 1])
    Ixz = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 2])
    Iyz = -np.sum(mass * coor_centered[:, 1] * coor_centered[:, 2])

    moment_inertial = np.linalg.eigvals(np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]))
    ABC_in_MHz = 505379.6963277365/np.array(moment_inertial)
    sorted_ABC = -np.sort(-ABC_in_MHz)

    return sorted_ABC

def text_converter(text):
    result = re.findall(re_coor, text)
    if not result:
        return
    mass = []
    coor = []
    for i in result:
        mass.append(mass_table[i[0]])
        coor.append(list(map(float, i[1:])))
    return np.array(mass), np.array(coor)


def coordination_generator(motion_inf_list, molecules):
    # to principal axis is done in get_mass_and_coor_vector
    mass, coor = get_mass_and_coor_vector(motion_inf_list, molecules)
    # principle_axis_coor = convert_to_main_axis(mass, coor)
    lines = []
    for m, c in zip(mass, coor):
        lines.append('{}\t{:>14.8f}\t{:>14.8f}\t{:>14.8f}'.format(convert_mass_to_atom_str(m), *c))
    return '\n'.join(lines)


def convert_mass_to_atom_str(mass):
    for key, val in mass_table.items():
        if val == mass:
            return key


def convert_to_main_axis(mass, coor):
    mass_sum = np.sum(mass)
    mass_center = np.sum(mass.reshape(-1, 1) * coor, 0) / mass_sum
    coor_centered = coor - mass_center

    Ixx = np.sum(mass * (coor_centered[:, 1] ** 2 + coor_centered[:, 2] ** 2))
    Iyy = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 2] ** 2))
    Izz = np.sum(mass * (coor_centered[:, 0] ** 2 + coor_centered[:, 1] ** 2))
    Ixy = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 1])
    Ixz = -np.sum(mass * coor_centered[:, 0] * coor_centered[:, 2])
    Iyz = -np.sum(mass * coor_centered[:, 1] * coor_centered[:, 2])

    w, v = np.linalg.eigh(np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]))

    for i in range(0, len(mass)):
        coor_centered[i] = np.matmul(v.T, coor_centered[i])

    return coor_centered

def optimize_one_done(future):
    global accu_num
    p_bar.update(1)
    if not future.result():
        return
    result, msg, idx = future.result()
    error = target_ABC - calc_ABC(*text_converter(result))
    if (not verify_coor_text(result, sid=sep_idx)) or (any(np.abs(error / target_ABC) > error_percent_discard_threshold)):
        # print('opt not passed')
        return
    sum_result.append((np.sum(error ** 2), msg, result))
    # print('passed')

    if len(sum_result) > save_batch_num:
        with open(save_fp, 'a') as f:
            for n, i in enumerate(sum_result):
                f.write('%s\n%.3e\t%s\n%s\n\n' % (accu_num+n, *i))
        accu_num += len(sum_result)
        sum_result.clear()
    # print(accu_num, error)

if __name__ == '__main__':
    target_ABC = np.array([1279.94604, 479.35125, 440.77197])
    t = '''C    -3.017483000000      0.498314000000      1.411246000000
C    -3.131057000000      0.433686000000      0.050862000000
S    -1.647857000000      0.817644000000     -0.722589000000
C    -0.858792000000      1.070554000000      0.780555000000
C    -1.708309000000      0.867754000000      1.829831000000
H    -3.835621000000      0.294147000000      2.087194000000
H    -3.999714000000      0.177316000000     -0.533293000000
H     0.185291000000      1.332607000000      0.807764000000
H    -1.410341000000      0.968505000000      2.863497000000

C     1.646544000000      0.009297000000     -1.895725000000
C     2.402502000000      0.778351000000     -1.059597000000
S     2.807378000000     -0.069745000000      0.379211000000
C     1.960323000000     -1.471741000000     -0.137457000000
C     1.394937000000     -1.287399000000     -1.366568000000
H     1.285541000000      0.350799000000     -2.854834000000
H     2.743313000000      1.789628000000     -1.211506000000
H     1.926374000000     -2.347328000000      0.490444000000
H     0.823010000000     -2.048198000000     -1.878480000000

O    -1.076167000000     -2.426866000000      0.879897000000
H    -0.510893000000     -1.881278000000      0.319742000000
H    -1.727930000000     -1.806813000000      1.228032000000
'''
    # x = calc_ABC(*text_converter(t))
    # x = [round(q, 6) for q in x]
    # print(', '.join(map(str, x)))
    # print('\n'.join(map(str, x)))
    # print(target_ABC - x, (target_ABC-x)/target_ABC * 100)
    # print(target_ABC - x, ['%.3e' % q for q in (target_ABC-x)/target_ABC])
    #
    # opt_coor, unc = optimize_coor(t, target_ABC, ignore_error=False)
    # print(opt_coor)
    # print(unc)
    # x = calc_ABC(*text_converter(opt_coor))
    # print(target_ABC - x, (target_ABC - x) / target_ABC * 100)
    #
    # optimize_coor_Quasi_Monte_Carlo_sample(t, target_ABC, loop_num=576000, acceptable_error=10000, save_file_path=r"D:\MW\data\程序优化结构测试\水六聚体\opt_qmc.xyz")
    # exit()
    save_fp = r'D:\MW\data\程序优化结构测试\苯甲醛-水\MTCR\2w_mtcr.xyz'
    loop_num = 1000000
    with open(save_fp, 'w') as f:
        f.write(f'sample space: {sample_space_trans}-{sample_space_rot}\nftol: {ftol}\njac: {jac}\nTarget ABC: {target_ABC}\nLoop number: {loop_num}\n\n')

    sep_idx = (9, 18)
    molecules = re.split(re_double_end_line, t)
    molecule_num = 0
    for molecule in molecules:
        if not molecule:
            continue
        molecule_num += 1

    if rot_method == 'quat':
        engine = RandomEngine(7)
    elif rot_method == 'euler':
        engine = RandomEngine(6)
    else:
        raise Exception()

    sum_result = []
    accu_num = 0
    pool = ProcessPoolExecutor(process_num)
    # can also use [0, 0, 0, 0, 0, 0, 1] as quat, seems no diff
    p_bar = tqdm(total=loop_num)
    loop_time_start = datetime.datetime.now()
    try:
        n = 0
        for i in range(loop_num):
            pool.submit(optimize_coor, t, target_ABC, True, True, n, engine.random(molecule_num - 1), sep_idx).add_done_callback(optimize_one_done)
            n += 1
    except KeyboardInterrupt:
        print('exiting')
    pool.shutdown(wait=True)
    loop_time_end = datetime.datetime.now()
    with open(save_fp, 'a') as f:
        if sum_result:
            for n, i in enumerate(sum_result):
                f.write('%s\n%.3e\t%s\n%s\n\n' % (accu_num + n, *i))
        f.write(f'Time consumed: {loop_time_end-loop_time_start}')

    p_bar.close()

