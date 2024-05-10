import numpy as np
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from filter_coor_file_with_atom_distance import verify_coor_text
from scipy.spatial.transform import Rotation
from scipy import optimize
from tqdm import tqdm

# http://www.nist.gov/pml/data/comp.cfm
mass_table = {
    'H': 1.00782503207,
    'D': 2.0141017778,
    '2H': 2.0141017778,

    'He': 4.0026032541,

    'C': 12.0,
    '13C': 13.0033548378,

    'N': 14.003074004,

    'O': 15.99491461956,
    '16O': 15.99491461956,
    '18O': 17.9991610,

    'F': 18.99840316,

    'Ne': 19.99244017,
    '20Ne': 19.99244017,
    '22Ne': 21.9913851,

    'P': 30.973761998,

    'S': 31.97207100,
    '34S': 33.96786690,

    'Cl': 34.96885268,
    '35Cl': 34.96885268,
    '37Cl': 36.96590259,

    '63Cu': 62.92959772,
    '65Cu': 64.92778970,
}
re_coor = re.compile(r'\s*(\d{0,3}[A-Z][a-z]?)\s+(\S+)\s+(\S+)\s+(\S+)\s*')
re_double_end_line = re.compile(r'\n\s*\n')
# re_file_split = re.compile(r'\s{2}\d{1,3}\n')  # crest
re_file_split = re_double_end_line
# re_file_split = re.compile(r'\s+\d{1,3}\n')

process_num = 15
error_percent_discard_threshold = 0.1
ftol = 1E-6  # ftol
jac = '2-point'
rot_method = 'euler'  # euler, quat
not_rot_flag = False
rot_to_trans_scale = 1
algorithm = 'trf'
max_iter_num = 300  # make it larger if max cycles exceeded

class Molecule:
    # must translate first then rotate
    @staticmethod
    def from_text(text, is_active=True):
        mass, coor = text_converter(text)
        return Molecule(mass, coor, 0, is_active)

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
    rot_param_num = 3 if len(motion_inf_list) % 6 == 0 else 4
    n = 0
    for molecule in molecules:
        if molecule.active_flag:
            motion_trans = motion_inf_list[n:n + 3]
            if not not_rot_flag:
                motion_rot = motion_inf_list[n + 3:n + rot_param_num + 3]
                n += rot_param_num + 3
                molecule.translate(motion_trans)
                molecule.rotate(motion_rot)
            else:
                n += 3
                molecule.translate(motion_trans)

        mass = np.append(mass, molecule.mass)
        coor = np.append(coor, molecule.coor, axis=0)
    principle_axis_coor = convert_to_main_axis(mass, coor)
    if uncertainty_analysis_flag:
        return principle_axis_coor.reshape(-1)
    return mass, principle_axis_coor


def optimize_coor(coor_text, target_ABC, k, isotope_target_ABC=None):
    # coor_text: molecule should be separated by two \n
    # X ...
    # X ...
    #
    # X ...

    # isotope_target_ABC: [(atom index, isotope type, A, B, C), ...]
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

    # can also use [0, 0, 0, 0, 0, 0, 1] as quat, seems no diff
    if rot_method == 'quat':
        initial_input = [0, 0, 0, 0, 0, 0, 1] * (len(molecule_list) - 1)
    elif rot_method == 'euler':
        if not_rot_flag:
            initial_input = [0, 0, 0] * (len(molecule_list) - 1)
            xscale = [1, 1, 1] * (len(molecule_list) - 1)
        else:
            initial_input = [0, 0, 0, 0, 0, 0] * (len(molecule_list) - 1)
            xscale = [1, 1, 1, rot_to_trans_scale, rot_to_trans_scale, rot_to_trans_scale] * (len(molecule_list) - 1)
            xscale = np.array(xscale)
    else:
        raise ValueError('bad rot method')
    # least square
    result = optimize.least_squares(calc_error, initial_input, args=(target_ABC, molecule_list, isotope_target_ABC),
                                    method=algorithm, jac=jac, ftol=ftol, max_nfev=max_iter_num*len(initial_input), x_scale=xscale)

    opt_coor = coordination_generator(result.x, molecule_list)
    # msg = '' if result.success else result.message
    msg = result.message
    return opt_coor, msg, k



def optimize_one_done(future):
    global sep_idx
    result, msg, k = future.result()
    error = target_ABC - calc_ABC(*text_converter(result))
    p_bar.update(1)
    if any(np.abs(error / target_ABC) > error_percent_discard_threshold) or not verify_coor_text(result, sid=sep_idx):
        # print(error, 'discard')
        return
    sum_result[k] = [error, result, msg]
    # print(k, error)


sum_result = {}
p_bar = None
def optimize_coor_file_list(fp, sep_idx, target_ABC, save_fp=None, count=0, isotope_target_ABC=None):
    global p_bar
    with open(fp) as f:
        t = f.read()

    title_repeat_num = 1
    coor_dict = {}
    for i in re.split(re_file_split, t):
        if not i:
            continue
        result = re.findall(re_coor, i)
        if not result:
            continue
        result = ['\t'.join(x) for x in result]
        for n, idx in enumerate(sep_idx):
            result.insert(idx+n, '')
        title = '[%s]' % i.split('\n')[0]
        if coor_dict.get(title):
            title += str(title_repeat_num)
            title_repeat_num += 1
        coor_dict[title] = '\n'.join(result)
        if count:
            if len(coor_dict) == count:
                break

    if not save_fp:
        save_fp = fp + save_suffix

    t0 = datetime.now()
    pool = ProcessPoolExecutor(process_num)

    p_bar = tqdm(total=len(coor_dict))
    for k, v in coor_dict.items():
        pool.submit(optimize_coor, v, target_ABC, k, isotope_target_ABC).add_done_callback(optimize_one_done)
    pool.shutdown(wait=True)
    p_bar.close()
    t1 = datetime.now()
    consumed_time = t1 - t0

    if sum_result:
        with open(save_fp, 'w', encoding='utf-8') as f:
            f.write(
                f'Input file: {file_path}\nRotation method: {rot_method}\nftol: {ftol}\njac: {jac}\nTarget ABC: {target_ABC}\nTime used: {str(consumed_time)}\n\n')
            for k, v in sum_result.items():
                error, result, msg = v
                f.write('%s\n%.3e\t%s\n%s\n\n' % (k, np.sum(error ** 2), msg, result))
    else:
        print('no result')

    # plt.plot([x[0] for x in sum_result.values()])
    # plt.show()


def calc_error(motion_inf_list, target_ABC, molecules, isotope_target_ABC_list=None):
    # motion_inf_list = [*7,*7,..]
    mass, coor = get_mass_and_coor_vector(motion_inf_list, molecules)
    ABC = calc_ABC(mass, coor)
    # scale of error effect gtol, large get more precise result, but slow
    ABC_error = (target_ABC - ABC) / target_ABC * 1000
    ABC_error = np.sum(ABC_error**2)

    iso_ABC_error = 0
    if isotope_target_ABC_list:
        for i in isotope_target_ABC_list:
            idx, iso_atom, *iso_tar_ABC = i
            ori_mass = mass[idx]

            mass[idx] = mass_table[iso_atom]
            iso_ABC = calc_ABC(mass, coor)
            single_iso_ABC_error = (iso_tar_ABC - iso_ABC) / iso_tar_ABC * 1000
            iso_ABC_error += np.sum(single_iso_ABC_error**2) * 1000

            mass[idx] = ori_mass

    error = ABC_error + iso_ABC_error
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
    lines = []
    for m, c in zip(mass, coor):
        lines.append('{}\t{:>11.8f}\t{:>11.8f}\t{:>11.8f}'.format(convert_mass_to_atom_str(m), *c))
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


if __name__ == '__main__':
    # experimental rotational constant of parent specie
    target_ABC = np.array([939.319, 584.097, 540.120])

    # ---------optimization of one coordinate---------
    # molecules should be separated by two line breaks
    text = '''C	-2.31625571	 1.47546368	 0.75444586
C	-1.77748083	 1.69052505	-0.50890374
C	-2.43245660	 0.18009941	 1.24648775
H	-2.64144776	 2.31478682	 1.35521933
H	-1.67677530	 2.69762615	-0.89258804
C	-1.35582380	 0.61904810	-1.28624769
H	-2.84494883	 0.00741849	 2.23213997
C	-2.01649937	-0.90278847	 0.48140704
H	-0.92314060	 0.76031252	-2.26665959
C	-1.48755545	-0.65719465	-0.77217668
H	-2.08519123	-1.91822639	 0.84628348
F	-1.07966209	-1.70565479	-1.52215872

C	 2.32068645	 1.45798984	-0.76602260
C	 2.40768001	 0.16058625	-1.25867027
C	 1.80795570	 1.68267473	 0.50647884
H	 2.64814781	 2.29163685	-1.37342819
H	 2.80029387	-0.01929141	-2.25112783
C	 1.98837975	-0.91495128	-0.48508496
H	 1.73017740	 2.69152454	 0.89083261
C	 1.38345071	 0.61869842	 1.29246655
H	 2.03466136	-1.93188983	-0.84931477
C	 1.48577129	-0.65983846	 0.77739122
H	 0.97000164	 0.76768946	 2.27998745
F	 1.07407760	-1.70122578	 1.53531897'''
    # in the format: (index, atom_type, A, B, C)
    # index starts from 0
    iso_abc = [
        (0, '13C', 934.653, 580.131, 535.823),
        (1, '13C', 933.993, 581.836, 536.660),
        (2, '13C', 936.594, 579.143, 536.751),
        (5, '13C', 935.819, 581.811, 538.820),
        (7, '13C', 937.499, 581.274, 537.358),
        (9, '13C', 937.570, 582.257, 538.582),
        (12, '13C', 934.653, 580.131, 535.823),
        (14, '13C', 933.993, 581.836, 536.660),
        (13, '13C', 936.594, 579.143, 536.751),
        (19, '13C', 935.819, 581.811, 538.820),
        (17, '13C', 937.499, 581.274, 537.358),
        (21, '13C', 937.570, 582.257, 538.582),
    ]
    result = optimize_coor(text, target_ABC, 0, isotope_target_ABC=iso_abc)
    print(result[0])
    print(result[1])
    print('new error:', target_ABC-calc_ABC(*text_converter(result[0])), (target_ABC-calc_ABC(*text_converter(result[0]))) / target_ABC)
    mass_in_array, coor_in_array = text_converter(result[0])
    for iso_idx, iso_atom, *iso_exp_rot_cons in iso_abc:
        ori_mass = mass_in_array[iso_idx]
        mass_in_array[iso_idx] = mass_table[iso_atom]
        iso_rot_cons = calc_ABC(mass_in_array, coor_in_array)
        print(f'----\n{iso_idx}-{iso_atom}:\n')
        iso_ABC_error = iso_exp_rot_cons - iso_rot_cons
        print(f'iso_error: {iso_ABC_error}; relative_error: {iso_ABC_error/iso_exp_rot_cons}')
        mass_in_array[iso_idx] = ori_mass
    exit()
    # ---------optimization of one coordinate---------

    # to optimization of coordinates list, comment out the above code
    # coordinates separation is defined in line 47-49
    file_path = r"D:\MW\data\HCl-H2O\calc\b3lyp_opt\B3LYP_1A2Ne_crest_conformers1to60_to_orca_coor_orca_extract.xyz"
    # sep_idx defines how to separate molecules, e. g. [2, 3] could be used to separate HCl-Ne-Ne
    sep_idx = [2, 3,]
    save_suffix = 'optimize_coor.xyz'
    optimize_coor_file_list(file_path, sep_idx, target_ABC)
