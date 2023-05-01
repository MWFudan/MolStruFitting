import numpy as np
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from filter_coor_file_with_atom_distance import verify_coor_text
from scipy.spatial.transform import Rotation
from scipy import optimize
from scipy.stats import qmc
from tqdm import tqdm

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
re_file_split = re.compile(r'\s{2}\d{1,3}\n')  # crest
# re_file_split = re_double_end_line

process_num = 15
error_percent_discard_threshold = 0.1
ftol = 1E-6  # ftol
jac = '2-point'
rot_method = 'euler'  # euler, quat
not_rot_flag = False
rot_to_trans_scale = 1
algorithm = 'trf'
max_iter_num = 200

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

    # not good
    # result = optimize.minimize(calc_error, initial_input, args=(target_ABC, molecule_list), method='L-BFGS-B', jac='3-point')

    # not good
    # result = optimize.minimize(calc_error, initial_input, args=(target_ABC, molecule_list), method='TNC',
    #                            jac='3-point')

    # not good
    # result = optimize.minimize(calc_error, initial_input, args=(target_ABC, molecule_list), method='trust-constr',
    #                            jac='3-point')

    bounds = [(-10, 10), (-10, 10), (-10, 10), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)] * len(molecule_list)
    # simplicial homology global optimization, very slow
    # result = optimize.shgo(lambda x, y=target_ABC, z=molecule_list: calc_error(x, y, z), bounds, options={'ftol': 1.0E-4, 'maxtime': 60})

    # direct, strange
    # result = optimize.direct(calc_error, bounds, args=(target_ABC, molecule_list), maxiter=10000)

    # not good
    # result = optimize.dual_annealing(calc_error, bounds, args=(target_ABC, molecule_list), maxiter=10000)

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
    # tmp
    # print('iso error:', np.array(iso_abc[0][2:]) - calc_ABC(*text_converter(result.replace('O', '18O'))))


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
        save_fp = fp + 'optimize_coor.xyz'

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
            iso_ABC_error += np.sum(single_iso_ABC_error**2)

            mass[idx] = ori_mass

    # error = target_ABC - ABC
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
    # principle_axis_coor = convert_to_main_axis(mass, coor)
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
    target_ABC = np.array([1488.44069,828.38884,664.21572])

    file_path = r"C:\Users\mike chen\Documents\WeChat Files\wxid_vom3jfqtyo9x22\FileStorage\File\2023-04\change_multistructures.xyz"

    iso_abc = [
        (18, '18O', 1242.416, 458.95570, 426.62361),
    ]
    sep_idx = [3, 6, 9, 12, 15, 18]  # separation index, ex. water dimer is 3
    optimize_coor_file_list(file_path, sep_idx, target_ABC)
    # result, msg, __ = optimize_coor(text, target_ABC, 0, isotope_target_ABC=iso_abc)
    # print(result,'\n', msg)
    # print('parent error:', target_ABC-calc_ABC(*text_converter(result)))
    # print('parent error:', np.array(iso_abc[0][2:]) - calc_ABC(*text_converter(result.replace('O','18O'))))
    exit()

    # di = r'D:\MW\data\SO2_H2O\coor\opt_try'
    '''parm = {
        '1s2w': ([3, 6],                (2089.5204, 1459.8550, 1194.6845)),
        '1s3w': ([3, 6,9],              (2089.5204, 1459.8550, 1194.6845)),
        '1s4w': ([3, 6,9,12],           (2089.5204, 1459.8550, 1194.6845)),
        '1s5w': ([3, 6, 9, 12, 15],     (2089.5204, 1459.8550, 1194.6845)),
        '1s6w': ([3, 6, 9, 12, 15,18],  (2089.5204, 1459.8550, 1194.6845)),
        '2s1w': ([3, 6],                (2089.5204, 1459.8550, 1194.6845)),
        '2s2w': ([3, 6,9],              (2089.5204, 1459.8550, 1194.6845)),
        '2s3w': ([3, 6,9,12],           (2089.5204, 1459.8550, 1194.6845)),
        '2s4w': ([3, 6, 9, 12, 15],     (2089.5204, 1459.8550, 1194.6845)),
        '2s5w': ([3, 6, 9, 12, 15,18],  (2089.5204, 1459.8550, 1194.6845)),
        '2s6w': ([3, 6,9,12,15,18,21],  (2089.5204, 1459.8550, 1194.6845)),
        '3s1w': ([3, 6,9],              (2089.5204, 1459.8550, 1194.6845)),
        '3s2w': ([3, 6,9,12],           (2089.5204, 1459.8550, 1194.6845)),
        '3s3w': ([3, 6, 9, 12, 15],     (2089.5204, 1459.8550, 1194.6845)),
        '3s4w': ([3, 6, 9, 12, 15,18],  (2089.5204, 1459.8550, 1194.6845)),
        '3s5w': ([3, 6,9,12,15,18,21],  (2089.5204, 1459.8550, 1194.6845)),
    }
    import os
    for k, v in parm.items():
        fn = '{}_crest_conformers.xyz'.format(k)
        ffn = os.path.join(di, fn)
        print(ffn)
        optimize_coor_file_list(ffn, v[0], np.array(v[1]))
        sum_result.clear()
    '''
