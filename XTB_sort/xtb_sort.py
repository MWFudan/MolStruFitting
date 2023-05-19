# exact data from orca output
import re
import os
import sys
import numpy as np
from tqdm import tqdm
from energy_calc import calculate_energy_xtb

# file path

re_coor = re.compile(r'\s*(\d{0,3}[A-Z][a-z]?)\s+(\S+)\s+(\S+)\s+(\S+)\s*')
re_double_end_line = re.compile(r'\n\s*\n')
atom_index_table = {
    'H': 1,
    'He': 2,
    'C': 6,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
}

def text_converter(text):
    result = re.findall(re_coor, text)
    if not result:
        return
    idx = []
    coor = []
    for i in result:
        idx.append(atom_index_table[i[0]])
        coor.append(list(map(float, i[1:])))
    return np.array(idx), np.array(coor)


def sort_one_file(file_path):
    global count
    with open(file_path) as f:
        t = f.read()

    output_dict = {}
    split_lines = re.split(re_double_end_line, t)
    total_len = len(split_lines)
    if total_len <= count:
        count = total_len
    with tqdm(total=count) as pbar:
        calc_num = 0
        for sep_coor in split_lines:
            r = text_converter(sep_coor)
            if r:
                energy = calculate_energy_xtb(*r)
                output_dict[sep_coor] = energy
                calc_num += 1
                pbar.update(1)
                if calc_num == count:
                    break

    return output_dict



if __name__ == '__main__':
    count = 99999
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    elif len(sys.argv) == 3:
        count = int(sys.argv[2])
    else:
        exit()
    
    results = sort_one_file(file_path)
    with open(file_path.replace('.xyz', 'xtb_sort.xyz'), 'w') as f:
        for t, e in sorted(results.items(), key=lambda x: x[1]):
            f.write(f'{e:5f}')
            f.write('\n')
            f.write(t)
            f.write('\n\n')
