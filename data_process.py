import os
import argparse

import mdtraj as md
import numpy as np


def main(top_file, traj_file, out_dir, test_ratio=0.1, seed=42):
    traj = md.load(traj_file, top=top_file)
    atom_idx = traj.topology.select('not water')
    atom_coord = traj.xyz[1:, atom_idx]
    enantiomer = atom_coord * np.array([-1, 1, 1], dtype=np.float32).reshape(1, 1, 3) + traj.unitcell_vectors[1:, 0:1]

    n_sample = atom_coord.shape[0]
    n_test = int(n_sample * test_ratio)
    n_train = n_sample - n_test
    generator = np.random.RandomState(seed)
    idx1, idx2 = np.arange(n_sample), np.arange(n_sample)
    generator.shuffle(idx1), generator.shuffle(idx2)
    train_data = np.concatenate([atom_coord[idx1[n_test:]], enantiomer[idx2[n_test:]]], axis=0) * 10
    test_data = np.concatenate([atom_coord[idx1[:n_test]], enantiomer[idx2[:n_test]]], axis=0) * 10
    train_label = np.concatenate([np.ones(n_train), np.zeros(n_train)]).astype(int)
    test_label = np.concatenate([np.ones(n_test), np.zeros(n_test)]).astype(int)

    atom_mapping = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
    table, bond = traj.topology.to_dataframe()
    element = np.array([atom_mapping[atom] for atom in table.element.values[atom_idx]], dtype=int)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez(os.path.join(out_dir, 'train.npz'), pos=train_data, label=train_label, z=element)
    np.savez(os.path.join(out_dir, 'test.npz'), pos=test_data, label=test_label, z=element)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Enantiomer dataset preprocessing')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--top-file', type=str, default='ala_1_md.pdb')
    parser.add_argument('--traj-file', type=str, default='ala_1_md.xtc')
    parser.add_argument('--out-dir', type=str, default='data/ala')
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    top_file = os.path.join(args.root, args.top_file)
    traj_file = os.path.join(args.root, args.traj_file)
    main(top_file, traj_file, args.out_dir, args.test_ratio, args.seed)
