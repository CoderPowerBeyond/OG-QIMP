import numpy as np

CHANNELS = ('sigma', 'pi', 'nonbonding')


def per_atom_ao(mol):
    s_idx = [[] for _ in range(mol.natm)]
    p_shells = [[] for _ in range(mol.natm)]
    offset = 0
    for sh in range(mol.nbas):
        l = mol.bas_angular(sh)
        n_ao = (l + 1) * (l + 2) // 2 if mol.cart else 2 * l + 1
        a = mol.bas_atom(sh)
        if l == 0:
            s_idx[a].extend(range(offset, offset + n_ao))
        elif l == 1:
            p_shells[a].append(list(range(offset, offset + n_ao)))
        offset += n_ao
    return s_idx, p_shells


def bond_frame(ri, rj):
    z = np.asarray(rj, dtype=float) - np.asarray(ri, dtype=float)
    norm = np.linalg.norm(z)
    if norm < 1e-8:
        return np.eye(3)
    z = z / norm
    ref = np.array([0.0, 0.0, 1.0])
    if abs(z @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def _edge_channels(S, buckets, i, j, ri, rj):
    s_i, ps_i = buckets[i]
    s_j, ps_j = buckets[j]

    R_i = bond_frame(ri, rj)
    R_j = bond_frame(rj, ri)

    ss = S[np.ix_(s_i, s_j)]

    total = float(np.abs(ss).sum())
    sigma = float(np.abs(ss).sum())
    pi = 0.0

    if s_i and ps_j:
        sp = S[np.ix_(s_i, [k for sh in ps_j for k in sh])]
        sp_loc = np.einsum('ske,de->skd', sp.reshape(len(s_i), len(ps_j), 3), R_j)
        total += float(np.abs(sp_loc).sum())
        sigma += float(np.abs(sp_loc[:, :, 2]).sum())
    if ps_i and s_j:
        ps = S[np.ix_([k for sh in ps_i for k in sh], s_j)]
        ps_loc = np.einsum('kbs,cb->kcs', ps.reshape(len(ps_i), 3, len(s_j)), R_i)
        total += float(np.abs(ps_loc).sum())
        sigma += float(np.abs(ps_loc[:, 2, :]).sum())
    for sh_i in ps_i:
        for sh_j in ps_j:
            pp_loc = R_i @ S[np.ix_(sh_i, sh_j)] @ R_j.T
            total += float(np.abs(pp_loc).sum())
            sigma += abs(pp_loc[2, 2])
            pi += float(np.abs(pp_loc[:2, :2]).sum())

    nonbonding = max(total - sigma - pi, 0.0)
    return float(sigma), float(pi), float(nonbonding)


def molecule_overlap_channels(symbols, coords, edges, charge=0):
    from pyscf import gto

    atom = [(s, tuple(float(c) for c in xyz)) for s, xyz in zip(symbols, coords)]
    mol = gto.M(atom=atom, basis='sto-3g', charge=int(charge), verbose=0, cart=False)
    S = mol.intor('int1e_ovlp')
    buckets = list(zip(*per_atom_ao(mol))) 

    coords = np.asarray(coords, dtype=float)
    out = {c: np.zeros(len(edges), dtype=np.float64) for c in CHANNELS}
    for e, (i, j) in enumerate(edges):
        sig, pi, nb = _edge_channels(S, buckets, int(i), int(j),
                                     coords[int(i)], coords[int(j)])
        out['sigma'][e] = sig
        out['pi'][e] = pi
        out['nonbonding'][e] = nb
    return out
