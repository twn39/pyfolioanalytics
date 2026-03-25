import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import from_mlab_linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def DBHTs(D, S, leaf_order=True):
    """
    Perform Direct Bubble Hierarchical Tree (DBHT) clustering.
    Reference: Riskfolio-Lib / Tomaso Aste.
    """
    Rpm, _, _, _, _ = PMFG_T2s(S)
    Apm = Rpm.copy()
    Apm[Apm != 0] = D[Apm != 0].copy()
    Dpm, _ = distance_wei(Apm)
    H1, Hb, Mb, CliqList, Sb = CliqHierarchyTree2s(Rpm, method1="uniqueroot")
    Mb = Mb[0 : CliqList.shape[0], :]
    Mv = np.empty((Rpm.shape[0], 0))
    for i in range(0, Mb.shape[1]):
        vec = np.zeros(Rpm.shape[0])
        vec[np.int32(np.unique(CliqList[Mb[:, i] != 0, :]))] = 1
        Mv = np.hstack((Mv, vec.reshape(-1, 1)))

    Adjv, T8 = BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)
    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Adjv, Mv)

    if leaf_order:
        Z = optimal_leaf_ordering(Z, squareform(D))

    return (T8, Rpm, Adjv, Dpm, Mv, Z)


def PMFG_T2s(W, nargout=3):
    N = W.shape[0]
    A = np.zeros((N, N))
    in_v = -1 * np.ones(N, dtype=np.int32)
    tri = np.zeros((2 * N - 4, 3))
    separators = np.zeros((N - 4, 3))

    s = np.sum(W * (W > np.mean(W)), axis=1)
    j = np.int32(np.argsort(s)[::-1].reshape(-1))

    in_v[0:4] = [int(x) for x in j[0:4]]
    ou_v = np.setdiff1d(np.arange(0, N), in_v)
    tri[0, :] = in_v[[0, 1, 2]]
    tri[1, :] = in_v[[1, 2, 3]]
    tri[2, :] = in_v[[0, 1, 3]]
    tri[3, :] = in_v[[0, 2, 3]]
    A[in_v[0], in_v[1]] = 1
    A[in_v[0], in_v[2]] = 1
    A[in_v[0], in_v[3]] = 1
    A[in_v[1], in_v[2]] = 1
    A[in_v[1], in_v[3]] = 1
    A[in_v[2], in_v[3]] = 1

    gain = np.zeros((N, 2 * N - 4))
    gain[ou_v, 0] = np.sum(W[np.ix_(ou_v, np.int32(tri[0, :]))], axis=1)
    gain[ou_v, 1] = np.sum(W[np.ix_(ou_v, np.int32(tri[1, :]))], axis=1)
    gain[ou_v, 2] = np.sum(W[np.ix_(ou_v, np.int32(tri[2, :]))], axis=1)
    gain[ou_v, 3] = np.sum(W[np.ix_(ou_v, np.int32(tri[3, :]))], axis=1)

    kk = 3
    for k in range(4, N):
        if len(ou_v) == 1:
            ve = ou_v[0]
            v = 0
            tr = np.argmax(gain[ou_v, :])
        else:
            gij = np.max(gain[ou_v, :], axis=0)
            v = np.argmax(gain[ou_v, :], axis=0)
            tr = np.argmax(np.round(gij, 6).flatten())
            ve = ou_v[v[tr]]
            v = v[tr]

        ou_v = ou_v[np.delete(np.arange(len(ou_v)), v)]
        in_v[k] = ve
        A[np.ix_([ve], np.int32(tri[tr, :]))] = 1
        separators[k - 4, :] = tri[tr, :]
        tri[kk + 1, :] = np.hstack((tri[tr, [0, 2]], ve))
        tri[kk + 2, :] = np.hstack((tri[tr, [1, 2]], ve))
        tri[tr, :] = np.hstack((tri[tr, [0, 1]], ve))
        gain[ve, :] = 0
        gain[ou_v, tr] = np.sum(W[np.ix_(ou_v, np.int32(tri[tr, :]))], axis=1)
        gain[ou_v, kk + 1] = np.sum(W[np.ix_(ou_v, np.int32(tri[kk + 1, :]))], axis=1)
        gain[ou_v, kk + 2] = np.sum(W[np.ix_(ou_v, np.int32(tri[kk + 2, :]))], axis=1)
        kk = kk + 2

    A = W * ((A + A.T) == 1)
    cliques = None
    if nargout > 3:
        cliques = np.vstack(
            (in_v[0:4].reshape(1, -1), np.hstack((separators, in_v[4:].reshape(-1, 1))))
        )

    return (A, tri, separators, cliques, None)


def distance_wei(L):
    n = len(L)
    D = np.ones((n, n)) * np.inf
    np.fill_diagonal(D, 0)
    B = np.zeros((n, n))
    for u in range(0, n):
        S = np.full(n, True, dtype=bool)
        L1 = L.copy()
        V = np.array([u])
        while 1:
            S[V] = False
            L1[:, V] = 0
            for v in V.tolist():
                _, T, _ = sp.find(L1[v, :])
                d = np.min(
                    np.vstack(
                        (D[np.ix_([u], T)], D[np.ix_([u], [v])] + L1[np.ix_([v], T)])
                    ),
                    axis=0,
                )
                wi = np.argmin(
                    np.vstack(
                        (D[np.ix_([u], T)], D[np.ix_([u], [v])] + L1[np.ix_([v], T)])
                    ),
                    axis=0,
                )
                D[np.ix_([u], T)] = d
                ind = T[wi == 2]
                B[u, ind] = B[u, v] + 1
            if D[u, S].size == 0:
                minD = np.empty((0, 0))
            else:
                minD = np.min(D[u, S])
                minD = np.array([minD])
            if minD.shape[0] == 0 or np.isinf(minD):
                break
            V = np.ravel(np.argwhere(D[u, :] == minD))
    return (D, B)


def CliqHierarchyTree2s(Apm, method1):
    N = Apm.shape[0]
    A = 1.0 * sp.csr_matrix(Apm != 0).toarray()
    _, _, clique = clique3(A)
    Nc = clique.shape[0]
    M = np.zeros((N, Nc))
    CliqList = clique.copy()
    Sb = np.zeros(Nc)
    for n in range(0, Nc):
        cliq_vec = CliqList[n, :]
        T, _ = FindDisjoint(A, cliq_vec)
        indx0 = np.argwhere(np.ravel(T) == 0)
        indx1 = np.argwhere(np.ravel(T) == 1)
        indx2 = np.argwhere(np.ravel(T) == 2)
        if len(indx1) > len(indx2):
            indx_s = np.vstack((indx2, indx0))
        else:
            indx_s = np.vstack((indx1, indx0))
        if indx_s.shape[0] != 0:
            Sb[n] = len(indx_s) - 3
        M[indx_s, n] = 1

    Pred = BuildHierarchy(M)
    Root = np.argwhere(Pred == -1)
    if method1.lower() == "uniqueroot":
        if len(Root) > 1:
            Pred = np.append(Pred[:], -1)
            Pred[Root] = len(Pred) - 1
        H = np.zeros((Nc + 1, Nc + 1))
        for n in range(0, len(Pred)):
            if Pred[n] != -1:
                H[n, np.int32(Pred[n])] = 1
        H = H + H.T
    else:
        H = np.zeros((Nc, Nc))
        for n in range(0, len(Pred)):
            if Pred[n] != -1:
                H[n, np.int32(Pred[n])] = 1
        H = H + H.T

    H2, Mb = BubbleHierarchy(Pred, Sb, A, CliqList)
    H2 = 1.0 * (H2 != 0)
    return (H, H2, Mb, CliqList, Sb)


def BuildHierarchy(M):
    Pred = -1 * np.ones(M.shape[1])
    for n in range(0, M.shape[1]):
        _, Children, _ = sp.find(M[:, n] == 1)
        ChildrenSum = np.sum(M[Children, :], axis=0)
        Parents = np.argwhere(np.ravel(ChildrenSum) == len(Children))
        Parents = Parents[Parents != n].flatten()
        if Parents.shape[0] != 0:
            ParentSum = np.sum(M[:, Parents], axis=0)
            a = np.argwhere(ParentSum == np.min(ParentSum)).flatten()
            if len(a) >= 1:
                Pred[n] = Parents[a[0]]
            else:
                return np.empty(0)
        else:
            Pred[n] = -1
    return Pred


def FindDisjoint(Adj, Cliq):
    N = Adj.shape[0]
    Temp = Adj.copy()
    T = np.zeros(N)
    IndxTotal = np.arange(0, N)
    IndxNot = np.argwhere(
        np.logical_and(IndxTotal != Cliq[0], IndxTotal != Cliq[1], IndxTotal != Cliq[2])
    )
    Temp[np.int32(Cliq), :] = 0
    Temp[:, np.int32(Cliq)] = 0
    d, _ = breadth(Temp, IndxNot[0])
    d[np.isinf(d)] = -1
    d[IndxNot[0]] = 0
    T[d == -1] = 1
    T[d != -1] = 2
    T[np.int32(Cliq)] = 0
    return (T, IndxNot)


def BubbleHierarchy(Pred, Sb, A, CliqList):
    Nc = Pred.shape[0]
    Root = np.argwhere(Pred == -1)
    CliqCount = np.zeros(Nc)
    CliqCount[Root] = 1
    Mb = np.empty((Nc, 0))
    if len(Root) > 1:
        TempVec = np.zeros((Nc, 1))
        TempVec[Root] = 1
        Mb = np.hstack((Mb, TempVec))
    while np.sum(CliqCount) < Nc:
        NxtRoot = np.empty((0, 1))
        for n in range(0, len(Root)):
            _, DirectChild, _ = sp.find(Pred == Root[n])
            TempVec = np.zeros((Nc, 1))
            TempVec[np.append(DirectChild, np.int32(Root[n])), 0] = 1
            Mb = np.hstack((Mb, TempVec))
            CliqCount[DirectChild] = 1
            for m in range(0, len(DirectChild)):
                if Sb[DirectChild[m]] != 0:
                    NxtRoot = np.vstack((NxtRoot, DirectChild[m]))
        Root = np.unique(NxtRoot)
    Nb = Mb.shape[1]
    H = np.zeros((Nb, Nb))
    for n in range(0, Nb):
        Indx = Mb[:, n] == 1
        JointSum = np.sum(Mb[Indx, :], axis=0)
        Neigh = JointSum >= 1
        H[n, Neigh] = 1
    H = H + H.T
    H = H - np.diag(np.diag(H))
    return (H, Mb)


def clique3(A):
    A = A - np.diag(np.diag(A))
    A = 1.0 * (A != 0)
    A2 = A @ A
    P = (1.0 * (A2 != 0)) * (1.0 * (A != 0))
    P = sp.csr_matrix(np.triu(P))
    r, c, _ = sp.find(P != 0)
    E = np.hstack((r.reshape(-1, 1), c.reshape(-1, 1)))
    K3 = {}
    for n in range(0, len(r)):
        i, j = r[n], c[n]
        a = A[i, :] * A[j, :]
        _, indx, _ = sp.find(a != 0)
        K3[n] = indx
    clique = np.zeros((1, 3))
    for n in range(0, len(r)):
        temp = K3[n]
        for m in range(0, len(temp)):
            candidate = np.sort(np.hstack((E[n, :], temp[m])))
            if not np.any(np.all(clique == candidate, axis=1)):
                clique = np.vstack((clique, candidate.reshape(1, -1)))
    clique = clique[np.lexsort((clique[:, 2], clique[:, 1], clique[:, 0]))]
    return (K3, E, clique[1:])


def breadth(CIJ, source):
    N = CIJ.shape[0]
    white, gray, black = 0, 1, 2
    color = np.zeros(N)
    distance = np.inf * np.ones(N)
    branch = np.zeros(N)
    color[source] = gray
    distance[source] = 0
    branch[source] = -1
    Q = np.array(source).reshape(-1)
    while Q.shape[0] != 0:
        u = Q[0]
        _, ns, _ = sp.find(CIJ[u, :])
        for v in ns:
            if distance[v] == np.inf:
                distance[v] = distance[u] + 1
            if color[v] == white:
                color[v] = gray
                distance[v] = distance[u] + 1
                branch[v] = u
                Q = np.hstack((Q, v))
        Q = Q[1:]
        color[u] = black
    return (distance, branch)


def BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList):
    Hc, Sep = DirectHb(Rpm, Hb, Mb, Mv, CliqList)
    N = Rpm.shape[0]
    _, indx, _ = sp.find(Sep == 1)
    if len(indx) > 1:
        Adjv = np.zeros((Mv.shape[0], len(indx)))
        for n in range(0, len(indx)):
            d, _ = breadth(Hc.T, indx[n])
            d[np.isinf(d)] = -1
            d[indx[n]] = 0
            r, _, _ = sp.find(Mv[:, d != -1] != 0)
            Adjv[np.unique(r), n] = 1
        Tc = -1 * np.ones(N)
        Bubv = Mv[:, indx]
        _, cv, _ = sp.find(np.sum(Bubv.T, axis=0).T == 1)
        _, uv, _ = sp.find(np.sum(Bubv.T, axis=0).T > 1)
        Mdjv = np.zeros((N, len(indx)))
        Mdjv[cv, :] = Bubv[cv, :]
        for v_idx in range(0, len(uv)):
            v = uv[v_idx]
            v_cont = np.sum(Rpm[:, v].reshape(-1, 1) * Bubv, axis=0).reshape(-1, 1)
            all_cont = (3 * (np.sum(Bubv, axis=0) - 2)).reshape(-1, 1)
            Mdjv[v, np.argmax(v_cont / all_cont)] = 1
        v, ci, _ = sp.find(Mdjv != 0)
        Tc[v] = ci
        Udjv = Dpm @ (Mdjv @ np.diag(1 / np.sum(Mdjv != 0, axis=0)))
        Udjv[Adjv == 0] = np.inf
        Tc[Tc == -1] = np.argmin(Udjv[np.sum(Mdjv, axis=1) == 0, :], axis=1)
    else:
        Tc = np.zeros(N)
        Adjv = np.ones((N, 1))
    return (Adjv, Tc)


def DirectHb(Rpm, Hb, Mb, Mv, CliqList):
    Hb_temp = 1 * (Hb != 0)
    r, c, _ = sp.find(sp.triu(sp.csr_matrix(Hb_temp)) != 0)
    CliqEdge = np.empty((0, 3))
    for n in range(0, len(r)):
        data = np.argwhere(np.logical_and(Mb[:, r[n]] != 0, Mb[:, c[n]] != 0))
        if data.shape[0] != 0:
            CliqEdge = np.vstack((CliqEdge, np.hstack((r[n], c[n], data.flatten()))))
    kb = np.sum(1 * (Hb_temp != 0), axis=0)
    Hc = np.zeros((Mv.shape[1], Mv.shape[1]))
    CliqEdge = np.int32(CliqEdge)
    for n in range(0, CliqEdge.shape[0]):
        Temp = Hb_temp.copy()
        Temp[int(CliqEdge[n, 0]), int(CliqEdge[n, 1])] = 0
        Temp[int(CliqEdge[n, 1]), int(CliqEdge[n, 0])] = 0
        d, _ = breadth(Temp, 0)
        d[np.isinf(d)] = -1
        d[0] = 0
        vo = np.int32(CliqList[int(CliqEdge[n, 2]), :])
        bleft = CliqEdge[n, 0:2][d[[int(x) for x in CliqEdge[n, 0:2]]] != -1]
        bright = CliqEdge[n, 0:2][d[[int(x) for x in CliqEdge[n, 0:2]]] == -1]
        vleft = np.setdiff1d(np.argwhere(Mv[:, d != -1] != 0)[:, 0], vo)
        vright = np.setdiff1d(np.argwhere(Mv[:, d == -1] != 0)[:, 0], vo)
        left, right = np.sum(Rpm[np.ix_(vo, vleft)]), np.sum(Rpm[np.ix_(vo, vright)])
        if left > right:
            Hc[np.ix_(bright, bleft)] = left
        else:
            Hc[np.ix_(bleft, bright)] = right
    Sep = np.double(np.sum(Hc.T, axis=0) == 0)
    Sep[np.logical_and(np.sum(Hc, axis=0) == 0, kb > 1)] = 2
    return (Hc, Sep)


def HierarchyConstruct4s(Rpm, Dpm, Tc, Adjv, Mv):
    N = Dpm.shape[0]
    kvec = np.int32(np.unique(Tc))
    LabelVec1 = np.arange(0, N)
    E = sp.csr_matrix(
        (np.ones(N), (np.arange(0, N), np.int32(Tc))),
        shape=(N, np.int32(np.max(Tc) + 1)),
    ).toarray()
    Z = np.empty((0, 3))
    for n in range(0, kvec.shape[0]):
        Mc = E[:, int(kvec[n])].reshape(-1, 1) * Mv
        Mvv = BubbleMember(Dpm, Rpm, Mv, Mc)
        _, Bub, _ = sp.find(np.sum(Mvv, axis=0) > 0)
        nc = np.sum(Tc == int(kvec[n])) - 1
        for m in range(0, Bub.shape[0]):
            _, V, _ = sp.find(Mvv[:, int(Bub[m])] != 0)
            if len(V) > 1:
                dpm, LabelVec = Dpm[np.ix_(V, V)], LabelVec1[V]
                LabelVec2 = LabelVec1.copy()
                for v in range(0, len(V) - 1):
                    PairLink, _ = LinkageFunction(dpm, LabelVec)
                    LabelVec[
                        np.logical_or(LabelVec == PairLink[0], LabelVec == PairLink[1])
                    ] = np.max(LabelVec1) + 1
                    LabelVec2[V] = LabelVec
                    Z = DendroConstruct(
                        Z, LabelVec1, LabelVec2, 1 / nc if nc > 0 else 1
                    )
                    nc -= 1
                    LabelVec1 = LabelVec2.copy()
        _, V, _ = sp.find(E[:, int(kvec[n])] != 0)
        if Bub.shape[0] > 1:
            dpm, LabelVec = Dpm[np.ix_(V, V)], LabelVec1[V]
            LabelVec2 = LabelVec1.copy()
            for b in range(0, Bub.shape[0] - 1):
                PairLink, _ = LinkageFunction(dpm, LabelVec)
                LabelVec[
                    np.logical_or(LabelVec == PairLink[0], LabelVec == PairLink[1])
                ] = np.max(LabelVec1) + 1
                LabelVec2[V] = LabelVec
                Z = DendroConstruct(Z, LabelVec1, LabelVec2, 1 / nc if nc > 0 else 1)
                nc -= 1
                LabelVec1 = LabelVec2.copy()
    LabelVec2 = LabelVec1.copy()
    dcl = np.ones(len(LabelVec1))
    for n in range(0, kvec.shape[0] - 1):
        PairLink, _ = LinkageFunction(Dpm, LabelVec1)
        LabelVec2[np.logical_or(LabelVec1 == PairLink[0], LabelVec1 == PairLink[1])] = (
            np.max(LabelVec1) + 1
        )
        dvu = dcl[LabelVec1 == PairLink[0]][0] + dcl[LabelVec1 == PairLink[1]][0]
        dcl[np.logical_or(LabelVec1 == PairLink[0], LabelVec1 == PairLink[1])] = dvu
        Z = DendroConstruct(Z, LabelVec1, LabelVec2, dvu)
        LabelVec1 = LabelVec2.copy()
    Z[:, 0:2] += 1
    return from_mlab_linkage(Z)


def LinkageFunction(d, labelvec):
    lvec = np.unique(labelvec)
    Links = np.empty((0, 3))
    for r in range(len(lvec) - 1):
        vecr = (labelvec == lvec[r]).flatten()
        for c in range(r + 1, len(lvec)):
            vecc = (labelvec == lvec[c]).flatten()
            x1 = np.logical_or(vecr, vecc)
            dd = d[np.ix_(x1, x1)]
            val = np.max(dd[dd != 0]) if np.any(dd != 0) else 0
            Links = np.vstack((Links, [lvec[r], lvec[c], val]))
    imn = np.argmin(Links[:, 2])
    return (Links[imn, 0:2], Links[imn, 2])


def BubbleMember(Dpm, Rpm, Mv, Mc):
    Mvv = np.zeros(Mv.shape)
    _, vu, _ = sp.find(np.sum(Mc.T, axis=0) > 1)
    _, v, _ = sp.find(np.sum(Mc.T, axis=0) == 1)
    Mvv[v, :] = Mc[v, :]
    for n in range(len(vu)):
        _, bub, _ = sp.find(Mc[vu[n], :] != 0)
        vu_bub = np.sum(Rpm[:, vu[n]].reshape(-1, 1) * Mv[:, bub], axis=0).T
        all_bub = np.diag(Mv[:, bub].T @ Rpm @ Mv[:, bub]) / 2
        Mvv[vu[n], bub[np.argmax(vu_bub / all_bub)]] = 1
    return Mvv


def DendroConstruct(Zi, LabelVec1, LabelVec2, LinkageDist):
    indx = LabelVec1 != LabelVec2
    return np.vstack(
        (Zi, np.hstack((np.sort(np.unique(LabelVec1[indx])), LinkageDist)))
    )
