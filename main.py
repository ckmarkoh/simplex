import numpy as np


def x_to_xbn(x, bv, nbv):
    return np.concatenate([x[bv], x[nbv]], axis=0)


def xbn_to_x(xbn, bv, nbv):
    pre_index_order = [x for x in enumerate(bv + nbv)]
    index_order = sorted(pre_index_order, key=lambda x: x[1])
    return xbn[[x[0] for x in index_order]]


def combination(a, m):
    assert len(a) >= m
    if len(a) == m:
        yield a
    elif m == 0:
        yield []
    else:
        for i in range(len(a) - m + 1):
            for b in combination(a[i + 1:], m - 1):
                yield [a[i]] + b


def update_x(A, c, x, bv):
    m, n = A.shape[0], A.shape[1]
    nbv = filter(lambda x: x not in bv, range(n))
    M = np.concatenate([np.concatenate([A[:, bv], A[:, nbv]], axis=1),
                        np.concatenate([np.zeros((m, n - m)), np.eye(n - m)], axis=1)],
                       axis=0)
    M_inv = np.linalg.inv(M)
    cbn = x_to_xbn(c, bv, nbv)
    xbn = x_to_xbn(x, bv, nbv)
    for q in range(len(nbv)):
        dq = M_inv[:, len(bv) + q]
        rq = np.dot(cbn[np.newaxis, :], dq)
        if rq < 0:
            alpha_array = map(lambda x: -x[0] / float(x[1]) if x[1] != 0 else np.inf,
                              zip(xbn[:len(bv)], dq[:len(bv)]))
            alpha = np.min(alpha_array)
            xbn_new = xbn + alpha * dq
            x_new = xbn_to_x(xbn_new, bv, nbv)
            bv_new = filter(lambda x: x != bv[np.argmin(alpha_array)], bv) + [nbv[q]]
            return x_new, bv_new, False
    return x, bv, True


def find_initial(A, b):
    m = A.shape[0]
    n = A.shape[1]
    v = range(n)
    # np.random.shuffle(v)
    v.reverse()
    for bv in combination(v, m):
        xbn = np.concatenate([np.dot(np.linalg.inv(A[:, bv]), b), np.zeros((n - m,))], axis=0)
        if np.sum(xbn < 0) == 0:
            nbv = filter(lambda x: x not in bv, range(n))
            x_new = xbn_to_x(xbn, bv, nbv)
            return x_new, bv
    pass


def run_simplex(A, b, c):
    x, bv = find_initial(A, b)
    stop = False
    obj = np.dot(c, x)
    while not stop:
        print "bv:%s, x:%s, obj:%s" % (bv, x, obj)
        x, bv, stop = update_x(A, c, x, bv)
        obj = np.dot(c, x)
    print "bv:%s, x:%s, obj:%s" % (bv, x, obj)
    print "Reach Minimum Value, STOP"
    return x, obj


def case1():
    A = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    b = np.array([1, 1])
    c = np.array([-1, -1, 0, 0])
    run_simplex(A, b, c)


def case2():
    A = np.array([
        [1, 1, 1, 0],
        [2, 1, 0, 1],
    ])
    b = np.array([40, 60])
    c = np.array([-1.5, -1, 0, 0])
    run_simplex(A, b, c)


if __name__ == '__main__':
    case1()
    case2()
