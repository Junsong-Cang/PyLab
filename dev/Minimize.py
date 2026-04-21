import numpy as np
from joblib import Parallel, delayed
import tqdm

def Chi2_example(theta):
    x0 = np.array([3, -1.5, 40.5, -1.5, 2.0])
    x0 = np.array([np.pi, -1.55, 40.51, -1.556, 2.02])
    
    dist = np.sqrt(np.sum((theta - x0)**2.0))
    return dist

def Minimize_5D(
        fun,
        bounds,
        verbose = 1,
        dx_min = 1E10,
        ncpu = 1):
    '''
    Minimize a 5D function using bisection.
    Caveat: not very good if have local minimum
    '''
    bisect = lambda x: [x[0], (x[0] + x[1])/2.0, x[1]]
    
    def find_best_match(params, param):
        n = len(params)
        dist = np.zeros(n)
        for idx in np.arange(0, n):
            param_here = params[idx, :]
            dist[idx] = np.sqrt(np.sum((param_here - param)**2.0))
        if np.min(dist) > 1E-20:
            index = np.nan
        else:
            index = np.argmin(dist)
        # print('best_idx', index, 'param =', param, 'best match', params[index, :])
        return index
    
    def find_junctions(xs):
        x0 = xs[0]
        mask = np.abs(xs - x0) > 1E-20
        x_new = xs[mask]
        
        x1 = x_new[0]
        mask = np.abs(x_new - x1) > 1E-20
        x_new = x_new[mask]
        r = np.array([x0, x1, x_new[0]])
        return np.sort(r)

    def bounds_to_params(B):
        n = int(3**5)
        params = np.zeros([n, 5])
        # print(np.shape(params))
        idx = 0
        for p0 in bisect(B[0]):
            for p1 in bisect(B[1]):
                for p2 in bisect(B[2]):
                    for p3 in bisect(B[3]):
                        for p4 in bisect(B[4]):
                            params[idx, :] = [p0, p1, p2, p3, p4]
                            idx = idx + 1
        return params
    
    def update_range(stats, pid):
        best_idx = np.argmin(stats[:, -1])
        best_params = stats[best_idx, 0:5]
        # xM = stats[best_idx, pid] # param at best
        xs = find_junctions(stats[:, pid])
        
        def update_param(x):
            p = np.zeros(5)
            for id in [0, 1, 2, 3, 4]:
                if id == pid:
                    p[id] = x
                else:
                    p[id] = best_params[id]
            return p
        '''
        p0 = best_params
        p1 = best_params
        p2 = best_params
        p0[pid] = xs[0]
        p1[pid] = xs[1]
        p2[pid] = xs[2]
        print('best_params', best_params)
        print('p0', p0)
        print('p1', p1)
        print('p2', p2)
        '''
        p0 = update_param(xs[0])
        p1 = update_param(xs[1])
        p2 = update_param(xs[2])

        Chi2_ax = stats[:, -1]
        c0 = Chi2_ax[find_best_match(stats[:, 0:5], p0)]
        c1 = Chi2_ax[find_best_match(stats[:, 0:5], p1)]
        c2 = Chi2_ax[find_best_match(stats[:, 0:5], p2)]
        Chi2_ax_here = np.array([c0, c1, c2])
        idx_max = np.argmax(Chi2_ax_here)
        # print('Chi2_ax_here', Chi2_ax_here)
        # print('xs', xs)
        new_range = []
        for idx in [0, 1, 2]:
            if not idx == idx_max:
                new_range.append(xs[idx])
        return new_range
    
    def update_ranges(stats):
        r = []
        for pid in [0, 1, 2, 3, 4]:
            r.append(update_range(stats, pid))
        return r
    
    def should_we_stop(params):
        dxs = np.zeros(5)
        for pid in np.arange(0, 5):
            xs = find_junctions(params[:, pid])
            dxs[pid] = (xs[1] - xs[0])/2.0
        dist = np.sqrt(np.sum(dxs**2.0))
        if dist > dx_min:
            stop = False
        else:
            stop = True
        r = {'stop': stop, 'dist': dist}
        return r

    def SerializeFun(params):
        def fun_mpi(idx):
            p = params[idx, :]
            return fun(p)
        return fun_mpi
    
    def find_params_to_run(stats, params):
        stats_array = np.array(stats)
        params_cache = stats_array[:, 0:5]
        n = len(params)
        new_stats = np.zeros([n, 6])
        for idx in np.arange(0, n):
            best_idx = find_best_match(params_cache, params[idx, :])
            new_stats[idx, 0:5] = params[idx, :]
            if np.isnan(best_idx):
                new_stats[idx, 5] = np.nan
            else:
                new_stats[idx, 5] = stats_array[best_idx, 5]
        mask = np.isnan(new_stats[:, 5])
        new_params = new_stats[mask, 0:5]
        return new_params

    # Initialize
    p = bounds_to_params(bounds)
    stats = []
    vals = Parallel(n_jobs=ncpu)(delayed(SerializeFun(p))(idx) for idx in tqdm.tqdm(range(len(p)), desc = 'Initializing', disable = not verbose))
    if True in np.isnan(vals): raise Exception("fun gives nan")
    for idx in np.arange(0, len(p)):
        stat = [p[idx][0], p[idx][1], p[idx][2], p[idx][3], p[idx][4], vals[idx]]
        stats.append(stat)
    
    go = 1
    stat_here = np.array(stats)
    iter_count = 1
    dist = np.inf
    while go:
        new_range = update_ranges(np.array(stat_here))
        # print('new_range', new_range)
        params = bounds_to_params(new_range)
        params_run = find_params_to_run(stats, params)
        n = len(params_run)
        MSG = "Run #{:.0f}, CVG = {:.1E}".format(iter_count, dist/dx_min)
        vals = Parallel(n_jobs=ncpu)(delayed(SerializeFun(params_run))(idx) for idx in tqdm.tqdm(range(n), desc = MSG, disable = not verbose))
        if True in np.isnan(vals): raise Exception("fun gives nan")

        # updating stats
        for idx in np.arange(0, n):
            new_stat = [params_run[idx, 0], params_run[idx, 1], params_run[idx, 2], params_run[idx, 3], params_run[idx, 4], vals[idx]]
            stats.append(new_stat)
        stat_here = np.zeros([len(params), 6])
        for idx in np.arange(0, len(params)):
            p = params[idx, :]
            stat_here[idx, 0:5] = p[:]
            idx_best = find_best_match(np.array(stats)[:, 0:5], p)
            stat_here[idx, 5] = stats[idx_best][5]
            # print(idx_best)
        '''
        print(find_junctions(params[:, 0]))
        print(find_junctions(params[:, 1]))
        print(find_junctions(params[:, 2]))
        print(find_junctions(params[:, 3]))
        print(find_junctions(params[:, 4]))
        '''
        iter_count = iter_count + 1
        go = not should_we_stop(params)['stop']
        if iter_count > 10:
            go = False
            print('Cannot find solution')
        dist = should_we_stop(params)['dist']

    stats = np.array(stats)
    idx_best = np.argmin(stats[:, -1])
    x = stats[idx_best, 0:5]
    r = {'x': x}
    return r
