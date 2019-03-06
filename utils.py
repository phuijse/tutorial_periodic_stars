import numpy as np
import matplotlib.pylab as plt

def create_noisy_xor(N_per_cluster=500, stddev_noise=0.4):
    data = stddev_noise*np.random.randn(4*N_per_cluster, 2)
    data[0*N_per_cluster:1*N_per_cluster, :] += [1.0, -1.0]
    data[1*N_per_cluster:2*N_per_cluster, :] += [-1.0, 1.0]
    data[2*N_per_cluster:3*N_per_cluster :] += [-1.0, -1.0]
    data[3*N_per_cluster:4*N_per_cluster, :] += [1.0, 1.0]
    #data = (data - np.mean(X, axis=0))/np.std(X, axis=0)
    labels = np.zeros(shape=(4*N_per_cluster,), dtype=int)
    labels[2*N_per_cluster:] = 1.0
    NP = np.random.permutation(4*N_per_cluster)
    return data[NP, :], labels[NP]

def featurize_lc(lc_data, period, phi_interp, sp=0.15): 
    mjd, mag, err = lc_data.T
    phi = np.mod(mjd, period)/period
    mag_interp = np.zeros_like(phi_interp)
    err_interp = np.zeros_like(phi_interp)
    w = 1.0/err**2
    for i in range(len(phi_interp)):
        gt = np.exp((np.cos(2.0*np.pi*(phi_interp[i] - phi)) -1)/sp**2)
        norm = np.sum(w*gt)
        mag_interp[i] = np.sum(w*gt*mag)/norm
        err_interp[i] = np.sqrt(np.sum(w*gt*(mag - mag_interp[i])**2)/norm)
    err_interp += np.sqrt(np.median(err**2))
    idx_max =  np.argmin(mag_interp)
    mag_interp = np.roll(mag_interp, -idx_max)
    err_interp = np.roll(err_interp, -idx_max)
    max_val = np.amax(mag_interp + err_interp)
    min_val = np.amin(mag_interp - err_interp)
    mag_interp = 2*(mag_interp - min_val)/(max_val - min_val) - 1
    err_interp = 2*err_interp/(max_val - min_val)
    return mag_interp, err_interp, [max_val, min_val, idx_max]

def defeaturize_lc(mag, err, norm):
    # center, scale, idx_max = norm[0], norm[1], norm[2]
    max_val, min_val, idx_max = norm[0], norm[1], norm[2]
    idx_max = int(idx_max)
    return 0.5*(np.roll(mag, idx_max) +1)*(max_val - min_val) + min_val, 0.5*np.roll(err, idx_max)*(max_val - min_val)


class live_metric_plotter:
    """
    This create and update the plots of the reconstruction error  and the KL divergence
    """
    def __init__(self, figsize=(7, 3)):
        self.fig, ax1 = plt.subplots(1, figsize=figsize, tight_layout=True)
        ax2 = ax1.twinx() 
        ax2.set_ylabel('KL qzx||pz (dotted)');
        ax1.set_ylabel('-log pxz (solid)')
        ax1.set_xlabel('Epoch')
        ax1.plot(0, alpha=0.75, linewidth=2, label='Train') 
        ax1.plot(0, alpha=0.75, linewidth=2, label='Validation')
        ax2.plot(0, alpha=0.75, linewidth=2, label='Train', linestyle='--') 
        ax2.plot(0, alpha=0.75, linewidth=2, label='Validation', linestyle='--')
        plt.legend(); plt.grid(); 
        self.axes = list([ax1, ax2])   
        
    def update(self, epoch, metrics):
        for i, ax in enumerate(self.axes):
            for j, line in enumerate(ax.lines):
                line.set_data(range(epoch+1), metrics[:epoch+1, j, i])
            ax.set_xlim([0, epoch])
            ax.set_ylim([np.amin(metrics[:epoch+1, :, i]), np.amax(metrics[:epoch+1, :, i])])
        self.fig.canvas.draw();
