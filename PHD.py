import numpy as np
import math
import operator
from copy import deepcopy
from scipy.stats import multivariate_normal as mvn
from collections import defaultdict
from scipy.spatial import distance
from confidence_ellipse import confidence_ellipse

class Gaussian():
    def __init__(self, w=0, m=[], P=[], label=0, z_history=[], z_history_len=0):
        self.w = w # Weight of Gaussian
        self.m = m # possitions of Gaussian
        self.P = P # Covariant matrix
        self.Eta = None
        self.S = None
        self.K = None
        self.P_new = None
        self.w_old = None
        self.label = label
        self.z_history = z_history
        self.z_history_len = z_history_len

        
    def __str__(self):  
        return "Gaussian l:% s w:% s" % (self.label,self.w)  

    def __repr__(self):
        return "Gaussian l:% s w:% s" % (self.label,self.w) 

class PHD():
    def __init__(self, targets=[], sources=[], F=[], Q=[], H=[], R=[], Ps=0.99, Pd=0.98, lam=0, U=4, T=1.e-5, T_plot=0.5, gamma=9.2, w_th=0.5, trajectory_prune=10):
        self.targets = targets # Array of Gassians (One for every target and for all new possible targetsl)
        self.sources = sources # Array of tuples (x,y) coordinates of new possible targets
        self.F = F # Same as A in KF
        self.H = H # Same as H in KF
        self.R = R # Same as R in KF
        self.Q = Q # Same as Q in KF
        self.Ps = Ps # Probability of survival
        self.Pd = Pd # Probability of detection
        self.lam = lam # intensity of clutter RFS
        self.T = T # Truncation threshold
        self.U = math.sqrt(U) # Merging threshold
        self.T_plot = T_plot # Treshold for weight
        self.label_counter = 1 # Counter of labels in filter 
        self.gamma = gamma # confidence coefficient
        self.measurements = set() # Measurements after gating
        self.w_th = w_th # Min weight of target to add trajectory
        self.trajectories = defaultdict(list) 
        self.trajectory_prune = trajectory_prune

    def predict(self, cov):
        for target in self.targets:
            target.w = self.Ps * target.w
            target.m = self.F.dot(target.m)
            target.P = self.Q + self.F.dot(target.P).dot(self.F.T)

        for source_spot in self.sources:
            self.targets.append(Gaussian(w=0.1, m=deepcopy(source_spot), P=cov, label=self.label_counter))
            self.label_counter += 1

    # (Observation Selection)
    def gate(self, measurements):
        self.measurements.clear()
        for target in self.targets:
            for z in measurements:
                d = distance.mahalanobis(z, self.H.dot(target.m), np.linalg.inv(self.H.dot(target.P).dot(self.H.T) + self.R))
                if d <= self.gamma:
                    self.measurements.add(z)
               
    def update(self):  
        for target in self.targets:
            target.Eta = deepcopy(self.H.dot(target.m))
            target.S = deepcopy(self.R + self.H.dot(target.P).dot(self.H.T))
            target.K = deepcopy(target.P.dot(self.H.T).dot(np.linalg.inv(target.S)))
            target.P_new = deepcopy((np.eye(target.K.shape[0]) - target.K.dot(self.H)).dot(target.P))
            
        # Step 4.1 (Update current)
        for target in self.targets:
            target.w_old = target.w
            target.w = (1 - self.Pd) * target.w
       
        # Step 4.2 (Create new)
        targets_copy = deepcopy(self.targets)
        for z in self.measurements:
            new_targets = []
            sum_weight = 0
            z = np.array(z)
            for target in targets_copy:
                new_target_w = self.Pd * target.w_old * mvn.pdf(z, target.Eta.flatten(), target.S)
                new_target_m = deepcopy(target.m + target.K.dot((z - target.Eta)))
                new_targets.append(Gaussian(w=new_target_w, m=new_target_m, P=target.P_new, label=target.label,z_history=deepcopy(target.z_history), z_history_len=target.z_history_len))  # NEW_BORN
                sum_weight += new_target_w

            for new_target in new_targets:
                new_target.w = new_target.w / (self.lam + sum_weight)
                if new_target.z_history_len == 2:
                    new_target.z_history.pop(0)
                    new_target.z_history_len -= 1
                new_target.z_history.append(z)
                new_target.z_history_len += 1
            self.targets.extend(deepcopy(new_targets))

        # Step 5 (Merge)
        I = deepcopy(self.targets)
        I[:] = [x for x in I if x.w > self.T]
        I.sort(key=operator.attrgetter('w'), reverse=True)
        I = I[:100]
        self.targets.clear()
        while(I != []):
            j = max(enumerate(I), key=lambda x: x[1].w)[0]
            L = []
            for i in range(len(I)):
                if (I[i].label == I[j].label) and (distance.mahalanobis(I[i].m, I[j].m, np.linalg.inv(I[i].P)) < self.U):
                    L.append(I[i])
            w_L = sum([l.w for l in L])
            m_L = deepcopy((1 / w_L) * sum([l.w * l.m for l in L]))
            P_L = deepcopy((1 / w_L) * sum([l.w * (l.P + np.outer((m_L - l.m), (m_L - l.m)).T) for l in L]))
            self.targets.append(Gaussian(w=w_L, m=m_L, P=P_L, label=I[j].label, z_history=deepcopy(I[j].z_history), z_history_len=I[j].z_history_len))
            for element in L:
                if element in I:
                    I.remove(element)

    def mergeLabels(self):
        I = deepcopy(self.targets)
        I[:] = [x for x in I if x.w > self.T]
        I.sort(key=operator.attrgetter('w'), reverse=True)
        I = I[:100]
        self.targets.clear()
        while(I != []):
            j = max(enumerate(I), key=lambda x: x[1].w)[0]
            L = []
            labels = []
            for i in range(len(I)):
                if all([np.allclose(x, y) for x, y in zip(I[i].z_history, I[j].z_history)]) and (distance.mahalanobis(I[i].m, I[j].m, np.linalg.inv(I[i].P)) < self.U):
                    L.append(I[i])
                    labels.append(I[i].label)
            w_L = sum([l.w for l in L])
            m_L = deepcopy((1 / w_L) * sum([l.w * l.m for l in L]))
            P_L = deepcopy((1 / w_L) * sum([l.w * (l.P + np.outer((m_L - l.m), (m_L - l.m)).T) for l in L]))
            labels.sort()
            self.targets.append(Gaussian(w=w_L, m=m_L, P=P_L, label=labels[0], z_history=deepcopy(I[j].z_history), z_history_len= I[j].z_history_len))
            for element in L:
                if element in I:
                    I.remove(element)

        self.targets.sort(key=operator.attrgetter('w'), reverse=True)
        unique_labels = []
        for target in self.targets:
            if target.label not in unique_labels:
                unique_labels.append(target.label)
            elif target.z_history_len == 2:
                target.label = self.label_counter
                self.label_counter += 1

        for target in self.targets:
            if target.w > self.w_th:
                self.trajectories[target.label].append(target)

    def fixTrajectories(self):
        tmp = deepcopy(self.trajectories)
        for key, value in tmp.items():
            if len(value) <= self.trajectory_prune:
                del self.trajectories[key]

    def plotPHD(self, ax):
        targets_to_plot = [x for x in self.targets if x.w > self.T_plot]
        for filter in targets_to_plot: 
            ax.plot(filter.m[0], filter.m[1], "+", color="black", label="PHD")
            ax.annotate('_____' + str(filter.label) +  '_____', xy=(filter.m[0], filter.m[1]), textcoords='data')
            confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax, edgecolor="blue")


