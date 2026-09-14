import os
import urllib.request
import zipfile
import io
import numpy as np
import scipy.optimize as optimize

def ingest_sparc_galaxy_data(galaxy_name="NGC6503", target_dir="sparc_raw_tables"):
    """Automated Public Database Table Puller"""
    url = "http://cwru.edu"
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir)
    try:
        with urllib.request.urlopen(url) as r:
            with zipfile.ZipFile(io.BytesIO(r.read())) as z:
                z.extract(f"{galaxy_name}_rotmod.txt", target_dir)
                m = np.genfromtxt(os.path.join(target_dir, f"{galaxy_name}_rotmod.txt"), skip_header=3)
                return m[:, 0], m[:, 1]
    except Exception as e: 
        print(f"Server Connection Issue: {e}")
        return None, None

class QCAUSGlobalAsymmetricFitter:
    def __init__(self, data): 
        self.data = data
        
    def dynamic_profile_prediction(self, r, omega, epsilon, m_t, m_d, r_core):
        scale = r_core * (1.0 - 0.2 * omega + 0.05 * np.log10(epsilon + 1e-15)) * np.sqrt(m_t / m_d)
        return 90.0 * (1.0 - np.exp(-r / max(scale, 0.02)))
        
    def joint_loss_function(self, vec):
        om, eps, mt, md = vec[0:4]
        if not (-1.0 <= om <= 1.0) or not (1e-15 <= eps <= 1e-2): 
            return 1e15
        return sum(np.sum(((y - self.dynamic_profile_prediction(x, om, eps, mt, md, r_c)) / (0.05*y+1e-3))**2) for r_c, (name, (x, y)) in zip(vec[4:], self.data.items()))
