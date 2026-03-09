import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================
# File paths
# ==========================
input_file = "/eos/user/z/zhilang/livnczz4l/MG_LHE_ppZZto4L_LO_theta_1e-4/result/update_MG_325300_1e-4_weighted_events.root"
output_file_sm  = "SM_LHEF.root"
output_file_liv = "LIV_LHEF.root"
tree_name = "LHEF"

os.makedirs("plots", exist_ok=True)

# ==========================
# Read ROOT tree
# ==========================
with uproot.open(input_file) as f:
    tree = f[tree_name]

    pid    = tree["Particle.PID"].array()
    status = tree["Particle.Status"].array()

    pt     = tree["Particle.PT"].array()
    eta    = tree["Particle.Eta"].array()
    phi    = tree["Particle.Phi"].array()
    mass   = tree["Particle.M"].array()
    pz     = tree["Particle.Pz"].array()
    energy = tree["Particle.E"].array()

    sm_values = tree["SM_Amplitude"].array()
    nc_values = tree["NC_Amplitude"].array()

# ==========================
# Event weights (LIV)
# ==========================
liv_weight = 1 + nc_values / sm_values

# ==========================
# Select Z1 and Z2 (PID=23, status=2)
# ==========================
is_Z = (pid == 23) & (status == 2)

Z_PT     = pt[is_Z]
Z_Eta    = eta[is_Z]
Z_Phi    = phi[is_Z]
Z_Mass   = mass[is_Z]
Z_Pz     = pz[is_Z]
Z_Energy = energy[is_Z]

mask_twoZ = (ak.num(Z_PT) == 2)

Z_PT     = Z_PT[mask_twoZ]
Z_Eta    = Z_Eta[mask_twoZ]
Z_Phi    = Z_Phi[mask_twoZ]
Z_Mass   = Z_Mass[mask_twoZ]
Z_Pz     = Z_Pz[mask_twoZ]
Z_Energy = Z_Energy[mask_twoZ]

liv_weight = liv_weight[mask_twoZ]

# ==========================
# Convert to NumPy arrays
# ==========================
z1_pt     = ak.to_numpy(Z_PT[:, 0])
z2_pt     = ak.to_numpy(Z_PT[:, 1])
z1_eta    = ak.to_numpy(Z_Eta[:, 0])
z2_eta    = ak.to_numpy(Z_Eta[:, 1])
z1_phi    = ak.to_numpy(Z_Phi[:, 0])
z2_phi    = ak.to_numpy(Z_Phi[:, 1])
z1_mass   = ak.to_numpy(Z_Mass[:, 0])
z2_mass   = ak.to_numpy(Z_Mass[:, 1])
z1_pz     = ak.to_numpy(Z_Pz[:, 0])
z2_pz     = ak.to_numpy(Z_Pz[:, 1])
z1_energy = ak.to_numpy(Z_Energy[:, 0])
z2_energy = ak.to_numpy(Z_Energy[:, 1])
liv_weight = ak.to_numpy(liv_weight)

print("Total events:", len(z1_pt))

# ==========================
# Save SM ROOT (all events, weight=1)
# ==========================
with uproot.recreate(output_file_sm) as f:
    f["LHEF"] = {
        "Z1_PT": z1_pt.astype(np.float32),
        "Z1_Eta": z1_eta.astype(np.float32),
        "Z1_Phi": z1_phi.astype(np.float32),
        "Z1_Mass": z1_mass.astype(np.float32),
        "Z1_Pz": z1_pz.astype(np.float32),
        "Z1_Energy": z1_energy.astype(np.float32),
        "Z2_PT": z2_pt.astype(np.float32),
        "Z2_Eta": z2_eta.astype(np.float32),
        "Z2_Phi": z2_phi.astype(np.float32),
        "Z2_Mass": z2_mass.astype(np.float32),
        "Z2_Pz": z2_pz.astype(np.float32),
        "Z2_Energy": z2_energy.astype(np.float32),
        "Weight": np.ones_like(z1_pt, dtype=np.float32)
    }

# ==========================
# Save LIV ROOT (only positive weights)
# ==========================
mask_pos = liv_weight > 0
print("Positive-weight LIV events:", mask_pos.sum())

with uproot.recreate(output_file_liv) as f:
    f["LHEF"] = {
        "Z1_PT": z1_pt[mask_pos].astype(np.float32),
        "Z1_Eta": z1_eta[mask_pos].astype(np.float32),
        "Z1_Phi": z1_phi[mask_pos].astype(np.float32),
        "Z1_Mass": z1_mass[mask_pos].astype(np.float32),
        "Z1_Pz": z1_pz[mask_pos].astype(np.float32),
        "Z1_Energy": z1_energy[mask_pos].astype(np.float32),
        "Z2_PT": z2_pt[mask_pos].astype(np.float32),
        "Z2_Eta": z2_eta[mask_pos].astype(np.float32),
        "Z2_Phi": z2_phi[mask_pos].astype(np.float32),
        "Z2_Mass": z2_mass[mask_pos].astype(np.float32),
        "Z2_Pz": z2_pz[mask_pos].astype(np.float32),
        "Z2_Energy": z2_energy[mask_pos].astype(np.float32),
        "Weight": liv_weight[mask_pos].astype(np.float32)
    }

# ==========================
# Plot SM vs LIV using weights
# ==========================
plt.figure(figsize=(15,10))
var_names = ["Z1_PT","Z1_Eta","Z1_Phi","Z1_Mass","Z1_Pz","Z1_Energy",
             "Z2_PT","Z2_Eta","Z2_Phi","Z2_Mass","Z2_Pz","Z2_Energy"]

vars_values_sm = [z1_pt, z1_eta, z1_phi, z1_mass, z1_pz, z1_energy,
                  z2_pt, z2_eta, z2_phi, z2_mass, z2_pz, z2_energy]

vars_values_liv = [z1_pt[mask_pos], z1_eta[mask_pos], z1_phi[mask_pos], z1_mass[mask_pos],
                   z1_pz[mask_pos], z1_energy[mask_pos],
                   z2_pt[mask_pos], z2_eta[mask_pos], z2_phi[mask_pos], z2_mass[mask_pos],
                   z2_pz[mask_pos], z2_energy[mask_pos]]

for i, (v_sm, v_liv) in enumerate(zip(vars_values_sm, vars_values_liv)):
    plt.subplot(4,3,i+1)
    plt.hist(v_sm, bins=50, histtype='step', label='SM')
    plt.hist(v_liv, bins=50, weights=liv_weight[mask_pos], histtype='step', label='LIV')
    plt.xlabel(var_names[i])
    plt.ylabel("Counts")
    plt.legend()

plt.tight_layout()
plt.savefig("plots/SM_vs_LIV_Z_variables.png", dpi=150)
plt.show()

print("Done. SM kept all events, LIV negative weights removed, ROOT files and plots created.")