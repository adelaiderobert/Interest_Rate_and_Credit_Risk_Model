import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


path = "statmon_E4_M1_M_2015_2025_SYN_v2.xlsx"   
sheet = "E4_Spot_CHF"

#Cleaning and preparing data 
df = pd.read_excel(path, sheet_name=sheet, header=0, engine="openpyxl")
df = df.dropna(how="all")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
df = df[df["Year"].notna() & df["Month"].notna()].copy()
df["Date"] = pd.to_datetime(dict(
    year=df["Year"].astype("Int64"),
    month=df["Month"].astype("Int64"),
    day=1
), errors="coerce")

req_cols = ["2 years", "3 years", "4 years", "5 years", "7 years", "10 years", "20 years", "30 years"]
maturity_cols = req_cols
taus = np.array([2, 3, 4, 5, 7, 10, 20, 30], dtype=float)
mask = (df["Date"] >= "2015-08-01") & (df["Date"] <= "2025-07-31")
df = df.loc[mask].copy()
july_mask = (df["Date"].dt.year == 2025) & (df["Date"].dt.month == 7)
row = df.loc[july_mask].iloc[-1]
obs = row[req_cols].astype(float).to_numpy()


# -----------------------------------
# (a) NS / NSS feature (loading) funcs
# -----------------------------------
def ns_features(tau, lam1):
    tau = np.asarray(tau, dtype=float)
    x1 = (1 - np.exp(-tau / lam1)) / (tau / lam1)
    x2 = x1 - np.exp(-tau / lam1)
    return np.column_stack([np.ones_like(tau), x1, x2])

def nss_features(tau, lam1, lam2):
    tau = np.asarray(tau, dtype=float)
    x1 = (1 - np.exp(-tau / lam1)) / (tau / lam1)
    x2 = x1 - np.exp(-tau / lam1)
    x3 = (1 - np.exp(-tau / lam2)) / (tau / lam2) - np.exp(-tau / lam2)
    return np.column_stack([np.ones_like(tau), x1, x2, x3])

def fit_linear(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    return beta, yhat, rmse

# -----------------------------------
# 3) Fit NS (grid over λ1) and NSS (grid over λ1, λ2)
# -----------------------------------
lam_grid  = np.geomspace(0.1, 15.0, 200)

# --- NS ---
best_ns_rmse = np.inf
best_ns = None
for lam1 in lam_grid:
    X = ns_features(taus, lam1)
    beta, yhat, rmse = fit_linear(X, obs)
    if rmse < best_ns_rmse:
        best_ns_rmse = rmse
        best_ns = (beta, lam1, yhat)

ns_beta, ns_lam1, ns_fit = best_ns

# --- NSS ---
best_nss_rmse = np.inf
best_nss = None
for lam1 in lam_grid:
    for lam2 in lam_grid:
        X = nss_features(taus, lam1, lam2)
        beta, yhat, rmse = fit_linear(X, obs)
        if rmse < best_nss_rmse:
            best_nss_rmse = rmse
            best_nss = (beta, lam1, lam2, yhat)

nss_beta, nss_lam1, nss_lam2, nss_fit = best_nss

# -----------------------------------
# Results
# -----------------------------------
print("Observed yields (July 2025):")
for t, y in zip(taus, obs):
    print(f"{t:>2.0f}y : {y:.6f}")

print("\n--- NS parameters ---")
print(f"beta0 = {ns_beta[0]:.8f}")
print(f"beta1 = {ns_beta[1]:.8f}")
print(f"beta2 = {ns_beta[2]:.8f}")
print(f"lambda1 = {ns_lam1:.6f}")
print(f"RMSE = {best_ns_rmse:.8f}")

print("\n--- NSS parameters ---")
print(f"beta0 = {nss_beta[0]:.8f}")
print(f"beta1 = {nss_beta[1]:.8f}")
print(f"beta2 = {nss_beta[2]:.8f}")
print(f"beta3 = {nss_beta[3]:.8f}")
print(f"lambda1 = {nss_lam1:.6f}")
print(f"lambda2 = {nss_lam2:.6f}")
print(f"RMSE = {best_nss_rmse:.8f}")

# -----------------------------
# Plots : observed vs fitted curves
# -----------------------------
plt.figure(figsize=(8,5))
tau_grid = np.linspace(0.25, 30.0, 250)
ns_curve  = ns_features(tau_grid, ns_lam1) @ ns_beta
nss_curve = nss_features(tau_grid, nss_lam1, nss_lam2) @ nss_beta

plt.scatter(taus, obs, label="Observed (Jul 2025)", zorder=3)
plt.plot(tau_grid, ns_curve,  label="NS fit",  linewidth=2)
plt.plot(tau_grid, nss_curve, label="NSS fit", linewidth=2)
plt.xlabel("Maturity (years)")
plt.ylabel("Zero-coupon yield")
plt.title("Yield Curve — Observed vs NS and NSS fits (July 2025)")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# (b) PCA of monthly yield changes
# ---------------------------------------------------------------
# Compute month-to-month changes Δy_t(T)  
dy = df[req_cols].astype(float).diff()
dy.index = df["Date"]
dy = dy.dropna(how="all")

# Covariance
cov_mat = np.cov(dy.values, rowvar=False, ddof=0)
cov_df = pd.DataFrame(cov_mat, index=req_cols, columns=req_cols)
print("\n=== Covariance matrix of monthly yield changes ===")
print(cov_df.round(6).to_string())

# PCA
pca = PCA(n_components=min(3, len(req_cols)))
pca.fit(dy.values)

eigenvalues = pca.explained_variance_
explained_ratio = pca.explained_variance_ratio_
cum_explained = explained_ratio.cumsum()

eig_df = pd.DataFrame({
    "Eigenvalue": eigenvalues,
    "Explained variance %": explained_ratio * 100.0,
    "Cumulative %": cum_explained * 100.0
}, index=[f"PC{i+1}" for i in range(len(eigenvalues))])

print("\n=== First three eigenvalues & variance explained ===")
print(eig_df.round(4).to_string())

loadings = pca.components_
loadings_df = pd.DataFrame(loadings.T, index=maturity_cols,
                           columns=[f"PC{i+1}" for i in range(loadings.shape[0])])
print("\n=== PCA loadings (eigenvectors) by maturity ===")
print(loadings_df.round(4).to_string())


# Plot PCA loadings 
colors = ["tab:blue", "tab:orange", "tab:green"]  # PC1, PC2, PC3

plt.figure(figsize=(8,5))
for i in range(min(3, loadings.shape[0])):
    plt.plot(taus, loadings[i, :], marker="o", linewidth=2,
             color=colors[i], label=f"PC{i+1}")
plt.axhline(0, linewidth=1, color="gray")
plt.xlabel("Maturity (years)")
plt.ylabel("Loading")
plt.title("PCA loadings vs maturity (monthly yield changes)")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------------
# (c) Compare NS theoretical factors vs empirical PCA loadings
# ---------------------------------------------------------------
ns_basis = ns_features(taus, ns_lam1) #I0, I1, I2 

plt.figure(figsize=(9,6))
for i in range(3):
    # PCA
    plt.plot(taus, loadings[i, :], marker="o", linewidth=2,
             color=colors[i], label=f"PCA PC{i+1}")
    # NS
    ns_label = ["NS level (theoretical)", "NS slope (theoretical)", "NS curvature (theoretical)"][i]
    plt.plot(taus, ns_basis[:, i], linestyle="--", linewidth=2,
             color=colors[i], label=ns_label)

plt.axhline(0, color="gray", linewidth=1)
plt.xlabel("Maturity (years)")
plt.ylabel("Normalized loading")
plt.title("Nelson–Siegel theoretical factors vs empirical PCA loadings")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()




