import os, re
import numpy as np
import matplotlib.pyplot as plt

# ---------- parameters: must match blockMeshDict and the generator ----------
case = "."
L, H = 40.0, 1.3          # domain length and height [m]
nx, ny = 800, 130         # cells in x and y
g, h, a0 = 9.81, 1.0, 0.1 # gravity, water depth, initial amplitude
t_fit_min = 1.0           # ignore the startup transient when fitting speed
# -----------------------------------------------------------------------------

dx, dy = L/nx, H/ny
xc = (np.arange(nx) + 0.5)*dx

def read_scalar(path):
    txt = open(path).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\d+\s*\(', txt)
    if m is None:                                    # uniform field
        u = re.search(r'internalField\s+uniform\s+([^;]+);', txt)
        return np.full(nx*ny, float(u.group(1)))
    body = txt[m.end(): txt.index(')', m.end())]
    return np.array(body.split(), dtype=float)

# collect time directories that contain alpha.water
times = []
for d in os.listdir(case):
    try:
        t = float(d)
    except ValueError:
        continue
    if os.path.isfile(os.path.join(case, d, "alpha.water")):
        times.append((t, d))
times.sort()
if not times:
    raise SystemExit("No time directories with alpha.water found "
                     "(did you run reconstructPar? is the output ascii?)")

t_arr, xcrest, amp, etas = [], [], [], []
for t, d in times:
    alpha = read_scalar(os.path.join(case, d, "alpha.water")).reshape(ny, nx)
    eta = alpha.sum(axis=0)*dy - h                   # surface elevation per column
    i = np.argmax(eta)
    # parabolic refinement of the crest position and height
    if 0 < i < nx-1:
        y0, y1, y2 = eta[i-1], eta[i], eta[i+1]
        den = y0 - 2*y1 + y2
        shift = 0.5*(y0 - y2)/den if den != 0 else 0.0
        xpk = xc[i] + shift*dx
        apk = y1 - 0.25*(y0 - y2)*shift
    else:
        xpk, apk = xc[i], eta[i]
    t_arr.append(t); xcrest.append(xpk); amp.append(apk); etas.append(eta)

t_arr = np.array(t_arr)
xcrest = np.unwrap(np.array(xcrest), period=L)       # undo wrapping at cyclic ends
amp = np.array(amp)

sel = t_arr >= t_fit_min
if sel.sum() < 2:
    sel = np.ones_like(t_arr, dtype=bool)
c_meas = np.polyfit(t_arr[sel], xcrest[sel], 1)[0]
c_th = np.sqrt(g*(h + a0))

print(f"times analysed      : {len(t_arr)}  (t = {t_arr[0]:g} to {t_arr[-1]:g} s)")
print(f"measured speed      : {c_meas:.4f} m/s")
print(f"sqrt(g(h+a0))       : {c_th:.4f} m/s")
print(f"amplitude start/end : {amp[0]:.4f} / {amp[-1]:.4f} m "
      f"({100*(amp[-1]/amp[0]-1):+.1f} %)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t_arr, xcrest, 'o-')
ax[0].set(xlabel="t [s]", ylabel="crest position, unwrapped [m]")
ax[1].plot(t_arr, amp, 'o-')
ax[1].axhline(a0, color='k', ls='--', lw=0.8)
ax[1].set(xlabel="t [s]", ylabel="amplitude [m]")

# last profile vs sech^2, centred on the crest
eta = etas[-1]
k = np.sqrt(3*amp[-1]/(4*h**3))
xs = (xc - xcrest[-1] + L/2) % L - L/2
order = np.argsort(xs)
ax[2].plot(xs[order], eta[order], '.', label="simulation")
ax[2].plot(xs[order], amp[-1]/np.cosh(k*xs[order])**2, '-', label="sech²")
ax[2].set(xlabel="x - x_crest [m]", ylabel="η [m]", xlim=(-15, 15))
ax[2].legend()

plt.tight_layout()
plt.savefig("soliton_analysis.png", dpi=150)
plt.show()
