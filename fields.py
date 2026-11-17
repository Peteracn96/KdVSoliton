import numpy as np

nx, ny = 800, 130
L, H = 40.0, 1.3
dx, dy = L/nx, H/ny
g, h, a, x0 = 9.81, 1.0, 0.1, 10.0
c = np.sqrt(g*(h + a))
k = np.sqrt(3*a/(4*h**3))

# cell centres, x fastest then y (blockMesh ordering)
xc = (np.arange(nx) + 0.5)*dx
yc = (np.arange(ny) + 0.5)*dy
X, Y = np.meshgrid(xc, yc)          # shape (ny, nx), flatten row by row
X, Y = X.ravel(), Y.ravel()

s = 1/np.cosh(k*(X - x0))
eta = a*s**2
etaX = -2*k*a*s**2*np.tanh(k*(X - x0))

alpha = np.clip((h + eta - (Y - 0.5*dy))/dy, 0, 1)
inwater = Y < h + eta
ux = np.where(inwater, c*eta/(h + eta), 0.0)
uy = np.where(inwater, -Y*c*etaX/h, 0.0)

n = nx*ny
header = lambda cls, obj: f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}

"""

with open("0/alpha.water", "w") as f:
    f.write(header("volScalarField", "alpha.water"))
    f.write("dimensions      [0 0 0 0 0 0 0];\n\n")
    f.write(f"internalField   nonuniform List<scalar> {n}\n(\n")
    f.write("\n".join(f"{v:.6g}" for v in alpha))
    f.write("\n)\n;\n\n")
    f.write("""boundaryField
{
    leftWall    { type cyclic; }
    rightWall   { type cyclic; }
    lowerWall   { type zeroGradient; }
    atmosphere  { type zeroGradient; }
    frontWall   { type empty; }
    backWall    { type empty; }
}
""")

with open("0/U", "w") as f:
    f.write(header("volVectorField", "U"))
    f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
    f.write(f"internalField   nonuniform List<vector> {n}\n(\n")
    f.write("\n".join(f"({vx:.6g} {vy:.6g} 0)" for vx, vy in zip(ux, uy)))
    f.write("\n)\n;\n\n")
    f.write("""boundaryField
{
    leftWall    { type cyclic; }
    rightWall   { type cyclic; }
    lowerWall   { type slip; }
    atmosphere  { type slip; }
    frontWall   { type empty; }
    backWall    { type empty; }
}
""")
