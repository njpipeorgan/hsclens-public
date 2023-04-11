# To generate slab.config, change the working directory to the simulation directory, and
# $ python .../generate-slab-config.py
# Modify the max redshift and h0 below if necessary.

import astropy, re, os, glob
import astropy.cosmology
import numpy as np

with open("./snapshots/snapshot.red") as f:
    zlist = np.array([float(z) for z in re.findall("z([0-9.]+)\n", f.read())])

workingDirectory = os.getcwd()
Om0, s8 = [float(v) for v in re.findall("/sim_Om(.+?)_si(.+?)_.+$",workingDirectory)[0]]

useFirstSnapshot = False
maxRedshift = 3.0
h0 = 0.6736

if not useFirstSnapshot:
    zlist = zlist[1:]

zlist = zlist[0:np.searchsorted(zlist, maxRedshift)]
splitingChi = np.array(list(map(lambda z: astropy.cosmology.FlatLambdaCDM(H0=100 * h0, Om0=Om0).comoving_distance(z).value * h0, np.concatenate(([0.0], (zlist[:-1] + zlist[1:])/2)))))
slabThickness = np.diff(splitingChi)
slabCenter = (splitingChi[1:] + splitingChi[:-1]) / 2

snapshotNames = sorted(glob.glob("./snapshots/snapshot.?????"))
snapshotNames.reverse()
if not useFirstSnapshot:
    snapshotNames = snapshotNames[1:]

slabRedshifts = list(map(lambda x: astropy.cosmology.z_at_value(
    astropy.cosmology.FlatLambdaCDM(H0=100 * h0, Om0=Om0).comoving_distance, x),
    slabCenter * astropy.units.Mpc / h0))

if (len(slabRedshifts) > 0 and type(slabRedshifts[0]).__name__ != "float" and type(slabRedshifts[0]).__name__ != "float64"):
    slabRedshifts = list(map(lambda z: z.value, slabRedshifts))

slabConfig = "filename\tomega_m\thubble_0\tredshift\tthickness\tdistance\n"
for i in range(len(slabThickness)):
    slabConfig += "{}\t{}\t{}\t{}\t{}\t{}\n".format(
        snapshotNames[i], Om0, h0, slabRedshifts[i], slabThickness[i], slabCenter[i])

with open("./slab.config", "w") as f:
    f.write(slabConfig)
