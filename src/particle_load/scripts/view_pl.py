import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


class CheckPL:
    def __init__(self, hdf5_pl_file):

        data = self.load_hdf5_file(hdf5_pl_file)

        self.find_glass_mass(data)

        self.plot_pl(data)

    def plot_pl(self, data):
        masses, counts = np.unique(data["masses"], return_counts=True)

        f, axarr = plt.subplots(1, 4, figsize=(14, 4))
        for i, m in enumerate(masses):
            if i > 3:
                break
            mask = np.where(
                (data["coords_z"] >= data["load_coords"][2] - data["radius"] * 100)
                & (data["coords_z"] <= data["load_coords"][2] + data["radius"] * 100)
                & (data["masses"] == m)
            )
            axarr[0].scatter(data["coords_x"][mask], data["coords_y"][mask], s=1)

            for j, (X, Y, xi, yi, Z, zi) in enumerate(
                zip(
                    ["coords_x", "coords_y", "coords_x"],
                    ["coords_y", "coords_z", "coords_z"],
                    [0, 1, 0],
                    [1, 2, 2],
                    ["coords_z", "coords_x", "coords_y"],
                    [2, 0, 1],
                )
            ):
                mask = np.where(
                    (data[Z] >= data["load_coords"][zi] - data["cell_length"] / 5.0)
                    & (data[Z] <= data["load_coords"][zi] + data["cell_length"] / 5.0)
                    & (data["masses"] == m)
                )
                axarr[1 + j].scatter(data[X][mask], data[Y][mask], s=1)

                axarr[1 + j].set_xlim(
                    data["load_coords"][xi] - data["radius"] * 1.5,
                    data["load_coords"][xi] + data["radius"] * 1.5,
                )
                axarr[1 + j].set_ylim(
                    data["load_coords"][yi] - data["radius"] * 1.5,
                    data["load_coords"][yi] + data["radius"] * 1.5,
                )
        plt.tight_layout(pad=0.1)
        plt.savefig("temp.png")
        plt.close()

    def load_hdf5_file(self, hdf5_pl_file):
        data = {}

        f = h5py.File(hdf5_pl_file, "r")
        data["coords_x"] = f["Coordinates_x"][...]
        data["coords_y"] = f["Coordinates_y"][...]
        data["coords_z"] = f["Coordinates_z"][...]
        data["masses"] = f["Masses"][...]
        data["load_coords"] = f["Header"].attrs["coords"]
        data["radius"] = f["Header"].attrs["radius"]
        data["cell_length"] = f["Header"].attrs["cell_length"]
        f.close()

        return data

    def find_glass_mass(self, data):
        masses, counts = np.unique(data["masses"], return_counts=True)
        return masses[0]


if __name__ == "__main__":
    hdf5_pl_file = sys.argv[1]

    CheckPL(hdf5_pl_file)
