from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class IndexReturn:
    in_bounds: bool
    in_bc: bool
    is_real: bool
    potential: float


class NumericalElectrostatics:
    def __init__(self, top_potential, bottom_potential):
        self.upper_location = 90
        self.lower_location = 115

        self.left_location = 75
        self.right_location = 125

        self.top_potential = top_potential
        self.bottom_potential = bottom_potential

        self.plate_thickness = 5

        self.x_grid_size = 200
        self.y_grid_size = 200

        self.dielectric_constant = 80

        self.dx = 0.001
        self.delta = 1 / self.dx**2

        self.material_array, self.dielectric_array = self.create_geometry()
        self.index_map = self.create_index_map()

        self.sparse_entries = []
        self.vector_entries = np.zeros(
            (len(self.material_array[self.material_array == 0]))
        )

    def create_plates(self, material_array, dielectric_array):
        material_array[
            self.lower_location : self.lower_location + self.plate_thickness,
            self.left_location : self.right_location,
        ] = self.bottom_potential
        material_array[
            self.upper_location : self.upper_location + self.plate_thickness,
            self.left_location : self.right_location,
        ] = self.top_potential

        dielectric_array[
            self.upper_location : self.lower_location,
            self.left_location : self.right_location,
        ] = self.dielectric_constant

        return material_array, dielectric_array

    def create_circle(
        self, radius, center, material_array, dielectric_array, potential
    ):
        center_y, center_x = center
        y, x = np.ogrid[: self.y_grid_size, : self.x_grid_size]
        distance_from_center = (y - center_y) ** 2 + (x - center_x) ** 2
        material_array[distance_from_center <= radius**2] = potential

        return material_array, dielectric_array

    def create_geometry(self):
        material_array = np.zeros((self.y_grid_size, self.x_grid_size))
        dielectric_array = np.ones((self.y_grid_size, self.x_grid_size))

        material_array, dielectric_array = self.create_circle(
            radius=50,
            center=(100, 10),
            material_array=material_array,
            dielectric_array=dielectric_array,
            potential=self.bottom_potential,
        )

        material_array, dielectric_array = self.create_circle(
            radius=50,
            center=(10, 100),
            material_array=material_array,
            dielectric_array=dielectric_array,
            potential=self.top_potential,
        )

        material_array, dielectric_array = self.create_circle(
            radius=50,
            center=(100, 190),
            material_array=material_array,
            dielectric_array=dielectric_array,
            potential=self.bottom_potential,
        )

        material_array, dielectric_array = self.create_circle(
            radius=50,
            center=(190, 100),
            material_array=material_array,
            dielectric_array=dielectric_array,
            potential=self.top_potential,
        )

        # material_array, dielectric_array = self.create_circle(
        #     radius=1,
        #     center=(1, 1),
        #     material_array=material_array,
        #     dielectric_array=dielectric_array,
        #     potential=1,
        # )

        # material_array, dielectric_array = self.create_plates(
        #     material_array=material_array, dielectric_array=dielectric_array
        # )

        return material_array, dielectric_array

    def check_index(self, y_index, x_index):
        if (
            y_index < 0
            or y_index > self.y_grid_size - 1
            or x_index < 0
            or x_index > self.x_grid_size - 1
        ):
            return IndexReturn(in_bounds=False, in_bc=False, is_real=False, potential=0)

        if self.material_array[y_index, x_index] != 0:
            return IndexReturn(
                in_bounds=True,
                in_bc=True,
                is_real=False,
                potential=self.material_array[y_index, x_index],
            )

        return IndexReturn(in_bounds=True, in_bc=False, is_real=True, potential=0)

    def get_position(self, current_index, center_index):

        y_index, x_index = current_index
        y_center, x_center = center_index

        global_current_index = y_index * self.y_grid_size + x_index
        global_center_index = y_center * self.y_grid_size + x_center

        index_return = self.check_index(y_index, x_index)

        if not index_return.in_bounds:
            matrix_entry = None
            vector_entry = None
            return 0

        center_position_index = self.index_map[global_center_index]

        if index_return.in_bc:
            matrix_entry = None
            vector_entry = -index_return.potential / self.delta
            self.vector_entries[center_position_index] = vector_entry
            return 0

        current_position_index = self.index_map[global_current_index]

        matrix_entry = (center_position_index, current_position_index, 1 / self.delta)
        vector_entry = 0

        self.sparse_entries.append(matrix_entry)

        return 0

    def create_index_map(self):
        index_map = {}
        current_index_count = 0
        for y_index in range(0, self.y_grid_size):
            for x_index in range(0, self.x_grid_size):
                index_return = self.check_index(y_index, x_index)

                global_index = y_index * self.y_grid_size + x_index

                if index_return.is_real:
                    index_map[global_index] = current_index_count
                    current_index_count += 1

        return index_map

    def solve_system(self):
        for y_index in range(0, self.y_grid_size):
            for x_index in range(0, self.x_grid_size):

                center_index = (y_index, x_index)

                center_return = self.check_index(y_index, x_index)

                if not center_return.is_real:
                    continue

                global_center_index = y_index * self.y_grid_size + x_index
                center_position_index = self.index_map[global_center_index]

                north_index = (y_index + 1, x_index)
                south_index = (y_index - 1, x_index)
                east_index = (y_index, x_index + 1)
                west_index = (y_index, x_index - 1)

                matrix_entry = (
                    center_position_index,
                    center_position_index,
                    -4 / self.delta,
                )
                self.sparse_entries.append(matrix_entry)

                self.get_position(north_index, center_index)
                self.get_position(south_index, center_index)
                self.get_position(east_index, center_index)
                self.get_position(west_index, center_index)

        rows = [entry[0] for entry in self.sparse_entries]
        cols = [entry[1] for entry in self.sparse_entries]
        data = [entry[2] for entry in self.sparse_entries]

        # print("Creating matrix!")

        sparse_matrix = coo_matrix((data, (rows, cols))).tocsc()

        plt.spy(sparse_matrix, markersize=5)
        plt.title("Sparse Matrix")
        plt.show()

        # print("Solving the system.")

        x = spsolve(A=sparse_matrix, b=self.vector_entries)

        potential_array = self.material_array.copy()

        # print("Extracting results!")

        result_counter = 0
        for y_index in range(0, self.y_grid_size):
            for x_index in range(0, self.x_grid_size):
                index_return = self.check_index(y_index, x_index)

                if index_return.is_real:
                    potential_array[y_index, x_index] = x[result_counter]
                    result_counter += 1

        return potential_array

    def plot_results(self, potential_array):
        gy, gx = np.gradient(potential_array)

        gy /= self.dielectric_array
        gx /= self.dielectric_array

        outlier_reduction_factor = 5

        gx[abs(gx) > outlier_reduction_factor * np.mean(abs(gx))] = (
            outlier_reduction_factor * np.mean(gx)
        )
        gy[abs(gy) > outlier_reduction_factor * np.mean(abs(gy))] = (
            outlier_reduction_factor * np.mean(gy)
        )

        # Create a meshgrid for arrow positions
        x = np.arange(potential_array.shape[1])
        y = np.arange(potential_array.shape[0])
        X, Y = np.meshgrid(x, y)

        # Plot the image
        plt.imshow(potential_array, cmap="jet", origin="lower")

        cbar = plt.colorbar()
        cbar.set_label("Potential (V)")

        step = 10

        # Plot the gradients as arrows
        plt.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            gx[::step, ::step],
            gy[::step, ::step],
            color="white",
            # scale=1,
            # width=0.005,
        )

        plt.xlabel("Position (mm)")
        plt.title("Capacitor with $\epsilon_r = 80$ dielectric")

        # plt.xlim([80, 120])
        # plt.ylim([80, 120])

        plt.savefig(
            f"numerical_electrostatics/figures/{datetime.now().strftime('%d%m%Y - ')}Numerical electrostatics - quadrupole - center detail.png",
            dpi=300,
        )
        plt.show()
