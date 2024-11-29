import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve

from numerical_electrostatics import NumericalElectrostatics


def ode_system(t, input_vector):
    x = input_vector[0]
    y = input_vector[1]

    vx = input_vector[2]
    vy = input_vector[3]

    f = 1e5  # Hz

    applied_voltage = 1000  # V

    top_potential = applied_voltage * np.sin(f * 2 * np.pi * t)
    bottom_potential = applied_voltage * np.cos(f * 2 * np.pi * t)

    electrostatics = NumericalElectrostatics(
        top_potential=top_potential, bottom_potential=bottom_potential
    )

    electrostatics.top_potential = top_potential
    electrostatics.bottom_potential = bottom_potential

    potential_array = electrostatics.solve_system()

    ky = np.array(
        [
            [0, 0, 1 / 12, 0, 0],
            [0, 0, -2 / 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 2 / 3, 0, 0],
            [0, 0, -1 / 12, 0, 0],
        ]
    )

    kx = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    gradient_x = convolve(potential_array, kx)
    gradient_y = convolve(potential_array, ky)

    step_size = electrostatics.dx
    x_grid_size = electrostatics.x_grid_size
    y_grid_size = electrostatics.y_grid_size

    x_max = x_grid_size * step_size
    y_max = y_grid_size * step_size

    x_interpolation_grid = np.linspace(0, x_max, x_grid_size)
    y_interpolation_grid = np.linspace(0, y_max, y_grid_size)

    interp_x = RegularGridInterpolator(
        (x_interpolation_grid, y_interpolation_grid), gradient_x, method="quintic"
    )
    interp_y = RegularGridInterpolator(
        (x_interpolation_grid, y_interpolation_grid), gradient_y, method="quintic"
    )

    particle_position = np.array([x, y])
    gradient_at_particle_x = interp_x(particle_position)
    gradient_at_particle_y = interp_y(particle_position)

    # Combine the gradients if needed
    gradient_at_particle = np.array([gradient_at_particle_x, gradient_at_particle_y])

    ion_mass = 137.9052472 * 1.66053906660e-27  # kg
    ion_charge = 1.602 * 1e-19

    particle_force = ion_charge * gradient_at_particle
    particle_acceleration = particle_force / ion_mass

    ax, ay = particle_acceleration[0].item(), particle_acceleration[1].item()

    print(
        f"t: {t}. Particle is at: {round(x, 5)},{round(y, 5)}, accelerating at {round(ax, 5)},{round(ay, 5)}. Speed: {round(vx, 5)},{round(vy,5)}"
    )

    return np.array([vx, vy, ax, ay])


def main():
    electrostatics = NumericalElectrostatics(top_potential=1, bottom_potential=-1)
    potential_array = electrostatics.solve_system()
    electrostatics.plot_results(potential_array)

    # step_size = electrostatics.dx
    # x_grid_size = electrostatics.x_grid_size
    # y_grid_size = electrostatics.y_grid_size

    # x_max = x_grid_size * step_size
    # y_max = y_grid_size * step_size

    # initial_conditions = np.array([0.10001, 0.100001, 0.0, 0.0])
    # t_span = (0, 7e-3)

    # # ode_system(t=0, input_vector=initial_conditions)

    # solution = solve_ivp(ode_system, t_span, initial_conditions, method="DOP853")

    # # Extent for the potential array to match the physical dimensions
    # extent = [0, x_max, 0, y_max]

    # plt.figure()

    # # Plotting the potential array
    # plt.imshow(
    #     potential_array, extent=extent, origin="lower", aspect="auto", cmap="viridis"
    # )

    # # Plotting the particle trajectory
    # plt.plot(solution.y[0], solution.y[1], "-x", color="red", label="Trajectory")

    # # plt.xlim([0.0249, 0.0254])
    # # plt.ylim([0.0243, 0.0251])

    # # Adding labels and legend
    # plt.xlabel("x position (mm)")
    # plt.ylabel("y position (mm)")
    # plt.legend()

    # plt.tight_layout()

    # # Show the plot
    # plt.colorbar(label="Potential")

    # plt.savefig(
    #     f"numerical_electrostatics/figures/{datetime.now().strftime('%d%m%Y - ')}Numerical electrostatics - quadrupole - Ba138 trapping - higher resolution.png",
    #     dpi=300,
    # )

    # plt.show()

    # # Extract time, vx, and vy from the solution
    # time = solution.t
    # vx = solution.y[2]  # vx is stored in the 3rd row of solution.y
    # vy = solution.y[3]  # vy is stored in the 4th row of solution.y

    # # Create the plot
    # plt.figure(figsize=(12, 6))

    # # Plot vx and vy
    # plt.plot(time, vx, label="Vx", color="b")
    # plt.plot(time, vy, label="Vy", color="r")

    # # Label the axes
    # plt.xlabel("Time")
    # plt.ylabel("Velocity")

    # # Add a title
    # plt.title("Velocity Vx and Vy over Time")

    # # Add a grid
    # plt.grid(True)

    # # Add a legend
    # plt.legend()

    # plt.savefig(
    #     f"numerical_electrostatics/figures/{datetime.now().strftime('%d%m%Y - ')}Numerical electrostatics - quadrupole - Ba138 trapping - velocities - higher resolution.png",
    #     dpi=300,
    # )

    # # Show the plot
    # plt.show()


if __name__ == "__main__":
    main()
