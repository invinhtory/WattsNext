import os
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aerosandbox as asb

# Create Optimization Animation
# INPUTS:
# history_dir (str): the directory to pull iterations from to create the animation
# output_file (str): filename of the output gif file
def create_optimization_animation(history_dir, output_file='optimization.gif'):
    # Load the full optimization history
    history_path = os.path.join(history_dir, "optimization_history.json")
    with open(history_path, 'r') as f:
        history = json.load(f)

    if not history:
        print("Error: Optimization history is empty.")
        return

    # Setup the figure, two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    cp_values = []
    iterations = []

    # helper function to generate a single frame of the gif
    def animate(frame):
        ax1.clear()
        ax2.clear()

        try:
            # Gets the data at a single frame
            data = history[frame]
            kulfan_params = data['parameters']

            # creates an airfoil to plot, and gets the list of coordinates of the airfoil
            foil = asb.KulfanAirfoil(
                lower_weights=kulfan_params["lower_weights"],
                upper_weights=kulfan_params["upper_weights"],
                leading_edge_weight=kulfan_params.get("leading_edge_weight", 0.5),
                TE_thickness=kulfan_params.get("TE_thickness", 0.002)
            )

            coords = foil.coordinates  # coords is shape (n, 2)
            x, y = coords[:, 0], coords[:, 1]

            # In the leftmost graph, plot the airfoil shape of this iteration's airfoil
            ax1.plot(x, y, 'b-', linewidth=2)
            ax1.set_title(f'Iteration {frame}')
            ax1.grid(True)
            ax1.axis('equal')
            ax1.set_xlim(-0.1, 1.1)
            ax1.set_ylim(-0.5, 0.5)

            # adds this iterations Cp value to the list of all previous CP values
            cp = data['CP']
            cp_values.append(cp)
            iterations.append(frame)
            
            # plots a graph of Cp values from the first iteration to the current iteration
            ax2.plot(iterations, cp_values, 'b-')
            ax2.set_title('Power Coefficient History')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('CP')
            ax2.grid(True)
            ax2.set_ylim(0, max(max(cp_values), 0.6))
            ax2.text(0.02, 0.98, f'Current CP: {cp:.4f}', transform=ax2.transAxes, verticalalignment='top')

        except Exception as e:
            print(f"Error in frame {frame}: {e}")
            ax1.text(0.5, 0.5, f'Frame error: {e}', ha='center', va='center')

    # Create and save animation
    anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=200, repeat=True)

    try:
        anim.save(output_file, writer='pillow')
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    plt.close()
