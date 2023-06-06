import matplotlib.pyplot as plt

def get_points_from_image(image_path):
    # Load the image
    image = plt.imread(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Initialize a list to store the selected points
    points = []

    def onclick(event):
        # Check if the mouse click is within the image boundaries
        if event.xdata is not None and event.ydata is not None:
            # Check the current toolbar mode
            if plt.get_current_fig_manager().toolbar.mode == '':  # Check for no active mode
                # Append the clicked point to the list
                points.append((event.xdata, event.ydata))

                # Plot a red circle at the clicked point
                ax.plot(event.xdata, event.ydata, 'ro')
                plt.draw()

                # Check if three points have been selected
                if len(points) == 3:
                    # Disconnect the click event listener
                    fig.canvas.mpl_disconnect(cid)

    # Connect the click event listener
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the image and wait for user input
    plt.show()

    # Return the selected points
    return points

# Example usage
image_path = "/home/matous/school_work/idp/idp-convert-coords/2022-10-06T16-34-42/frame_0.jpeg"
selected_points = get_points_from_image(image_path)

# Print the selected points
print("Selected points:", selected_points)
