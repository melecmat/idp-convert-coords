import matplotlib.pyplot as plt
import argparse
import cv2

def get_points_from_image(image):
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
                points.append([event.xdata, event.ydata])

                # Plot a red circle at the clicked point
                ax.plot(event.xdata, event.ydata, 'ro')

                # Number the point
                point_number = len(points)
                ax.text(event.xdata, event.ydata, str(point_number), color='red', fontsize=12)

                plt.draw()

    # Connect the click event listener
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the image and wait for user input
    plt.show()

    # Return the selected points
    return points

parser = argparse.ArgumentParser(description="input a list of point on an image")
parser.add_argument("video_path")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
ret, image = cap.read()
cap.release()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
selected_points = get_points_from_image(image)

fig, ax = plt.subplots()
ax.imshow(image)
for i, point in enumerate(selected_points):
    ax.plot(point[0], point[1], 'ro')
    ax.text(point[0], point[1], str(i+1), color='red', fontsize=12)
    plt.savefig("./numbered_points_image.png")

# Print the selected points
print("Selected points:", selected_points)
