from imports import *


@typed
def dataset_features(num_frames: int, num_groups: int, num_points: int, side: int):
    dataset = np.empty((num_frames, num_groups, num_points, 5))
    group_color = np.random.random((num_groups, 3))
    color_speed = np.random.randn(num_groups, 3) / 100
    center_x = 0.5 * side
    center_y = 0.5 * side
    for frame in range(num_frames):
        group_color += color_speed

        for group in range(num_groups):
            group_angle = 3 * 2 * np.pi * (frame / num_frames + group / num_groups)
            group_radius = 0.3 * side
            for point in range(num_points):
                point_angle = (
                    2
                    * np.pi
                    * (
                        frame / num_frames * (1 + 3 * point / num_points)
                        + point / num_points
                    )
                )
                point_radius = 0.1 * side
                x = (
                    center_x
                    + group_radius * np.cos(group_angle)
                    + point_radius * np.cos(point_angle)
                )
                y = (
                    center_y
                    + group_radius * np.sin(group_angle)
                    + point_radius * np.sin(point_angle)
                )

                dataset[frame, group, point, 0] = x
                dataset[frame, group, point, 1] = y
                dataset[frame, group, point, 2:] = (np.sin(group_color[group]) + 1) / 2

    return dataset


@typed
def dataset_pixels(num_frames: int, num_groups: int, num_points: int, side: int, radius: float):
    features = dataset_features(num_frames, num_groups, num_points, side)
    pixels = np.ones((num_frames, side, side, 3))
    for frame in range(num_frames):
        for group in range(num_groups):
            for point in range(num_points):
                x, y = features[frame, group, point, 0], features[frame, group, point, 1]
                dist = np.fromfunction(lambda i, j: np.hypot(i - x, j - y) / radius, (side, side))
                gauss = ein.rearrange(np.exp(-dist**2), "x y -> x y 1")
                color = ein.rearrange(features[frame, group, point, 2:], "c -> 1 1 c")

                pixels[frame] = gauss * color + (1 - gauss) * pixels[frame]
    return pixels


def render_images(images):
    fig, ax = plt.subplots()
    current_image = 0
    im = ax.imshow(images[current_image])
    def update_image(index):
        im.set_data(images[index])
        fig.canvas.draw()

    # Function to handle keypress events
    def on_key(event):
        nonlocal current_image
        if event.key == "right":
            # Move to the next image
            current_image = (current_image + 1) % len(images)
            update_image(current_image)
        elif event.key == "left":
            # Move to the previous image
            current_image = (current_image - 1) % len(images)
            update_image(current_image)

    # Connect the keypress event handler
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


images = dataset_pixels(600, 5, 5, 64, 2.0)
render_images(images)

import imageio
imageio.mimwrite("output.gif", images, format="GIF", duration=0.1)
