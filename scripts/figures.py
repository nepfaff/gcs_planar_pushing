import zarr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_name = "outputs/2023-05-13/10-36-08/initial_positions_2023-05-13_10-36-08.zarr"
initial_positions = zarr.open_group(file_name, mode="r")
object_pos = initial_positions["object_pos"][:]
robot_pos = initial_positions["robot_pos"][:]

# plt.scatter(object_pos[:, 0], object_pos[:, 1], label="object")
# plt.scatter(robot_pos[:, 0], robot_pos[:, 1], label="robot")
# plt.legend()
# plt.show()
# print(object_pos)
# print(robot_pos)


# Color plot success
# Success 0.1
# df = pd.read_csv("data/wandb_export_2023-05-15T11_46_42.522-04_00.csv")
# plt.scatter(object_pos[:, 0],  object_pos[:, 1], marker="s", c=df["success_ratio"], label="object")
# plt.scatter(robot_pos[:, 0], robot_pos[:, 1], c=df["success_ratio"], label="robot")
# Success 0.2
succes_02 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0.6, 1, 0, 1, 1, 1])
plt.scatter(object_pos[:, 0], object_pos[:, 1], marker="s", c=succes_02, label="object")
# plt.scatter(robot_pos[:, 0], robot_pos[:, 1], c=succes_02, label="robot")
# plt.legend()
plt.colorbar()
plt.show()

# # Images overlay
# training_file_name = "outputs/2023-05-03/00-07-29/replay_2023-05-03_00-07-29.zarr"
# training_data = zarr.open_group(training_file_name, mode="r")
# episode_ends = training_data["meta"]["episode_ends"][:]
# finger_positions = training_data["data"]["state"][:]
# images = training_data["data"]["img"][:]
# # actual epsiode end is episode_ends[i]-1
# episode_starts = np.concatenate([[0], episode_ends[:-1]])
# start_images = images[episode_starts]
# ave_image = np.mean(start_images, axis=0)

# # # Create a numpy array of floats to store the average (assume RGB images)
# arr=np.zeros((96,96,4),np.float)
# N = 20
# # Build up average pixel intensities, casting each image as an array of floats
# for image in start_images:
#     image_rgba = np.concatenate((image, np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * 255), axis=2)

#     white_pixels = np.all(image == [255, 255, 255], axis=-1)  # Create a boolean mask for white pixels
#     image_rgba[white_pixels] = [0, 0, 0, 0]  # Set alpha to 0 for white pixels

#     arr=arr+image_rgba/N
# # Round values in array and cast as 8-bit integer
# arr=np.array(np.round(arr),dtype=np.uint8)


# plt.imshow(arr)
plt.show()
