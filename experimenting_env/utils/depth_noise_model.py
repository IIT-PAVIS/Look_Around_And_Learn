import numpy as np
import cv2


def add_depth_noise(depth, rgb):
	img = ((depth / 10.0) * 255).astype(np.uint8)
	img_col = rgb

	edges = cv2.Canny(img, 200, 300,apertureSize = 5)
	# edges_col = cv2.Canny(img_col, 100, 200,apertureSize = 5)
	# edges += edges_col

	mask=img.copy()
	mask.fill(0)
	minLineLength = 10
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, 100, 10)

	# uniform noise
	noise_size = int(depth.shape[0] / 4)
	noise = np.random.uniform(low=0, high=0.05, size=(noise_size, noise_size, 1) )
	depth += cv2.resize(noise, tuple(depth.shape[0:2])).reshape( depth.shape[0], depth.shape[1], depth.shape[2] )

	# depth shadows
	if lines is not None:
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(mask, (x1,y1), (x2,y2), 255, 1)

		for i in range(480):
			for j in range(640):
				if mask[i][j]>0:
					cv2.circle(depth, (j,i), 2, 0, -1)
					if np.random.uniform() > 0.8:
						cv2.circle(depth, (j,i), np.random.randint(low=2, high=4), 0, -1)

	return depth