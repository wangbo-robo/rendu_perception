import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry


cam = PinholeCamera(4032, 3024, 816*0.25, 818*0.25, 506*0.25, 344*0.25)
vo = VisualOdometry(cam)

traj = np.zeros((600,600,3), dtype=np.uint8)

for img_id in range(4035,4074):
	img = cv2.imread("pic\\IMG_"+str(img_id)+".JPG", 1)
	img = cv2.resize(img, [int(4032*0.25), int(3024*0.25)], interpolation= cv2.INTER_AREA)
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	vo.get_rgb_img(img)
	vo.update(img_gray, img_id)


	cur_t = vo.cur_t
	if(img_id > 4035):
		x, y, z = cur_t[0], cur_t[1], cur_t[2]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x)+290, int(z)+90
	true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

	cv2.circle(traj, (draw_x,draw_y), 1, (255,255,255), 1)
	cv2.circle(traj, (true_x,true_y), 1, (0,255,0), 2)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

cv2.imwrite('map.png', traj)