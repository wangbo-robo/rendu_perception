import numpy as np 
import cv2
import time
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500
counter=0

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
	def __init__(self, cam):
		self.counter = 0
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		#self.detector = cv2.SIFT_create()
		#self.detector = cv2.ORB_create(500, nlevels=15)
	def featureTracking(image_ref, image_cur, px_ref, rgb = None):
		kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1], currFeatures, status, track_error
		st = st.reshape(st.shape[0])
		kp1 = px_ref[st == 1]      # get the useful keypoint in last frame
		kp2 = kp2[st == 1]		   # get the useful keypoint in new frame
		# visualize_flux(rgb, kp1, kp2)

		return kp1, kp2	
	def get_rgb_img(self, img):
		self.img_rgb = img

	def processFirstFrame(self):
		start = time.perf_counter()
		self.px_ref = self.detector.detect(self.new_frame)
		#self.px_ref = cv2.goodFeaturesToTrack(self.new_frame,5000,0.01,10)
		#local_px_ref = np.int0(self.px_ref)
		#for i in local_px_ref:
		#	x,y = i.ravel()
		#	cv2.circle(self.img_rgb,(x,y),3,255,-1)
		img = cv2.drawKeypoints(self.img_rgb, self.px_ref, self.img_rgb)
		cv2.imshow("img",self.img_rgb)
		cv2.waitKey(0)
		cv2.imwrite('results/lol' + str(self.counter) + '.png', img)
		self.counter+=1
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		scale = 10.0
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.cur_t = self.cur_t + scale * self.cur_R.dot(t)
		self.cur_R = R.dot(self.cur_R)
		print(self.cur_t)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		#assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame