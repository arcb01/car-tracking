# import the necessary packages
from threading import Thread
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import sys

from queue import Queue

class FileVideoStream:
	def __init__(self, path, queueSize=128, fps = 30):
        #queueSize : The maximum number of frames to store in the queue. This value defaults to 128 frames, but you depending on (1) the frame dimensions of your video and (2) the amount of memory you can spare, you may want to raise/lower this value.

		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		fps = 1
		self.stream = cv2.VideoCapture(path)
		print(fps)
		print(self.stream.get(cv2.CAP_PROP_FPS))
		self.stream.set(cv2.CAP_PROP_FPS, fps)
		print(self.stream.get(cv2.CAP_PROP_FPS))
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)


	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self


	def update(self):
        #This method is responsible for reading and decoding frames from the video file, along with maintaining the actual queue data structure
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)


	def read(self):
		# return next frame in the queue
		return self.Q.get()


	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


if __name__ == '__main__':
	fvs = FileVideoStream('videos/output7.mp4').start()
	time.sleep(1.0)
	# start the FPS timer
	fps = FPS().start()

	# loop over frames from the video file stream
	while fvs.more():
		# grab the frame from the threaded video file stream, resize it, and convert it to grayscale (while still retaining 3 channels)
		frame = fvs.read()
		frame = imutils.resize(frame, width=450)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = np.dstack([frame, frame, frame])

		# display the size of the queue on the frame
		#cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
			#(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	

		# show the frame and update the FPS counter
		cv2.imshow("Frame", frame)
		cv2.waitKey(1)
		fps.update()