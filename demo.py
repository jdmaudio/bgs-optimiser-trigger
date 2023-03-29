import time
import threading
import queue
from collections import deque

import cv2
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.lines import Line2D
import numpy as np

from cloud_estimator import *
import bgs_optimizer
import pysky360
from movingobjects import MovingCircle


def create_mask(frame, radius, center_x, center_y):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return mask

def apply_mask(frame, mask):
    return cv2.bitwise_and(frame, frame, mask=mask)

def update_plot(new_x, new_y, update_scatter=False):
    global line, scatter_points, scatter_x_data, scatter_y_data
    global ccr_plot, scatter_plot, x_data, y_data, max_frames_plot

    # Shift the data in the arrays to the left
    x_data[:-1] = x_data[1:]
    y_data[:-1] = y_data[1:]

    # Add the new data to the end of the arrays
    x_data[-1] = new_x
    y_data[-1] = new_y

    # Plot only the most recent data
    line.set_data(x_data[-max_frames_plot:], y_data[-max_frames_plot:])

    if update_scatter:
        scatter_x_data.append(new_x)
        scatter_y_data.append(new_y)
        scatter_points.set_offsets(np.c_[scatter_x_data, scatter_y_data])

    # Update annotations
    deviations = [averages[i + 1] - averages[i] for i in range(len(averages) - 1)]
    deviations.insert(0, 0)  # Add 0 as the first deviation since there is no previous point
    update_annotations(deviations)

    # Update the x-axis limits conditionally
    ax.set_xlim(new_x - max_frames_plot if new_x >= max_frames_plot else 0, new_x)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()

def update_annotations(deviations):
    # Clear existing annotations
    for txt in annotations:
        txt.remove()
    annotations.clear()

    # Add new annotations with deviations
    for x, y, dev in zip(averages_frames, averages, deviations):
        annotation = mtext.Annotation(f"{dev:.2f}%", (x, y), xytext=(5, -15), textcoords='offset points', fontsize=7)
        ax.add_artist(annotation)
        annotations.append(annotation)

# Put your own videos here 
# (note: this script is setup for day time cloud estimations only)
# 's' key will skip video
video_files = ["../videos/9d569219-7c26-481f-b86b-adb4d79c83b9.mkv", # 100%
            #    "../videos/2cd44331-d19f-488a-b187-9391e3799723.mkv", # 75%
               "../videos/3fb22b3c-0cf9-4c33-ac7a-a7e549ad0d2c.mkv", # 48%
               "../videos/6aeed808-3465-4109-b8a7-fad9c6f8524a.mkv", # 8%
            #    "../videos/1da6c3ef-0693-4e15-8f6d-14b2360514aa.mkv", # 27%
               "../videos/1af3523f-99ef-4132-8d07-2833977d7caa.mkv", # 45%
            #    "../videos/3fb22b3c-0cf9-4c33-ac7a-a7e549ad0d2c.mkv", # 48%
               "../videos/22ae4ec0-00cf-44bc-a4e1-f51d1544d43c.mkv"] # 95%
            #    "../videos/1df03abd-0646-4a09-b200-ed42626ac67b.mkv"]  # 35%      

print("\nRunning BGS Optimizer Demo")

# Video and cloud cover settings
frame_size = (800, 800)
radius_of_mask = int(min(frame_size[1], frame_size[0]) * 0.43)
center_x = int(frame_size[1] / 2)
center_y = int(frame_size[0] / 2)

time_interval = 20  # Time interval in seconds
queue_size = 6     # Number of frames to average CC
ccr_running_ave = deque(maxlen=queue_size)
threshold = 8      # How much CC deviation (%) to trigger optimizer

# Initial vibe parameters 
vibeThreshold = 8
vibeBGSSamples = 16
vibeReqBGSamples = 1
vibeLearningRate = 2

# frames to pass to optimizer
frame_buffer_size = 125 
frame_buffer = deque(maxlen=frame_buffer_size)

# Plot parameters (rolling window)
max_frames_plot = 3500 # number of points along x-axis 
fig, ax = plt.subplots()
ax.set_title('Cloud Cover levels')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Cloud Cover %')
ax.set_ylim(0, 105)
ax.set_xlim(0, max_frames_plot)  

line = Line2D([], [], color='grey',  linestyle='--', lw=0.5)
ax.add_line(line)
plt.show(block=False)

x_data = np.zeros(max_frames_plot)
y_data = np.zeros(max_frames_plot)

scatter_points = ax.scatter([], [], color='blue', marker='x', s=50)
scatter_x_data = []
scatter_y_data = []
annotations = []

# Optimization result
result_queue = queue.Queue()

# Read first frame to compute mask
capture = cv2.VideoCapture(video_files[0])
last_check_time = time.time() 

while not capture.isOpened():
    capture = cv2.VideoCapture(video_files[0])
    cv2.waitKey(1000)
    print("Wait for the header")

_, first_frame = capture.read()
first_frame = cv2.resize(first_frame, frame_size)
mask = create_mask(first_frame, radius_of_mask, center_x, center_y)

# Some moving objects to demo BGS
circles = [MovingCircle(center_x, center_y, radius_of_mask, frame_size[1], frame_size[0]) for i in range(10)]

cloud_cover_ratios = []
averages = []
averages_frames = []
frame_count = 1 

cloud_clover = 100
prev_running_avg = cloud_clover

for video_file in video_files:

    while not capture.isOpened():
        capture = cv2.VideoCapture(video_file)
        cv2.waitKey(1000)

    frame_buffer.clear()
    
    algorithm = pysky360.Vibe()
    parameters = algorithm.getParameters()
    params_and_setters = [
        (vibeThreshold, parameters.setThreshold),
        (vibeBGSSamples, parameters.setBGSamples),
        (vibeReqBGSamples, parameters.setRequiredBGSamples),
        (vibeLearningRate, parameters.setLearningRate),
    ]
    vibeLearningRate = min(vibeLearningRate, vibeBGSSamples)

    for value, setter in params_and_setters:
        setter(value)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        frame_masked = apply_mask(frame, mask)
        frame_buffer.append(frame_masked.copy())

        for circle in circles:
            circle.move()
            circle.draw(frame_masked, (10, 10, 20))

        ce = DayTimeCloudEstimator()

        if frame_count % 15 == 0 or time.time() - last_check_time > time_interval:
            cloud_clover = ce.estimate(frame_masked)
            update_plot(frame_count, cloud_clover, False)

            # This could probably be every 5-10 mins 
            if time.time() - last_check_time > time_interval:
                ccr_running_ave.append(cloud_clover)
                # update_plot(frame_count, cloud_clover, True)

            if len(ccr_running_ave) == queue_size:
                running_avg = sum(ccr_running_ave) / queue_size
                averages.append(running_avg)
                averages_frames.append(frame_count)
                
                print(f"Avg CC% (over {queue_size} frames): {round(running_avg, 1)}% | Deviation (since param change): {prev_running_avg - running_avg:.1f}% at frame {frame_count}")
                update_plot(frame_count, running_avg, True)
            
                if abs(prev_running_avg - running_avg) > threshold and len(frame_buffer) == frame_buffer.maxlen:
                    prev_running_avg = running_avg         
                    parameters = algorithm.getParameters()
                    optimization_thread = threading.Thread(target=bgs_optimizer.optimize, args=(result_queue, frame_buffer, [parameters.getThreshold(), parameters.getBGSamples(), parameters.getRequiredBGSamples(), parameters.getLearningRate()]))
                    optimization_thread.start()
                    cv2.destroyWindow("video")
                    cv2.destroyWindow("foregroundMask")
                    optimization_thread.join() # Waits for optimization to finish

                last_check_time = time.time()
                ccr_running_ave = deque(maxlen=queue_size)


        foreground_mask = algorithm.apply(frame_masked)
        foreground_mask = cv2.medianBlur(foreground_mask, 3) # Optional
        cv2.imshow('foregroundMask', foreground_mask)      
        # cv2.moveWindow('foregroundMask', 800, 10)

        if not result_queue.empty():
            optimization_result = result_queue.get()
            algorithm = pysky360.Vibe()
            parameters = algorithm.getParameters()
            vibeThreshold, vibeBGSSamples, vibeReqBGSamples, vibeLearningRate = map(int, optimization_result)
            vibeLearningRate = min(vibeLearningRate, vibeBGSSamples)

            for value, setter in params_and_setters:
                setter(value)

            prev_running_avg = running_avg

        frame_count += 1
        plt.pause(0.005)

        cv2.putText(frame_masked, "CC: " + str(round(cloud_clover, 1)) + "%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow('video', frame_masked)
        # cv2.moveWindow('video', 10, 10)

        key = cv2.waitKey(2) & 0xFF
        if key == ord('s'):
            break

    capture.release()

cv2.destroyAllWindows()