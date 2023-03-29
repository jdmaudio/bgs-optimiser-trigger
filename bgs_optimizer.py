import nlopt
import pysky360
import numpy as np
import cv2
import metrics
import math
import itertools

class BGSTestSequence: # Perhaps could have a base class to allow us to develop different sequences?
    def __init__(self, num_objects, radii, angle, contrast, phase_shifts):
        self.num_objects = num_objects
        self.radii = radii
        self.angle = angle
        self.contrast = contrast
        self.phase_shifts = phase_shifts

    def __call__(self, frame):
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        angular_step = 2 * math.pi / self.num_objects  

        # Initialize the ground truth 
        ground_truth = np.zeros((height, width), dtype=np.uint8)

        for circle_idx, object_idx in itertools.product(range(len(self.radii)), range(self.num_objects)):
            phase_shift = self.phase_shifts[circle_idx]
            object_angle = self.angle + phase_shift + angular_step * object_idx
            x = int(center_x + self.radii[circle_idx] * math.cos(object_angle))
            y = int(center_y + self.radii[circle_idx] * math.sin(object_angle))

            object_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(object_image, (x, y), 1, (255, 255, 255), -1)
            cv2.circle(ground_truth, (x, y), 1, 255, -1)

            # Change the contrast of the objects
            object_contrast = cv2.addWeighted(object_image, self.contrast, np.zeros(object_image.shape, object_image.dtype), 1 - self.contrast, 0)
            mask = cv2.inRange(object_contrast, (1, 1, 1), (255, 255, 255))

            # Combine the object with the frame using the mask
            frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame, object_contrast)

        return frame, ground_truth
    

def objective_function(params, grad, args):

    frames = args
    frames = np.array(frames)
    x_dim = frames[0].shape[1] // 2
    threshold, BGSamples, requiredBGSample, learningRate = params[0], params[1], params[2], params[3]

    if int(learningRate) > int(BGSamples):
        learningRate = int(BGSamples)

    # Set BGS parameters
    bgsalgorithm = pysky360.Vibe()
    parameters = bgsalgorithm.getParameters()
    parameters.setThreshold(int(threshold))
    parameters.setBGSamples(int(BGSamples))
    parameters.setRequiredBGSamples(int(requiredBGSample))
    parameters.setLearningRate(int(learningRate))

    # Initialize variables
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    num_objects = 8
    circle_radii = [x_dim*0.3, x_dim*0.55, x_dim*0.8]
    phase_shifts = [0, math.pi / num_objects, 2 * math.pi / num_objects]

    for frame_count, frame in enumerate(frames): 

        # Calculate the angle and contrast for the current frame
        contrast = frame_count / len(frames)
        angle = 2 * math.pi * contrast
        
        # Draw moving objects on the frame
        test_sequence = BGSTestSequence(num_objects, circle_radii, angle, contrast, phase_shifts)
        frame_with_objects, ground_truth = test_sequence(frame)

        foreground_mask = bgsalgorithm.apply(frame_with_objects)
        foreground_mask = cv2.medianBlur(foreground_mask, 3) # Optional

        cv2.putText(frame_with_objects, "Optimizing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame_with_objects)
        # cv2.moveWindow('frame', 10, 10)
        cv2.imshow('foreground_mask', foreground_mask)
        # cv2.moveWindow('foreground_mask', 800, 10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cm = metrics.ConfusionMatrix(ground_truth, foreground_mask)
        TP, TN, FP, FN = cm.get()

        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

    temp = metrics.Metrics(total_TP, total_TN, total_FP, total_FN)
    fitness = ((temp.precision * 2) + (temp.recall)) / 3 # Favours precision (less false positives)
    # fitness = temp.MCC
    # fitness = temp.f1_score
    print(f"Fitness: {np.round(fitness, 4)} | Params: {np.round(params)}")
    return fitness

# ---------- NOT USED ----------------------
def my_constraint(x, grad):
    # Constraint 1: Param 3 <= Param 2
    if x[2] > x[1]:
        return -1.0
    # Constraint 2: Param 4 <= Param 2
    if x[3] >= x[1]:
        return -1.0
    # Constraint 3: 4 is power of two
    return 0.0 if is_power_of_two(int(x[3])) else -1.0

def is_power_of_two(n: int) -> bool:
    return False if n <= 0 else (n & (n - 1)) == 0
# ------------------------------------------


def optimize(result_queue, frame_buffer, initial_params):

    print("\nRunning global optimizer...")
    opt = nlopt.opt(nlopt.GN_CRS2_LM, 4)
    opt.set_max_objective(lambda x, grad: objective_function(x, grad, frame_buffer))
    lower_bounds = [5, 2, 1, 2]
    upper_bounds = [80, 32, 2, 16]
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_maxeval(7)
    result = opt.optimize(initial_params)
    maxf = opt.last_optimum_value()
    print(f"Best: {np.round(maxf, 4)} | Using params: {np.round(result)}\n")
    cv2.destroyAllWindows()
    result_queue.put(result)