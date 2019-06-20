import cv2
import time
import numpy as np
import argparse
import csv
from terminalPrintColors import tcolors 

# Parse Command Line Arguments -------------------------------------------------
parser = argparse.ArgumentParser(
        description='Human Gesture Recognition Using OpenPose and Dynamic Time Warping')
parser.add_argument('--video', default='sample_video.avi', help='RGB Video')
parser.add_argument('--outcsv', default='pose.csv', help='CSV file where key points are written to')
parser.add_argument('--dbgprints', default='false', help='Print additional debug info in terminal')
parser.add_argument('--dbgimgs', default='false', help='Save keypoint image for every frame')
args = parser.parse_args()

# Important indices in the feature vector
COCO_NECK_IDX = 1
COCO_SHOULDER_LEFT_IDX = 5
COCO_SHOULDER_RIGHT_IDX = 2
DRAW_COORDS = False # draw coordinates in debug image

np.set_printoptions(precision=2) # prettier printing of decimals

#print("\033[37;42m Human Gesture Recognition Using OpenPose and Dynamic Time Warping " + tcolors.ENDC + '\n')

# Set up everything we need for the COCO model
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# These are the key points between which connections are drawn
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],
               [1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

print(tcolors.HEADER + "Reading video: " + tcolors.ENDC + args.video)

threshold = 0.1

input_source = args.video
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

t = time.time()
# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setInput(inpBlob)

output = net.forward()
print(tcolors.HEADER + "Time taken by network [s]: " + tcolors.ENDC + "{:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

vid_writer = cv2.VideoWriter(args.video + '_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

frameCount = 0

while cv2.waitKey(1) < 0:
    try:
        frameCount += 1
        t = time.time()
        hasFrame, frame = cap.read()
        
        if not hasFrame:
            break
            
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()  
        
        # Empty list to store the detected keypoints
        points = []
        
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                points.append((int(x), int(y)))
            else:
                raise Exception(tcolors.ERR + "One or more key points could not be located. Skipping this image..." + tcolors.ENDC + '\n')

        # Normalization:
        #  - neck key point is the origin
        #  - everything scaled with respect to distance between shoulders (=1)
        shoulderDistanceVec = abs(np.subtract(points[COCO_SHOULDER_LEFT_IDX],points[COCO_SHOULDER_RIGHT_IDX]))
        shoulderDistance = np.linalg.norm(shoulderDistanceVec) # simple vector norm, i.e. sqrt(x² + y²)

        normalizedPoints = []
        for p in points:
            #                          scale     translate
            normalizedPoints.append(np.divide(np.subtract(p, points[COCO_NECK_IDX]), shoulderDistance))

        if args.dbgprints == 'true':
            print(tcolors.DBG + 'Raw points (without normalization to neck being the origin):')
            print(points, '\n')
            print('Normalized points (neck as origin, shoulder-to-shoulder = 1):')
            print(normalizedPoints, '\n' + tcolors.ENDC)
            print(tcolors.DBG + "Shoulder distance [px]:" + tcolors.ENDC + str(shoulderDistance))


        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            # black outline
            cv2.circle(frame, points[COCO_NECK_IDX], 8, (0, 0, 0), thickness=5, lineType=cv2.FILLED)
            cv2.circle(frame, points[COCO_SHOULDER_LEFT_IDX], 8, (0, 0, 0), thickness=5, lineType=cv2.FILLED)
            cv2.circle(frame, points[COCO_SHOULDER_RIGHT_IDX], 8, (0, 0, 0), thickness=5, lineType=cv2.FILLED)

            cv2.circle(frame, points[COCO_NECK_IDX], 8, (0, 255, 192), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[COCO_SHOULDER_LEFT_IDX], 8, (255, 128, 64), thickness=2, lineType=cv2.FILLED)
            cv2.circle(frame, points[COCO_SHOULDER_RIGHT_IDX], 8, (255, 128, 64), thickness=2, lineType=cv2.FILLED)

            if points[partA] and points[partB]: # check if key points are 'None' (e.g. because they are not visible)
                cv2.line(frame, points[partA], points[partB], (0, 0, 0), 5)
                cv2.line(frame, points[partA], points[partB], (64, 255, 64), 3)
                cv2.circle(frame, points[partA], 8, (0, 0, 0), thickness=5, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 0), thickness=5, lineType=cv2.FILLED)
                cv2.circle(frame, points[partA], 8, (255, 0, 255), thickness=2, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (255, 0, 255), thickness=2, lineType=cv2.FILLED)
            
        if DRAW_COORDS:
            for pair in POSE_PAIRS:
                if pair[0]: # check if key points are 'None' (e.g. because they are not visible)
                    cv2.putText(frame, str(normalizedPoints[pair[0]]), points[pair[0]], font, 0.35, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, str(normalizedPoints[pair[0]]), points[pair[0]], font, 0.35, (0,255,0), 1, cv2.LINE_AA)


        # write pose keypoints to csv file with mode 'append' TODO: Maybe use .npz instead if files get too large
        flatPoints = [item for sublist in normalizedPoints for item in sublist]
        with open(args.outcsv, mode='a') as pose_csv_file:
            pose_csv_writer = csv.writer(pose_csv_file, delimiter=',')
            pose_csv_writer.writerow(flatPoints)

        print(tcolors.HEADER + "Frame " + tcolors.ENDC + tcolors.DBG + str(frameCount) + tcolors.ENDC + ", output key points have been written to: " + tcolors.ENDC +  args.outcsv)
        print(tcolors.HEADER + "Total time taken [s]: " + tcolors.ENDC + "{:.3f} \n".format(time.time() - t))
        cv2.imshow('Output-Skeleton', frame)
        vid_writer.write(frame)
    except:
        continue

print(tcolors.DBG + "Done with video " + tcolors.ENDC + args.video + '\n')
vid_writer.release()
