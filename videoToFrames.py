import cv2
import argparse
from terminalPrintColors import tcolors 

parser = argparse.ArgumentParser(
        description='Video to Frame Image Writer')
parser.add_argument('--video', default='sample.mp4', help='Input video')
parser.add_argument('--outputfolder', default='frames', help='Folder where the frames are written to')
parser.add_argument('--rotate', default='false', help='Rotate images 90 deg cc-wise')
parser.add_argument('--prefix', default='sample', help='Frame file name prefix')
args = parser.parse_args()

print(tcolors.DBG + "Video to frame image writer started...")
print("(NOTE: The output folder has to exist beforehand.)" + tcolors.ENDC)
print(tcolors.HEADER + "Reading video: " + tcolors.ENDC + args.video)
print(tcolors.HEADER + "Writing frames to: " + tcolors.ENDC + args.outputfolder)

input_source = args.video
cap = cv2.VideoCapture(input_source)

i = 0
while True:
    hasFrame, frame = cap.read()
    if hasFrame:
        #cv2.imwrite('/' + args.outputfolder + '/' + args.prefix + str(i) + '.jpg', frame)
        if args.rotate == 'true':
            frame = cv2.flip(cv2.transpose(frame, 0), 0)
        cv2.imwrite(args.outputfolder + '/' + args.prefix + str(i) + '.jpg', frame)
        i += 1
    else:
        if i == 0:
            print(tcolors.ERR + "No frames were extracted. Check if the path to the video is correct...")
            exit()
        else:
            print(str(i) + tcolors.DBG + " frames have been extracted." + tcolors.ENDC)
            exit()
