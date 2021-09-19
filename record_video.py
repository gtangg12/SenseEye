import argparse
import time
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Recording settings.')

    parser.add_argument('-o', '--output_path', default='output.mp4', type=str,
        help='')
    parser.add_argument('-n', '--num_frames', default=1500, type=int,
        help='');
    parser.add_argument('--fps', default=24, type=int,
        help='')

    return parser.parse_args()


def main():
    args = parse_args()

    video_cap = cv2.VideoCapture(1)
    cap_prop = lambda x : int(video_cap.get(x))

    width, height = \
        cap_prop(cv2.CAP_PROP_FRAME_WIDTH), cap_prop(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Camera dimensions: {}x{}".format(height, width))

    start = time.time()
    frames = []
    while True:
        success, frame = video_cap.read()
        if not success or len(frames) > args.num_frames - 1:
            break
        frames.append(frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    print ("Recording time taken : {0} seconds".format(time.time() - start))

    video_out = cv2.VideoWriter(
        args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

    for frame in frames:
        video_out.write(frame)
    print("Number of frames written: " + str(len(frames)))


if __name__ == '__main__':
    main()
