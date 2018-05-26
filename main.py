import cv2
import numpy as np
from lk import estimate_motion, MotionEstimate


def draw_motion_estimate(frame: np.array, motion: MotionEstimate):
    """
    Draws a vector on the frame representing the motion estimate.
    :param scale_factor: the factor by which the displayed image is scaled compared to the gray image.
    """
    (vx, vy), center_row, center_col = motion

    SCALE_FACTOR = 12
    end_row = int(center_row + vx * SCALE_FACTOR)
    end_col = int(center_col - vy * SCALE_FACTOR)

    cv2.line(frame, (center_col, center_row), (end_col, end_row), (0, 255, 0, 0), 1)


def run():
    REGION_SIZE = 15
    DOWNSCALE = 0.05

    video = cv2.VideoCapture('test.mp4')

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    output_size = (int(frame_width * DOWNSCALE), int(frame_height * DOWNSCALE))
    out = cv2.VideoWriter('output.mov', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps/3, output_size)

    # Used to display the number of frames completed.
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # The consecutive greyscale frames used to perform motion analysis.
    motion_frame, prev_motion_frame = None, None

    while video.isOpened():
        print('Frame {}/{}'.format(current_frame, num_frames))

        ret, frame = video.read()

        if not ret:
            break

        downsampled_frame = cv2.resize(frame, (0,0), fx=DOWNSCALE, fy=DOWNSCALE)

        prev_motion_frame = motion_frame
        motion_frame = cv2.cvtColor(downsampled_frame, cv2.COLOR_BGR2GRAY)

        # Perform motion analysis
        if prev_motion_frame is not None and motion_frame is not None:
            motions = estimate_motion(prev_motion_frame, motion_frame, REGION_SIZE)

            for m in motions:
                draw_motion_estimate(downsampled_frame, m)


        out.write(downsampled_frame)
        cv2.imshow('Motion', downsampled_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_frame += 1

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print('Done')


if __name__ == '__main__':
    run()
