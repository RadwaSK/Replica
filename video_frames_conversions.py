from os.path import join, isfile
import os
import cv2 as cv
import tqdm


def video_to_frames(video_path, frames_path, out_fps, in_fps=60):
    """
    Divide the video frames into images with the specified fps

    :param video_path: input path of the video
    :param frames_path: output path of the folder of images
    :param out_fps: output (required) frames per second
    :param in_fps: input frames per second
    :return: None
    """
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    in_sample_rate = int(in_fps)
    if out_fps == 0:
        out_fps = 3
    out_sample_rate = int(out_fps)

    sample_rate = out_sample_rate / in_sample_rate
    current_sample = sample_rate
    name = 0
    drop_frames_counter = 0

    video = cv.VideoCapture(video_path)
    frames_count = video.get(cv.CAP_PROP_FRAME_COUNT)

    print("\tResampling video: {}\n\t\tSample Rate:{}\n\t\tTotal number of frames: {}"
          .format(video_path, sample_rate, frames_count))

    for i in tqdm.trange(int(frames_count)):
        ret, frame = video.read()
        if ret:
            if current_sample >= 1:
                new_image_path = join(frames_path, 'F_' + str(name).zfill(9) + ".jpg")
                cv.imwrite(new_image_path, frame)
                current_sample -= 1

            current_sample += sample_rate
            name += 1
        else:
            drop_frames_counter += 1
            print("\tDropping frame {}/{}, total number of dropped frames is {}"
                  .format(name + drop_frames_counter, frames_count, drop_frames_counter))

    video.release()
    print("\tfinished sampling.")


def frames_to_video(frames_path, video_path, fps):
    frame_array = []
    files = [f for f in os.listdir(frames_path) if isfile(join(frames_path, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename = frames_path + '/' + files[i]
        #reading each files
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv.VideoWriter(video_path,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
