from pose_estimation.pose_estimation import get_poses
from segmentation.segmentation import segment_images
from GAN.Pix2PixModel import callGAN
from video_frames_conversions import video_to_frames, frames_to_video
import os
import sys


if __name__ == '__main__':
    if len(sys.argv) == 1:
        quick_run = 0
        target_video_path = input('Enter target video path: ')
        source_video_path = input('Enter source video path: ')
    else:
        quick_run = int(sys.argv[1]) == 1
        target_video_path = sys.argv[2]
        source_video_path = sys.argv[3]
    
    source_name = source_video_path[source_video_path.rfind('/') + 1 : source_video_path.rfind('.')]
    target_name = target_video_path[target_video_path.rfind('/') + 1 : target_video_path.rfind('.')]
    mixed_name = target_name + '_' + source_name

    if not os.path.exists('fake_videos'):
        os.makedirs('fake_videos')

    fake_video_path = 'fake_videos/' + mixed_name + '.mp4'

    target_frames_path = 'datasets/' + mixed_name + '/train/train_img'
    target_segmented_path = 'datasets/' + target_name + '_segmented'
    target_poses_path = 'datasets/' + mixed_name + '/train/train_label'

    source_frames_path = 'datasets/' + mixed_name + '/test/test_img'
    source_segmented_path = 'datasets/' + source_name + '_segmented'
    source_poses_path = 'datasets/' + mixed_name + 'test/test_label'

    if quick_run:
        video_to_frames(target_video_path, target_frames_path, 30, 30)
        video_to_frames(source_video_path, source_frames_path, 30, 30)

        segment_images(target_frames_path, target_segmented_path)
        get_poses(target_frames_path, target_segmented_path, target_poses_path)

        segment_images(source_frames_path, source_segmented_path)
        get_poses(source_frames_path, source_segmented_path, source_poses_path)

        callGAN(False, target_name, 0, True, True, True, 0, 0, 0, 0)

        result_frames_path = 'testing_samples/0'

        frames_to_video(result_frames_path, fake_video_path, 30)

    else:
        in_fps = int(input('Enter fps of target/source videos: '))
        out_fps = int(input('Enter fps sampling for output of target/source videos: '))

        video_to_frames(target_video_path, target_frames_path, out_fps, in_fps)
        video_to_frames(source_video_path, source_frames_path, out_fps, in_fps)

        run_mode = int(input('Enter 0 to run segmentation on target, 1 on source, 2 if not: '))
        if run_mode == 0:
            print("Running segmentation on", target_frames_path, '-->', target_segmented_path)
            segment_images(target_frames_path, target_segmented_path)
        if run_mode == 1:
            print("Running segmentation on", source_frames_path, '-->', source_segmented_path)
            segment_images(source_frames_path, source_segmented_path)
        else:
            run_mode = int(input('Enter 0 to run pose estimation on target, 1 on source, 2 if not: '))
            if run_mode == 0:
                print('\nSegmented target frames path should be in:', target_segmented_path, '!!!')
                print("Running pose estimation on", target_frames_path, target_segmented_path, '-->', target_poses_path)
                get_poses(target_frames_path, target_segmented_path, target_poses_path)
            if run_mode == 1:
                print('\nSegmented source frames path should be in:', source_segmented_path, '!!!')
                print("Running pose estimation on", source_frames_path, source_segmented_path, '-->', source_poses_path)
                get_poses(source_frames_path, source_segmented_path, source_poses_path)
            else:
                run_mode = int(input('Enter 1 to run GAN, 0 if not: '))
                if run_mode == 1:
                    train = int(input('Enter 1 to train, 0 if not')) == 1
                    test = int(input('Enter 1 to test, 0 if not')) == 1
                    print("Running GAN")
                    callGAN(False, mixed_name, 0, train, test, True, 0, 0, 0 , 0)
    


