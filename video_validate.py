import cv2

def video_validate(video_path):
    video = cv2.VideoCapture(video_path)

    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    # checks if video has 30 fps
    if video_fps == 0:
        return "Video file is not correct."
    elif video_fps != 30:
        return "Video does not have 30 fps."

    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # in frames
    video_len_sec = video_length / video_fps # in seconds
    
    # checks if video is not longer than 15 minutes
    if video_len_sec > 15*60:
        return "Video is longer than 15 minutes."

    video_width = int(video.get(3))
    video_height = int(video.get(4))

    if video_width != 1920 and video_height != 1080:
        return "Video resolution is different than Full HD."

    return 0 # Returns 0 if video is correct
