import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from ..facelandmark.LandmarkMapping import wflw, multipie

wflw_set = {l.name for l in wflw}
multipie_set = {l.name for l in multipie}

hrnet_avail_lmks = {"WFLW"}
dlib_avail_lmks = {"multipie"}


class CheckConfig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.checklist = []
        self.config_check(self.cfg)

    def config_check(self, cfg):
        # Check debug options
        if not isinstance(cfg["setting"]["verbose"], bool):
            logger.error('verbose option can only be a boolean.')
            
        if not isinstance(cfg["setting"]["display_image"], bool):
            logger.error('display_image option can only be a boolean.')
            

        # Check face_det options 
        if cfg["face_det"]["selector"] not in {'lfd', 'opencv_dnn'}:
            logger.error(f'selector does not support {cfg["face_det"]}.')

        if not(0 < cfg["face_det"]["min_face_ratio"] < 1 ) or not(0 < cfg["face_det"]["max_face_ratio"] < 1):
            logger.error("Face ratio must be between 0 and 1.")
            
        # Check face_det: lfd options
        if cfg["face_det"]["selector"] == 'lfd':
            if not isinstance(cfg["face_det"]["lfd"]["use_gpu"], bool):
                logger.error("use_gpu option can only be boolean.")
            
            if not isinstance(cfg["face_det"]["lfd"]["candidate_size"], (int, float)) \
                or not isinstance(cfg["face_det"]["lfd"]["input_size"], (int, float)) \
                or not isinstance(cfg["face_det"]["lfd"]["threshold"], (int, float)):
                logger.error("candidate_size, input_size and threshold can only be numbers.")
                
            if cfg["face_det"]["lfd"]["candidate_size"] < 0 or cfg["face_det"]["lfd"]["input_size"] < 0 or \
                cfg["face_det"]["lfd"]["threshold"] < 0:
                logger.error("candidate_size, input_size and threshold can only be values more than 0.")
                
        # Check face_det: opencv_dnn options
        if cfg["face_det"]["selector"] == "opencv_dnn":
            if not isinstance(cfg["face_det"]["opencv_dnn"]["use_gpu"], bool):
                logger.error("use_gpu option can only be boolean.")
                
            if (not isinstance(cfg["face_det"]["opencv_dnn"]["threshold"], float)) \
                or cfg["face_det"]["opencv_dnn"]["threshold"] < 0:
                logger.error("threshold option can only be values more than 0.")
                
        # Check landmark_det
        if cfg["landmark_det"]["selector"] not in {'dlib', 'hrnet'}:
            logger.error(f'selector does not support {cfg["landmark_det"]["selector"]}.')
    
        # Check landmark_det: hrnet options
        if cfg["landmark_det"]["selector"] == 'hrnet':
            if not isinstance(cfg["landmark_det"]["hrnet"]["use_gpu"], bool):
                logger.error("use_gpu option can only be boolean.")
            if cfg["landmark_det"]["hrnet"]["landmark"] not in hrnet_avail_lmks:
                logger.error(f'{cfg["landmark_det"]["hrnet"]["landmark"]} landmark not supported.')
            elif cfg["landmark_det"]["hrnet"]["landmark"] == "WFLW" and \
                not any(x in cfg["landmark_det"]["hrnet"]["features"]["points"] for x in wflw_set):
                logger.error("selected feature does not exist. See list of available points.")

        # Check landmark_det: dlib options
        elif cfg["landmark_det"]["selector"] == 'dlib':
            if not isinstance(cfg["landmark_det"]["dlib"]["use_gpu"], bool):
                logger.error("use_gpu option can only be boolean.")
            if cfg["landmark_det"]["dlib"]["landmark"] not in dlib_avail_lmks:
                logger.error(f'{cfg["landmark_det"]["dlib"]["landmark"]} landmark not supported.')
            elif (cfg["landmark_det"]["dlib"]["landmark"] == "multipie") and \
                not any(x in cfg["landmark_det"]["dlib"]["features"]["points"] for x in multipie_set):
                logger.error("selected feature does not exist. See list of available points.")

        # Check foreground_seg options
        if cfg["foreground_seg"]["selector"] not in {"u2net", "u2netp"}:
            logger.error("selected model does not exist.")
        elif (not isinstance(cfg["foreground_seg"]["u2net"]["use_gpu"], bool)) or \
             (not isinstance(cfg["foreground_seg"]["u2netp"]["use_gpu"], bool)):
             logger.error("use_gpu option can only be boolean.")

        # Check face_parsing options
        if cfg["face_parsing"]["selector"] not in {"parsenet"}:
            logger.error("selected model does not exist.")
        if (not isinstance(cfg["face_parsing"]["part_thres"], (int, float))) or \
            (not isinstance(cfg["face_parsing"]["angle_sym"], (int, float))):
            logger.error("part_thres and angle_sym options must be a number above 0.")

        # Individual check settings
        ## Image size check
        if cfg["image_size_check"]["im_height"] < 0 or cfg["image_size_check"]["im_width"] < 0:
            logger.error("Image height and width must be a number above 0.")

        ## Gaze check
        if not (0 < cfg["gaze_check"]["blinking_thresh"] < 1) or \
            not (0 < cfg["gaze_check"]["left_thresh"] < 1 ) or \
            not (0 < cfg["gaze_check"]["right_thresh"] < 1):
            logger.error("Threshold options must be between 0 and 1.")

        ## Hair Cover Eye  check
        if not (0 < cfg["hair_cover_eye_check"]["glasses_thres"] < 1) or \
            cfg["hair_cover_eye_check"]["part_thres"] < 0 or \
            cfg["hair_cover_eye_check"]["hairtouch_thres"] < 0 or \
            not (0 < cfg["hair_cover_eye_check"]["haircover_thres"] < 1):
            logger.error("Absolute thresholds must be positive, and decimal thresholds must be between 0 and 1.")

        ## Background Check
        if not (0 < cfg["background_check"]["background_thres"] < 1):
            logger.error("background_thres option must be between 0 and 1.")

        ## Frame Cover Eye check
        if not (0 < cfg["frame_cover_eye_check"]["glasses_thres"] < 1):
            logger.error("glasses_thres option must be between 0 and 1")

        # Check range for dilation kernel size and dilation_iter

        ## Exposure check
        if not (0 < cfg["exposure_check"]["hist"]["over_exp_thres"] <= 1):
            logger.error("exposure_thres option must be between 0 and 1")
        
        ## Mouth Open check
        if not (0 < cfg["mouth_open_check"]["ratio"] < 1):
            logger.error("ratio option must be between 0 and 1.")
 
        
            
            


