import enum

class wflw(enum.Enum):
    BOTTOM_FACE_CONTOUR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    LEFT_EYEBROW = [42, 43, 44, 45, 46, 47, 48, 49, 50]
    RIGHT_EYEBROW = [33, 34, 35, 36, 37, 38, 39, 40, 41]
    NOSE = [51, 52, 53, 54]
    LEFT_EYE = [68, 69, 70, 71, 72, 73, 74, 75]
    RIGHT_EYE = [60, 61, 62, 63, 64, 65, 66, 67]
    OUTER_LIP = [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88]
    INNER_LIP = [89, 90, 91, 92, 93, 94, 95]
    LEFT_PUPIL = [97]
    RIGHT_PUPIL = [96]

class multipie(enum.Enum):
    BOTTOM_FACE_CONTOUR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    LEFT_EYEBROW = [22, 23, 24, 25, 26]
    RIGHT_EYEBROW = [17, 18, 19, 20, 21]
    NOSE = [27, 28, 29, 30, 31, 32, 33, 34, 35]
    LEFT_EYE = [42, 43, 44, 45, 46, 47]
    RIGHT_EYE = [36, 37, 38, 39, 40, 41]
    OUTER_LIP = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    INNER_LIP = [60, 61, 62, 63, 64, 65, 66, 67]
    LEFT_PUPIL = []
    RIGHT_PUPIL = []

class aflw(enum.Enum):
    BOTTOM_FACE_CONTOUR = [12, 16, 20]
    LEFT_EYEBROW = [3, 4, 5]
    RIGHT_EYEBROW = [0, 1, 2]
    NOSE = [13, 14, 15]
    LEFT_EYE = [9, 11]
    RIGHT_EYE = [6, 8]
    OUTER_LIP = []
    INNER_LIP = [17, 18, 19]
    LEFT_PUPIL = [7]
    RIGHT_PUPIL = [10]
