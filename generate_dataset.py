import random
# from torchvision import datasets, transforms
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
# import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import string
import imutils
import shutil
import datetime


def load_words(img, prob=1):
    """
    Loads random number of strings on input img.
    Strings loaded will be numbers for 25% prob, for training on detecting numbers better
    Side effect: Some words may be cropped away due to rotation of canvas, good for simulating occlusion

    Parameters
    ----------
    img : PIL.JpegImagePlugin.JpegImageFile
        Original image to generate watermarked and worded mask

    prob: probability of loading words

    Returns
    -------
    img : torch.Tensor
        Image loaded with random strings
    original_img: torch.Tensor
        Clean image to generate word mask
    """

    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_original = img.copy()

    # some images will not have words loaded
    if random.random() > prob:
        img_original = (torch.from_numpy(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255).to(
        torch.float32)
        return img_original, img_original

    height, width, _ = img.shape
    colored_canvas, canvas = np.zeros(img.shape, np.uint8), np.zeros((height, width), np.uint8)

    num_words = random.randint(1, 3)
    for _ in range(num_words):
        # upper and lower bound for text length
        len_words = random.randint(5, 20)

        # add digits with char
        string_list = (string.digits + string.ascii_letters
                        if random.random() < 0.5
                        else string.ascii_letters)
        rand_string = ''.join(random.choices(string_list, k=len_words))

        # upper and lower bound for position of word
        rand_width = int(random.uniform(0.1, 0.9) * width)
        rand_height = int(random.uniform(0.1, 0.9) * height)
        pos = (rand_width, rand_height)

        # random font
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, 
                 cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
        font = random.choice(fonts)

        # random font scale
        fontScale = random.uniform(0.1, 2)

        # randomly have 1/4 rgb and 3/4 white text
        fontColor = ((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                    if random.random() < 0.25
                    else (255,255,255))

        #random thickness
        thickness = random.randint(1, 3)
        
        # random linetype
        lineType = random.choice([cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA])

        if random.random() < 0.5:
            # add text to canvas without rotation
            cv2.putText(canvas, rand_string, pos, font, fontScale, (255, 255, 255), thickness, lineType)
            cv2.putText(colored_canvas, rand_string, pos, font, fontScale, fontColor, thickness, lineType)
        else:
            # add text to canvas with rotation
            # rotate canvas
            rotate_angle = random.randint(-180, 180)
            canvas = imutils.rotate(canvas, rotate_angle)
            colored_canvas = imutils.rotate(colored_canvas, rotate_angle)

            # add text to canvas
            cv2.putText(canvas, rand_string, pos, font, fontScale, (255, 255, 255), thickness, lineType)
            cv2.putText(colored_canvas, rand_string, pos, font, fontScale, fontColor, thickness, lineType)

            # rotate canvas back to original pos
            canvas = imutils.rotate(canvas, -rotate_angle)
            colored_canvas = imutils.rotate(colored_canvas, -rotate_angle)

    alpha = random.uniform(0.3, 1)
    _, canvas_mask = cv2.threshold(canvas, 50, 255, cv2.THRESH_BINARY)
    img[np.where(canvas_mask == 255)] = cv2.addWeighted(img, 1-alpha, colored_canvas, alpha, 0)[np.where(canvas_mask == 255)]

    img = (torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255).to(torch.float32)
    img_original = (torch.from_numpy(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) / 255).to(
        torch.float32)

    return img, img_original


def load_watermark(img_original, word_img, watermark_path, watermark_files, prob=1):
    """
    Loads a random watermark on original image and worded image

    Parameters
    -----------
    img_original : torch.Tensor
        Cleaned image from dataset
    word_img : torch.Tensor
        Image loaded with only words, for generating binary masks
    watermark_path : str
        Path containing watermarks
    watermark_files : list
        Contains watermark filenames
    prob: probability of loading watermark

    Returns
    -------
    word_img : torch.Tensor
        Image loaded with watermarks and words, to be used for training
    img_original : torch.Tensor
        Image loaded with only watermarks, for generating binary masks

    """

    _, img_height, img_width = word_img.shape
    img_original = img_original.clone()
    word_img = word_img.clone()

    # some images will not have watermarks loaded
    if random.random() > prob:
        return word_img, img_original

    logo_id = random.randint(0, len(watermark_files) - 1)
    logo = Image.open(osp.join(watermark_path, watermark_files[logo_id]))
    logo = logo.convert('RGBA')

    rotate_angle = random.randint(0, 360)
    logo_rotate = logo.rotate(rotate_angle, expand=True)

    logo_height = random.randint(img_height//4, img_height//2)  # maybe tweak 256/4
    logo_width = int(logo_height * (random.uniform(0.5, 2)))  # mutiply 0.5 ~ 2
    if logo_width > img_width:
        logo_width = img_width
    logo_resize = logo_rotate.resize((logo_height, logo_width))

    transform_totensor = transforms.Compose([transforms.ToTensor()])
    # img = torch.from_numpy(img)
    logo = transform_totensor(logo_resize)

    alpha = random.uniform(0.3, 1)  # 0.8
    start_height = random.randint(0, img_height - logo_height)
    start_width = random.randint(0, img_width - logo_width)

    img_original[:, start_width:start_width + logo_width, start_height:start_height + logo_height] *= \
        (1.0 - alpha * logo[3:4, :, :]) + logo[:3, :, :] * alpha * logo[3:4, :, :]
    word_img[:, start_width:start_width + logo_width, start_height:start_height + logo_height] *= \
        (1.0 - alpha * logo[3:4, :, :]) + logo[:3, :, :] * alpha * logo[3:4, :, :]

    return word_img, img_original


def solve_mask(img, img_target):
    """
    Generates binary mask based from difference between img and img target
    Parameters
    ----------
    img : torch.Tensor
    img_target : torch.Tensor

    Returns
    -------
    mask : numpy.ndarray
    """
    img1 = np.asarray(img.permute(1, 2, 0).cpu())
    img2 = np.asarray(img_target.permute(1, 2, 0).cpu())
    img3 = abs(img1 - img2)
    mask = img3.sum(2) > (15.0 / 255.0)
    return mask.astype(int)


def main():
    results_folder = 'data'  # folder to store generated dataset, needs to contain photos and watermarks
    # generated_per_file = 1  # number of pictures generated from one photo

    labels = ['train', 'val']
    for label in labels:

        print(f">> Generating images for {label} set")

        results_path = f"{results_folder}/{label}"

        # required
        photo_path = osp.join(results_path, "photos")          # photos are unique between train/val/test
        watermark_path = osp.join(results_path, "watermarks")  # watermarks are unique between train/val/test

        photo_files = sorted(os.listdir(photo_path))
        watermark_files = sorted([f for f in os.listdir(watermark_path) if f[-15:]!='Zone.Identifier'])

        # generated
        watermark_mask_path = osp.join(results_path, 'mask_watermark')
        words_mask_path = osp.join(results_path, 'mask_word')
        img_input_path = osp.join(results_path, 'input')

        shutil.rmtree(watermark_mask_path, ignore_errors=True)
        shutil.rmtree(words_mask_path, ignore_errors=True)
        shutil.rmtree(img_input_path, ignore_errors=True)

        os.makedirs(watermark_mask_path, exist_ok=True)
        os.makedirs(img_input_path, exist_ok=True)
        os.makedirs(words_mask_path, exist_ok=True)

        i = 1
        blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(1,1))

        for photo in tqdm(photo_files):
            if photo[-15:] == 'Zone.Identifier':
                continue

            indiv_photo_path = osp.join(photo_path, photo)
            img = Image.open(indiv_photo_path)
            img = img.resize((256, 256))

            # hard negative for all photos that are newly added
            if (os.path.getmtime(indiv_photo_path) < datetime.date(2021,9,20).timestamp() or 
                os.path.getmtime(indiv_photo_path) > datetime.date(2021,9,22).timestamp()):
                img = np.array(img).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = ((torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            .permute(2, 0, 1) / 255)
                            .to(torch.float32))
                img = blurrer(img)
                save_id = f'{i}.jpg'
                cv2.imwrite(osp.join(img_input_path, save_id),
                            cv2.cvtColor(np.array(img.permute(1, 2, 0) * 255), cv2.COLOR_BGR2RGB))
                i += 1
                continue

            # 5/7 of input images are hard negatives
            if random.random() < (5/7):
                img = np.array(img).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = ((torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            .permute(2, 0, 1) / 255)
                            .to(torch.float32))
                img = blurrer(img)
                save_id = f'{i}.jpg'
                cv2.imwrite(osp.join(img_input_path, save_id),
                            cv2.cvtColor(np.array(img.permute(1, 2, 0) * 255), cv2.COLOR_BGR2RGB))
                i += 1

            else: # 2/7 of input images are positives
                # for _ in range(generated_per_file):

                word_img, img_original = load_words(img, prob=1)
                watermarked_and_word_img, watermarked_img = load_watermark(img_original, word_img, watermark_path,
                                                                        watermark_files, prob=1)

                img_original = blurrer(img_original)
                word_img = blurrer(word_img)
                watermarked_img = blurrer(watermarked_img)
                watermarked_and_word_img = blurrer(watermarked_and_word_img)

                # # half of positive input images are stitched positives
                # if random.random() < 0.5:
                img_comb_free = torch.cat([img_original, img_original], dim=2)

                if random.random() > 0.5:
                    # append img_original to the left
                    img_comb_mask = torch.cat([watermarked_and_word_img, img_original], dim=2)
                    watermarked_comb_mask = torch.cat([watermarked_img, img_original], dim=2)
                    word_comb_mask = torch.cat([word_img, img_original], dim=2)
                else:
                    # append img_original to the right
                    img_comb_mask = torch.cat([img_original, watermarked_and_word_img], dim=2)
                    watermarked_comb_mask = torch.cat([img_original, watermarked_img], dim=2)
                    word_comb_mask = torch.cat([img_original, word_img], dim=2)
                
                # # half of positive input images are hard positives, ie no stitching
                # # else: 
                # img_comb_free = img_original
                # img_comb_mask = watermarked_and_word_img
                # watermarked_comb_mask = watermarked_img
                # word_comb_mask = word_img
                
                # solve for binary masks
                watermarked_mask = solve_mask(watermarked_comb_mask, img_comb_free)
                word_mask = solve_mask(word_comb_mask, img_comb_free)

                '''saving'''
                save_id = f'{i}.jpg'
                cv2.imwrite(osp.join(img_input_path, save_id),
                            cv2.cvtColor(np.array(img_comb_mask.permute(1, 2, 0) * 255), cv2.COLOR_BGR2RGB))

                cv2.imwrite(osp.join(watermark_mask_path, save_id),
                            np.concatenate((watermarked_mask[:, :, np.newaxis],
                                            watermarked_mask[:, :, np.newaxis],
                                            watermarked_mask[:, :, np.newaxis]), 2) * 256.0)

                cv2.imwrite(osp.join(words_mask_path, save_id),
                            np.concatenate((word_mask[:, :, np.newaxis],
                                            word_mask[:, :, np.newaxis],
                                            word_mask[:, :, np.newaxis]), 2) * 256.0)

                i += 1


if __name__ == "__main__":
    main()
