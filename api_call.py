'''
small script for calling bccaas api
'''
import base64
import datetime

import requests

import pytz


def get_localtime_str():
    # Set Singapore local time
    singapore_time = pytz.timezone('Asia/Singapore')
    datetime_now = datetime.datetime.now()
    singapore_datetime_now = datetime_now.astimezone(singapore_time)
    return singapore_datetime_now.isoformat()

images = ['S6866304J - after.jpg', 
          'S8158453B - after.jpg', 
          'S8158453B - before.JPG', 
          'S9448749H - before.jpg', 
          'S9615909I - after.jpg'] 

new = ['G4033731N - before.jpg',
       'S7624984I - before.jpg',
       'S8319776E -before.JPG',
       'S9184694B - before.JPG',
       'S9448749H - before.jpg']
       

check_urls = "http://127.0.0.1:8080/invocations"

dataset_dir = 'MOM'

print("OLD")
for im_path in images:
    if im_path[-4:].lower() != '.jpg':
        continue
    with open(f'{dataset_dir}/{im_path}', "rb") as image_file:
        b64str = base64.b64encode(image_file.read()).decode('utf-8')
        PARAMS = {
                    'image_base64': b64str,
                    'datetime': get_localtime_str(),
                    "debug_image": False
                }
        response = requests.post(url=check_urls, json=PARAMS)
        check_results = response.json()['results']
        print(im_path)
        print(check_results['watermark_check'])

print("NEW")
for im_path in new:
    if im_path[-4:].lower() != '.jpg':
        continue
    with open(f'{dataset_dir}/{im_path}', "rb") as image_file:
        b64str = base64.b64encode(image_file.read()).decode('utf-8')
        PARAMS = {
                    'image_base64': b64str,
                    'datetime': get_localtime_str(),
                    "debug_image": False
                }
        response = requests.post(url=check_urls, json=PARAMS)
        check_results = response.json()['results']
        print(im_path)
        print(check_results['watermark_check'])
