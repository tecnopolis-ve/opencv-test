#@title
import numpy as np
import cv2
import imutils
import random
import string
import urllib.request
import random as r
import skimage
from PIL import Image
import requests
from io import BytesIO
from google.colab.patches import cv2_imshow
from main import DocumentOcr

class TestDocumentOcr(object):
  def get_img(self, url):
    #Send user agent to bypass forbidden code
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Platform; Security; OS-or-CPU; Localization; rv:1.4) Gecko/20030624 Netscape/7.1 (ax)'})
    return np.array(Image.open(BytesIO(response.content)))
  def gen_code(self, codelen):
    char_set = string.ascii_uppercase + string.digits
    return ' '.join(random.sample(char_set*codelen, codelen))
  #Mock data
  def mock(self):
    
    #Load card image and cut card background
    card = imutils.url_to_image("https://storage.googleapis.com/interview-tm/card.png")
    card_bg = imutils.url_to_image("https://storage.googleapis.com/interview-tm/futuristic_cpu.jpg")
    avatar = self.get_img("https://www.thispersondoesnotexist.com/image")
    card_bg = (cv2.resize(card_bg, (1000, 1000)) * 0.8).astype(np.uint8)
    card_mask =  cv2.cvtColor(card, cv2.COLOR_BGRA2GRAY)
    card_mask = cv2.threshold(card_mask, 100, 255, cv2.THRESH_BINARY)[-1]

    #Change tint color
    hsv = cv2.cvtColor(card_bg, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = r.randint(0, 200)
    hsv[:,:,1] = 50
    card_bg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    image_roi_rand = self.copy_with_mask(card, card_bg, r.randint(0, 200), r.randint(0, 200), card_mask)

    #Paste avatar
    ay = 50
    ax = 120
    avatar = cv2.resize(avatar, (200, 230))
    avatar =  cv2.cvtColor(avatar, cv2.COLOR_RGB2GRAY)
    avatar =  cv2.cvtColor(avatar, cv2.COLOR_GRAY2BGRA)
    image_roi_rand[ay:ay+avatar.shape[0], ax:ax+avatar.shape[1]] = avatar


    #Put random text codes
    text_offset = r.randint(0, 40)
    codelen = r.randint(5, 8)
    tcolor = 255
    light_color = (tcolor, tcolor, tcolor, 255)
    dark_color = (tcolor-255, tcolor-255, tcolor-255, 255)
    code = self.gen_code(codelen)
    #cv2.rectangle(image_roi_rand, (int(410), int(320) + text_offset), (int(700), int(360) + text_offset), (0,0,255,255), 3)
    cv2.rectangle(image_roi_rand, (int(410), int(220) + text_offset), (int(700), int(260) + text_offset), (0,0,255,255), 3)
    cv2.putText(image_roi_rand, self.gen_code(codelen), (int(120), int(350) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
    cv2.putText(image_roi_rand, "CODE / M", (int(120), int(310) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 1)
    cv2.putText(image_roi_rand, self.gen_code(codelen), (int(420), int(350) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
    cv2.putText(image_roi_rand, "CODE / Z", (int(420), int(310) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 1)
    cv2.putText(image_roi_rand, code, (int(420), int(250) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
    cv2.putText(image_roi_rand, "CODE / Y", (int(420), int(210) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 1)
    cv2.putText(image_roi_rand, self.gen_code(27), (int(120), int(400) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)
    cv2.putText(image_roi_rand, self.gen_code(codelen), (int(420), int(90) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_color, 3)
    cv2.putText(image_roi_rand, self.gen_code(codelen), (int(420), int(160) + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)

    #cv2.rectangle(image_roi_rand, (310,50), (444,90), (0,0, 255), thickness= 4)

    #Random rotate
    if r.randint(0, 1):
      image_roi_rand = cv2.rotate(image_roi_rand, cv2.ROTATE_180)


    #Deform card
    dest_shape = (int(856/1.6), int(539/1.6))
    max_dis = 60
    height, width, channels = image_roi_rand.shape
    src = np.array([[0, 0], [0, height], [width, height], [width, 0]], np.float32)
    dst = np.array([[r.randint(0, max_dis), r.randint(0, max_dis)], [r.randint(0, max_dis), dest_shape[1] - r.randint(0, max_dis)], [dest_shape[0]-r.randint(0, max_dis), dest_shape[1]-r.randint(0, max_dis)], [dest_shape[0]-r.randint(0, r.randint(0, max_dis)), r.randint(0, max_dis)]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    image_roi_rand = cv2.warpPerspective(image_roi_rand, M, (dest_shape[0], dest_shape[1]))
    image_roi_rand_gray = cv2.cvtColor(image_roi_rand, cv2.COLOR_BGRA2GRAY)
    thresh = cv2.threshold(image_roi_rand_gray, 20, 255, cv2.THRESH_BINARY)[-1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    mask = np.zeros((dest_shape[1], dest_shape[0]), np.uint8)
    cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    image_roi_rand = cv2.bitwise_and(image_roi_rand, image_roi_rand, mask=mask)

    #Load background
    background_url = "https://storage.googleapis.com/interview-tm/t" + str(r.randint(0, 1)) + ".jpg"
    image = imutils.url_to_image(background_url)   
    image = cv2.resize(image, (1000,1000))
    size = image.shape

    #Paste card
    margin = 20
    x = np.clip(random.randrange(margin, 500), 0, size[1] - image_roi_rand.shape[1] - margin)
    y = np.clip(random.randrange(margin, 500), 0, size[0] - image_roi_rand.shape[1] - margin)
    image = self.copy_transparent(image, image_roi_rand, x, y)


    #Add some noise
    image = cv2.blur(image, (2, 2))
    image = (skimage.util.random_noise(image, mode='gaussian', var = 0.003) * 255).astype(np.uint8)

    code = code.replace(" ", "")
    
    return image, code
  def copy_transparent(self, background, overlay, x, y):
    b,g,r,a = cv2.split(overlay)
    overlay_color = cv2.merge((b,g,r))
    h, w, _ = overlay_color.shape
    roi = background[y:y+h, x:x+w]
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(a))
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = a)
    background[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    return background
  def copy_with_mask(self, background, overlay, x, y, mask):
    h,w,_ = background.shape
    overlay = overlay[y:y+h, x:x+w]
    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    out = cv2.merge((overlay, mask))
    return out

  #Run test
  def run(self, func):
      image, code = self.mock()
      print('Code to read: ' + code)
      readed_code = func(image)
      assert readed_code == code, 'Error: ' + str(readed_code) + ' != ' + code
      print('Success: test_code ' + readed_code)

def main():
    test = TestDocumentOcr()
    document_ocr = DocumentOcr()
    test.run(document_ocr.extract_code)

if __name__ == '__main__':
    main()
