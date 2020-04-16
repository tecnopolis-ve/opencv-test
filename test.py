#@title 
import numpy as np
import cv2
import imutils
import random
import string
import urllib.request
from google.colab.patches import cv2_imshow
from main import DocumentOcr

class TestDocumentOcr(object):
    #Mock data
    def mock(self):
        global image_roi
        if not 'image_roi' in globals():
            image_roi = imutils.url_to_image("https://docs.google.com/uc?export=download&id=1TMmEYfFKbQ7bmneukBlelQCvgTo1DNsP")
        image_roi_rand = image_roi.copy()

        char_set = string.ascii_uppercase + string.digits
        code = ''.join(random.sample(char_set*6, 6))
        cv2.putText(image_roi_rand, code, (int(315), int(80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        #cv2.rectangle(image_roi_rand, (310,50), (444,90), (0,0, 255), thickness= 4)

        image_roi_rand = imutils.rotate_bound(image_roi_rand, random.randrange(-25, 25))

        size = 1000, 1000, 3
        margin = 20

        x = np.clip(random.randrange(margin, 300), 0, size[1] - image_roi_rand.shape[1] - margin)
        y = np.clip(random.randrange(margin, 300), 0, size[0] - image_roi_rand.shape[1] - margin)

        image = np.zeros(size, dtype=np.uint8)
        image[y: y + image_roi_rand.shape[0], x: x + image_roi_rand.shape[1]] = image_roi_rand
        #cv2_imshow(image_roi_rand)
        return image, code

    #Run test
    def run(self, func):
        image, code = self.mock()
        readed_code = func(image)
        assert readed_code == code, 'Error: ' + str(readed_code) + ' != ' + code
        print('Success: test_code ' + readed_code)

def main():
    test = TestDocumentOcr()
    document_ocr = DocumentOcr()
    test.run(document_ocr.extract_code)

if __name__ == '__main__':
    main()