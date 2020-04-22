import pytesseract
import cv2
import numpy as np

class DocumentOcr(object):
    
    def extract_code(self, image):

        # Muestro la imagen original
        cv2.imshow("Imagen", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Preproceso la imagen: se elimina la saturación
        # El área de interés corresponde a la zona de mayor luz, los espacios en negro se omiten
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # se calcula el área de interés (explicado arriba) y con esto se determina el ángulo de inclinación
        # para posteriormente poder poner la imagen derecha y luego pasar el OCR
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        coords = cv2.findNonZero(rotated)
        x, y, w, h = cv2.boundingRect(coords)
        rect = rotated[y:y+h, x:x+w]

        # Se pasa a monocromática la imagen para mejorar las probabilidades de detección OCR
        bw_image = cv2.threshold(rect, 64, 255, cv2.THRESH_BINARY)[1]

        # Se selecciona el área de interés que el OCR va a analizar
        image_roi = bw_image[50:100, 300:500] 
        text = pytesseract.image_to_string(image_roi)

        cv2.imshow("Imagen", image_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return text.upper()