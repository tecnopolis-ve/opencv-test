import pytesseract
import cv2
import numpy as np
import imutils
import skimage.exposure
import time

class DocumentOcr(object):

	def extract_code(self, image):

		# Copio la imagen original
		original_image = image.copy()

		# Se eliminan algunas sombras presentes en el background para reducir la posibilidad de falsos positivos
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		range1 = (0, 5, 50)
		range2 = (160, 50, 255)

		mask = cv2.inRange(hsv, range1, range2)
		mask = 255 - mask

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		mask = cv2.GaussianBlur(mask, (0,0), sigmaX=1, sigmaY=1, borderType = cv2.BORDER_DEFAULT)
		mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,255))
		mask = cv2.bitwise_not(mask)

		image[mask==0] = (255,255,255)

		cv2.imshow("Imagen", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
		sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Se elimina la saturación
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (1, 1), 0)

		thresh = cv2.Canny(gray, 125, 225)
		thresh = cv2.dilate(thresh, kernel, 24)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
		thresh = cv2.bitwise_not(thresh)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		thresh = cv2.dilate(thresh, sqKernel, 512)
		thresh = cv2.threshold(thresh, 185, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		thresh = cv2.dilate(thresh, kernel, 128)

		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		roi = None

		c = max(contours, key = cv2.contourArea)

		(x, y, w, h) = cv2.boundingRect(c)

		roi = original_image[y:y + h, x:x + w].copy()
		roi_copy = cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		#FASE FINAL

		roi = cv2.GaussianBlur(roi, (3,3), 0)
		img_hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

		# Blue color
		low_blue = np.array([85, 20, 20])
		high_blue = np.array([140, 255, 255])
		blue_mask = cv2.inRange(img_hsv, low_blue, high_blue)
		blue = cv2.bitwise_and(roi, roi, mask=blue_mask)

		# Green color
		low_green = np.array([30, 20, 20])
		high_green = np.array([75, 255, 255])
		green_mask = cv2.inRange(img_hsv, low_green, high_green)
		green = cv2.bitwise_and(roi, roi, mask=green_mask)

		blue_mask = cv2.GaussianBlur(blue_mask, (15, 15), 0)
		green_mask = cv2.GaussianBlur(green_mask, (15, 15), 0)

		kernelx2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

		blue_mask = cv2.threshold(blue_mask, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		blue_mask = cv2.dilate(blue_mask, kernelx2, 64)

		green_mask = cv2.threshold(green_mask, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		green_mask = cv2.dilate(green_mask, kernelx2, 64)

		blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
		green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

		h_blue,w_blue = blue_mask.shape
		blue_dimension = h_blue*w_blue

		h_green,w_green = green_mask.shape
		green_dimension = h_green*w_green

		bm = cv2.countNonZero(blue_mask)
		gm = cv2.countNonZero(green_mask)

		bmp = 100 * float(bm)/float(blue_dimension)
		gmp = 100 * float(gm)/float(green_dimension)

		if(bmp <= 50):
			blue_mask = cv2.bitwise_not(blue_mask)

		if(gmp <= 50):
			green_mask = cv2.bitwise_not(green_mask)

		contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(contours, key = cv2.contourArea)
		(b_x, b_y, b_w, b_h) = cv2.boundingRect(c)
		blue_mask = blue_mask[b_y:b_y + b_h, b_x:b_x + b_w].copy()

		contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(contours, key = cv2.contourArea)
		(g_x, g_y, g_w, g_h) = cv2.boundingRect(c)
		green_mask = green_mask[g_y:g_y + g_h, g_x:g_x + g_w].copy()

		# calculos adicionales
		h,w,z = roi.shape
		roi_dimension = h*w
		minimum_roi_white = 60
		maximum_roi_white = 95
		maximum_roi_black = 35

		bm = cv2.countNonZero(blue_mask)
		gm = cv2.countNonZero(green_mask)

		bmp = 100 * float(bm)/float(roi_dimension)
		gmp = 100 * float(gm)/float(roi_dimension)

		masks = [
			{
				'mask': blue_mask,
				'total': bmp,
				'name': 'blue_mask',
				'coords': (b_x, b_y, b_w, b_h)
			},
			{
				'mask': green_mask,
				'total': gmp,
				'name': 'green_mask',
				'coords': (g_x, g_y, g_w, g_h)
			}
		]

		max_white = []

		for item in masks:
			h, w = item['mask'].shape
			ar = w/h

			if(minimum_roi_white <= item['total'] <= maximum_roi_white and 1.2 <= ar <= 2.4 and w <= 750):
				max_white.append(item)

		max_white = sorted(max_white, key = lambda i: i['total'], reverse=True)[1 if len(max_white) > 1 else 0]

		final_mask = None

		if(minimum_roi_white <= max_white['total'] <= maximum_roi_white):
			print("Mascara blanca")
			final_mask = max_white
		else:
			print("Imposible determinar mascara")

		if(final_mask is not None):

			cv2.imshow("final_mask", final_mask['mask'])
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			# Se determina la posicion del documento conociendo donde se encuentra la foto
			# la foto es un spot oscuro en la mascara y dependiendo del cuadrante donde se encuentra
			# se rota o no
			(filtered_x, filtered_y, filtered_w, filtered_h) = final_mask['coords']
			image_mask = final_mask['mask']

			CROP_W_SIZE  = 2
			CROP_H_SIZE = 2

			filtered_dimension_quarter = filtered_h*filtered_w//4
			center = (filtered_w // 2, filtered_h // 2)
			height = filtered_h
			width = filtered_w 

			pieces = []

			for ih in range(CROP_H_SIZE):
				for iw in range(CROP_W_SIZE):
					x = width // CROP_W_SIZE * iw 
					y = height // CROP_H_SIZE * ih
					h = (height // CROP_H_SIZE)
					w = (width // CROP_W_SIZE)
					piece = image_mask[y:y+h, x:x+w].copy()
					weight = cv2.countNonZero(piece)
					pieces.append(filtered_dimension_quarter-weight)

			angulo = 90
			indice = np.argmax(pieces)

			roi_filtered = roi[filtered_y:filtered_y + filtered_h, filtered_x:filtered_x + filtered_w].copy()

			if(indice > 0):
				indice -= 1
				tupla_tamano = (filtered_h, filtered_w)

				if(angulo * indice % 180 == 0):
					tupla_tamano = (filtered_w, filtered_h)

				M = cv2.getRotationMatrix2D(center, angulo * indice, 1.0)
				image_mask = cv2.warpAffine(image_mask, M, tupla_tamano)
				roi_filtered = cv2.warpAffine(roi_filtered, M, tupla_tamano)

			# Postprocesado adicional

			post_image = roi_filtered.copy()
			# Se elimina la saturación
			gray = cv2.cvtColor(post_image, cv2.COLOR_BGR2GRAY)
			kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
			gray = cv2.filter2D(gray, -1, kernel)

			# Se calcula el tamaño de la imagen final
			h, w = gray.shape

			h_pos = h//2-20
			w_pos = w//2-20

			image_pre_ocr = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)[1]
			image_roi = cv2.rectangle(gray, (w_pos, h_pos), (w_pos+250, h_pos+55), (0, 255, 0), 3)

			ocr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
			image_pre_ocr = cv2.dilate(image_pre_ocr, ocr_kernel, 1)

			image_ocr = image_pre_ocr[h_pos:h_pos+55, w_pos:w_pos+250].copy()
			image_ocr = cv2.threshold(image_ocr, 0, 255, cv2.THRESH_BINARY)[1]

			# corregir la inclinacion del texto
			# se calcula el área de interés (explicado arriba) y con esto se determina el ángulo de inclinación
			# para posteriormente poder poner la imagen derecha y luego pasar el OCR
			coords = np.column_stack(np.where(image_ocr > 0))
			angle = cv2.minAreaRect(coords)[-1]
			if angle < -45:
				angle = -(90 + angle)
			else:
				angle = -angle

			(h, w) = image_ocr.shape[:2]
			center = (w // 2, h // 2)
			M = cv2.getRotationMatrix2D(center, angle, 1.0)
			rotated = cv2.warpAffine(image_ocr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

			coords = cv2.findNonZero(rotated)
			x, y, w, h = cv2.boundingRect(coords)
			rect = rotated[y:y+h, x:x+w]
			rect = cv2.bitwise_not(rect)
			
			rect = cv2.threshold(rect, 128, 255, cv2.THRESH_BINARY)[1]

			cv2.imshow("PRE OCR", image_pre_ocr)
			cv2.imshow("ROI", image_roi)
			cv2.imshow("OCR", rect)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			text = pytesseract.image_to_string(rect)

		return text.upper()