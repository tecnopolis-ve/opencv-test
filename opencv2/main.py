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

		# blackhat = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

		# gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
		# gradX = np.absolute(gradX)
		# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
		# gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
		# thresh = cv2.Canny(blackhat, 200, 255)

		# cv2.imshow("Imagen", thresh)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)
		#thresh = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		thresh = cv2.bitwise_not(thresh)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		thresh = cv2.dilate(thresh, sqKernel, 512)
		thresh = cv2.threshold(thresh, 185, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		thresh = cv2.dilate(thresh, kernel, 128)

		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# for cnt in contours:
		# 	cv2.drawContours(thresh, [cnt], 0, 255, 8)

		roi = None

		c = max(contours, key = cv2.contourArea)

		(x, y, w, h) = cv2.boundingRect(c)

		# pX = int((x + w) * 0.03)
		# pY = int((y + h) * 0.03)
		# (x, y) = (x - pX, y - pY)
		# (w, h) = (w + (pX * 2), h + (pY * 2))

		roi = original_image[y:y + h, x:x + w].copy()
		roi_copy = cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the output images
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

		cv2.imshow("blue_mask", blue_mask)
		cv2.imshow("green_mask", green_mask)

		# # cv2.erode(red_mask,kernel, 8)
		# # cv2.dilate(red_mask,kernel, 32)

		# # cv2.erode(blue_mask,kernel, 8)
		# # cv2.dilate(blue_mask,kernel, 32)

		# # cv2.erode(green_mask,kernel, 8)
		# # cv2.dilate(green_mask,kernel, 32)

		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

		# red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
		# blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
		# green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

		blue_mask = cv2.GaussianBlur(blue_mask, (15, 15), 0)
		green_mask = cv2.GaussianBlur(green_mask, (15, 15), 0)

		cv2.imshow('blue_mask2g', blue_mask)
		cv2.imshow('green_mask2g', green_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		kernelx2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

		blue_mask = cv2.threshold(blue_mask, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		blue_mask = cv2.dilate(blue_mask, kernelx2, 64)

		green_mask = cv2.threshold(green_mask, 160, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		green_mask = cv2.dilate(green_mask, kernelx2, 64)

		cv2.imshow('blue_mask2g', blue_mask)
		cv2.imshow('green_mask2g', green_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

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

		print(bmp, gmp)

		if(bmp <= 50):
			blue_mask = cv2.bitwise_not(blue_mask)

		if(gmp <= 50):
			green_mask = cv2.bitwise_not(green_mask)

		cv2.imshow('antes_rojoblue_mask2', blue_mask)
		cv2.imshow('antes_rojogreen_mask2', green_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(contours, key = cv2.contourArea)
		(b_x, b_y, b_w, b_h) = cv2.boundingRect(c)
		blue_mask = blue_mask[b_y:b_y + b_h, b_x:b_x + b_w].copy()

		contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(contours, key = cv2.contourArea)
		(g_x, g_y, g_w, g_h) = cv2.boundingRect(c)
		green_mask = green_mask[g_y:g_y + g_h, g_x:g_x + g_w].copy()

		cv2.imshow('blue_mask2', blue_mask)
		cv2.imshow('green_mask2', green_mask)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

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
			mask_attr = item['mask'].shape
			ar = w/h

			print(item['name'], item['total'], ar, w)

			if(minimum_roi_white <= item['total'] <= maximum_roi_white and 1.4 <= ar <= 2.4 and w < 650):
				max_white.append(item)
			# else:
				# min_white.append(item)

		print(max_white)

		max_white = sorted(max_white, key = lambda i: i['total'], reverse=True)[1 if len(max_white) > 1 else 0]
		# min_white = min(min_white, key=lambda x:x['total'])

		final_mask = None

		if(minimum_roi_white <= max_white['total'] <= maximum_roi_white):
			print("Mascara blanca")
			final_mask = max_white
		# elif (min_white['total'] <= maximum_roi_black):
		# 	print("Mascara negra")
		# 	final_mask = cv2.bitwise_not(min_white['mask'])
		else:
			print("Imposible determinar mascara")

		cv2.imshow("final_mask", final_mask['mask'])
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		# red_mask = cv2.bitwise_not(red_mask)
		# blue_mask = cv2.bitwise_not(blue_mask)
		# green_mask = cv2.bitwise_not(green_mask)

		cv2.imshow("roi", roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		if(final_mask is not None):

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

			# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
			# gray = clahe.apply(gray)
			#gray = cv2.GaussianBlur(gray, (1, 1), 0)

			kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
			gray = cv2.filter2D(gray, -1, kernel)

			# Se calcula el tamaño de la imagen final
			h, w = gray.shape

			h_pos = h//2-20
			w_pos = w//2-20

			image_pre_ocr = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
			image_roi = cv2.rectangle(gray, (w_pos, h_pos), (w_pos+250, h_pos+55), (0, 255, 0), 3)
			# cv2.imshow("OCR", thresh)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			ocr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
			image_pre_ocr = cv2.dilate(image_pre_ocr, ocr_kernel, 1024)

			cv2.imshow("OCR", image_pre_ocr)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			image_ocr = image_pre_ocr[h_pos:h_pos+55, w_pos:w_pos+250].copy()
			image_ocr = cv2.threshold(image_ocr, 245, 255, cv2.THRESH_BINARY)[1]

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

			kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

			cv2.imshow("PRE OCR", image_pre_ocr)
			cv2.imshow("ROI", image_roi)
			cv2.imshow("OCR", rect)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

			text = pytesseract.image_to_string(rect)

		# red_mask = cv2.erode(red_mask,kernel, 8)
		# blue_mask = cv2.erode(blue_mask,kernel, 8)
		# green_mask = cv2.erode(green_mask,kernel, 8)

		# red_mask = cv2.dilate(red_mask,kernel, 64)
		# blue_mask = cv2.dilate(blue_mask,kernel, 64)
		# green_mask = cv2.dilate(green_mask,kernel, 64)

		# # cv2.imshow("Red3", red_mask)
		# # cv2.imshow("Blue3", blue_mask)
		# # cv2.imshow("Green3", green_mask)

	#####/////////////////////////////////////////////////

		# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# #thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# thresh = cv2.Canny(gray, 100, 120)
		# # thresh = cv2.dilate(thresh, sqKernel, 1)
		# thresh = cv2.GaussianBlur(thresh, (13,13), 0)
		# thresh = cv2.threshold(thresh, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# thresh = cv2.bitwise_not(thresh)

		# cv2.imshow('sharpen', thresh)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# thresh = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)



		# cv2.imshow('sharpen', thresh)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# thresh = cv2.GaussianBlur(thresh, (27,27), 0)
		# thresh = cv2.threshold(thresh, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		# sharpen = cv2.filter2D(blur, -1, sharpen_kernel)



		# thresh = cv2.threshold(sharpen,100,255, cv2.THRESH_BINARY_INV)[1]
		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

		# cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

		# min_area = 100
		# max_area = 1500
		# image_number = 0
		# for c in cnts:
		# 	area = cv2.contourArea(c)
		# 	if area > min_area and area < max_area:
		# 		x,y,w,h = cv2.boundingRect(c)
		# 		ROI = image[y:y+h, x:x+h]
		# 		cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
		# 		cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
		# 		image_number += 1

		# cv2.imshow('sharpen', sharpen)
		# cv2.imshow('close', close)
		# cv2.imshow('thresh', thresh)
		# cv2.waitKey()

		# # # cv2.imshow("Red", red_mask)
		# # # cv2.imshow("Blue", blue_mask)
		# # # cv2.imshow("Green", green_mask)


		# # # cv2.erode(red_mask,kernel, 8)
		# # # cv2.dilate(red_mask,kernel, 32)

		# # # cv2.erode(blue_mask,kernel, 8)
		# # # cv2.dilate(blue_mask,kernel, 32)

		# # # cv2.erode(green_mask,kernel, 8)
		# # # cv2.dilate(green_mask,kernel, 32)

		# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

		# # red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
		# # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
		# # green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)


		# # # cv2.imshow("Red2", red_mask)
		# # # cv2.imshow("Blue2", blue_mask)
		# # # cv2.imshow("Green2", green_mask)

		# # red_mask = cv2.erode(red_mask,kernel, 8)
		# # blue_mask = cv2.erode(blue_mask,kernel, 8)
		# # green_mask = cv2.erode(green_mask,kernel, 8)

		# # red_mask = cv2.dilate(red_mask,kernel, 64)
		# # blue_mask = cv2.dilate(blue_mask,kernel, 64)
		# # green_mask = cv2.dilate(green_mask,kernel, 64)

		# # # cv2.imshow("Red3", red_mask)
		# # # cv2.imshow("Blue3", blue_mask)
		# # # cv2.imshow("Green3", green_mask)

		# # cv2.waitKey(0)
		# # cv2.destroyAllWindows()

		# # # set my output img to zero everywhere except my mask


		# # # # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# # # # thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# # # cv2.imshow("ROI", roi)
		# # # cv2.waitKey(0)
		# # # cv2.destroyAllWindows()

		# # # dilate = cv2.dilate(thresh,kernel,6)
		# # # erode = cv2.erode(dilate,kernel,6)

		# # # morph_img = thresh.copy()
		# # # cv2.morphologyEx(src=erode, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)

		# # # cv2.imshow("ROI", thresh)
		# # # cv2.waitKey(0)
		# # # cv2.destroyAllWindows()

		# # # coords = np.column_stack(np.where(thresh > 0))
		# # # angle = cv2.minAreaRect(coords)[-1]
		# # # if angle < -45:
		# # # 	angle = -(90 + angle)
		# # # else:
		# # # 	angle = -angle

		# # # (h, w) = image.shape[:2]
		# # # center = (w // 2, h // 2)
		# # # M = cv2.getRotationMatrix2D(center, angle, 1.0)
		# # # rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		# # # coords = cv2.findNonZero(rotated)
		# # # x, y, w, h = cv2.boundingRect(coords)
		# # # rect = rotated[y:y+h, x:x+w]




        # # # # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # # # thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # # # # # se calcula el área de interés (explicado arriba) y con esto se determina el ángulo de inclinación
        # # # # # para posteriormente poder poner la imagen derecha y luego pasar el OCR
        # # # # coords = np.column_stack(np.where(thresh > 0))
        # # # # angle = cv2.minAreaRect(coords)[-1]
        # # # # if angle < -45:
        # # # #     angle = -(90 + angle)
        # # # # else:
        # # # #     angle = -angle

        # # # # (h, w) = image.shape[:2]
        # # # # center = (w // 2, h // 2)
        # # # # M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # # # # rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # # # # coords = cv2.findNonZero(rotated)
        # # # # x, y, w, h = cv2.boundingRect(coords)
        # # # # rect = rotated[y:y+h, x:x+w]

        # # # # # Se pasa a monocromática la imagen para mejorar las probabilidades de detección OCR
        # # # # bw_image = cv2.threshold(rect, 64, 255, cv2.THRESH_BINARY)[1]

        # # # # # Se selecciona el área de interés que el OCR va a analizar
        # # # # image_roi = bw_image[50:100, 300:500] 
        # # # # text = pytesseract.image_to_string(image_roi)

        # # # # cv2.imshow("Imagen", image_roi)
        # # # # cv2.waitKey(0)
        # # # cv2.destroyAllWindows()


		return text.upper()