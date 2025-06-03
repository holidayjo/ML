
import os
import cv2
import numpy as np
import svgwrite
from skimage import measure

# 이미지 경로 설정 (입력 PNG 파일)
img_path = r'C:/Users/23411e/Desktop/Image Process/svg/sample data/Sketch(PNG)/sketch_1 (3).png'

# 이미지 읽기 (그레이스케일 모드)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"이미지를 불러올 수 없습니다: {img_path}")
    exit()

# 이진화 처리 (검정 배경, 흰색 선)
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# 외곽선 추출 (skimage는 더 부드럽게 추출됨)
contours = measure.find_contours(binary, 0.8)

# SVG 크기를 설정하기 위해 이미지 크기 가져오기
height, width = binary.shape

# SVG 출력 디렉토리 설정
output_dir = r'C:/Users/23411e/Desktop/Image Process/svg/sample data/test'
os.makedirs(output_dir, exist_ok=True)

# SVG 출력 경로 설정
output_path = os.path.join(output_dir, 'output_vectorized.svg')

# SVG 객체 생성 (크기 및 보기 영역 설정)
dwg = svgwrite.Drawing(output_path, size=(f'{width}px', f'{height}px'), profile='tiny')
dwg.viewbox(width=width, height=height)

# 모든 외곽선을 SVG에 추가
for contour in contours:
    points = [(float(x[1]), float(x[0])) for x in contour]
    if len(points) > 1:
        path_data = 'M ' + ' L '.join([f'{x:.2f},{y:.2f}' for x, y in points]) + ' Z'
        dwg.add(dwg.path(d=path_data, fill='none', stroke='black', stroke_width=1))

# SVG 파일 저장
dwg.save()
print(f"벡터 이미지가 저장되었습니다: {output_path}")
