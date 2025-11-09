import cv2

image_path = "internal_assets\\extra_dataset\\images\\classes-Clothing-Accessories\\bracelet\\0.56_aacb233a-84df-46e9-873f-9fd35b3d906f_e468832eb89d4b48acadaae61f4c6feb_master.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(image_path)

x1, y1, x2, y2, conf = 95.969337, 1.304102, 1149.659180, 825.284180, 0.9176
label = "person"

pt1 = (int(x1), int(y1))
pt2 = (int(x2), int(y2))
cv2.rectangle(img, pt1, pt2, (0,255,0), 2)
cv2.putText(img, f"{label} {conf:.2f}", (pt1[0], pt1[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, lineType=cv2.LINE_AA)

cv2.imwrite("output_check.jpg", img)
print("Saved: output_check.jpg")
