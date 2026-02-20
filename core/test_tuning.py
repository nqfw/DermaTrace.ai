import cv2

cat_img = cv2.resize(cv2.imread(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\test images\calm-orange-cat-0410-5700096.jpg"), (600, 450))
texture_score = cv2.Laplacian(cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
print(f"Cat Texture Score: {texture_score}")

img2 = cv2.resize(cv2.imread(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\test images\istockphoto-638880710-612x612.jpg"), (600, 450))
print(f"Istock Texture Score: {cv2.Laplacian(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()}")

ham = cv2.imread(r"C:\Users\lenovo\OneDrive\Desktop\HACKATHON\data\HAM10000 dataset\HAM10000_images_part_1\ISIC_0024306.jpg")
print(f"HAM Texture Score: {cv2.Laplacian(cv2.cvtColor(cv2.resize(ham, (600, 450)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()}")
