# DermaTrace.ai

Traditional AI for skin cancer is a "black box" that doctors often distrust. Our model uses OpenCV and Grad-CAM to generate visual heatmaps, highlighting exactly which lesion features the AI is analyzing. By transforming raw probability scores into transparent, clinical evidence, we bridge the trust gap between AI and medical professionals.

Our model was trained on the HAM10000 dataset, with further bias corrections done using the fitzpatrick dataset for non-dermoscopic images. DermaTrace.ai uses two models, a double pipeline intelligence which ensures that image detection is done accurately depending on the quality of the assigned images (dermoscopic vs non-dermoscopic). 

A set of 30 skin images, as well as a few non-skin images have been attached for sample testing, including 10 images from the Mednode dataset, on which the model has not been NOT trained on, in order to audit the models' actual capabilities and ensure that it has not been overfitted for the specific dataset, also proving generalization.

Dermatrace.ai aims to be an assistive technology for dermatologists to detect the early onset of skin-cancer and differentiate it early. It also serves as a tool for doctors in rural areas with a lack of access to proper medical technology, enabling them to diagnose cancer early. This makes it an effective tool for high-end dermoscopy clinics as well as rural areas with a lack of access to medical grade equipment.

Safety gates such as identification of the image being skin or not, along with a confidence rating, and top 2 possibilities ensure that the model detects the conditions accurately, assisting dermatologists to make a proper and informed diagnosis, with the help of our eXplainable AI (XAI) program. 

MST analysis of provided images has also been added with a +-2 accuracy of over 95% tested on the a cleaned fitzpatrick set with pre-graded skin tones, mitigating bias among darker skin tones, that the AI has been trained on. While we acknowledge the fact that DermaTrace.ai might not work as accurately on darker skin tones (MST 7-10), we have done our best to remove bias for those given skin tones.

DermaTrace.ai is an assistive XAI tool designed to support clinical decision-making. It is not a standalone diagnostic device. Results should always be interpreted by a qualified medical professional.