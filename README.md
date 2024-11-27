## Handwritten-Captcha-detector

# A captcha detector which uses Deep Learning techniques to detech handwritten captchas.

- Two different Convolutional Neural Networks are trained to predict characters from an input image.
- The characters set used was "A C E H J K L M P Q R T U W X Y d h n 0 2 3 5 6 8 9" since there was confusion among characters like (1,L,I),(9,q,g),(0,O,o), etc so, the characters causing this were removed and 26 were selected.
- Data was augmented using ImageDataGenerator in Keras
- Augmentation included rotation of characters by 45 degrees, width_shift of 0.1, height_shift of 0.1, shear of 0.2, and zoom of 0.1 magnitudes.
- The first model trained on images achieved a training accuracy of ~90% and validation accuracy of ~82%
- The second model trained on EMNIST dataset achieved a training accuracy of ~95% and validation accuracy of ~85%
- **Weighted Ensembling** was used to get the best out of both the models.
- The captcha image was preprocessed using OpenCV library in python to remove noise, separate characters, etc and then each character was individually fed into the Neural Network for prediction.
- The final ensembled model was able to identify captchas even with a noisy background, dark background etc.
