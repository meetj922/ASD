This is a Autism Spectrum Disorder Detection Web Application using Facial Images.
VGG19, MobileNet, Exception and InceptionV3 models were loaded and similar modifications(early stopping, freezing layers and learing rate scheduler) were made.
The top 2 models(VGG19 and MobileNet) were used to create an ensemble model using weighted average.
The ensemble model was able to get an accuracy of 95.3%

Steps to run the application:
1. git clone the repo to your local machine.
2. Change the paths in the "models.ipynb" file according to your local machine.
3. Run "models.ipynb"
4. After the previous step is complete run "app.py".
