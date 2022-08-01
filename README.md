# App-Based-Covid-Detection-Using-CT-and-XRay-Images
This project is an implementation of an app-based covid-19 diagnosis system 
## Project description
The first step in the treatment of COVID-19 is to screen patients in primary health centers or hospitals. Although the final diagnosis still relies mainly on transcription-polymerase chain reaction (PCR) tests, in case of people with strong respiratory symptoms the election protocol nowadays in hospitals relays on medical imaging, as it is simple and fast, thus helping doctors to identify diseases and their effects more quickly. Following this protocol, patients that are suspected to suffer COVID-19 undergoes first an X-Ray session and then, in the case that more details are needed, they take a CT-scan session. As a result of this protocol, computed tomography scan (CT scan) and X-ray images are being widely used on the clinic as alternative diagnostic tools for detecting COVID-19.

Our goal was to implement a diagnosis sytem that classifies X-ray and CT images of patient's lungs into 2 categories: Covid and Non-Covid classes.

# About the datasets
* We used 2 datasets for this project. For implemeting CT-based model, we used [SARS-COV-2](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) dataset containing 1252 CT scans that are positive for SARS-CoV-2 infection (COVID-19) and 1230 CT scans for patients non-infected by SARS-CoV-2, 2482 CT scans in total.
