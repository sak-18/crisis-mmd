# Crisis MMD

1) Install requirements: `pip install -r requirements.txt`.

2) Run `python app.py` and go to `http://127.0.0.1:5000`.

You might need access to a few files/folders not on the repository that you can find in the dropbox link.
- `data_dump/all_images_data_dump.npy`
- `data_image`
- `model` (contains all the trained models for ensemble)
- `env` (might not be necessary after installing the requirements with pip)

## Tree structure of the project

```
.
├── Readme.md
├── __pycache__
│   ├── aidrtokenize.cpython-310.pyc
│   ├── aidrtokenize.cpython-37.pyc
│   ├── aidrtokenize.cpython-38.pyc
│   ├── aidrtokenize.cpython-39.pyc
│   ├── crisis_data_generator_image_optimized.cpython-310.pyc
│   ├── crisis_data_generator_image_optimized.cpython-37.pyc
│   ├── crisis_data_generator_image_optimized.cpython-38.pyc
│   ├── crisis_data_generator_image_optimized.cpython-39.pyc
│   ├── data_process_multimodal_pair.cpython-310.pyc
│   ├── data_process_multimodal_pair.cpython-37.pyc
│   ├── data_process_multimodal_pair.cpython-38.pyc
│   ├── data_process_multimodal_pair.cpython-39.pyc
│   ├── data_process_new.cpython-310.pyc
│   └── data_process_new.cpython-38.pyc
├── aidrtokenize.py
├── app.py
├── crisis_data_generator_image_optimized.py
├── data_dump
│   └── all_images_data_dump.npy
├── data_image
│   ├── california_wildfires
│   ├── hurricane_harvey
│   ├── hurricane_irma
│   ├── hurricane_maria
│   ├── iraq_iran_earthquake
│   ├── mexico_earthquake
│   └── srilanka_floods
├── data_process_multimodal_pair.py
├── data_process_new.py
├── dmd
│   ├── multimodal
│   └── readme.txt
├── env
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   └── pyvenv.cfg
├── metadata
│   ├── task_humanitarian_text_img_agreed_lab_dev.tsv
│   ├── task_humanitarian_text_img_agreed_lab_test.tsv
│   ├── task_humanitarian_text_img_lab_test.tsv
│   ├── task_informative_text_img_agreed_lab_dev.tsv
│   ├── task_informative_text_img_agreed_lab_test.tsv
│   ├── task_informative_text_img_lab_test.tsv
│   └── task_severity_test.tsv
├── model
│   ├── hum_multimodal_paired_agreed_lab.tokenizer
│   ├── humanitarian_cnn_keras_09-04-2022_05-10-03.hdf5
│   ├── humanitarian_cnn_keras_09-04-2022_05-10-03.tokenizer
│   ├── humanitarian_image_vgg16_ferda.hdf5
│   ├── info_multimodal_paired_agreed_lab.tokenizer
│   ├── informative_image.hdf5
│   ├── informativeness_cnn_keras.hdf5
│   ├── informativeness_cnn_keras_09-04-2022_04-26-49.hdf5
│   ├── informativeness_cnn_keras_09-04-2022_04-26-49.tokenizer
│   ├── model_info_x copy.hdf5
│   ├── model_info_x.hdf5
│   ├── model_info_x1.hdf5
│   ├── model_info_x2.hdf5
│   ├── model_severe_x.hdf5
│   ├── model_severe_x1.hdf5
│   ├── model_severe_x2.hdf5
│   ├── model_x.hdf5
│   ├── model_x1.hdf5
│   ├── model_x2.hdf5
│   ├── severity_cnn_keras_21-07-2022_08-14-32.hdf5
│   ├── severity_cnn_keras_21-07-2022_08-14-32.tokenizer
│   └── severity_image.hdf5
├── performance_measures
│   ├── humanitarian.csv
│   ├── informative.csv
│   └── severity.csv
├── requirements.txt
├── static
│   ├── Capture.PNG
│   ├── Capture1.PNG
│   ├── base.html
│   ├── bg.jpg
│   ├── bg.png
│   ├── capture3.png
│   ├── case1.png
│   ├── case1_1.PNG
│   ├── case1_Arch.PNG
│   ├── case2.png
│   ├── case2_2.PNG
│   ├── case2_arch.PNG
│   ├── case3.png
│   ├── case3_3.PNG
│   ├── case3_arch.png
│   ├── classification.PNG
│   ├── classification1.png
│   ├── css
│   ├── data_image
│   ├── data_image_wrong
│   ├── heatmap.jpg
│   ├── humanitarian.csv
│   ├── index.html
│   ├── informative.csv
│   ├── js
│   ├── result.html
│   ├── severity.csv
│   ├── text-removebg-preview.png
│   ├── text.jpg
│   ├── vgg-removebg-preview.png
│   ├── visualize.html
│   └── visualize.jpg
├── stop_words
│   └── stop_words_english.txt
└── templates
    ├── base.html
    ├── index.html
    ├── result.html
    ├── temp
    └── visualize.html
```