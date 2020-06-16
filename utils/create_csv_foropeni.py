import os
import pandas
from tqdm import tqdm
from proprcessing import test_XML,find_parentImage
import json

basepath = "F:/openi"
imagedir = "F:/openi/NLMCXR_png"
report_root = os.path.join(basepath,'ecgen-radiology')
to_basepath = "../data"
to_imgdir = to_basepath+'/'+'imgs'
to_reportdir = to_basepath+'/'+'reports/ecgen-radiology'

label_file = 'label.json'
with open(label_file) as json_file:
    label = json.load(json_file)

# id for patients
subject_ids = []
report_id = []
# The path of the image
imagepaths = []
# id for the report
reportpaths = []
# id for the image
dicom_ids = []

direction = []

# For each chest report
for name in tqdm(os.listdir(report_root)):
    id = os.path.splitext(name)[0]
    report_file = os.path.join(report_root,name)
    # If the report is with correct format
    if test_XML(report_file):
        image_ids = find_parentImage(report_file)
        img_names = [image_id + '.png' for image_id in image_ids if image_id+'.png' in label.keys()]
        if len(img_names)==2:
            reportpaths.append(to_reportdir + '/' + name)
            report_id.append(id)
            for img_n in img_names:
                image_id = img_n.split('.')[0]
                img_path = imagedir+'/'+img_n
                assert os.path.exists(img_path), "{} image not exists".format(img_path)
                subject_ids.append(id)
                dicom_ids.append(image_id)
                direction.append(label[img_n])
                imagepaths.append(to_imgdir+'/'+img_n)

image_data = {'subject_id':subject_ids,
              #'study_id':subject_ids,
              'dicom_id':dicom_ids,
              'path':imagepaths,
              'direction':direction}
img_df = pandas.DataFrame(image_data)
img_df.to_csv(os.path.join('E:\pyProject\\text-to-image\config','openi_images.csv'))

report_data = {'subject_id':report_id,
              #'study_id':subject_ids,
              'path':reportpaths}
rp_df = pandas.DataFrame(report_data)
rp_df.to_csv(os.path.join('E:\pyProject\\text-to-image\config','openi_reports.csv'))