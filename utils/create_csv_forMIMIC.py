from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from proprcessing import *
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    all_text_cvs = pd.read_csv('../MIMIC/physionet.org/files/mimic-cxr/2.0.0/files/mimic_cxr_sectioned.csv')
    report_csv_name = '../MIMIC/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz'
    report_csv = pd.read_csv(report_csv_name)
    img_csv_name='../MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz'
    img_csv = pd.read_csv(img_csv_name)
    p10_csv = img_csv[img_csv.subject_id<11000000]
    img_dir = '../MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10'
    report_dir = '../MIMIC/physionet.org/files/mimic-cxr/2.0.0'
    study_ids = []
    dicom_ids = []
    report_ids = []
    imagepaths = []
    direction = []
    reportpaths = []
    for study_id in tqdm(p10_csv.study_id):
        meta_study = p10_csv[p10_csv.study_id==study_id]
        PA_study = meta_study[meta_study.ViewPosition=='PA']
        AP_study = meta_study[meta_study.ViewPosition=='AP']
        LA_study = meta_study[meta_study.ViewPosition=='LATERAL']
        report_study = report_csv[report_csv.study_id==study_id].iloc[0]
        if study_id in study_ids:
            continue

        if PA_study.shape[0]>0:
            F_study = PA_study.iloc[0]
        elif AP_study.shape[0]>0:
            F_study = AP_study.iloc[0]
        else:
            continue

        if LA_study.shape[0]>0:
            L_study = LA_study.iloc[0]
        else:
            continue

        report = all_text_cvs[all_text_cvs.study==study_id].iloc[0]

        if type(report.findings)==float or type(report.impression)==float:
            print('NAN')
            continue

        study_ids.append(study_id)
        study_ids.append(study_id)
        dicom_ids.append(L_study.dicom_id)
        dicom_ids.append(F_study.dicom_id)
        imagepaths.append('{}/p{}/s{}/_{}.jpg'.format(img_dir,
                                                L_study.subject_id,
                                                L_study.study_id,
                                                L_study.dicom_id))
        imagepaths.append('{}/p{}/s{}/_{}.jpg'.format(img_dir,
                                                    F_study.subject_id,
                                                    F_study.study_id,
                                                    F_study.dicom_id))
        direction.append('L')
        direction.append('F')
        reportpaths.append('{}/{}'.format(report_dir,
                                          report_study.path))
        report_ids.append(report_study.study_id)

    image_data = {'subject_id': study_ids,
                  'dicom_id': dicom_ids,
                  'path': imagepaths,
                  'direction': direction}
    img_df = pd.DataFrame(image_data)
    img_df.to_csv(os.path.join('./config', 'MIMIC_p10_images.csv'))

    report_data = {'subject_id': report_ids,
                   'path': reportpaths}
    rp_df = pd.DataFrame(report_data)
    rp_df.to_csv(os.path.join('./config', 'MIMIC_p10_reports.csv'))