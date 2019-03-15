import numpy as np
import pandas as pd
import SimpleITK as sitk

def get_label(mask):
    mask_bbox = []

    label = sitk.ConnectedComponent(mask)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(label,mask)
    mask_num = len(stats.GetLabels())

    for l in stats.GetLabels():
        bbox = stats.GetBoundingBox(l)
        bbox = np.array(bbox).astype('float')

        x = bbox[0] + bbox[3]
        y = bbox[1] + bbox[4]
        z = bbox[2] + bbox[5]

        bounding_box = [bbox[0],bbox[1],x,y,bbox[2],z]
        mask_bbox.append(bounding_box)

    return mask_bbox,mask_num

def compute_area_bbox(mask_bbox):
    area_bx = []
    for k in range(len(mask_bbox)):
        areaBbox = (mask_bbox[k][2] - mask_bbox[k][0]) * (mask_bbox[k][3] - mask_bbox[k][1]) * (mask_bbox[k][5] - mask_bbox[k][4])
        area_bx.append(areaBbox)

    return area_bx

def compute_overlap(whz):
    overlap = []
    for kk in range(len(whz)):
        area_overlap = abs(whz[kk][0] * whz[kk][1] * whz[kk][2])
        overlap.append(area_overlap)
    return overlap

def compute_PQ(iou_sum,tp,fp,fn):
    PQ = iou_sum / (tp + fn/2 + fp/2)
    return PQ

if __name__ == "__main__":
    gt_mask = sitk.ReadImage('C:/Users/12sigma/Desktop/ROI_and_output/2_27/HCC_917227_predict_cn.nii.gz')
    gt_array = sitk.GetArrayFromImage(gt_mask)   #  16-bit unsigned integer
    gt_mask = sitk.Cast(gt_mask,sitk.sitkUInt8)
    # gt_array = gt_array[49,:,:]
    # gt_array_img = sitk.GetImageFromArray(gt_array)

    pre_mask = sitk.ReadImage('C:/Users/12sigma/Desktop/ROI_and_output/2_27/HCC_917227_predict.nii.gz')
    pre_array = sitk.GetArrayFromImage(pre_mask)   #  64-bit float
    pre_mask = sitk.Cast(pre_mask,sitk.sitkUInt8)
    # pre_array = pre_array[49,:,:]
    # pre_array_img = sitk.GetImageFromArray(pre_array)

    gt_mask_bbox,gt_mask_num = get_label(gt_mask)
    pre_mask_bbox,pre_mask_num = get_label(pre_mask) 
    print(gt_mask_bbox,pre_mask_num)    
    
    xyz = []
    for i in range (len(gt_mask_bbox)):
        for j in range (len(pre_mask_bbox)):
            x1 = max(gt_mask_bbox[i][0],pre_mask_bbox[j][0])
    #         x1 = max(gt_i_bbox[l][0],pre_i_bbox[l][0])
            y1 = max(gt_mask_bbox[i][1],pre_mask_bbox[j][1])
            
            x2 = min(gt_mask_bbox[i][2],pre_mask_bbox[j][2])
            y2 = min(gt_mask_bbox[i][3],pre_mask_bbox[j][3])
            
            z1 = max(gt_mask_bbox[i][4],pre_mask_bbox[j][4])
            z2 = min(gt_mask_bbox[i][5],pre_mask_bbox[j][5]) 
            x_y_z = [x1,y1,x2,y2,z1,z2]

            xyz.append(x_y_z)
    
    whz = []
    for k in range(len(xyz)):
        width = abs((xyz[k][2] - xyz[k][0]))
        height = abs((xyz[k][3] - xyz[k][1]))
        z = abs((xyz[k][5] - xyz[k][4]))

        w_h_z = [width,height,z]
        whz.append(w_h_z)    

    overlap = compute_overlap(whz)
    gt_bx = compute_area_bbox(gt_mask_bbox)
    pre_bx = compute_area_bbox(pre_mask_bbox)

    combined_1 = []
    for h in range (len(gt_bx)):
        for hh in range(len(pre_bx)):
            combined_1_1 = gt_bx[h] + pre_bx[hh]
            combined_1.append(combined_1_1)

    combined = []
    for m in range (len(combined_1)):
        area_combined = abs(combined_1[m] - overlap[m])
        combined.append(area_combined)

    iou_m = []    
    for m in range(len(combined)):
        iou = overlap[m] / combined[m]
        iou_m.append(iou)

    tp = 0
    PQ = 0 
    for i in range (len(iou_m)):
        if iou_m[i] > 0.5:
            tp = tp + 1
            iou_sum = iou_m[i]
            
            fn = gt_mask_num - tp
            fp = pre_mask_num - tp

    PQ = compute_PQ(iou_sum,tp,fp,fn)
    print('*************')
    print('PQ = ',PQ)
    print('*************')