import torch
import cv2
import numpy as np
from torch.nn import functional as F



#
def add_prediction_to_image_base(image,regression_point,target_point):
    image = image.squeeze(0)
    image = image.permute(1,2,0)
    image = image.cpu().data.numpy()
    image = image * 255.
    image_size = 512

    regression_point = regression_point.cpu().data.numpy()
    target_point = target_point.cpu().data.numpy()
    dist = np.sqrt(np.power((regression_point[0][0] * image_size - target_point[0][0] * image_size), 2) + np.power((regression_point[0][1] * image_size - target_point[0][1] * image_size), 2))

    image = cv2.circle(image, (int(regression_point[0][0] * image_size), int(regression_point[0][1] * image_size)), 5, (255, 0, 0),-1)
    image = cv2.circle(image, (int(target_point[0][0] * image_size), int(target_point[0][1] * image_size)), 5, (0, 0, 255), -1)
    cv2.putText(image, "DIST:" + str(round(dist, 4)), (5, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image / 255.

    return image



def add_prediction_to_image_batch(images,prediction_points,target_points):
    image_sp = torch.split(images, split_size_or_sections=1, dim=0)
    prediction_point_sp = torch.split(prediction_points, split_size_or_sections=1, dim=0)
    target_point_sp = torch.split(target_points, split_size_or_sections=1, dim=0)

    return_sp = []
    for i in range(len(image_sp)):
        return_sp.append(add_prediction_to_image_base(image_sp[i], prediction_point_sp[i], target_point_sp[i]))

    return torch.cat(return_sp, dim=0)




def GetCenterfromHeatMap(heatmap):
    width,height = heatmap.size()[2:4]
    _,pos = F.max_pool2d(heatmap,(width,height),return_indices=True)
    pos = pos.squeeze(-1).squeeze(-1).cpu().data.numpy()
    pos = np.concatenate([((pos+1)%width),((pos+1)//height+1)],axis=1)

    return pos/width