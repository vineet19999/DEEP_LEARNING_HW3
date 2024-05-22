import torch
import torchvision
import datetime
import os
import time
import numpy as np
from DeepDataMiningLearning.detection import utils
#from DeepDataMiningLearning.detection.coco_eval import CocoEvaluator
#from DeepDataMiningLearning.detection.coco_utils import get_coco_api_from_dataset
from pycocotools.coco import COCO #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util
import copy
import io
from contextlib import redirect_stdout
from tqdm.auto import tqdm

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys()))) #[139]
        self.img_ids.extend(img_ids) #[139]

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions): #predictions, key=image_id, val=dict
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"] #10,4
            boxes = utils.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist() #list of 10
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results #create a list of 10 dicts, each with "image_id", and one box (4)

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = utils.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results

def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    # for _ in range(10):
    #     if isinstance(dataset, torchvision.datasets.CocoDetection):
    #         break
    #     if isinstance(dataset, torch.utils.data.Subset):
    #         dataset = dataset.dataset
    # if isinstance(dataset, torchvision.datasets.CocoDetection):
    #     return dataset.coco
     return convert_to_coco_api(dataset)

def convert_to_coco_api2(ds):#mykittidetectiondataset
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ds_len=len(ds)
    print("convert to coco api:")
    progress_bar = tqdm(range(ds_len))
    for img_idx in range(ds_len):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2] #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
        progress_bar.update(1)
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    #print("convert_to_coco_api",dataset["categories"])
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def convert_to_coco_api(ds):#mykittidetectiondataset
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ds_len=len(ds)
    print("convert to coco api:")
    progress_bar = tqdm(range(ds_len))
    for img_idx in range(ds_len):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx] #img is [3, 1280, 1920], 
        image_id = targets["image_id"] #68400
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone() #torch.Size([23, 4])
        bboxes[:, 2:] -= bboxes[:, :2] #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist() #23 list of [536.0, 623.0, 51.0, 18.0]
        labels = targets["labels"].tolist() #torch.Size([23]) -> list 23 [1,1,1]
        areas = targets["area"].tolist() #torch.Size([23]) -> list 23 []
        iscrowd = targets["iscrowd"].tolist() #torch.Size([23]) -> list
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i] #int
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
        progress_bar.update(1)
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    #print("convert_to_coco_api",dataset["categories"])
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
    
def simplemodelevaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset) #go through the whole dataset, convert_to_coco_api
    #coco = convert_to_coco_api(data_loader.dataset)
    iou_types = ["bbox"] #_get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    evalprogress_bar = tqdm(range(len(data_loader)))

    for images, targets in data_loader: #images, targets are a tuple (tensor, )
        images = list(img.to(device) for img in images) #list of torch.Size([3, 426, 640]), len=1
        #targets: len=1 dict (image_id=139), boxes[20,4], labels[20]
        model_time = time.time()
        outputs = model(images) #len1 list of dict boxes tensor (10x4), labels tensor[10], scores

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs] #len1 list of dicts with tensors
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)} #dict, key=139, val=dict[boxes] 10,4
        #print("res:", res)
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        evalprogress_bar.update(1)

    # gather the stats from all processes
    #coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.inference_mode()
def modelevaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = utils._get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        #print("res:", res)
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def yoloconvert_to_coco_api(ds):#mykittidetectiondataset
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    ds_len=len(ds)
    print("convert to coco api:")
    progress_bar = tqdm(range(ds_len))
    #for img_idx in range(ds_len):
    for batch in ds:
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        #img, targets = ds[img_idx]
        #batch = ds[img_idx]
        img = batch['img'] #[3, 640, 640] [1, 3, 640, 640]
        image_id = batch['image_id'][0] #targets["image_id"] 0
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = batch['bboxes'].clone() #normalized xc, yc, width, height
        #bboxes = targets["boxes"].clone()
        W=640
        H=640
        bboxes[:,0]=bboxes[:,0]*W
        bboxes[:,1]=bboxes[:,1]*H
        bboxes[:,2]=bboxes[:,2]*W
        bboxes[:,3]=bboxes[:,3]*H
        bboxes[:,0]=bboxes[:,0]-bboxes[:,2]/2 #-w/2: xmin
        bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2 #-H/2: ymin
        #bboxes[:, 2:] -= bboxes[:, :2] #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist()
        labels = batch['cls'].tolist()
        #labels = targets["labels"].tolist()
        areas = batch["area"][0].tolist()
        #areas = targets["area"].tolist()
        iscrowd = batch["iscrowd"][0].tolist()
        #iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
        progress_bar.update(1)
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    #print("convert_to_coco_api",dataset["categories"])
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds

def yoloevaluate(model, data_loader, preprocess, device):

    cpu_device = torch.device("cpu")
    model.eval()

    #coco = get_coco_api_from_dataset(data_loader.dataset) #go through the whole dataset, convert_to_coco_api
    #coco = yoloconvert_to_coco_api(data_loader)
    iou_types = ["bbox"] #_get_iou_types(model)
    #coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    evalprogress_bar = tqdm(range(len(data_loader)))

    all_res=[]
    for batch in data_loader:
        targets={}
        #convert from yolo data format to COCO
        img = batch['img'] # [1, 3, 640, 640]
        img_dict = {}
        image_id = batch['image_id'][0] #batch['image_id'] is tuple, get the 0-th element
        targets["image_id"]=image_id
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2] #img is CHW
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)

        box=batch['bboxes'] #normalized xc, yc, width, height
        box=torchvision.ops.box_convert(box, 'cxcywh', 'xyxy')#xcenter, ycenter,wh, to xyxy
        box=box.numpy()
        (H, W, C)=(640,640,3) #batch['orig_shape'][0] #tuple
        originalshape = [[H, W, C]]
        box[:,0]=box[:,0]*W
        box[:,1]=box[:,1]*H
        box[:,2]=box[:,2]*W
        box[:,3]=box[:,3]*H
        targets['boxes'] = box #xmin, ymin, xmax, ymax
        oneimg=torch.squeeze(img, 0) #[3, 640, 640]
        targets['labels']=batch['cls'].numpy()
        #vis_example(targets, oneimg, filename='result1.jpg')

        bboxes = batch['bboxes'].clone() #normalized xc, yc, width, height
        bboxes[:,0]=bboxes[:,0]*W
        bboxes[:,1]=bboxes[:,1]*H
        bboxes[:,2]=bboxes[:,2]*W
        bboxes[:,3]=bboxes[:,3]*H
        bboxes[:,0]=bboxes[:,0]-bboxes[:,2]/2 #-w/2: xmin
        bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2 #-H/2: ymin
        #[xmin, ymin, xmax, ymax] in torch to [xmin, ymin, width, height] in COCO
        bboxes = bboxes.tolist()
        labels = batch['cls'].tolist()
        #labels = targets["labels"].tolist()
        areas = batch["area"][0].tolist()
        #areas = targets["area"].tolist()
        iscrowd = batch["iscrowd"][0].tolist()
        #iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    
        #Inference
        batch['img']=preprocess(batch['img']) #batch['img'] = batch['img'].to(device)
        #img is already a tensor, preprocess function only do device

        #images = list(img.to(device) for img in images) #list of torch.Size([3, 426, 640]), len=1
        #targets: len=1 dict (image_id=139), boxes[20,4], labels[20]
        model_time = time.time()
        #outputs = model(images) #len1 dict boxes (10x4), labels[10], scores
        imgtensors = batch['img']
        preds = model(imgtensors)
        imgsize = imgtensors.shape[2:] #640, 640
        outputs = preprocess.postprocess(preds, imgsize, originalshape)
        #outputs["boxes"] (xmin, ymin, xmax, ymax) format ["scores"] ["labels"]

        #vis_example(outputs[0], oneimg, filename='result2.jpg')

        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        targets = [targets] #make it a list
        res = {target["image_id"]: output for target, output in zip(targets, outputs)} #dict, key=139, val=dict[boxes] 10,4
        #print("res:", res) #image_id: output['boxes'] ['scores'] ['labels']
        evaluator_time = time.time()
        all_res.append(res)
        #coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        evalprogress_bar.update(1)

    #for coco evaluation
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()

    coco_evaluator = CocoEvaluator(coco_ds, iou_types)
    for res in all_res:
        coco_evaluator.update(res)
        

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    #torch.set_num_threads(n_threads)
    #return coco_evaluator

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
def vis_example(onedetection, imgtensor, filename='result.jpg'):
    #labels = [names[i] for i in detections["labels"]] #classes[i]
    #img=im0.copy() #HWC (1080, 810, 3)
    #img_trans=im0[..., ::-1].transpose((2,0,1))  # BGR to RGB, HWC to CHW
    #imgtensor = torch.from_numpy(img_trans.copy()) #[3, 1080, 810]
    #pred_bbox_tensor=torchvision.ops.box_convert(torch.from_numpy(onedetection["boxes"]), 'xywh', 'xyxy')
    pred_bbox_tensor=torch.from_numpy(onedetection["boxes"])
    #pred_bbox_tensor=torch.from_numpy(onedetection["boxes"])
    print(pred_bbox_tensor)
    pred_labels = onedetection["labels"].astype(str).tolist()
    #img: Tensor of shape (C x H x W) and dtype uint8.
    #box: Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
    #labels: Optional[List[str]]
    imgtensor_uint=torchvision.transforms.functional.convert_image_dtype(imgtensor, torch.uint8)
    box = draw_bounding_boxes(imgtensor_uint, boxes=pred_bbox_tensor,
                            labels=pred_labels,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    # save a image using extension
    im = im.save(filename)

from DeepDataMiningLearning.detection import utils
from DeepDataMiningLearning.detection.dataset import get_dataset
from DeepDataMiningLearning.detection.models import create_detectionmodel
class args:
    data_path = '/data/cmpe249-fa23/coco/' #'/data/cmpe249-fa23/COCOoriginal/' # #'/data/cmpe249-fa23/WaymoCOCO/' #'/data/cmpe249-fa23/coco/'
    annotationfile = '/data/cmpe249-fa23/coco/train2017.txt'
    weights = None
    test_only = True
    backend = 'PIL' #tensor
    use_v2 = False
    dataset = 'yolo'#'coco'
if __name__ == "__main__":
    is_train =False
    is_val =True
    datasetname='yolo'#'coco' #'waymococo' #'yolo'
    dataset, num_classes=get_dataset(datasetname, is_train, is_val, args)
    print("train set len:", len(dataset))
    test_sampler = torch.utils.data.SequentialSampler(dataset) #RandomSampler(dataset)#torch.utils.data.SequentialSampler(dataset)
    new_collate_fn = utils.mycollate_fn #utils.mycollate_fn
    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=test_sampler, num_workers=1, collate_fn=new_collate_fn
    )
    # for batch in data_loader_test:
    #     print(batch.keys()) #['img', 'bboxes', 'cls', 'batch_idx']
    #     break
    # #batch=next(iter(data_loader_test))
    #print(batch.keys())

    device='cuda:0'
    model, preprocess, classes = create_detectionmodel('yolov8', num_classes=80, trainable_layers=0, ckpt_file='/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt', fp16=False, device= device)
    model.to(device)

    yoloevaluate(model, data_loader_test, preprocess, device)
    #simplemodelevaluate(model, data_loader_test, device)