import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import py_sod_metrics
import cv2


class EvaluationMetricsV2:
    def __init__(self):
        self.SM = py_sod_metrics.Smeasure()
        self.EM = py_sod_metrics.Emeasure()
        self.FM = py_sod_metrics.Fmeasure()
        self.WFM = py_sod_metrics.WeightedFmeasure()
        self.MAE = py_sod_metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255]
        gt: [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        sm = self.SM.get_results()["sm"]
        # mean E-measure and E-measure Curve
        _em = self.EM.get_results()["em"]
        em_curve = np.flip(_em["curve"])
        emMean = _em['curve'].mean()
        emMax = _em['curve'].max()
        emAdp = _em['adp']
        # F-measure curve and PR-curve
        _fm = self.FM.get_results()
        # F-measure
        fm = _fm["fm"]
        fmMean = fm["curve"].mean()
        fmMax = fm["curve"].max()
        fmAdp = fm["adp"]
        fm_curve = np.flip(fm["curve"])
        pr = _fm["pr"]
        p = np.flip(pr["p"])
        r = np.flip(pr["r"])
        # weighted F-measure
        wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            'sm': sm,

            'emMean': emMean,
            'emAdp': emAdp,
            'emMax': emMax,
            'em_curve': em_curve,

            'fmMean': fmMean,
            'fmMax': fmMax,
            'fmAdp': fmAdp,
            'fm_curve': fm_curve,

            'wfm': wfm,
            'mae': mae,

            'p': p,
            'r': r
        }


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=4))


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_boxes_on_image_and_save(raw_image, boxes, save_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, dpi=300)
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image_and_save(raw_image, boxes, input_points, input_labels=None, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_multi_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')


def show_points(coords, labels, ax, marker_size=25):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', s=marker_size,
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
               linewidth=1.25)


def show_multi_points(multi_coords, multi_labels, ax, marker_size=50):
    for coords, labels in zip(multi_coords, multi_labels):
        show_points(coords, labels, ax, marker_size=marker_size)


def show_masks_on_image(raw_image, masks, scores, save_vis=False, vis_path=None, show=True):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i + 1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    if show:
        plt.show()
    if save_vis:
        plt.savefig(vis_path)
    plt.close()


def save_evaluation(mask, out_dir, gt_path, metric, save_mask=False):
    '''
    :param mask: pytorch bool tensor
    :return: None
    '''
    pred = mask.numpy().astype(np.uint8)
    with open(gt_path, 'rb') as f:
        gt = np.array(Image.open(f).convert('L'))
    metric.step(pred=pred, gt=gt)
    if save_mask:
        pred = Image.fromarray(pred * 255, mode='L')
        pred.save(out_dir)


def save_evaluation_sam(mask, out_dir, gt_path, metric, save_mask=False):
    '''
    :param mask: pytorch bool tensor
    :return: None
    '''
    with open(gt_path, 'rb') as f:
        gt = np.array(Image.open(f).convert('L'))
    metric.step(pred=mask, gt=gt)
    if save_mask:
        pred = Image.fromarray(mask * 255, mode='L')
        pred.save(out_dir)


def save_evaluation_hqsam(mask, out_dir, gt_path, metric, save_mask=False):
    '''
    :param mask: pytorch bool tensor
    :return: None
    '''
    with open(gt_path, 'rb') as f:
        gt = np.array(Image.open(f).convert('L'))
    metric.step(pred=mask, gt=gt)
    if save_mask:
        pred = Image.fromarray(mask * 255, mode='L')
        pred.save(out_dir)


def save_evaluation_hqsam_superpixel(mask, out_dir, gt_path, metric, save_mask=False, resize_tuple=None):
    '''
    :param mask: pytorch bool tensor
    :return: None
    '''
    with open(gt_path, 'rb') as f:
        img = Image.open(f).convert('L')
        if resize_tuple != img.size:
            img = img.resize(resize_tuple)
        gt = np.array(img)
    metric.step(pred=mask, gt=gt)
    if save_mask:
        pred = Image.fromarray(mask * 255, mode='L')
        pred.save(out_dir)


def save_mask(mask, out_dir):
    pred = mask[0].astype(np.uint8)
    if save_mask:
        pred = Image.fromarray(pred * 255, mode='L')
        pred.save(out_dir)


def save_evaluation_logit_hqsam(mask, out_dir, gt_path, metric, save_mask=False):
    '''
    :param mask: pytorch bool tensor
    :return: None
    '''
    pred = 1. / (1 + np.exp(-mask[0]))
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    with open(gt_path, 'rb') as f:
        gt = np.array(Image.open(f).convert('L'))
    metric.step(pred=pred, gt=gt)
    if save_mask:
        # pred = Image.fromarray((pred * 255).astype('np.uint8'), mode='L')
        # pred.save(out_dir)
        cv2.imwrite(out_dir, pred * 255)


def save_evaluation_logit(pred, out_dir, gt_path, metric, save_mask=False):
    '''
    :param mask: pytorch bool tensor
    :return: None
    '''
    with open(gt_path, 'rb') as f:
        gt = np.array(Image.open(f).convert('L'))
    metric.step(pred=pred, gt=gt)
    if save_mask:
        # pred = Image.fromarray((pred * 255).astype('np.uint8'), mode='L')
        # pred.save(out_dir)
        cv2.imwrite(out_dir, pred * 255)


# from GenPromp/datasets/evaluation/cam.py line:28
# github.com/callsys/GenPromp

def get_bboxes(cam, cam_thr=None):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
        # sorted(contours, key=cv2.contourArea, reverse=True)
        # areas = []
        # for contour in contours:
        #     areas.append(cv2.contourArea(contour))
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox  # len(contours), sorted(areas, reverse=True)  # , thr_gray_heatmap, len(contours)


def get_multi_bboxes(cam, cam_thr=0.3, value_thr_in_bbx=0.9):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    boxes = []
    if len(contours) != 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            max_in_contour = np.array([cam[index] for index in get_all_index_from_contour(contour, cam)]).max()
            # print(max_in_contour)
            if max_in_contour >= value_thr_in_bbx * np.max(cam) and area != 0:
                x, y, w, h = cv2.boundingRect(contour)
                estimated_bbox = [x, y, x + w, y + h]
                boxes.append(estimated_bbox)
    else:
        estimated_bbox = [0, 0, 1, 1]
        boxes.append(estimated_bbox)

    return boxes


def get_multi_bboxes_area_thr(cam, cam_thr=None, value_thr_in_bbx=None, area_thr=None):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    boxes = []
    if len(contours) != 0:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(sorted_contours[0])
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            max_in_contour = np.array([cam[index] for index in get_all_index_from_contour(contour, cam)]).max()
            # print(max_in_contour)
            if max_in_contour >= value_thr_in_bbx * np.max(cam) and area > area_thr * max_area:
                x, y, w, h = cv2.boundingRect(contour)
                estimated_bbox = [x, y, x + w, y + h]
                boxes.append(estimated_bbox)
    else:
        estimated_bbox = [0, 0, 1, 1]
        boxes.append(estimated_bbox)

    return boxes


def get_one_bbox(cam, cam_thr=None):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
        # sorted(contours, key=cv2.contourArea, reverse=True)
        # areas = []
        # for contour in contours:
        #     areas.append(cv2.contourArea(contour))
    else:
        estimated_bbox = [0, 0, 1, 1]

    return [estimated_bbox]  # len(contours), sorted(areas, reverse=True)  # , thr_gray_heatmap, len(contours)


def get_limited_bboxes(cam, cam_thr=None, value_thr_in_bbx=None, bbx_max_num=50):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    # thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    boxes = []
    if len(contours) != 0:
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:bbx_max_num]:
            area = cv2.contourArea(contour)
            max_in_contour = np.array([cam[index] for index in get_all_index_from_contour(contour, cam)]).max()
            # print(max_in_contour)
            if max_in_contour >= value_thr_in_bbx * np.max(cam) and area != 0:
                x, y, w, h = cv2.boundingRect(contour)
                estimated_bbox = [x, y, x + w, y + h]
                boxes.append(estimated_bbox)
    else:
        estimated_bbox = [0, 0, 1, 1]
        boxes.append(estimated_bbox)
    if len(boxes) == 0:
        boxes.append([0, 0, 1, 1])

    return boxes


def get_all_index_from_contour(contour, cam):
    mask = np.zeros_like(cam)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    contour_pixels = cv2.findNonZero(mask).tolist()
    return [(x[0][1], x[0][0]) for x in contour_pixels]


def get_points_labels_from_bbx(bbxes, attn, sample_point=5):
    '''
    :param bbx: [[x,y,w,h],...,[x,y,w,h]] B x 4
    :return:
    points: B x N x 2
    labels: B x N
    '''
    points = []
    labels = []
    if bbxes == [[0, 0, 1, 1]]:
        return None, None
    if sample_point <= 0:
        return None, None
    for bbx in bbxes:
        x, y, x1, y2 = bbx
        w = x1 - x
        h = y2 - y
        bbx_region = attn[y:y + h, x:x + w]

        sorted_indice = np.argsort(bbx_region.flatten())
        fore_indice = sorted_indice[-sample_point:]
        back_indice = sorted_indice[:sample_point]
        fore_points = np.unravel_index(fore_indice, bbx_region.shape)
        back_points = np.unravel_index(back_indice, bbx_region.shape)

        # assert len(back_points[0]) == random_sample_point, 'deterministic sampling!'

        input_points_fore = [[a + x, b + y] for a, b in zip(fore_points[1], fore_points[0])]
        input_points_back = [[a + x, b + y] for a, b in zip(back_points[1], back_points[0])]
        # prevent the num of points in the box < sample_points
        if len(input_points_fore) < sample_point:
            input_points_fore.extend([input_points_fore[0]] * (sample_point - len(input_points_fore)))
            input_points_back.extend([input_points_back[0]] * (sample_point - len(input_points_back)))
        input_points = input_points_fore + input_points_back
        points.append(input_points)
        label = [1] * sample_point + [0] * sample_point
        labels.append(label)

    return points, labels


def get_points_labels_from_bbx_without_background_points(bbxes, attn, sample_point=5):
    '''
    :param bbx: [[x,y,w,h],...,[x,y,w,h]] B x 4
    :return:
    points: B x N x 2
    labels: B x N
    '''
    points = []
    labels = []
    if bbxes == [[0, 0, 1, 1]]:
        return None, None
    for bbx in bbxes:
        x, y, x1, y2 = bbx
        w = x1 - x
        h = y2 - y
        bbx_region = attn[y:y + h, x:x + w]

        sorted_indice = np.argsort(bbx_region.flatten())
        fore_indice = sorted_indice[-sample_point:]
        fore_points = np.unravel_index(fore_indice, bbx_region.shape)

        # assert len(back_points[0]) == random_sample_point, 'deterministic sampling!'

        input_points_fore = [[a + x, b + y] for a, b in zip(fore_points[1], fore_points[0])]
        # prevent the num of points in the box < sample_points
        if len(input_points_fore) < sample_point:
            input_points_fore.extend([input_points_fore[0]] * (sample_point - len(input_points_fore)))
        input_points = input_points_fore
        points.append(input_points)
        label = [1] * sample_point
        labels.append(label)

    return points, labels


def save_attn_with_bbx_points(attn_img, save_path):
    bbox = get_multi_bboxes(attn_img, cam_thr=0.3, value_thr_in_bbx=0.9)
    points, lables = get_points_labels_from_bbx(bbox, attn_img, sample_point=5)
    show_points_and_boxes_on_image_and_save(attn_img, bbox, points, lables, save_path=save_path)
