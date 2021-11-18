"""Performance measure code (get_intersection_and_union) adapted from https://github.com/dvlab-research/PFENet.git
"""
import os

import torch
import torch.nn.functional as F
import torchvision as tv

from fss.utils.recursive_functions import recursive_detach, recursive_to
from fss.utils.debugging import print_tensor_statistics, revert_imagenet_normalization, COLOR_RED, COLOR_WHITE

# These are used to not feed in annotations into model
OBSERVABLE_DATA_KEYS = {
    'support_images',
    'support_segmentations',
    'query_images',
    'support_classes',
    'query_classes',
    'identifier'
}

        
class FSSEvaluator:
    def __init__(self, visualization_path, device):
        self._visualization_path = visualization_path
        self._device = device
        self._issued_warnings = set()

    def _force_model_mode(self, model):
        old_model_mode = {
            'training': model.training,
            'calculate_losses': model.calculate_losses,
        }
        model.eval()
        model.calculate_losses = False
        return old_model_mode
    def _restore_model_mode(self, model, old_model_mode):
        model.train(old_model_mode['training'])
        model.calculate_losses = old_model_mode['calculate_losses']

    def _get_visualization(self, images, segmentations):
        B, N, H, W = segmentations.size()
        background = (segmentations == 0).cpu().detach().float().view(B, N, 1, H, W)
        target = (segmentations == 1).cpu().detach().float().view(B, N, 1, H, W)
        ignore = (segmentations == 255).cpu().detach().float().view(B, N, 1, H, W)
        visualization = (background * images
                         + target * (0.5 * images + 0.5 * COLOR_RED)
                         + ignore * COLOR_WHITE)
        visualization = (visualization * 255).byte()
        return visualization
            
    def _visualize_episode(self, model_output, data):
        query_images = revert_imagenet_normalization(data['query_images'].cpu().detach())
        support_images = revert_imagenet_normalization(data['support_images'].cpu().detach())
        qq = data['original_query_segmentations']
        support_anno_vis = self._get_visualization(support_images, data['support_segmentations'])
        query_pred_vis = self._get_visualization(query_images, model_output['query_segmentations'])
        query_anno_vis = self._get_visualization(query_images, data['query_segmentations'])
        B, Q, _, H, W = query_images.size()
        _, S, _, _, _ = support_images.size()

        if not os.path.exists(self._visualization_path):
            os.makedirs(self._visualization_path)
        for b in range(B):
            for s in range(S):
                fpath = os.path.join(self._visualization_path, f"eval_b{b}_supp_n{s}_anno.png")
                tv.io.write_png(support_anno_vis[b, s], fpath)
            for q in range(Q):
                fpath = os.path.join(self._visualization_path, f"eval_b{b}_query_n{q}_pred.png")
                tv.io.write_png(query_pred_vis[b, q], fpath)
                fpath = os.path.join(self._visualization_path, f"eval_b{b}_query_n{q}_anno.png")
                tv.io.write_png(query_anno_vis[b, q], fpath)
        
    def _get_single_intersection_and_union(self, pred, anno, C):
        """Note that these calculations would in theory deal with multi-class examples. However,
        we have primarily worked with a single class at a time. C is therefore always 2 in this
        function. The caller will then pick out the element at index 1 in the outputs.
        Args:
            pred (torch.LongTensor [H, W])
            anno (torch.LongTensor [H, W])
        Returns:
            torch.LongTensor [C]
            torch.LongTensor [C]
        """
        pred = pred.reshape(-1)
        anno = anno.reshape(-1)
        pred[anno == 255] = 255
        true_and_positive = torch.histc(pred[pred == anno], bins=C, min=0, max=C-1)
        positive          = torch.histc(pred, bins=C, min=0, max=C-1)
        true              = torch.histc(anno, bins=C, min=0, max=C-1)
        true_or_positive  = positive + true - true_and_positive
        return true_and_positive, true_or_positive
    
    def _get_intersection_and_union(self, preds, annos):
        intersections = []
        unions = []
        B, Q, C, H1, W1 = preds['query_segscores'].size()
#        print("\n----")
        for b in range(B):
            assert Q == 1
            q = 0
            pred = preds['query_segscores'][b, q]
            anno = annos['original_query_segmentations'][b, q]
            og_size = annos['original_query_sizes'][b, q].tolist()
            pred = F.interpolate(pred[None], size=tuple(og_size), mode='bilinear')[0].argmax(dim=0)
            anno = anno[:og_size[0],:og_size[1]]
            intersection, union = self._get_single_intersection_and_union(pred, anno, C)
            intersection = intersection[1] # We have a single class in channel 1
            union = union[1]               # We have a single class in channel 1
            intersections.append(intersection)
            unions.append(union)
#            print(f"b{b} intersection {intersection:.3f} union {union:.3f}")
        return intersections, unions

    def _get_class_ids(self, anno):
        """We assume that all query examples have the same class, and that there is only a single class
        """
        class_ids = anno['query_classes'][:, 0, 0].tolist()
        return class_ids
    
    def _init_kpi(self, class_ids):
        return {idx: {'intersection': 0, 'union': 0} for idx in class_ids}
    def _update_kpi(self, kpi, pred, anno):
        intersections, unions = self._get_intersection_and_union(pred, anno)
        class_ids = self._get_class_ids(anno)
        for intersection, union, class_idx in zip(intersections, unions, class_ids):
            kpi[class_idx]['intersection'] += intersection
            kpi[class_idx]['union'] += union
        return kpi
    
    def _print_kpi(self, kpi, episode):
        iou = [(elem['intersection'] + 1e-5) / (elem['union'] + 1e-5) for elem in kpi.values()]
        mean_iou = sum(iou) / len(iou)
        print(f"IoU after {episode} episodes: {mean_iou}")        
    def _aggregate_kpi(self, kpi, evaluation_identifier):
        iou = [(elem['intersection'] + 1e-5) / (elem['union'] + 1e-5) for elem in kpi.values()]
        for elem in iou:
            print(f"Class evaluated with IoU: {elem}")
        mean_iou = sum(iou) / len(iou)
        print(f"{evaluation_identifier} aggregated over all classes. IoU: {mean_iou}")
        return {'IoU': mean_iou}

    def _evaluate_episode(self, model, data, kpi, num_classes, visualize=False):
        data = recursive_to(data, self._device)
        data_for_model = {key: val for key, val in data.items() if key in OBSERVABLE_DATA_KEYS}
        model_out, _, _ = model(data_for_model)
        if visualize and self._visualization_path is not None:
            self._visualize_episode(model_out, data)
        kpi = self._update_kpi(kpi, model_out, data)
        return kpi

    def evaluate(self, model, dataloader, evaluation_identifier=""):
        old_model_mode = self._force_model_mode(model)
        class_ids = dataloader.dataset.get_classes()
        kpi = self._init_kpi(class_ids)
        with torch.no_grad():
            for episode, data in enumerate(dataloader):
                kpi = self._evaluate_episode(model, data, kpi, class_ids, visualize=(episode == 0))
                
                if (episode + 1) % 10 == 0:
                    self._print_kpi(kpi, episode + 1)
            kpi = self._aggregate_kpi(kpi, evaluation_identifier)

        self._restore_model_mode(model, old_model_mode)

