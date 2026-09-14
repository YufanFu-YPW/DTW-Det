import torch 
import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    images = torch.cat([x[0][None] for x in items], dim=0)
    targets = [x[1] for x in items]
    # 如果 targets 中包含我们生成的 gauss_map，将其抽取并单独堆叠
    if targets and 'gt_distribution_map' in targets[0]:
        # 从每个 target 字典中弹出 gauss_map，并沿 Batch 维度拼接
        # 得到张量形状为 [B, 1, H, W]
        gauss_gt = torch.stack([t.pop('gt_distribution_map') for t in targets], dim=0)
        # 将分离好的 gauss_maps 作为第三个返回值输出
        return images, targets, gauss_gt
    return images, targets
    # return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]
