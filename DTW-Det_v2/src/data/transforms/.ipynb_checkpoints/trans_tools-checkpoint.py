import torch

class GaussGTDistributionGenerator:
    """
    高斯GT分布生成器
    """
    def __init__(self, img_size=(640, 640), sigma_ratio=1.2, overlay=False, gauss_norm = False):
        self.sigma_ratio = sigma_ratio
        self.overlay = overlay
        self.img_size = img_size
        self.gauss_norm = gauss_norm

    def __call__(self, bboxes):
        H, W = self.img_size
        # 初始化高斯分布图
        gauss_dist_map = torch.zeros((H, W), dtype=torch.float32)

        for box in bboxes:
            x_center, y_center, width, height = box  # 归一化的[cx,xy,w,h]
            x_center_px = int(x_center * W)
            y_center_px = int(y_center * H)
            w_px = max(int(width * W), 1)
            h_px = max(int(height * H), 1)

            sigma_x = max(w_px * self.sigma_ratio / 2, 1.0)  # 高斯核x方向标准差
            sigma_y = max(h_px * self.sigma_ratio / 2, 1.0)  # 高斯核y方向标准差
            # 生成高斯核
            gauss_kernel = self._gaussian_kernel(sigma_x, sigma_y, norm=self.gauss_norm)
            self._paste_kernel(gauss_dist_map, gauss_kernel, x_center_px, y_center_px, overlay=self.overlay)

        if gauss_dist_map.max() > 0:
            gauss_dist_map = gauss_dist_map / gauss_dist_map.max()
        return gauss_dist_map.unsqueeze(0)

    def _gaussian_kernel(self, sigma_x, sigma_y, norm):
        """生成高斯核"""
        kernel_w = int(6 * sigma_x) + 1
        kernel_h = int(6 * sigma_y) + 1
        # 确保为奇数存在中心点
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
        # 生成网格
        x = torch.arange(kernel_w, dtype=torch.float32) - (kernel_w // 2)
        y = torch.arange(kernel_h, dtype=torch.float32) - (kernel_h // 2)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        # 计算高斯核
        kernel = torch.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))

        if norm:
            # 归一化(总能量归一化)
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
        return kernel

    def _paste_kernel(self, canvas, kernel, x_c, y_c, overlay):
        """粘贴核，处理边界裁剪问题"""
        if kernel.numel() == 0:
            return

        H, W = canvas.shape
        k_h, k_w = kernel.shape
        r_x = k_w // 2
        r_y = k_h // 2
        # 画布上的坐标
        x1 = max(x_c - r_x, 0)
        y1 = max(y_c - r_y, 0)
        x2 = min(x_c + r_x + 1, W)
        y2 = min(y_c + r_y + 1, H)
        # 核上的坐标
        k_x1 = max(r_x - (x_c - x1), 0)
        k_y1 = max(r_y - (y_c - y1), 0)
        k_x2 = k_w - max((x_c + r_x + 1) - x2, 0)
        k_y2 = k_h - max((y_c + r_y + 1) - y2, 0)

        patch = kernel[k_y1:k_y2, k_x1:k_x2]
        # 尺寸匹配校验
        ph, pw = y2 - y1, x2 - x1
        if patch.shape != (ph, pw):
            patch = patch[:ph, :pw]
        # 叠加高斯核
        if overlay:
            canvas[y1:y2, x1:x2] += patch
        else:
            canvas[y1:y2, x1:x2] = torch.max(canvas[y1:y2, x1:x2], patch)