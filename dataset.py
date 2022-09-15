import os
from PIL import Image
from torch.utils.data import Dataset
# import random

# random.seed(1)


class Clothes(Dataset):
    def __init__(self, data_dir, transform=None):
        """
            Clothes Dataset
            :param data_dir: str, 數據集所在路徑
            :param transform: torch.transform，數據預處理, 默認不進行處理
            self.data_info: (圖片路徑, 標籤)的列表(全部圖片), [(), (), ...]
        """
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 圖片需轉RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    # 返回所有圖片的路徑和標籤
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            if dirs:
                for sub_dir in dirs:
                    file_names = os.listdir(os.path.join(root, sub_dir))
                    img_names = list(filter(lambda x: x.endswith('.jpg'), file_names))

                    for img_name in img_names:
                        path_img = os.path.join(root, sub_dir, img_name)
                        label = sub_dir
                        data_info.append((path_img, int(label)))
        return data_info