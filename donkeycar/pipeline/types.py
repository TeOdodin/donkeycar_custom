import os
from typing import Any, List, Optional, TypeVar, Tuple

import numpy as np
from donkeycar.config import Config
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import load_image, load_pil_image, normalize_image, train_test_split, process_image
from typing_extensions import TypedDict

X = TypeVar('X', covariant=True)

TubRecordDict = TypedDict(
    'TubRecordDict',
    {
        'cam/image_array': str,
        'user/angle': float,
        'user/throttle': float,
        'user/mode': str,
        'imu/acl_x': Optional[float],
        'imu/acl_y': Optional[float],
        'imu/acl_z': Optional[float],
        'imu/gyr_x': Optional[float],
        'imu/gyr_y': Optional[float],
        'imu/gyr_z': Optional[float],
    }
)


class TubRecord(object):
    def __init__(self, config: Config, base_path: str,
                 underlyings: List[TubRecordDict]) -> None:
        self.config = config
        self.base_path = base_path
        self.underlyings = underlyings
        self._image: Optional[Any] = None

    def image(self, cached=True, as_nparray=True) -> np.ndarray:
        """Loads the image for you

        Args:
            cached (bool, optional): whether to cache the image. Defaults to True.
            as_nparray (bool, optional): whether to convert the image to a np array of uint8.
                                         Defaults to True. If false, returns result of Image.open()

        Returns:
            np.ndarray: [description]
        """
        if self._image is None:
            images = []
            for underlying in self.underlyings:
                image_path = underlying['cam/image_array']
                full_path = os.path.join(self.base_path, 'images', image_path)

                if as_nparray:
                    _image = load_image(full_path, cfg=self.config)
                else:
                    # If you just want the raw Image
                    _image = load_pil_image(full_path, cfg=self.config)
                _image = process_image(_image)
                images.append(_image)
            _image = np.stack((images[0],images[1],images[2],images[3]),axis=2)
            _image = _image.reshape(1, _image.shape[0], _image.shape[1], _image.shape[2]) #1*80*80*4  
            if cached:
                self._image = _image
        else:
            _image = self._image
        return _image
    def __repr__(self) -> str:
        return repr(self.underlyings)


class TubDataset(object):
    '''
    Loads the dataset, and creates a train/test split.
    '''

    def __init__(self, config: Config, tub_paths: List[str],
                 shuffle: bool = True) -> None:
        self.config = config
        self.tub_paths = tub_paths
        self.shuffle = shuffle
        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = list()

    def train_test_split(self) -> Tuple[List[TubRecord], List[TubRecord]]:
        print(f'Loading tubs from paths {self.tub_paths}')
        self.records.clear()
        img1 = None
        img2 = None
        img3 = None
        img4 = None
        for tub in self.tubs:
            for underlying in tub:
                if img3 == None:
                    img1 = underlying
                    img2 = underlying
                    img3 = underlying
                    img4 = underlying
                else:
                    img1 = img2
                    img2 = img3
                    img3 = img4
                    img4 = underlying
                record = TubRecord(self.config, tub.base_path,
                                   underlyings=[img1, img2, img3, img4])
                self.records.append(record)

        return train_test_split(self.records, shuffle=self.shuffle,
                                test_size=(1. - self.config.TRAIN_TEST_SPLIT))
