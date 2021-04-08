"""

keras.py

Methods to create, use, save and load pilots. Pilots contain the highlevel
logic used to determine the angle and throttle of a vehicle. Pilots can
include one or more models to help direct the vehicles motion.

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import donkeycar as dk
from donkeycar.utils import normalize_image, linear_bin, process_image, linear_unbin
from donkeycar.pipeline.types import TubRecord

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2DTranspose
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Optimizer

ONE_BYTE_SCALE = 1.0 / 255.0

# type of x
XY = Union[float, np.ndarray, Tuple[float, ...], Tuple[np.ndarray, ...]]


class KerasPilot(ABC):
    """
    Base class for Keras models that will provide steering and throttle to
    guide a car.
    """
    def __init__(self) -> None:
        self.model: Optional[Model] = None
        self.optimizer = "adam"
        self.s_t = None
        print(f'Created {self}')

    def load(self, model_path: str) -> None:
        self.model = keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        assert self.model, 'Model not set'
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        assert self.model, 'Model not set'
        if optimizer_type == "adam":
            self.model.optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            self.model.optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            self.model.optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)

    def get_input_shape(self) -> tf.TensorShape:
        assert self.model, 'Model not set'
        return self.model.inputs[0].shape

    def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        norm_arr = normalize_image(img_arr)
        x_t = process_image(norm_arr)

        if not self.s_t:
            self.s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
            # In Keras, need to reshape
            self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2]) #1*80*80*4
        else:
            x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1) #1x80x80x1
            self.s_t = np.append(x_t, self.s_t[:, :, :, :3], axis=3) #1x80x80x4
            
        return self.inference(self.s_t, other_arr), 0.5

    @abstractmethod
    def inference(self, img_arr: np.ndarray, other_arr: np.ndarray) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Virtual method to be implemented by child classes for inferencing

        :param img_arr:     float32 [0,1] numpy array with normalized image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        pass

    def train(self,
              model_path: str,
              train_data: 'BatchSequence',
              train_steps: int,
              batch_size: int,
              validation_data: 'BatchSequence',
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5) -> tf.keras.callbacks.History:
        """
        trains the model
        """
        model = self._get_train_model()
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]

        history: Dict[str, Any] = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False
        )
        return history

    def _get_train_model(self) -> Model:
        """ Model used for training, could be just a sub part of the model"""
        return self.model

    def x_transform(self, record: TubRecord) -> XY:
        img_arr = record.image(cached=True)
        return img_arr

    def y_transform(self, record: TubRecord) -> XY:
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        return {'img_in': x}

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self) -> Dict[str, np.typename]:
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self):
        """ Used in tf.data, assume all types are doubles"""
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self) -> Optional[Dict[str, tf.TensorShape]]:
        return None

    def __str__(self) -> str:
        """ For printing model initialisation """
        return type(self).__name__


class KerasLinear(KerasPilot):
    """
    The KerasLinear pilot uses one neuron to output a continous value via the
    Keras Dense layer with linear activation. One each for steering and
    throttle. The output is not bounded.
    """
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3)):
        super().__init__()
        img_rows , img_cols = 80, 80
        self.model = Sequential()
        self.model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_rows,img_cols,4), name='conv_1'))  #80*80*4
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', name='conv_2'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_3'))
        self.model.add(Activation('relu'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(512, name='dense_1'))
        self.model.add(Activation('relu'))

        # 15 categorical bins for steering angles
        self.model.add(Dense(15, activation="linear", name="angle"))
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def inference(self, img_arr, other_arr):
        # print("get action")
        #print("return max q prediction")
        q_value = self.model.predict(img_arr)
        # convert q array to steering value
        return linear_unbin(q_value[0]), 0.7

    def x_transform(self, record: TubRecord) -> XY:
        img_arr = super().x_transform(record)
        # we need to return the image data first
        return img_arr

    def y_transform(self, record: TubRecord):
        angle: float = record.underlyings[0]['user/angle']
        # throttle: float = record.underlyings[0]['user/throttle']
        return angle

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        # if isinstance(y, tuple):
        angle = y
        return {'angle': angle}
        # else:
        #     raise TypeError('Expected tuple')

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle': tf.TensorShape([])})
        return shapes