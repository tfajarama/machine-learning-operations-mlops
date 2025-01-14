"""Tuner module"""

# Define imports
from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import tensorflow_transform as tft

from loan_approval_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name
)

# Declare namedtuple field names
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def gzip_reader_fn(filenames):
    '''Load compressed dataset

    Args:
      filenames - filenames of TFRecords to load

    Returns:
      TFRecordDataset loaded from the filenames
    '''

    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records

    Args:
      file_pattern - List of files or patterns of file paths containing Example records.
      tf_transform_output - transform output graph
      num_epochs - Integer specifying the number of times to read through the dataset.
              If None, cycles through the dataset forever.
      batch_size - An int representing the number of records to combine in a single batch.

    Returns:
      A dataset of dict elements, (or a tuple of dict elements and label).
      Each dict maps feature keys to Tensor or SparseTensor objects.
    '''

    # Get feature specification based on transform output
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # Create batches of features and labels
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))

    return dataset


def model_builder(hp):
    '''
    Builds the model and sets up the hyperparameters to tune.

    Args:
      hp - Keras tuner object

    Returns:
      model with hyperparameters to tune
    '''

    # # Initialize the Sequential API and start stacking the layers
    # model = keras.Sequential()

    # one-hot categorical features and stack input layers
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            keras.Input(shape=(dim + 1,), name=transformed_name(key))
        # model.add(keras.Input(shape=(dim + 1,), name=transformed_name(key)))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            keras.Input(shape=(1,), name=transformed_name(feature))
        # model.add(keras.Input(shape=(1,), name=transformed_name(feature)))
        )

    # Concatenate the inputs
    concatenate = keras.layers.concatenate(input_features)

    # Tune the number of units in the hidden Dense layers
    # Choose an optimal value for each layer
    hp_units_1 = hp.Int('units_1', min_value=128, max_value=256, step=128)
    # model.add(keras.layers.Dense(units=hp_units_1, activation='relu', name='dense_1'))
    deep = keras.layers.Dense(units=hp_units_1, activation='relu', name='dense_1')(concatenate)

    hp_units_2 = hp.Int('units_2', min_value=64, max_value=128, step=64)
    # model.add(keras.layers.Dense(units=hp_units_2, activation='relu', name='dense_2'))
    deep = keras.layers.Dense(units=hp_units_2, activation='relu', name='dense_2')(deep)

    hp_units_3 = hp.Int('units_3', min_value=32, max_value=64, step=32)
    # model.add(keras.layers.Dense(units=hp_units_3, activation='relu', name='dense_3'))
    deep = keras.layers.Dense(units=hp_units_3, activation='relu', name='dense_3')(deep)

    # Add output layer
    # model.add(keras.layers.Dense(1, activation='sigmoid'))
    outputs = keras.layers.Dense(1, activation='sigmoid')(deep)

    # Build model
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
      fn_args: Holds args as name/value pairs.

        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.

    Returns:
      A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                      model , e.g., the training and validation dataset. Required
                      args depend on the above tuner's implementation.
    """

    # Define tuner search strategy
    tuner = kt.Hyperband(model_builder,
                         objective='val_binary_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=fn_args.working_dir,
                         project_name='loan_approval_tuning')
    # tuner = kt.RandomSearch(model_builder,
    #                      objective='val_binary_accuracy',
    #                      max_epochs=10,
    #                      factor=3,
    #                      directory=fn_args.working_dir,
    #                      project_name='loan_approval_tuning')

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Use input_fn() to extract input features and labels from the train and val set
    train_set = input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
