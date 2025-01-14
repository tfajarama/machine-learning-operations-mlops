"""Training module
"""

import os

from tensorflow import keras
from keras.utils.vis_utils import plot_model
# from typing import NamedTuple, Dict, Text, Any, List
# from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

from loan_approval_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)


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
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))

    return dataset


# def model_builder(hp):
def model_builder():
    '''
    Builds the model and sets up the hyperparameters.

    Args:
      hp - Keras tuner object

    Returns:
      model with hyperparameters
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

    # # Get the number of units from the Tuner results
    # hp_units_1 = hp.get('units_1')
    # model.add(keras.layers.Dense(units=hp_units_1, activation='relu'))
    # deep = keras.layers.Dense(units=hp_units_1, activation='relu')(concatenate)
    deep = keras.layers.Dense(units=128, activation='relu')(concatenate)

    # hp_units_2 = hp.get('units_2')
    # model.add(keras.layers.Dense(units=hp_units_2, activation='relu'))
    # deep = keras.layers.Dense(units=hp_units_2, activation='relu')(deep)
    deep = keras.layers.Dense(units=64, activation='relu')(deep)

    # hp_units_3 = hp.get('units_3')
    # model.add(keras.layers.Dense(units=hp_units_3, activation='relu'))
    # deep = keras.layers.Dense(units=hp_units_3, activation='relu')(deep)
    deep = keras.layers.Dense(units=32, activation='relu')(deep)

    # Add output layer
    # model.add(keras.layers.Dense(1, activation='sigmoid'))
    outputs = keras.layers.Dense(1, activation='sigmoid')(deep)

    # Build model
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    # # Get the learning rate from the Tuner results
    # hp_learning_rate = hp.get('learning_rate')

    # Setup model for training
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Print the model summary
    model.summary()

    return model


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


# def run_fn(fn_args: FnArgs) -> None:
#     """Defines and trains the model.
#     Args:
#       fn_args: Holds args as name/value pairs. Refer here for the complete attributes:
#       https://www.tensorflow.org/tfx/api_docs/python
#       /tfx/components/trainer/fn_args_utils/FnArgs#attributes
#     """
#
#     # Callback for TensorBoard
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=fn_args.model_run_dir, update_freq='batch')
#
#     # Load transform output
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
#
#     # Create batches of data good for 10 epochs
#     train_set = input_fn(fn_args.train_files[0], tf_transform_output, 10)
#     val_set = input_fn(fn_args.eval_files[0], tf_transform_output, 10)
#
#     # Load best hyperparameters
#     hp = fn_args.hyperparameters.get('values')
#
#     # Build the model
#     model = model_builder(hp)
#
#     # Train the model
#     model.fit(
#         x=train_set,
#         validation_data=val_set,
#         callbacks=[tensorboard_callback]
#     )
# TFX Trainer will call this function.
def run_fn(fn_args):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 64)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = model_builder()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
