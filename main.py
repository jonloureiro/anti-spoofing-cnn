################################################################################
# git: jonloureiro/anti-spoofing-cnn
################################################################################

import sys

from tensorflow import keras

import pandas as pd

# MODEL ########################################################################

def modelVGG(activation):
    base = keras.applications.VGG16(
        weights     = 'imagenet',
        include_top = False,
        input_shape = (203, 203, 3)
    )

    for layer in base.layers:
        layer.trainable = False

    x = keras.layers.Flatten()(base.output)
    x = keras.layers.Dense(1024)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1)(x)
    x = keras.layers.Activation(activation)(x)

    return keras.Model(
        inputs  = base.input,
        outputs = x
    )

def modelDefault(activation):
    return keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape = (203, 203, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size = (2, 2)),
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size = (2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1),
        keras.layers.Activation(activation)
    ])

def model(base, activation):
    if base == 'vgg':
        return modelVGG(activation)
    else:
        return modelDefault(activation)
        


# GENERATORS ###################################################################

def generator(mode):
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    return datagen.flow_from_dataframe(
        dataframe   = pd.read_csv(mode + '.csv', sep=','),
        directory   = './Detectedface',
        x_col       = 'id',
        y_col       = 'label',
        class_mode  = 'other',
        target_size = (203, 203),
        batch_size  = 2
    )

# ARG ##########################################################################

def catchArg():
    hasArg = len(sys.argv) > 1 and sys.argv[1] == '-vgg'
    return 'vgg' if hasArg else 'default'

# MAIN #########################################################################

if __name__ == '__main__':
    arg = catchArg()
    activation = 'softmax'

    print('\n\n')

    train_generator = generator('train')
    test_generator = generator('test')

    print('\n\n')

    model = model(arg, activation)

    print('\n\n')

    model.summary()

    print('\n\n')

    model.compile(
        optimizer = 'Adam',
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )

    model.fit_generator(
        generator        = train_generator,
        steps_per_epoch  = len(train_generator),
        epochs           = 2,
        validation_data  = test_generator,
        validation_steps = len(test_generator)
    )

    print('\n\n')

    model.save_weights(arg + '_' + activation + '_weights.h5')