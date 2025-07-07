

"""
General checks to simplify jupyter notebooks and GUI
"""

def check_packages():
    """ Wrapper for check_yaml and check_keras_version """
    check_yaml()
    check_keras_version()


def check_yaml():
    """ Check if ruamel.yaml is installed, otherwise notify user with instructions """

    try:
        import ruamel.yaml
        yaml = ruamel.yaml.YAML(typ='rt')
    except ModuleNotFoundError:
        print('\nModuleNotFoundError: The package "ruamel.yaml" does not seem to be installed on this PC.',
              'This package is necessary to load the configuration files of the models.\n',
              'Please install it with "pip install ruamel.yaml"')
        return

    print('\tYAML reader installed (version {}).'.format(ruamel.yaml.__version__))

def check_keras_version():
    """ Import tensorflow and check tf.keras version """
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        print('ModuleNotFoundError: The package "tensorflow" does not seem to be installed on this PC.',
              'Please install tensorflow with "pip install tensorflow".')
        return

    print('\ttensorflow installed (version {}).'.format(tf.__version__))
    print('\ttf.keras API version: {}'.format(tf.keras.__version__))

    # Optionally, check for minimum version
    min_tf_version = (2, 8)
    tf_version_tuple = tuple(map(int, tf.__version__.split(".")[:2]))
    if tf_version_tuple < min_tf_version:
        print(f'Warning: TensorFlow version >= {min_tf_version[0]}.{min_tf_version[1]} is recommended for best compatibility.')
