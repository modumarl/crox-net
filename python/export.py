
import logging


import os
from docopt import docopt

from unitytrainers.trainer_controller import TrainerController

if __name__ == '__main__':
    logger = logging.getLogger("unityagents")
    _USAGE = '''
    Usage:
      export (<savePath>) <outFile>)
      export --help

    '''
    options = docopt(_USAGE)
    logger.info(options)

    ckpt = tf.train.get_checkpoint_state(savePath)
    freeze_graph.freeze_graph(input_graph=savePath + '/raw_graph_def.pb',
                            input_binary=True,
                            input_checkpoint=ckpt.model_checkpoint_path,
                            output_node_names=target_nodes,
                            output_graph=outFile,
                            clear_devices=True, initializer_nodes="", input_saver="",
                            restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")



