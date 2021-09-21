import numpy as np
import onnxruntime as ort


class SimpleModelRunner:
    def __init__(self, filename):
        """Generates function to run ONNX model with one set of inputs and outputs

            Args:
                filename (str): Path to ONNX model on disk
            Returns:
                function

            Examples:
                >>> runner = SimpleModelRunner('path_to_model.onnx')
                >>> predictions = runner(input_data)
        """
        self.filename = filename

    def __call__(self, input_data):
        # get ONNX predictions
        sess = ort.InferenceSession(self.filename)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        onnx_input = {input_name: input_data.astype(np.float32)}
        pred_onnx = sess.run([output_name], onnx_input)[0]
        return pred_onnx