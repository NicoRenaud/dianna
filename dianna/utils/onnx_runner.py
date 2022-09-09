import onnxruntime as ort


class SimpleModelRunner:
    """Runs an onnx model with a set of inputs and outputs."""
    def __init__(self, filename, preprocess_function=None, multiple_inputs=False):
        """
        Generates function to run ONNX model with one set of inputs and outputs.

        Args:
            filename (str): Path to ONNX model on disk
            preprocess_function (callable, optional): Function to preprocess input data with

        Returns:
            function

        Examples:
            >>> runner = SimpleModelRunner('path_to_model.onnx')
            >>> predictions = runner(input_data)
        """
        self.filename = filename
        self.preprocess_function = preprocess_function
        self.multiple_inputs = multiple_inputs

    def __call__(self, input_data):
        # get ONNX predictions
        sess = ort.InferenceSession(self.filename)
        output_name = sess.get_outputs()[0].name

        if self.preprocess_function is not None:
            input_data = self.preprocess_function(input_data)

        if self.multiple_inputs:
            
            onnx_input = {}
            input_names = [input.name for input in sess.get_inputs()]
            input_data.reverse()
            for idx, input_name in enumerate(input_names):
                onnx_input[input_name] = input_data[idx].astype('float32')

        else:
            
            input_name = sess.get_inputs()[0].name
            onnx_input = {input_name: input_data}

        pred_onnx = sess.run([output_name], onnx_input)[0]
        return pred_onnx
