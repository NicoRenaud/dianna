import onnxruntime as ort


class SimpleModelRunner:
    """Runs an onnx model with a set of inputs and outputs."""
    def __init__(self, filename, preprocess_function=None, multiple_inputs = False):
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
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        if self.preprocess_function is not None:
            input_data = self.preprocess_function(input_data)

        if self.multiple_inputs:
            
            onnx_input = {}
            onnx_input['input_3'] = input_data[1].astype('float32')
            onnx_input['input_4'] = input_data[0].astype('float32')        

        else:
            onnx_input = {input_name: input_data}

        pred_onnx = sess.run([output_name], onnx_input)[0]
        return pred_onnx
