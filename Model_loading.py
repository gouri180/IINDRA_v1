from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Singleton class for TrOCR model and processor
class OCRModelSingleton:
    _instance = None

    def __init__(self):
        if OCRModelSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            OCRModelSingleton._instance = self
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

    @staticmethod
    def get_instance():
        if OCRModelSingleton._instance is None:
            OCRModelSingleton()
        return OCRModelSingleton._instance

# Automatically initialize the shared processor and model
ocr_model_instance = OCRModelSingleton.get_instance()
processor_tr_ocr = ocr_model_instance.processor
trocr_model = ocr_model_instance.model
