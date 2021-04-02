import visualkeras
from tensorflow.keras.models import load_model
from PIL import ImageFont
model = load_model('model/model_mnist.h5')

font = ImageFont.truetype("Avenir.ttc", 12)
#visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model, legend=True, to_file='model_mnist.png', font=font) # write to disk
#visualkeras.layered_view(model, to_file='output.png').show() # write and show