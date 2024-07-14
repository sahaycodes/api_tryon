from flask import Flask, request, send_file
from flask_cors import CORS
from gradio_client import Client, handle_file
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gradio Client
client = Client("levihsu/OOTDiffusion")

@app.route("/simple/", methods=["POST"])
def generate_image_simple():
    vton_img = request.files['vton_img']
    garm_img = request.files['garm_img']

    # Save uploaded files temporarily
    vton_img_path = f"temp_{vton_img.filename}"
    garm_img_path = f"temp_{garm_img.filename}"

    vton_img.save(vton_img_path)
    garm_img.save(garm_img_path)

    # Use Gradio client to predict
    result = client.predict(
        vton_img=handle_file(vton_img_path),
        garm_img=handle_file(garm_img_path),
        n_samples=1,
        n_steps=20,
        image_scale=2,
        seed=-1,
        api_name="/process_hd"
    )

    # Extract the image path from the result dictionary
    image_info = result[0]
    image_path = image_info['image']

    # Load the generated image and save it to a temporary file
    output_image_path = 'generated_image.png'
    Image.open(image_path).save(output_image_path)

    # Clean up temporary files
    os.remove(vton_img_path)
    os.remove(garm_img_path)

    # Return the generated image as a response
    return send_file(output_image_path, mimetype='image/png', as_attachment=True)


@app.route("/fullbody/", methods=["POST"])
def generate_image_fullbody():
    vton_img = request.files['vton_img']
    garm_img = request.files['garm_img']
    type = int(request.form.get('type', 1))

    category_input = ['Upper-body', 'Lower-body', 'Dress'][type]

    # Save uploaded files temporarily
    vton_img_path = f"temp_{vton_img.filename}"
    garm_img_path = f"temp_{garm_img.filename}"

    vton_img.save(vton_img_path)
    garm_img.save(garm_img_path)

    # Use Gradio client to predict
    result = client.predict(
        vton_img=handle_file(vton_img_path),
        garm_img=handle_file(garm_img_path),
        n_samples=1,
        category=category_input,
        n_steps=20,
        image_scale=2,
        seed=-1,
        api_name="/process_dc"
    )

    # Extract the image path from the result dictionary
    image_info = result[0]
    image_path = image_info['image']

    # Load the generated image and save it to a temporary file
    output_image_path = 'generated_image.png'
    Image.open(image_path).save(output_image_path)

    # Clean up temporary files
    os.remove(vton_img_path)
    os.remove(garm_img_path)

    # Return the generated image as a response
    return send_file(output_image_path, mimetype='image/png', as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4578, debug=True)

