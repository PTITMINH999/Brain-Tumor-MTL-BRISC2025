from flask import Flask, render_template, request, redirect
import os
import uuid

from src.inference.pipeline import run_flask_inference
from src.inference.model_loader import load_all_models

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# load model 1 lần
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    load_all_models()


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        # MRI
        file = request.files.get("file")

        if file is None or file.filename == "":
            return redirect(request.url)

        image_filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(
            UPLOAD_FOLDER,
            image_filename
        )

        file.save(image_path)

        # GT MASK
        mask_file = request.files.get("mask")

        mask_path = None

        if mask_file and mask_file.filename != "":

            mask_filename = (
                f"{uuid.uuid4().hex}_mask.png"
            )

            mask_path = os.path.join(
                UPLOAD_FOLDER,
                mask_filename
            )

            mask_file.save(mask_path)

            print("GT MASK SAVED:", mask_path)

        result = run_flask_inference(
            image_path=image_path,
            output_folder=RESULT_FOLDER,
            gt_mask_path=mask_path
        )

        return render_template(
            "result.html",
            image_url=result["image_url"],
            overlay_url=result["overlay_url"],
            summary=result["summary"]
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)