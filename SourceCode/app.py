from flask import Flask, render_template, request, redirect
import os
import uuid

from src.inference.pipeline import run_flask_inference
from src.inference.model_loader import load_all_models

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    load_all_models() 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return redirect(request.url)

        # tránh trùng tên file
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = run_flask_inference(filepath, "static/results")


        return render_template(
            "result.html",
            image_url=result["image_url"],
            summary=result["summary"]
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)