import typer
from codecarbon import OfflineEmissionsTracker

from planktolab.pipeline import run_train_kfolds, run_detect_suspect, run_train, run_resize_images, InferenceOutputType, run_inference


app = typer.Typer()

@app.command()
def resize_images(input_path: str, output_path: str, size: int = 224):
    run_resize_images(input_path, output_path, size)

@app.command()
def run_kfold(
        input_path: str,
        model_name: str,
        kfold: int = 5,
        output_path: str = None,
        max_epoch: int = 20,
        #method: str = "self_confidence",
        #threshold: float = 0.5,
        batch_size: int = 64,
        image_size: int = 128
    ):
    run_train_kfolds(input_path, model_name, kfold, output_path, max_epoch, batch_size, size=image_size)
    #run_detect_suspect(output_path, kfold, max_epoch, method, threshold)

@app.command()
def detect_suspect(
        kfold_path: str,
        output_path: str,
        method: str = "self_confidence",
        threshold: float = 0.5
    ):
    run_detect_suspect(kfold_path, output_path, method, threshold)

@app.command()
def train(
        train_path: str,
        model_name: str,
        output_path: str = None,
        max_epoch: int = 1,
        batch_size:int = 64,
        val_path: str = None,
        test_path: str = None,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        image_size: int = 128,
    ):
    print("hello")
    tracker = OfflineEmissionsTracker(country_iso_code="FRA", log_level="error" )
    tracker.start()

    run_train(train_path, model_name, output_path, max_epoch, batch_size, val_path, test_path, val_ratio, test_ratio, image_size=image_size)

    emissions = tracker.stop()
    print(f"Emissions: {emissions} kg CO2")

@app.command()
def inference(
        model_path: str,
        image_path: str,
        output_path: str,
        image_size: int = 128,
        output_type: InferenceOutputType = InferenceOutputType.csv
    ):

    run_inference(model_path, image_path, output_path=output_path, image_size=image_size, output_type=output_type)

if __name__ == "__main__":
    print("coucuocoucouo")
    app()