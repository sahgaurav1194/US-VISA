"""Quick local entry point: python demo.py trains the full pipeline."""

from us_visa.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    TrainPipeline().run_pipeline()
