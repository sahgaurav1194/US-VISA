"""
FastAPI app exposing the trained model.

GET  /        -> simple HTML form
GET  /train   -> triggers the full training pipeline
POST /        -> reads the form and returns the visa prediction
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from us_visa.constants import APP_HOST, APP_PORT, CURRENT_YEAR
from us_visa.pipeline.prediction_pipeline import USvisaClassifier, USvisaData
from us_visa.pipeline.training_pipeline import TrainPipeline

app = FastAPI(title="US Visa Approval Prediction")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """Pulls the submitted form fields off the request."""

    def __init__(self, request: Request):
        self.request = request

    async def get_usvisa_data(self):
        form = await self.request.form()
        return USvisaData(
            continent=form.get("continent"),
            education_of_employee=form.get("education_of_employee"),
            has_job_experience=form.get("has_job_experience"),
            requires_job_training=form.get("requires_job_training"),
            no_of_employees=int(form.get("no_of_employees")),
            region_of_employment=form.get("region_of_employment"),
            prevailing_wage=float(form.get("prevailing_wage")),
            unit_of_wage=form.get("unit_of_wage"),
            full_time_position=form.get("full_time_position"),
            company_age=CURRENT_YEAR - int(form.get("yr_of_estab")),
        )


@app.get("/", tags=["ui"])
async def index(request: Request):
    return templates.TemplateResponse("usvisa.html", {"request": request, "context": "Fill the form"})


@app.get("/train", tags=["pipeline"])
async def train_route():
    try:
        TrainPipeline().run_pipeline()
        return Response("Training successful!")
    except Exception as e:
        return Response(f"Error occurred: {e}")


@app.post("/", tags=["pipeline"])
async def predict_route(request: Request):
    try:
        form = DataForm(request)
        usvisa_data = await form.get_usvisa_data()
        df = usvisa_data.get_usvisa_input_data_frame()
        status = USvisaClassifier().predict(df)
        return templates.TemplateResponse(
            "usvisa.html", {"request": request, "context": f"Visa Status: {status}"}
        )
    except Exception as e:
        return Response(f"Error occurred: {e}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
