# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* AgeDetection/*.py api/*.py

black:
	@black scripts/* AgeDetection/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr AgeDetection-*.dist-info
	@rm -fr AgeDetection.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstall:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

PACKAGE_NAME=AgeDetection
FILENAME=trainer

# ----------------------------------
#      Google Cloud Platform
# ----------------------------------

# project id - replace with your Project's ID
PROJECT_ID=wagon-bootcamp-313018

# bucket name - follow the convention of wagon-ml-[YOUR_LAST_NAME]-[BATCH_NUMBER]
BUCKET_NAME=wagon-ml-pereira-566

# Choose your region https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}






run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

predict:
	@python -m ${PACKAGE_NAME}.predict

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload




# ----------------------------------
#           Dockerfile
# ----------------------------------

docker_build:
	@docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} .

docker_run:
	@docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_push:
	@docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_deploy:
	@gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region europe-west1
