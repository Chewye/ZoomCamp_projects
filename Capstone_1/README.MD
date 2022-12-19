# Capstone 1 project

## Problem description

We was predict loss package mail in Russia.

The shipment is packed, transported to a warehouse and transported between sorting points. If the delivery is coming from abroad, then there are additional operations at foreign and Russian customs.

Large logistics hubs help speed up and reduce the cost of delivery. There, small cargoes are sorted and sent to neighboring regions or collected in new containers for shipment to other macro-regions. Thus, postal items travel through the network of sorting centers, like blood through capillaries, and eventually reach any point in the country.
Despite the high level of the security system, there is still a risk of loss or damage to shipments: there are a lot of movements and operations with parcels, in addition, the human factor can play a role in the process.
Accurate prediction of losses and their localization will increase the reliability of the system - guarantee the delivery of items on time and reduce transportation costs. 

We chose CatBoost is a high-performance open source library for gradient boosting on decision trees, because catboost good work with categorical features. Second model we choose classical linear regression for catboost model evaluation. Metrick choosed ROC-AUC, because it good work with disbalance data.

## Project include:
1. [Dataset](https://drive.google.com/file/d/116LQFYRbHTrqB_ui1o7ZfFJUGAajRzng/view?usp=share_link)
2. model.ipynb - Jupiter Notebook with EDA and matching between linear regression and gradient boosting on decision trees (catboost).
3. model.bin - pickled catboost model.
4. thr.bin - pickled best threshold for model.
5. server-flask.py - script for create web service on Flask.
6. request_localhost.py - script for request on web service predict in localhost.
7. Pipfile and Pipfile.lock - dependencies python virtual environment (pipenv).
8. Dockerfile - file fo build docker images with our predict service.
9. [Containerized application](https://hub.docker.com/r/chewye/zoom_camp/tags)

## Run containerized application:
1. Install [docker](https://www.docker.com/)
2. Open linux terminal
3. Pull IMAGE docker `docker pull chewye/zoom_camp:latest`
4. Run image `docker run -it --rm -p 9696:9696 chewye/zoom_camp`
5. For test application run script server-flask.py `python request_localhost.py`

## Build container:
If you want change service and build container:
1. Copy files from github `https://github.com/Chewye/ZoomCamp_projects.git` or use `git@github.com:Chewye/ZoomCamp_projects.git`. 
3. Open folder `Capstone_1` with Linux Terminal.
4. Make changes if need
5. Build docker image by using command `docker build -t <name_images> .`
6. Run image using command `docker run -it --rm -p 9696:9696 <name_images>`

## Cloud deployment:
For cloud i used Virtual Private Server (VPS) on Ubuntu 22

1. Install [docker](https://www.docker.com/) on VPS

2. Pull IMAGE docker`docker pull chewye/zoom_camp:latest`

![pull image in cloud](/Capstone_1/img/Pull.png)

3. Show  pull image in cloud

![show image in cloud](/Capstone_1/img/show.png)

4. Run image `docker run -it --rm -p 9696:9696 chewye/zoom_camp`

![run image in cloud](/Capstone_1/img/run.png)

5. Change request script `request_localhost.py` on your local machine, need change url on IP your VPS (my VPS 79.133.181.183)

![change script](/Capstone_1/img/change_script.png)

6. Run request script on local machine and receive predict `python request_localhost.py`

![result](/Capstone_1/img/result.png)

