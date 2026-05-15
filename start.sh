docker build -t embedding-query-python-app .
docker run -it --rm --network="host" --env-file /home/debian/docker/embedding-query/.env --name embedding-query embedding-query-python-app

