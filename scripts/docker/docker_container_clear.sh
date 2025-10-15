SERVER_NAME=$1
docker stop $SERVER_NAME && docker rm $SERVER_NAME