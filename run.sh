# gunicorn -b 0.0.0.0:9876 -w 8 -k uvicorn.workers.UvicornWorker search_all_dev:app
uvicorn search_all_dev:app --host 0.0.0.0 --port 9876 --reload
# uvicorn search_all_dev:app --host 0.0.0.0 --port 9876 --worker 8