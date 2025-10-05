# -------- Makefile --------
IMAGE=ai-client-insights
PORT?=8000

# local run (no Docker)
run:
	uvicorn src.app.main:app --host 0.0.0.0 --port $(PORT)

# build container
build:
	docker build -t $(IMAGE) .

# run container (with model cache mounted)
serve:
	docker run -it --rm -p $(PORT):8000 \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		$(IMAGE)

# clean dangling containers/images
clean:
	docker stop $$(docker ps -aq) 2>/dev/null || true
	docker rm $$(docker ps -aq) 2>/dev/null || true
	docker system prune -f

